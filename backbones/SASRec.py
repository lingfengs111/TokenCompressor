"""SASRec backbone with meta-patch, projection head, and optional PEFT adapters."""

import math
import logging
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from backbones.patch import MetaPatch
from backbones.modules import (
    apply_head as apply_head_fn,
    expand_shared_token_bank,
    ScoreCalibrationHead,
    head_parameters as head_parameters_fn,
    sequence_summary,
    shared_token_init_std_from_config,
    shared_token_len_from_config,
    strip_leading_tokens,
)

logger = logging.getLogger("train-sasrec-meta-patch")


def _sdpa_kernel_context(enable_flash: bool, enable_mem_efficient: bool, enable_math: bool):
    try:
        from torch.nn.attention import sdpa_kernel

        return sdpa_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem_efficient,
            enable_math=enable_math,
        )
    except Exception:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem_efficient,
            enable_math=enable_math,
        )


def _parse_block_selection(spec: str, num_blocks: int) -> Set[int]:
    text = str(spec or "none").strip().lower()
    if not text or text == "none" or num_blocks <= 0:
        return set()
    if text == "all":
        return set(range(num_blocks))

    selected: Set[int] = set()
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        if token == "first":
            selected.add(0)
            continue
        if token == "last":
            selected.add(num_blocks - 1)
            continue
        idx = int(token)
        if idx < 0:
            idx += num_blocks
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"attn_lora_blocks index out of range: {token}")
        selected.add(idx)
    return selected


def _normalize_attention_norm(value: str) -> str:
    norm = str(value or "softmax").strip().lower()
    aliases = {
        "sdpa": "softmax",
        "sm": "softmax",
        "custom": "softmax_custom",
        "manual_softmax": "softmax_custom",
        "softmax_manual": "softmax_custom",
        "sm1": "softmax1",
        "softmax_1": "softmax1",
        "softmax+1": "softmax1",
    }
    norm = aliases.get(norm, norm)
    if norm not in {"softmax", "softmax_custom", "softmax1"}:
        raise ValueError(f"Unsupported SASRec attention_norm: {value}")
    return norm


class RelativePositionalBias(nn.Module):
    """Learned relative-position bias table, matching the HSTU parameterization."""

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("RelativePositionalBias requires max_seq_len > 0.")
        self.max_seq_len = int(max_seq_len)
        self.bias = nn.Parameter(torch.empty(2 * self.max_seq_len - 1))
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}.")
        t = F.pad(self.bias[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(1, seq_len, 3 * seq_len - 2)
        radius = (2 * seq_len - 1) // 2
        return t[..., radius:-radius].squeeze(0).to(dtype=dtype)


def _custom_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    extra_bias: Optional[torch.Tensor],
    is_causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Scaled dot-product attention with standard softmax using the manual softmax1 path."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if extra_bias is not None:
        scores = scores + extra_bias.to(dtype=scores.dtype)

    valid_mask: Optional[torch.Tensor] = None
    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        if attn_mask.dtype == torch.bool:
            valid_mask = attn_mask
        else:
            attn_bias = attn_mask.to(dtype=scores.dtype)
            scores = scores + attn_bias
            mask_floor = torch.finfo(attn_bias.dtype).min / 2
            valid_mask = attn_bias > mask_floor
    elif is_causal:
        q_len = q.size(-2)
        kv_len = k.size(-2)
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(-1)
        k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
        valid_mask = (k_pos <= q_pos).view(1, 1, q_len, kv_len)

    scores = scores.to(torch.float32)
    if valid_mask is not None:
        valid_mask = valid_mask.to(dtype=torch.bool)
        scores = torch.where(valid_mask, scores, torch.full_like(scores, float("-inf")))

    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max)
    if valid_mask is not None:
        exp_scores = torch.where(valid_mask, exp_scores, torch.zeros_like(exp_scores))
    denom = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, exp_scores / denom.clamp_min(torch.finfo(exp_scores.dtype).tiny), exp_scores)
    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p, training=training)
    return torch.matmul(probs.to(dtype=v.dtype), v)


def _softmax1_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    extra_bias: Optional[torch.Tensor],
    is_causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Scaled dot-product attention with softmax1 normalization."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if extra_bias is not None:
        scores = scores + extra_bias.to(dtype=scores.dtype)

    valid_mask: Optional[torch.Tensor] = None
    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        if attn_mask.dtype == torch.bool:
            valid_mask = attn_mask
        else:
            attn_bias = attn_mask.to(dtype=scores.dtype)
            scores = scores + attn_bias
            mask_floor = torch.finfo(attn_bias.dtype).min / 2
            valid_mask = attn_bias > mask_floor
    elif is_causal:
        q_len = q.size(-2)
        kv_len = k.size(-2)
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(-1)
        k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
        valid_mask = (k_pos <= q_pos).view(1, 1, q_len, kv_len)

    scores = scores.to(torch.float32)
    if valid_mask is not None:
        valid_mask = valid_mask.to(dtype=torch.bool)
        scores = torch.where(valid_mask, scores, torch.full_like(scores, float("-inf")))

    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max)
    if valid_mask is not None:
        exp_scores = torch.where(valid_mask, exp_scores, torch.zeros_like(exp_scores))
    probs = exp_scores / (1.0 + exp_scores.sum(dim=-1, keepdim=True))
    if dropout_p > 0.0:
        probs = F.dropout(probs, p=dropout_p, training=training)
    return torch.matmul(probs.to(dtype=v.dtype), v)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q/k using current sequence slots."""
    head_dim = q.size(-1)
    rope_dim = head_dim - (head_dim % 2)
    if rope_dim <= 0:
        return q, k
    positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (float(base) ** (torch.arange(0, rope_dim, 2, device=q.device, dtype=torch.float32) / rope_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype=q.dtype).view(1, 1, q.size(-2), rope_dim // 2)
    sin = freqs.sin().to(dtype=q.dtype).view(1, 1, q.size(-2), rope_dim // 2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_rope = x[..., :rope_dim]
        x_even = x_rope[..., 0::2]
        x_odd = x_rope[..., 1::2]
        x_rot = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        if rope_dim == head_dim:
            return x_rot
        return torch.cat([x_rot, x[..., rope_dim:]], dim=-1)

    return rotate(q), rotate(k)


class LowRankLinearAdapter(nn.Module):
    """LoRA-style low-rank delta initialized to preserve the base linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LowRankLinearAdapter rank must be > 0.")
        self.rank = int(rank)
        self.scale = float(alpha if alpha > 0 else rank) / float(rank)
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


class InputEmbeddingDelta(nn.Module):
    """Input-only low-rank embedding delta so scorer embeddings stay anchored."""

    def __init__(self, num_embeddings: int, hidden_units: int, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("InputEmbeddingDelta rank must be > 0.")
        self.scale = float(alpha if alpha > 0 else rank) / float(rank)
        self.rows = nn.Embedding(num_embeddings, rank, padding_idx=0)
        self.proj = nn.Linear(rank, hidden_units, bias=False)
        nn.init.normal_(self.rows.weight, mean=0.0, std=0.02)
        if self.rows.padding_idx is not None:
            with torch.no_grad():
                self.rows.weight[self.rows.padding_idx].zero_()
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.rows(input_ids)) * self.scale


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(
        self,
        hidden_units: int,
        num_heads: int,
        dropout_rate: float,
        use_flash_attention: bool = True,
        attention_norm: str = "softmax",
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        enable_relative_attention_bias: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_units % num_heads == 0

        # Combined QKV projection
        self.c_attn = nn.Linear(hidden_units, 3 * hidden_units)
        # Output projection
        self.c_proj = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.use_flash_attention = use_flash_attention
        self.attention_norm = _normalize_attention_norm(attention_norm)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self._flash_fallback_warned = False
        self.rel_pos_bias = (
            RelativePositionalBias(max_length)
            if enable_relative_attention_bias and max_length is not None
            else None
        )
        self.c_attn_lora = (
            LowRankLinearAdapter(hidden_units, 3 * hidden_units, lora_rank, lora_alpha) if lora_rank > 0 else None
        )
        self.c_proj_lora = (
            LowRankLinearAdapter(hidden_units, hidden_units, lora_rank, lora_alpha) if lora_rank > 0 else None
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, hidden units

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        if self.c_attn_lora is not None:
            qkv = qkv + self.c_attn_lora(x)
        q, k, v = qkv.split(self.hidden_units, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        if self.use_rope:
            q, k = _apply_rope(q, k, base=self.rope_base)
        rel_bias = None
        if self.rel_pos_bias is not None:
            rel_bias = self.rel_pos_bias(T, dtype=q.dtype).view(1, 1, T, T)

        # Causal self-attention (prefer flash when available; disable under forward AD)
        try:
            from torch.autograd import forward_ad

            fwd_ad_enabled = forward_ad.is_enabled()
        except Exception:
            fwd_ad_enabled = False

        use_causal = attn_mask is None
        attn_bias = attn_mask

        if self.attention_norm == "softmax_custom":
            y = _custom_softmax_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                extra_bias=rel_bias,
                is_causal=use_causal,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                training=self.training,
            )
        elif self.attention_norm == "softmax1":
            y = _softmax1_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                extra_bias=rel_bias,
                is_causal=use_causal,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                training=self.training,
            )
        else:
            if rel_bias is not None:
                y = _custom_softmax_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_bias,
                    extra_bias=rel_bias,
                    is_causal=use_causal,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    training=self.training,
                )
            elif q.is_cuda and self.use_flash_attention and not fwd_ad_enabled and attn_bias is None:
                try:
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=use_causal, dropout_p=self.attn_dropout.p if self.training else 0.0
                    )
                except RuntimeError:
                    if not self._flash_fallback_warned:
                        logger.warning("Flash attention failed; falling back to math kernel.")
                        self._flash_fallback_warned = True
                    with _sdpa_kernel_context(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        y = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_bias, is_causal=use_causal, dropout_p=self.attn_dropout.p if self.training else 0.0
                        )
            elif q.is_cuda:
                with _sdpa_kernel_context(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_bias, is_causal=use_causal, dropout_p=self.attn_dropout.p if self.training else 0.0
                    )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, is_causal=use_causal, dropout_p=self.attn_dropout.p if self.training else 0.0
                )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        proj_input = y
        y = self.c_proj(proj_input)
        if self.c_proj_lora is not None:
            y = y + self.c_proj_lora(proj_input)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network)."""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-LN architecture."""

    def __init__(
        self,
        hidden_units: int,
        num_heads: int,
        dropout_rate: float,
        use_flash_attention: bool = True,
        attention_norm: str = "softmax",
        attn_lora_rank: int = 0,
        attn_lora_alpha: float = 1.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        enable_relative_attention_bias: bool = False,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = CausalSelfAttention(
            hidden_units,
            num_heads,
            dropout_rate,
            use_flash_attention=use_flash_attention,
            attention_norm=attention_norm,
            lora_rank=attn_lora_rank,
            lora_alpha=attn_lora_alpha,
            use_rope=use_rope,
            rope_base=rope_base,
            enable_relative_attention_bias=enable_relative_attention_bias,
            max_length=max_length,
        )
        self.ln_2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = MLP(hidden_units, dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: LayerNorm -> Sub-layer -> Residual
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model."""

    def __init__(self, config, item_num: int):
        super().__init__()

        self.config = config  # Store config for later use
        self.item_num = item_num
        self.max_seq_length = config.max_seq_length
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len
        self.attention_norm = _normalize_attention_norm(getattr(config, "sasrec_attention_norm", "softmax"))
        self.use_rope = bool(getattr(config, "sasrec_use_rope", False))
        self.rope_base = float(getattr(config, "sasrec_rope_base", 10000.0))
        self.enable_relative_attention_bias = bool(getattr(config, "sasrec_enable_relative_attention_bias", False))
        self.persrec_enable = bool(getattr(config, "persrec_enable", False))
        self.persrec_num_tokens = int(getattr(config, "persrec_num_tokens", 0) or 0)
        self.persrec_pretrain_len = int(getattr(config, "persrec_pretrain_len", 0) or 0)
        self.persrec_recent_len = int(getattr(config, "persrec_recent_len", 0) or 0)
        self.shared_prefix_len = shared_token_len_from_config(config)
        self.shared_prefix_init_std = shared_token_init_std_from_config(config)

        # Embedding layers
        # +2 to reserve PAD=0 and UNK=1
        self.item_emb = nn.Embedding(item_num + 2, config.hidden_units, padding_idx=0)
        input_emb_lora_rank = int(getattr(config, "input_emb_lora_rank", 0) or 0)
        input_emb_lora_alpha = float(getattr(config, "input_emb_lora_alpha", input_emb_lora_rank or 1.0))
        self.input_emb_lora = (
            InputEmbeddingDelta(item_num + 2, config.hidden_units, input_emb_lora_rank, input_emb_lora_alpha)
            if input_emb_lora_rank > 0
            else None
        )
        if self.shared_prefix_len > 0:
            self.shared_prefix_tokens = nn.Parameter(
                torch.randn(1, self.shared_prefix_len, self.hidden_units) * self.shared_prefix_init_std
            )
        else:
            self.register_parameter("shared_prefix_tokens", None)
        if self.persrec_enable:
            if self.shared_prefix_len > 0:
                raise ValueError("shared_prefix_len is not supported together with persrec_enable.")
            if self.persrec_num_tokens <= 0:
                raise ValueError("persrec_num_tokens must be > 0 when persrec_enable=True.")
            if self.persrec_pretrain_len <= 0 or self.persrec_recent_len <= 0:
                raise ValueError("persrec_pretrain_len/recent_len must be set when persrec_enable=True.")
            if self.persrec_pretrain_len + self.persrec_recent_len != self.max_seq_length:
                logger.warning(
                    "PersRec lengths mismatch: pretrain(%s)+recent(%s) != max_seq_length(%s).",
                    self.persrec_pretrain_len,
                    self.persrec_recent_len,
                    self.max_seq_length,
                )
            self.pos_emb = nn.Embedding(
                config.max_seq_length + self.shared_prefix_len + self.persrec_num_tokens + 1,
                config.hidden_units,
                padding_idx=0,
            )
            self.persrec_tokens = nn.Parameter(
                torch.randn(1, self.persrec_num_tokens, self.hidden_units)
            )
        else:
            self.pos_emb = nn.Embedding(
                config.max_seq_length + self.shared_prefix_len + 1,
                config.hidden_units,
                padding_idx=0,
            )
        self.score_calibration = ScoreCalibrationHead(
            item_num + 2,
            max_seq_length=self.max_seq_length,
            enable_item_bias=bool(getattr(config, "enable_score_item_bias", False)),
            enable_length_bias=bool(getattr(config, "enable_score_length_bias", False)),
            enable_length_scale=bool(getattr(config, "enable_score_length_scale", False)),
            bucket_size=int(getattr(config, "score_length_bucket_size", 20) or 20),
        )
        attn_max_length = (
            self.max_seq_length
            + self.shared_prefix_len
            + (self.persrec_num_tokens if self.persrec_enable else self.patch_len)
        )
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # Transformer blocks
        attn_lora_rank = int(getattr(config, "attn_lora_rank", 0) or 0)
        attn_lora_alpha = float(getattr(config, "attn_lora_alpha", attn_lora_rank or 1.0))
        attn_lora_blocks = _parse_block_selection(
            getattr(config, "attn_lora_blocks", "all" if attn_lora_rank > 0 else "none"),
            config.num_blocks,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_units,
                    config.num_heads,
                    config.dropout_rate,
                    use_flash_attention=config.use_flash_attention,
                    attention_norm=self.attention_norm,
                    attn_lora_rank=attn_lora_rank if i in attn_lora_blocks else 0,
                    attn_lora_alpha=attn_lora_alpha,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                    enable_relative_attention_bias=self.enable_relative_attention_bias,
                    max_length=attn_max_length,
                )
                for i in range(config.num_blocks)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_units, eps=1e-8)

        # Trainable projection head (only head is adapted in inner loop)
        self.proj_linear = nn.Linear(config.hidden_units, config.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(config.hidden_units, eps=1e-8)

        # Meta-patch module (outer loop)
        self.meta_patch = MetaPatch(config)

        # Initialize weights
        self.apply(self._init_weights)
        self.score_calibration.reset_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Don't initialize padding_idx
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def _freeze_backbone(self) -> None:
        """Freeze the transformer body (backbone)."""
        for p in self.item_emb.parameters():
            p.requires_grad = False
        for p in self.score_calibration.parameters():
            p.requires_grad = False
        if self.input_emb_lora is not None:
            for p in self.input_emb_lora.parameters():
                p.requires_grad = False
        if self.shared_prefix_tokens is not None:
            self.shared_prefix_tokens.requires_grad_(False)
        for p in self.pos_emb.parameters():
            p.requires_grad = False
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.ln_f.parameters():
            p.requires_grad = False

    def _input_item_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        item_embs = self.item_emb(input_ids)
        if self.input_emb_lora is not None:
            item_embs = item_embs + self.input_emb_lora(input_ids)
        return item_embs

    def _shared_token_count(self) -> int:
        if self.shared_prefix_tokens is None:
            return 0
        return int(self.shared_prefix_tokens.size(1))

    def _shared_token_states(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        shared_len = self._shared_token_count()
        if shared_len <= 0:
            return None
        shared_states = expand_shared_token_bank(
            self.shared_prefix_tokens,
            batch_size,
            device=device,
            dtype=dtype,
        )
        positions = torch.arange(1, shared_len + 1, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        return shared_states + self.pos_emb(positions).to(dtype)

    def _build_item_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
        use_prefix_tail_pos = bool(getattr(self.config, "prefix_tail_positions", False))
        if use_prefix_tail_pos and prefix_len > 0 and seq_length <= self.max_seq_length:
            prefix_len = min(prefix_len, seq_length)
            tail_len = max(seq_length - prefix_len, 0)
            pos_custom = torch.zeros_like(positions)
            if prefix_len > 0:
                prefix_pos = torch.arange(1, prefix_len + 1, dtype=torch.long, device=input_ids.device)
                pos_custom[:, :prefix_len] = prefix_pos
            if tail_len > 0:
                tail_start = max(self.max_seq_length - tail_len + 1, 1)
                tail_pos = torch.arange(tail_start, tail_start + tail_len, dtype=torch.long, device=input_ids.device)
                pos_custom[:, prefix_len:] = tail_pos
            positions = pos_custom
        elif getattr(self.config, "right_align_positions", True):
            offset = max(self.max_seq_length - seq_length, 0)
            if offset > 0:
                positions = positions + offset
        positions = positions * (input_ids != 0).long()
        shared_len = self._shared_token_count()
        if shared_len > 0:
            positions = torch.where(positions > 0, positions + shared_len, positions)
        return positions

    def _build_patch_positions(self, input_ids: torch.Tensor, item_positions: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        if self.patch_len <= 0:
            return item_positions.new_zeros((batch_size, 0))
        shared_len = self._shared_token_count()
        max_pos = self.max_seq_length + shared_len
        min_start = shared_len + 1
        max_start = max(min_start, max_pos - self.patch_len + 1)
        large = max_pos + self.patch_len + 1

        prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
        patch_after_prefix = bool(getattr(self.config, "patch_after_prefix", False))
        if patch_after_prefix and prefix_len > 0:
            prefix_eff = min(prefix_len, item_positions.size(1))
            prefix_part = item_positions[:, :prefix_eff]
            tail_part = item_positions[:, prefix_eff:]
            prefix_last = (
                prefix_part.max(dim=1).values
                if prefix_part.numel() > 0
                else item_positions.new_full((batch_size,), shared_len)
            )
            if tail_part.numel() > 0:
                tail_min = torch.where(tail_part > 0, tail_part, torch.full_like(tail_part, large)).min(dim=1).values
                start = torch.minimum(prefix_last + 1, tail_min - self.patch_len)
            else:
                start = prefix_last + 1
        else:
            first_item = torch.where(item_positions > 0, item_positions, torch.full_like(item_positions, large)).min(dim=1).values
            fallback = item_positions.new_full((batch_size,), max_start)
            start = torch.where(first_item < large, first_item - self.patch_len, fallback)

        start = torch.clamp(start, min=min_start, max=max_start)
        offsets = torch.arange(self.patch_len, dtype=item_positions.dtype, device=item_positions.device)
        return start.unsqueeze(1) + offsets.unsqueeze(0)

    def _patch_token_start(self, seq_length: int) -> int:
        start = self._shared_token_count()
        prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
        if bool(getattr(self.config, "patch_after_prefix", False)) and prefix_len > 0:
            start += min(prefix_len, seq_length)
        return start

    def _strip_patch_tokens(self, hidden_states: torch.Tensor, seq_length: int) -> torch.Tensor:
        if self.patch_len <= 0:
            return hidden_states
        patch_start = self._patch_token_start(seq_length)
        patch_end = min(hidden_states.size(1), patch_start + self.patch_len)
        return torch.cat([hidden_states[:, :patch_start, :], hidden_states[:, patch_end:, :]], dim=1)

    def _strip_shared_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return strip_leading_tokens(hidden_states, self._shared_token_count())

    def _sequence_summary(self, item_embs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return last-item embedding and sequence lengths."""
        pool = getattr(self.config, "gating_pool", "last")
        return sequence_summary(item_embs, input_ids, pool=pool)

    def _build_persrec_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Use _build_persrec_attention_mask_with_len instead.")

    def _build_persrec_attention_mask_with_len(
        self, input_ids: torch.Tensor, pretrain_len: int
    ) -> torch.Tensor:
        if not self.persrec_enable:
            raise RuntimeError("PersRec attention mask requested but persrec is disabled.")
        if self.persrec_num_tokens <= 0:
            raise RuntimeError("PersRec requires persrec_num_tokens > 0.")
        device = input_ids.device
        dtype = self.item_emb.weight.dtype
        batch_size, seq_len = input_ids.size()
        total_len = seq_len + self.persrec_num_tokens

        item_valid = input_ids != 0
        token_valid = torch.ones((batch_size, self.persrec_num_tokens), device=device, dtype=torch.bool)
        valid = torch.cat(
            [
                item_valid[:, : pretrain_len],
                token_valid,
                item_valid[:, pretrain_len :],
            ],
            dim=1,
        )

        base = torch.tril(torch.ones((total_len, total_len), device=device, dtype=torch.bool))
        mask = base.unsqueeze(0).unsqueeze(1)
        mask = mask & valid.unsqueeze(1).unsqueeze(2)
        mask = mask & valid.unsqueeze(1).unsqueeze(3)

        pre_end = pretrain_len - 1
        recent_start = pretrain_len + self.persrec_num_tokens
        if recent_start < total_len and pre_end >= 0:
            mask[:, :, recent_start:, : pre_end + 1] = False

        attn_bias = torch.where(mask, 0.0, torch.finfo(dtype).min)
        return attn_bias

    def _build_persrec_embeddings(
        self, item_embs: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = input_ids.size()
        pretrain_len, _ = self._resolve_persrec_lengths(seq_length)
        if self.persrec_pretrain_len + self.persrec_recent_len != seq_length:
            logger.warning(
                "PersRec expects seq_len=%s (pretrain=%s + recent=%s) but got %s; using pretrain_len=%s.",
                self.persrec_pretrain_len + self.persrec_recent_len,
                self.persrec_pretrain_len,
                self.persrec_recent_len,
                seq_length,
                pretrain_len,
            )

        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        offset = 0
        if getattr(self.config, "right_align_positions", True):
            offset = max(self.max_seq_length - seq_length, 0)
            if offset > 0:
                positions = positions + offset
        positions = positions * (input_ids != 0).long()

        if self.persrec_num_tokens > 0:
            recent_pos = positions[:, pretrain_len :]
            recent_pos = torch.where(recent_pos > 0, recent_pos + self.persrec_num_tokens, recent_pos)
            positions = torch.cat(
                [positions[:, : pretrain_len], recent_pos], dim=1
            )

        pos_embs = self.pos_emb(positions)
        item_with_pos = item_embs + pos_embs

        token_positions = (
            torch.arange(1, self.persrec_num_tokens + 1, dtype=torch.long, device=input_ids.device)
            + offset
            + pretrain_len
        )
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        token_embs = self.persrec_tokens.expand(batch_size, -1, -1) + self.pos_emb(token_positions)

        pre = item_with_pos[:, : pretrain_len, :]
        recent = item_with_pos[:, pretrain_len :, :]
        hidden_states = torch.cat([pre, token_embs, recent], dim=1)
        attn_mask = self._build_persrec_attention_mask_with_len(input_ids, pretrain_len)
        return hidden_states, attn_mask

    def _build_persrec_target_sequence(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return raw item/token targets and an item-only loss mask.

        This mirrors the original PerSRec training setup more closely:
        the flattened sequence is [pretrain items, learnable tokens, recent items],
        query positions correspond to that flattened sequence, and only item
        positions participate in the loss. The final item position is masked out
        because it has no next target.
        """
        batch_size, seq_length = input_ids.size()
        pretrain_len, _ = self._resolve_persrec_lengths(seq_length)
        item_embs = self.item_emb(input_ids)
        token_embs = self.persrec_tokens.expand(batch_size, -1, -1)
        targets = torch.cat(
            [item_embs[:, :pretrain_len, :], token_embs, item_embs[:, pretrain_len:, :]],
            dim=1,
        )

        item_valid = input_ids != 0
        token_mask = torch.zeros(
            (batch_size, self.persrec_num_tokens), dtype=torch.bool, device=input_ids.device
        )
        loss_mask = torch.cat(
            [item_valid[:, :pretrain_len], token_mask, item_valid[:, pretrain_len:]],
            dim=1,
        )
        if loss_mask.numel() > 0:
            has_valid = loss_mask.any(dim=1)
            last_from_end = loss_mask.flip(1).float().argmax(dim=1)
            last_idx = loss_mask.size(1) - 1 - last_from_end
            loss_mask = loss_mask.clone()
            loss_mask[has_valid, last_idx[has_valid]] = False
        return targets, loss_mask, pretrain_len

    def _build_persrec_negative_sequence(
        self, neg_ids: torch.Tensor, pretrain_len: int
    ) -> torch.Tensor:
        """Insert dummy token slots into negative embeddings to match query layout."""
        neg_embs = self.item_emb(neg_ids)
        if neg_embs.dim() == 4:
            filler = neg_embs.new_zeros(
                neg_embs.size(0), self.persrec_num_tokens, neg_embs.size(2), neg_embs.size(3)
            )
        else:
            filler = neg_embs.new_zeros(neg_embs.size(0), self.persrec_num_tokens, neg_embs.size(2))
        return torch.cat([neg_embs[:, :pretrain_len], filler, neg_embs[:, pretrain_len:]], dim=1)

    def _resolve_persrec_lengths(self, seq_length: int) -> Tuple[int, int]:
        recent_len = min(self.persrec_recent_len, seq_length)
        pretrain_len = max(seq_length - recent_len, 0)
        return pretrain_len, recent_len

    def head_parameters(self) -> List[torch.Tensor]:
        return head_parameters_fn(self.config, self.proj_linear, self.proj_ln)

    def apply_head(self, hidden_states: torch.Tensor, head_params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return apply_head_fn(hidden_states, self.config, self.proj_linear, self.proj_ln, head_params=head_params)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            input_ids: Item sequences [batch_size, seq_length]

        Returns:
            Hidden states [batch_size, seq_length, hidden_units]
        """
        batch_size, seq_length = input_ids.size()

        # Item embeddings
        item_embs = self._input_item_embeddings(input_ids)
        item_embs *= self.hidden_units**0.5  # Scale by sqrt(d) as in Transformer

        # Patch embeddings (meta-patch)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary, user_ids=user_ids)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(
                    seq_summary, patch_params, user_ids=user_ids
                )
        else:
            patch_emb = item_embs.new_zeros((batch_size, 0, self.hidden_units))
            gating_weights = None

        attn_mask = None
        if self.persrec_enable:
            if use_patch and self.patch_len > 0:
                raise ValueError("PersRec is not supported together with meta-patch in this implementation.")
            hidden_states, attn_mask = self._build_persrec_embeddings(item_embs, input_ids)
        else:
            # Positional embeddings
            positions = self._build_item_positions(input_ids)
            shared_states = self._shared_token_states(batch_size, device=input_ids.device, dtype=item_embs.dtype)

            if use_patch and self.patch_len > 0:
                pos_embs_item = self.pos_emb(positions)
                item_with_pos = item_embs + pos_embs_item
                if bool(getattr(self.config, "patch_use_position_embeddings", False)):
                    patch_positions = self._build_patch_positions(input_ids, positions)
                    patch_emb = patch_emb + self.pos_emb(patch_positions).to(patch_emb.dtype)
                prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
                patch_after_prefix = bool(getattr(self.config, "patch_after_prefix", False))
                if patch_after_prefix and prefix_len > 0:
                    prefix_len = min(prefix_len, seq_length)
                    prefix_part = item_with_pos[:, :prefix_len, :]
                    tail_part = item_with_pos[:, prefix_len:, :]
                    pieces = []
                    if shared_states is not None:
                        pieces.append(shared_states)
                    pieces.extend([prefix_part, patch_emb, tail_part])
                    hidden_states = torch.cat(pieces, dim=1)
                else:
                    pieces = []
                    if shared_states is not None:
                        pieces.append(shared_states)
                    pieces.extend([patch_emb, item_with_pos])
                    hidden_states = torch.cat(pieces, dim=1)
            else:
                pos_embs = self.pos_emb(positions)
                hidden_states = item_embs + pos_embs
                if shared_states is not None:
                    hidden_states = torch.cat([shared_states, hidden_states], dim=1)

        hidden_states = self.emb_dropout(hidden_states)

        # Pass through transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = gradient_checkpoint(block, hidden_states, attn_mask)
            else:
                hidden_states = block(hidden_states, attn_mask=attn_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if return_gating:
            return hidden_states, gating_weights
        return hidden_states

    def get_gating_weights(
        self,
        input_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        item_embs = self._input_item_embeddings(input_ids)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if patch_params is None:
            _, weights = self.meta_patch(seq_summary, user_ids=user_ids)
        else:
            _, weights = self.meta_patch.forward_with_eta(seq_summary, patch_params, user_ids=user_ids)
        return weights

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        neg_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ):
        if pos_ids is not None and neg_ids is not None:
            return self.training_step(
                input_ids,
                pos_ids,
                neg_ids,
                patch_params=patch_params,
                user_ids=user_ids,
                return_gating=return_gating,
                use_patch=use_patch,
            )
        return self.forward_features(
            input_ids,
            user_ids=user_ids,
            patch_params=patch_params,
            return_gating=return_gating,
            use_patch=use_patch,
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        user_ids: Optional[torch.Tensor] = None,
        use_patch: bool = True,
        use_head: bool = True,
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            candidate_ids: Candidate items to score [batch_size, num_candidates]

        Returns:
            Scores for each candidate [batch_size, num_candidates]
        """
        hidden_states = self.forward_features(
            input_ids, user_ids=user_ids, patch_params=patch_params, use_patch=use_patch
        )
        if use_patch and self.patch_len > 0:
            hidden_states = self._strip_patch_tokens(hidden_states, input_ids.size(1))
        hidden_states = self._strip_shared_tokens(hidden_states)
        if self.persrec_enable and self.persrec_num_tokens > 0:
            pre, _ = self._resolve_persrec_lengths(input_ids.size(1))
            k = self.persrec_num_tokens
            hidden_states = torch.cat([hidden_states[:, :pre, :], hidden_states[:, pre + k :, :]], dim=1)
        final_hidden = hidden_states[:, -1, :]  # [B, H]
        if use_head:
            final_hidden = self.apply_head(final_hidden, head_params=head_params)
        candidate_embs = self.item_emb(candidate_ids)  # [B, C, H]
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)  # [B, C]
        seq_lengths = (input_ids != 0).sum(dim=1)
        scores = self.score_calibration(scores, candidate_ids, seq_lengths)
        return scores

    def training_step(
        self,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        user_ids: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
        return_loss_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Training step with positive and negative items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            pos_ids: Positive items (next items) [batch_size, seq_length]
            neg_ids: Negative items (sampled) [batch_size, seq_length]

        Returns:
            pos_logits: Scores for positive items [batch_size, seq_length]
            neg_logits: Scores for negative items [batch_size, seq_length]
        """
        hidden_states, gating_weights = self.forward_features(
            input_ids,
            user_ids=user_ids,
            patch_params=patch_params,
            return_gating=True,
            use_patch=use_patch,
        )
        if use_patch and self.patch_len > 0:
            hidden_states = self._strip_patch_tokens(hidden_states, input_ids.size(1))
        hidden_states = self._strip_shared_tokens(hidden_states)
        projected = self.apply_head(hidden_states, head_params=head_params)
        if self.persrec_enable and self.persrec_num_tokens > 0:
            target_seq, loss_mask, pretrain_len = self._build_persrec_target_sequence(input_ids)
            pos_embs = projected.new_zeros(projected.shape)
            pos_embs[:, :-1, :] = target_seq[:, 1:, :].to(projected.dtype)
            neg_embs = self._build_persrec_negative_sequence(neg_ids, pretrain_len).to(projected.dtype)
        else:
            pos_embs = self.item_emb(pos_ids).to(projected.dtype)  # [B, T, H]
            neg_embs = self.item_emb(neg_ids).to(projected.dtype)  # [B, T, H] or [B, T, K, H]
            loss_mask = pos_ids != 0

        pos_logits = (projected * pos_embs).sum(dim=-1)  # [B, T]
        if neg_embs.dim() == 4:
            neg_logits = (projected.unsqueeze(2) * neg_embs).sum(dim=-1)  # [B, T, K]
        else:
            neg_logits = (projected * neg_embs).sum(dim=-1)  # [B, T]
        seq_lengths = (input_ids != 0).sum(dim=1)
        pos_logits = self.score_calibration(pos_logits, pos_ids, seq_lengths)
        neg_logits = self.score_calibration(neg_logits, neg_ids, seq_lengths)

        if return_gating and return_loss_mask:
            return pos_logits, neg_logits, gating_weights, loss_mask
        if return_loss_mask:
            return pos_logits, neg_logits, loss_mask
        if return_gating:
            return pos_logits, neg_logits, gating_weights
        return pos_logits, neg_logits
