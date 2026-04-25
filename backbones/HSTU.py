"""Dense HSTU backbone adapted for ID-only soft-patch training."""

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

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
from backbones.patch import MetaPatch

logger = logging.getLogger("train-hstu-meta-patch")


class RelativePositionalBias(nn.Module):
    """Relative positional bias matching the original HSTU parameterization."""

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("RelativePositionBias requires max_seq_len > 0.")
        self.max_seq_len = int(max_seq_len)
        self.bias = nn.Parameter(torch.empty(2 * self.max_seq_len - 1))
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}.")
        # self.bias 只存 1 个长度为 2*max_seq_len-1 的相对距离表。
        # 这里把它展开成当前序列长度对应的 [T, T] 相对位置偏置矩阵。
        t = F.pad(self.bias[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(1, seq_len, 3 * seq_len - 2)
        radius = (2 * seq_len - 1) // 2
        return t[..., radius:-radius].squeeze(0).to(dtype=dtype)


class HSTUBlock(nn.Module):
    """Single dense HSTU block."""

    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        linear_activation: str = "silu",
        normalization: str = "rel_bias",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        enable_relative_attention_bias: bool = False,
        max_length: Optional[int] = None,
        parametric_block_norm: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if linear_hidden_dim <= 0:
            raise ValueError("linear_hidden_dim must be > 0.")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if linear_activation not in {"silu", "none"}:
            raise ValueError(f"Unsupported linear_activation: {linear_activation}")
        if normalization not in {"rel_bias", "hstu_rel_bias", "softmax_rel_bias", "softmax1_rel_bias"}:
            raise ValueError(f"Unsupported normalization: {normalization}")

        self.embedding_dim = int(embedding_dim)
        self.linear_dim = int(linear_hidden_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.dropout_ratio = float(dropout_ratio)
        self.linear_activation = linear_activation
        self.normalization = normalization
        self.concat_ua = bool(concat_ua)
        self.epsilon = float(epsilon)
        self.parametric_block_norm = bool(parametric_block_norm)

        out_dim = self.linear_dim * self.num_heads * (3 if self.concat_ua else 1)
        qkv_dim = self.linear_dim * 2 * self.num_heads + self.attention_dim * 2 * self.num_heads
        self.uvqk = nn.Parameter(torch.empty(self.embedding_dim, qkv_dim))
        nn.init.normal_(self.uvqk, mean=0.0, std=0.02)

        self.o = nn.Linear(out_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.o.weight)
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)

        self.attn_dropout = nn.Dropout(attn_dropout_ratio)
        self.rel_pos_bias = (
            RelativePositionalBias(max_length) if enable_relative_attention_bias and max_length is not None else None
        )
        if self.parametric_block_norm:
            self.norm_input = nn.LayerNorm(self.embedding_dim, eps=self.epsilon)
            self.norm_attn_output = nn.LayerNorm(self.linear_dim * self.num_heads, eps=self.epsilon)
        else:
            self.norm_input = None
            self.norm_attn_output = None

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_input is not None:
            return self.norm_input(x)
        return F.layer_norm(x, normalized_shape=(self.embedding_dim,), eps=self.epsilon)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_attn_output is not None:
            return self.norm_attn_output(x)
        return F.layer_norm(x, normalized_shape=(self.linear_dim * self.num_heads,), eps=self.epsilon)

    def _match_attention_mask(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.bool)
        if logits.dim() == 3 and mask.dim() == 4 and mask.size(1) == 1:
            return mask[:, 0]
        if logits.dim() == 4 and mask.dim() == 3:
            return mask.unsqueeze(1)
        if logits.dim() == 4 and mask.dim() == 4 and mask.size(1) == 1:
            return mask.expand(-1, logits.size(1), -1, -1)
        if logits.dim() != mask.dim():
            raise ValueError(
                f"Attention mask rank mismatch: logits has {logits.dim()} dims but mask has {mask.dim()} dims."
            )
        return mask

    def _softmax1_attention(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, T] or [B, H, T, T]
        # softmax1 = exp(z_i) / (1 + sum_j exp(z_j))
        mask = self._match_attention_mask(logits, attention_mask)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        max_logits = masked_logits.max(dim=-1, keepdim=True).values
        max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
        max_logits = torch.maximum(max_logits, torch.zeros_like(max_logits))
        # Use masked logits here; exp(inf) * 0 on masked positions can become NaN.
        exp_shifted = torch.exp(masked_logits - max_logits) * mask.to(dtype=logits.dtype)
        denom = torch.exp(-max_logits) + exp_shifted.sum(dim=-1, keepdim=True)
        weights = exp_shifted / denom.clamp_min(1e-12)
        return self.attn_dropout(weights)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        normed_x = self._norm_input(x)
        uvqk = torch.matmul(normed_x, self.uvqk)
        if self.linear_activation == "silu":
            uvqk = F.silu(uvqk)

        # uvqk -> u/v/q/k
        # u,v: [B, T, H*D_v]
        # q,k: [B, T, H*D_a]
        u, v, q, k = torch.split(
            uvqk,
            [
                self.linear_dim * self.num_heads,
                self.linear_dim * self.num_heads,
                self.attention_dim * self.num_heads,
                self.attention_dim * self.num_heads,
            ],
            dim=-1,
        )

        if self.normalization in {"rel_bias", "hstu_rel_bias"}:
            # HSTU 原始路径：加相对位置偏置，但不用 softmax。
            q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
            qk_attn = torch.einsum("bthd,bshd->bhts", q, k)
            if self.rel_pos_bias is not None:
                # rel_pos_bias(seq_len): [T, T]
                qk_attn = qk_attn + self.rel_pos_bias(seq_len, qk_attn.dtype).unsqueeze(0).unsqueeze(0)
            # qk_attn: [B, H, T, T]
            qk_attn = F.silu(qk_attn) / float(max(seq_len, 1))
            qk_attn = qk_attn * attention_mask.to(dtype=qk_attn.dtype)
            attn_output = torch.einsum("bhts,bshd->bthd", qk_attn, v).reshape(
                batch_size, seq_len, self.num_heads * self.linear_dim
            )
        elif self.normalization in {"softmax_rel_bias", "softmax1_rel_bias"}:
            # softmax 路径：
            # - softmax_rel_bias: 标准 softmax
            # - softmax1_rel_bias: softmax1
            # 这里也按 head 维分别计算 attention，而不是把所有 head 通道混成一张 [B, T, T] 图。
            q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
            qk_attn = torch.einsum("bthd,bshd->bhts", q, k)
            if self.rel_pos_bias is not None:
                qk_attn = qk_attn + self.rel_pos_bias(seq_len, qk_attn.dtype).unsqueeze(0).unsqueeze(0)
            # qk_attn: [B, H, T, T]
            qk_attn = qk_attn / math.sqrt(self.attention_dim)
            if self.normalization == "softmax1_rel_bias":
                qk_attn = self._softmax1_attention(qk_attn, attention_mask)
            else:
                mask = self._match_attention_mask(qk_attn, attention_mask)
                qk_attn = qk_attn.masked_fill(~mask, float("-inf"))
                qk_attn = F.softmax(qk_attn, dim=-1)
                qk_attn = self.attn_dropout(qk_attn)
                qk_attn = qk_attn * mask.to(dtype=qk_attn.dtype)
            attn_output = torch.einsum("bhts,bshd->bthd", qk_attn, v).reshape(
                batch_size, seq_len, self.num_heads * self.linear_dim
            )
        else:
            raise ValueError(f"Unknown normalization method {self.normalization}")

        if self.concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        return self.o(F.dropout(o_input, p=self.dropout_ratio, training=self.training)) + x


class HSTU(nn.Module):
    """Dense HSTU backbone with SASRec-compatible training and inference interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.max_seq_length = int(config.max_seq_length)
        self.hidden_units = int(config.hidden_units)
        self.patch_len = int(getattr(config, "patch_len", 0) or 0)
        self.persrec_enable = bool(getattr(config, "persrec_enable", False))
        self.persrec_num_tokens = int(getattr(config, "persrec_num_tokens", 0) or 0)
        self.persrec_pretrain_len = int(getattr(config, "persrec_pretrain_len", 0) or 0)
        self.persrec_recent_len = int(getattr(config, "persrec_recent_len", 0) or 0)
        self.shared_prefix_len = shared_token_len_from_config(config)
        self.shared_prefix_init_std = shared_token_init_std_from_config(config)
        self.total_seq_length = self.max_seq_length + self.patch_len + (
            self.persrec_num_tokens if self.persrec_enable else 0
        )
        self.total_seq_length += self.shared_prefix_len

        num_heads = int(getattr(config, "num_heads", 1))
        default_subdim = max(1, self.hidden_units // max(num_heads, 1))
        linear_dim = int(getattr(config, "hstu_linear_dim", default_subdim) or default_subdim)
        attention_dim = int(getattr(config, "hstu_attention_dim", default_subdim) or default_subdim)
        linear_activation = str(getattr(config, "hstu_linear_activation", "silu") or "silu").lower()
        normalization = str(getattr(config, "hstu_normalization", "rel_bias") or "rel_bias").lower()
        concat_ua = bool(getattr(config, "hstu_concat_ua", False))
        epsilon = float(getattr(config, "hstu_epsilon", 1e-6))
        attn_dropout = float(getattr(config, "hstu_attn_dropout", getattr(config, "dropout_rate", 0.0)))
        enable_relative_attention_bias = bool(getattr(config, "hstu_enable_relative_attention_bias", False))
        parametric_block_norm = bool(getattr(config, "hstu_parametric_block_norm", False))

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

        self.item_emb = nn.Embedding(item_num + 2, self.hidden_units, padding_idx=0)
        self.score_calibration = ScoreCalibrationHead(
            item_num + 2,
            max_seq_length=self.max_seq_length,
            enable_item_bias=bool(getattr(config, "enable_score_item_bias", False)),
            enable_length_bias=bool(getattr(config, "enable_score_length_bias", False)),
            enable_length_scale=bool(getattr(config, "enable_score_length_scale", False)),
            bucket_size=int(getattr(config, "score_length_bucket_size", 20) or 20),
        )
        if self.shared_prefix_len > 0:
            self.shared_prefix_tokens = nn.Parameter(
                torch.randn(1, self.shared_prefix_len, self.hidden_units) * self.shared_prefix_init_std
            )
        else:
            self.register_parameter("shared_prefix_tokens", None)
        pos_table_len = (
            self.max_seq_length + self.shared_prefix_len + self.persrec_num_tokens + 1
            if self.persrec_enable
            else self.max_seq_length + self.shared_prefix_len + 1
        )
        self.pos_emb = nn.Embedding(pos_table_len, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=self.hidden_units,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    dropout_ratio=config.dropout_rate,
                    attn_dropout_ratio=attn_dropout,
                    linear_activation=linear_activation,
                    normalization=normalization,
                    concat_ua=concat_ua,
                    epsilon=epsilon,
                    enable_relative_attention_bias=enable_relative_attention_bias,
                    max_length=self.total_seq_length,
                    parametric_block_norm=parametric_block_norm,
                )
                for _ in range(int(config.num_blocks))
            ]
        )

        self.ln_f = nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.proj_linear = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.meta_patch = MetaPatch(config)
        if self.persrec_enable:
            self.persrec_tokens = nn.Parameter(torch.randn(1, self.persrec_num_tokens, self.hidden_units))

        self.apply(self._init_weights)
        self.score_calibration.reset_parameters()

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _freeze_backbone(self) -> None:
        for p in self.item_emb.parameters():
            p.requires_grad = False
        for p in self.score_calibration.parameters():
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

    def _sequence_summary(self, item_embs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pool = getattr(self.config, "gating_pool", "last")
        return sequence_summary(item_embs, input_ids, pool=pool)

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
                pos_custom[:, :prefix_len] = torch.arange(1, prefix_len + 1, dtype=torch.long, device=input_ids.device)
            if tail_len > 0:
                tail_start = max(self.max_seq_length - tail_len + 1, 1)
                pos_custom[:, prefix_len:] = torch.arange(
                    tail_start,
                    tail_start + tail_len,
                    dtype=torch.long,
                    device=input_ids.device,
                )
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

    def _patch_token_start(self, seq_length: int) -> int:
        start = self._shared_token_count()
        prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
        if bool(getattr(self.config, "patch_after_prefix", False)) and prefix_len > 0:
            start += min(prefix_len, seq_length)
        return start

    def head_parameters(self) -> List[torch.Tensor]:
        return head_parameters_fn(self.config, self.proj_linear, self.proj_ln)

    def apply_head(self, hidden_states: torch.Tensor, head_params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return apply_head_fn(hidden_states, self.config, self.proj_linear, self.proj_ln, head_params=head_params)

    def _build_attention_mask(self, valid_tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # valid_tokens: [B, L]
        # 这里只记录每个位置是否是有效 token（非 PAD）。
        batch_size, seq_len = valid_tokens.size()
        # causal: [L, L]
        # 下三角因果 mask，保证位置 i 只能看见 <= i 的位置。
        causal = torch.tril(torch.ones((seq_len, seq_len), device=valid_tokens.device, dtype=torch.bool))
        # 先扩成 [1, 1, L, L]，后面再和 batch 维的有效位做广播。
        # 这里保留第 2 维的原因是 attention 模块通常希望 mask 形状是 [B, H/1, Q, K]，
        # 这样可以直接在 head 维上广播；如果做成 3 维 [B, Q, K]，后续很多实现都得额外再补一维。
        mask = causal.unsqueeze(0).unsqueeze(1)
        # valid.unsqueeze(1).unsqueeze(2): [B, 1, 1, L]，限制 key 侧只能落在有效 token 上。
        mask = mask & valid_tokens.unsqueeze(1).unsqueeze(2)
        # valid.unsqueeze(1).unsqueeze(3): [B, 1, L, 1]，限制 query 侧本身也必须是有效 token。
        mask = mask & valid_tokens.unsqueeze(1).unsqueeze(3)
        # 最终 mask: [B, 1, L, L]
        return mask.to(dtype=dtype).view(batch_size, 1, seq_len, seq_len)

    def _build_persrec_attention_mask_with_len(self, input_ids: torch.Tensor, pretrain_len: int) -> torch.Tensor:
        if not self.persrec_enable:
            raise RuntimeError("PersRec attention mask requested but persrec is disabled.")
        if self.persrec_num_tokens <= 0:
            raise RuntimeError("PersRec requires persrec_num_tokens > 0.")

        batch_size, seq_len = input_ids.size()
        total_len = seq_len + self.persrec_num_tokens
        # item_valid: [B, L]
        item_valid = input_ids != 0
        # token_valid: [B, T]，PersRec token 槽位始终视为有效。
        token_valid = torch.ones((batch_size, self.persrec_num_tokens), device=input_ids.device, dtype=torch.bool)
        # valid: [B, L+T]，对应 [pretrain][token][recent] 的完整布局。
        valid = torch.cat([item_valid[:, :pretrain_len], token_valid, item_valid[:, pretrain_len:]], dim=1)

        # causal: [L+T, L+T]
        causal = torch.tril(torch.ones((total_len, total_len), device=input_ids.device, dtype=torch.bool))
        # mask: [1, 1, L+T, L+T] -> 广播后变成 [B, 1, L+T, L+T]
        mask = causal.unsqueeze(0).unsqueeze(1)
        # 这里和普通 attention mask 一样，分别对 key/query 两个方向施加有效位约束。
        # line 389 之所以还是 4 维，不是因为它“多存了东西”，而是为了和 attention 的 [B, H/1, Q, K]
        # 接口保持一致，head 维用长度 1 占位即可。
        mask = mask & valid.unsqueeze(1).unsqueeze(2)
        mask = mask & valid.unsqueeze(1).unsqueeze(3)

        pre_end = pretrain_len - 1
        recent_start = pretrain_len + self.persrec_num_tokens
        if recent_start < total_len and pre_end >= 0:
            # PersRec 的关键约束：recent 不能直接看 pretrain raw items，
            # 只能通过中间插入的 persrec tokens 间接访问长历史信息。
            # recent_start: recent 区间在 [pretrain][token][recent] 中的起点。
            # :pre_end+1: 旧 pretrain raw items 的列区间。
            # 这里置 False 后，recent -> pretrain 的边会被整块切断。
            mask[:, :, recent_start:, : pre_end + 1] = False
        return mask.to(dtype=self.item_emb.weight.dtype).view(batch_size, 1, total_len, total_len)

    def _resolve_persrec_lengths(self, seq_length: int) -> Tuple[int, int]:
        recent_len = min(self.persrec_recent_len, seq_length)
        pretrain_len = max(seq_length - recent_len, 0)
        return pretrain_len, recent_len

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

        # PersRec 的实际输入布局是：
        # [pretrain items][persrec tokens][recent items]
        # 所以 recent 部分的位置编码需要整体后移，为中间插入的 token 腾出位置。
        # item_embs: [B, L, D]
        # positions: [B, L]
        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        offset = 0
        if getattr(self.config, "right_align_positions", True):
            offset = max(self.max_seq_length - seq_length, 0)
            if offset > 0:
                positions = positions + offset
        positions = positions * (input_ids != 0).long()

        recent_pos = positions[:, pretrain_len:]
        recent_pos = torch.where(recent_pos > 0, recent_pos + self.persrec_num_tokens, recent_pos)
        positions = torch.cat([positions[:, :pretrain_len], recent_pos], dim=1)

        # item_with_pos: [B, L, D]
        item_with_pos = item_embs + self.pos_emb(positions)

        token_positions = (
            torch.arange(1, self.persrec_num_tokens + 1, dtype=torch.long, device=input_ids.device)
            + offset
            + pretrain_len
        )
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
        # token_embs: [B, T, D]
        token_embs = self.persrec_tokens.expand(batch_size, -1, -1) + self.pos_emb(token_positions)

        # hidden_states: [B, L+T, D]
        hidden_states = torch.cat(
            [item_with_pos[:, :pretrain_len, :], token_embs, item_with_pos[:, pretrain_len:, :]],
            dim=1,
        )
        # attention mask 和序列布局必须一起改；只插 token 不改 mask 会破坏 PersRec 设计。
        attn_mask = self._build_persrec_attention_mask_with_len(input_ids, pretrain_len)
        return hidden_states, attn_mask

    def _build_persrec_target_sequence(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch_size, seq_length = input_ids.size()
        pretrain_len, _ = self._resolve_persrec_lengths(seq_length)
        # targets: [B, L+T, D]，与 PersRec 前向后的 hidden_states 严格对齐。
        item_embs = self.item_emb(input_ids)
        token_embs = self.persrec_tokens.expand(batch_size, -1, -1)
        targets = torch.cat(
            [item_embs[:, :pretrain_len, :], token_embs, item_embs[:, pretrain_len:, :]],
            dim=1,
        )

        item_valid = input_ids != 0
        token_mask = torch.zeros((batch_size, self.persrec_num_tokens), dtype=torch.bool, device=input_ids.device)
        # token 槽位是“中介记忆”，不是直接监督的预测目标，所以这里明确从 loss mask 里去掉。
        loss_mask = torch.cat(
            [item_valid[:, :pretrain_len], token_mask, item_valid[:, pretrain_len:]],
            dim=1,
        )
        if loss_mask.numel() > 0:
            has_valid = loss_mask.any(dim=1)
            last_from_end = loss_mask.flip(1).float().argmax(dim=1)
            last_idx = loss_mask.size(1) - 1 - last_from_end
            loss_mask = loss_mask.clone()
            # next-item 训练里最后一个有效位置没有“下一个 item”可预测，所以要屏蔽掉。
            loss_mask[has_valid, last_idx[has_valid]] = False
        # loss_mask: [B, L+T]
        return targets, loss_mask, pretrain_len

    def _build_persrec_negative_sequence(self, neg_ids: torch.Tensor, pretrain_len: int) -> torch.Tensor:
        # neg_ids 常见形状有两种：
        # 1. [B, L]：每个位置 1 个负样本
        # 2. [B, L, K]：每个位置 K 个负样本
        neg_embs = self.item_emb(neg_ids)
        if neg_embs.dim() == 4:
            # neg_embs: [B, L, K, D] -> filler: [B, T, K, D]
            filler = neg_embs.new_zeros(
                neg_embs.size(0), self.persrec_num_tokens, neg_embs.size(2), neg_embs.size(3)
            )
        else:
            # neg_embs: [B, L, D] -> filler: [B, T, D]
            filler = neg_embs.new_zeros(neg_embs.size(0), self.persrec_num_tokens, neg_embs.size(2))
        # 输出和 target 一样插入 token 槽位，形状变成 [B, L+T, D] 或 [B, L+T, K, D]。
        return torch.cat([neg_embs[:, :pretrain_len], filler, neg_embs[:, pretrain_len:]], dim=1)

    def _strip_patch_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.patch_len <= 0:
            return hidden_states
        item_len = max(hidden_states.size(1) - self.patch_len - self._shared_token_count(), 0)
        patch_start = self._patch_token_start(item_len)
        patch_end = min(hidden_states.size(1), patch_start + self.patch_len)
        return torch.cat([hidden_states[:, :patch_start, :], hidden_states[:, patch_end:, :]], dim=1)

    def _strip_shared_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return strip_leading_tokens(hidden_states, self._shared_token_count())

    def strip_patch_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._strip_patch_tokens(hidden_states)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        batch_size, seq_length = input_ids.size()

        # input_ids: [B, L]
        # item_embs: [B, L, D]
        item_embs = self.item_emb(input_ids)
        item_embs = item_embs * (self.hidden_units**0.5)

        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary, user_ids=user_ids)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params, user_ids=user_ids)
        else:
            patch_emb = item_embs.new_zeros((batch_size, 0, self.hidden_units))
            gating_weights = None
        # patch_emb: [B, P, D]，其中 P = patch_len；gating_weights 常见为 [B, num_patches]

        attn_mask = None
        if self.persrec_enable:
            if use_patch and self.patch_len > 0:
                raise ValueError("PersRec is not supported together with meta-patch in this implementation.")
            # PersRec 模式下，输入结构和 mask 都换成专用版本。
            hidden_states, attn_mask = self._build_persrec_embeddings(item_embs, input_ids)
        else:
            positions = self._build_item_positions(input_ids)
            shared_states = self._shared_token_states(batch_size, device=input_ids.device, dtype=item_embs.dtype)
            item_with_pos = item_embs + self.pos_emb(positions)
            item_valid = input_ids != 0
            # shared_states: [B, S, D] 或 None
            # item_with_pos: [B, L, D]
            # item_valid: [B, L]

            if use_patch and self.patch_len > 0:
                prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
                patch_after_prefix = bool(getattr(self.config, "patch_after_prefix", False))
                if patch_after_prefix and prefix_len > 0:
                    # 旧 prefix 设定：shared 在最前，patch 插在 prefix 和 tail 之间。
                    prefix_len = min(prefix_len, seq_length)
                    hidden_parts = []
                    valid_parts = []
                    if shared_states is not None:
                        hidden_parts.append(shared_states)
                        valid_parts.append(
                            torch.ones((batch_size, self._shared_token_count()), device=input_ids.device, dtype=torch.bool)
                        )
                    hidden_parts.extend([item_with_pos[:, :prefix_len, :], patch_emb, item_with_pos[:, prefix_len:, :]])
                    valid_parts.extend(
                        [
                            item_valid[:, :prefix_len],
                            torch.ones((batch_size, self.patch_len), device=input_ids.device, dtype=torch.bool),
                            item_valid[:, prefix_len:],
                        ]
                    )
                    # hidden_states / valid_tokens 分别是 [B, S+prefix+P+tail, D] / [B, S+prefix+P+tail]
                    hidden_states = torch.cat(hidden_parts, dim=1)
                    valid_tokens = torch.cat(valid_parts, dim=1)
                else:
                    # 无 prefix 时，patch 直接放在 item 之前；shared 仍然位于最前面。
                    hidden_parts = []
                    valid_parts = []
                    if shared_states is not None:
                        hidden_parts.append(shared_states)
                        valid_parts.append(
                            torch.ones((batch_size, self._shared_token_count()), device=input_ids.device, dtype=torch.bool)
                        )
                    hidden_parts.extend([patch_emb, item_with_pos])
                    valid_parts.extend(
                        [
                            torch.ones((batch_size, self.patch_len), device=input_ids.device, dtype=torch.bool),
                            item_valid,
                        ]
                    )
                    # hidden_states / valid_tokens 分别是 [B, S+P+L, D] / [B, S+P+L]
                    hidden_states = torch.cat(hidden_parts, dim=1)
                    valid_tokens = torch.cat(valid_parts, dim=1)
            else:
                hidden_states = item_with_pos
                valid_tokens = item_valid
                if shared_states is not None:
                    hidden_states = torch.cat([shared_states, hidden_states], dim=1)
                    valid_tokens = torch.cat(
                        [
                            torch.ones((batch_size, self._shared_token_count()), device=input_ids.device, dtype=torch.bool),
                            valid_tokens,
                        ],
                        dim=1,
                    )

        hidden_states = self.emb_dropout(hidden_states)
        if attn_mask is None:
            attn_mask = self._build_attention_mask(valid_tokens, dtype=hidden_states.dtype)
        # hidden_states: [B, L_total, D]
        # attn_mask: [B, 1, L_total, L_total]

        for block in self.blocks:
            if getattr(self.config, "use_gradient_checkpointing", False) and self.training and hidden_states.requires_grad:
                hidden_states = gradient_checkpoint(block, hidden_states, attn_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, attn_mask)

        hidden_states = self.ln_f(hidden_states)
        if return_gating:
            return hidden_states, gating_weights
        return hidden_states

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
        hidden_states = self.forward_features(
            input_ids,
            user_ids=user_ids,
            patch_params=patch_params,
            use_patch=use_patch,
        )
        if use_patch and self.patch_len > 0:
            hidden_states = self._strip_patch_tokens(hidden_states)
        hidden_states = self._strip_shared_tokens(hidden_states)
        if self.persrec_enable and self.persrec_num_tokens > 0:
            pre, _ = self._resolve_persrec_lengths(input_ids.size(1))
            k = self.persrec_num_tokens
            # 最终打分不直接使用 patch/shared/PersRec token 自身，
            # 而是删掉这些辅助 token 后，取最后一个 item hidden 去打分。
            hidden_states = torch.cat([hidden_states[:, :pre, :], hidden_states[:, pre + k :, :]], dim=1)
        # hidden_states: [B, L_item, D]
        final_hidden = hidden_states[:, -1, :]
        # final_hidden: [B, D]
        if use_head:
            final_hidden = self.apply_head(final_hidden, head_params=head_params)
        # candidate_embs: [B, C, D]，scores: [B, C]
        candidate_embs = self.item_emb(candidate_ids)
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)
        seq_lengths = (input_ids != 0).sum(dim=1)
        return self.score_calibration(scores, candidate_ids, seq_lengths)

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
    ):
        hidden_states, gating_weights = self.forward_features(
            input_ids,
            user_ids=user_ids,
            patch_params=patch_params,
            return_gating=True,
            use_patch=use_patch,
        )
        if use_patch and self.patch_len > 0:
            hidden_states = self._strip_patch_tokens(hidden_states)
        hidden_states = self._strip_shared_tokens(hidden_states)

        # projected: [B, L_total, D]
        projected = self.apply_head(hidden_states, head_params=head_params)
        if self.persrec_enable and self.persrec_num_tokens > 0:
            # PersRec 下 target/negative 序列也要插入 token 槽位，才能和前向 hidden 一一对齐。
            target_seq, loss_mask, pretrain_len = self._build_persrec_target_sequence(input_ids)
            pos_embs = projected.new_zeros(projected.shape)
            pos_embs[:, :-1, :] = target_seq[:, 1:, :].to(projected.dtype)
            neg_embs = self._build_persrec_negative_sequence(neg_ids, pretrain_len).to(projected.dtype)
        else:
            pos_embs = self.item_emb(pos_ids).to(projected.dtype)
            neg_embs = self.item_emb(neg_ids).to(projected.dtype)
            loss_mask = pos_ids != 0

        # pos_logits: [B, L_total]
        pos_logits = (projected * pos_embs).sum(dim=-1)
        if neg_embs.dim() == 4:
            # neg_embs: [B, L_total, K, D] -> neg_logits: [B, L_total, K]
            neg_logits = (projected.unsqueeze(2) * neg_embs).sum(dim=-1)
        else:
            # neg_embs: [B, L_total, D] -> neg_logits: [B, L_total]
            neg_logits = (projected * neg_embs).sum(dim=-1)
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
