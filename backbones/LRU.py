"""LRU backbone adapted for soft-patch training."""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.patch import MetaPatch
from backbones.peft import Adapter


def _force_real(x: torch.Tensor) -> torch.Tensor:
    """Force real conversion; raise if torch.real is unsupported."""
    try:
        return torch.real(x)
    except Exception as exc:
        raise RuntimeError("torch.real failed in _force_real (likely functorch/complex incompatibility)") from exc


class LRUEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_units: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_units, padding_idx=0)
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.embed_dropout = nn.Dropout(dropout)

    def get_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x > 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.get_mask(x)
        x = self.token(x)
        x = self.embed_dropout(x)
        # Ensure LN params and input are aligned for mixed-dtype / complex-safe environments.
        x = _force_real(x)
        if self.layer_norm.weight.is_complex():
            weight = self.layer_norm.weight.real
            bias = self.layer_norm.bias.real if self.layer_norm.bias is not None else None
            x = F.layer_norm(x, self.layer_norm.normalized_shape, weight, bias, self.layer_norm.eps)
        else:
            if x.dtype != self.layer_norm.weight.dtype:
                x = x.to(self.layer_norm.weight.dtype)
            x = self.layer_norm(x)
        return x, mask


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _force_real(x)  # <--- 必须确保这一行存在且生效
        # Always align dtype to linear weights
        x = x.to(self.w_1.weight.dtype)
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)


class LRULayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, use_bias: bool = True, r_min: float = 0.8, r_max: float = 0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        self.out_vector = nn.Identity()

        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i: int, h: torch.Tensor, lamb: torch.Tensor, mask: torch.Tensor, B: int, L: int, D: int):
        # Parallel algorithm, see: https://kexue.fm/archives/9554
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, : l // 2], h[:, l // 2 :]

        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1 : l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = _force_real(x)
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        x_real = _force_real(x)
        h = self.in_proj(x_real.to(torch.cfloat)) * gamma

        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        out = self.out_proj(h).real
        x = self.dropout(out) + self.out_vector(x_real)
        # LayerNorm does not support complex; also align dtype with LN params.
        x = _force_real(x)
        if self.layer_norm.weight.is_complex():
            weight = self.layer_norm.weight.real
            bias = self.layer_norm.bias.real if self.layer_norm.bias is not None else None
            return F.layer_norm(x, self.layer_norm.normalized_shape, weight, bias, self.layer_norm.eps)
        if x.dtype != self.layer_norm.weight.dtype:
            x = x.to(self.layer_norm.weight.dtype)
        return self.layer_norm(x)


class LRUBlock(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        dropout: float,
        attn_dropout: float,
        adapter_dim: int = 0,
        adapter_dropout: float = 0.0,
        adapter_activation: str = "gelu",
        adapter_init: str = "zero",
    ):
        super().__init__()
        self.lru_layer = LRULayer(d_model=hidden_units, dropout=attn_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden_units, d_ff=hidden_units * 4, dropout=dropout)
        self.adapter = None
        if adapter_dim and adapter_dim > 0:
            self.adapter = Adapter(
                hidden_units,
                adapter_dim,
                dropout=adapter_dropout,
                activation=adapter_activation,
                init=adapter_init,
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.lru_layer(x, mask)
        x = _force_real(x)
        x = x.to(self.feed_forward.w_1.weight.dtype)
        x = self.feed_forward(x)
        if self.adapter is not None:
            x = self.adapter(x)
        return x


class LRU(nn.Module):
    """Linear Recurrent Units backbone with SASRec-compatible interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len

        num_blocks = getattr(config, "lru_num_blocks", config.num_blocks)
        dropout = getattr(config, "lru_dropout", config.dropout_rate)
        attn_dropout = getattr(config, "lru_attn_dropout", config.dropout_rate)
        self.embedding = LRUEmbedding(item_num + 2, self.hidden_units, dropout)
        self.patch_dropout = nn.Dropout(dropout)

        adapter_enabled = getattr(config, "enable_adapter", False)
        adapter_dim = int(getattr(config, "adapter_dim", max(4, self.hidden_units // 8))) if adapter_enabled else 0
        adapter_dropout = float(getattr(config, "adapter_dropout", 0.0))
        adapter_activation = getattr(config, "adapter_activation", "gelu")
        adapter_init = getattr(config, "adapter_init", "zero")
        self.blocks = nn.ModuleList(
            [
                LRUBlock(
                    self.hidden_units,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    adapter_dim=adapter_dim,
                    adapter_dropout=adapter_dropout,
                    adapter_activation=adapter_activation,
                    adapter_init=adapter_init,
                )
                for _ in range(num_blocks)
            ]
        )

        # Optional final layer norm to align with SASRec-style heads
        self.ln_f = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Trainable projection head (only head is adapted in inner loop)
        self.proj_linear = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Meta-patch module (outer loop)
        self.meta_patch = MetaPatch(config)

        # Initialize LRU-specific weights
        self._init_lru_weights()

    def _init_lru_weights(self) -> None:
        # Truncated normal init from the original LRURec implementation
        mean, std, lower, upper = 0.0, 0.02, -0.04, 0.04
        with torch.no_grad():
            l = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
            u = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0

            for name, p in self.named_parameters():
                if name.startswith("meta_patch.") or name.startswith("proj_"):
                    continue
                if "layer_norm" in name or "params_log" in name:
                    continue
                if torch.is_complex(p):
                    p.real.uniform_(2 * l - 1, 2 * u - 1)
                    p.imag.uniform_(2 * l - 1, 2 * u - 1)
                    p.real.erfinv_()
                    p.imag.erfinv_()
                    p.real.mul_(std * math.sqrt(2.0))
                    p.imag.mul_(std * math.sqrt(2.0))
                    p.real.add_(mean)
                    p.imag.add_(mean)
                else:
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.0))
                    p.add_(mean)

    def _freeze_backbone(self) -> None:
        for p in self.embedding.parameters():
            p.requires_grad = False
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.ln_f.parameters():
            p.requires_grad = False

    def _sequence_summary(self, item_embs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = input_ids != 0
        lengths = valid_mask.sum(dim=1)
        pool = getattr(self.config, "gating_pool", "last")
        if pool == "mean":
            denom = lengths.clamp_min(1).unsqueeze(1).to(item_embs.dtype)
            mask = valid_mask.unsqueeze(-1).to(item_embs.dtype)
            mean_emb = (item_embs * mask).sum(dim=1) / denom
            return mean_emb, lengths
        seq_len = input_ids.size(1)
        if seq_len == 0:
            last_idx = torch.zeros_like(lengths)
        else:
            last_from_end = valid_mask.flip(1).float().argmax(dim=1)
            last_idx = (seq_len - 1) - last_from_end
        batch_idx = torch.arange(item_embs.size(0), device=item_embs.device)
        last_emb = item_embs[batch_idx, last_idx]
        return last_emb, lengths

    def head_parameters(self) -> List[torch.Tensor]:
        params = list(self.proj_linear.parameters())
        if getattr(self.config, "head_use_ln", True):
            params += list(self.proj_ln.parameters())
        return params

    def apply_head(self, hidden_states: torch.Tensor, head_params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if not getattr(self.config, "enable_projection_head", True):
            return hidden_states
        use_gelu = getattr(self.config, "head_use_gelu", True)
        use_ln = getattr(self.config, "head_use_ln", True)
        if head_params is None:
            delta = self.proj_linear(hidden_states)
            if use_gelu:
                delta = F.gelu(delta)
            if use_ln:
                delta = self.proj_ln(delta)
        else:
            needed = 2 + (2 if use_ln else 0)
            if len(head_params) < needed:
                raise ValueError("head_params length does not match head configuration")
            w, b = head_params[0], head_params[1]
            delta = F.linear(hidden_states, w, b)
            if use_gelu:
                delta = F.gelu(delta)
            if use_ln:
                ln_w, ln_b = head_params[2], head_params[3]
                delta = F.layer_norm(delta, (self.hidden_units,), ln_w, ln_b, eps=1e-8)
        if getattr(self.config, "head_residual", False):
            return hidden_states + delta
        return delta

    def _pad_to_pow2(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_len = x.size(1)
        if seq_len <= 1:
            return x, mask, seq_len
        log2_L = int(np.ceil(np.log2(seq_len)))
        target_len = 2 ** log2_L
        pad_len = target_len - seq_len
        if pad_len <= 0:
            return x, mask, seq_len
        x = F.pad(x, (0, 0, pad_len, 0, 0, 0))
        mask = F.pad(mask, (pad_len, 0, 0, 0))
        return x, mask, seq_len

    def forward_features(
        self,
        input_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        # Embedding
        item_embs, mask = self.embedding(input_ids)

        # Patch embeddings (meta-patch)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
            patch_emb = self.patch_dropout(patch_emb)
            patch_mask = torch.ones((input_ids.size(0), self.patch_len), device=input_ids.device, dtype=mask.dtype)
            hidden_states = torch.cat([patch_emb, item_embs], dim=1)
            mask = torch.cat([patch_mask, mask], dim=1)
        else:
            hidden_states = item_embs
            gating_weights = None

        hidden_states, mask, orig_len = self._pad_to_pow2(hidden_states, mask)

        for block in self.blocks:
            hidden_states = block(hidden_states, mask)

        # remove padding
        hidden_states = hidden_states[:, -orig_len:]

        # Optional final layer norm
        hidden_states = self.ln_f(hidden_states)

        if return_gating:
            return hidden_states, gating_weights
        return hidden_states

    def get_gating_weights(self, input_ids: torch.Tensor, patch_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        item_embs, _ = self.embedding(input_ids)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if patch_params is None:
            _, weights = self.meta_patch(seq_summary)
        else:
            _, weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
        return weights

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        neg_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ):
        if pos_ids is not None and neg_ids is not None:
            return self.training_step(
                input_ids,
                pos_ids,
                neg_ids,
                patch_params=patch_params,
                return_gating=return_gating,
                use_patch=use_patch,
            )
        return self.forward_features(
            input_ids,
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
        use_patch: bool = True,
        use_head: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.forward_features(input_ids, patch_params=patch_params, use_patch=use_patch)
        if use_patch and self.patch_len > 0:
            hidden_states = hidden_states[:, self.patch_len :, :]
        final_hidden = hidden_states[:, -1, :]
        if use_head:
            final_hidden = self.apply_head(final_hidden, head_params=head_params)
        candidate_embs = self.embedding.token(candidate_ids)
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)
        return scores

    def training_step(
        self,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden_states, gating_weights = self.forward_features(
            input_ids, patch_params=patch_params, return_gating=True, use_patch=use_patch
        )
        if use_patch and self.patch_len > 0:
            hidden_states = hidden_states[:, self.patch_len :, :]

        projected = self.apply_head(hidden_states, head_params=head_params)
        pos_embs = self.embedding.token(pos_ids)
        neg_embs = self.embedding.token(neg_ids)

        pos_logits = (projected * pos_embs).sum(dim=-1)
        if neg_embs.dim() == 4:
            neg_logits = (projected.unsqueeze(2) * neg_embs).sum(dim=-1)
        else:
            neg_logits = (projected * neg_embs).sum(dim=-1)

        if return_gating:
            return pos_logits, neg_logits, gating_weights
        return pos_logits, neg_logits
