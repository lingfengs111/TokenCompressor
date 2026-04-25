"""Shared helper utilities for backbones."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_summary(
    item_embs: torch.Tensor,
    input_ids: torch.Tensor,
    pool: str = "last",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sequence summary embedding and lengths.

    pool: "last" or "mean"
    """
    valid_mask = input_ids != 0
    lengths = valid_mask.sum(dim=1)
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


def shared_token_len_from_config(config) -> int:
    return int(getattr(config, "shared_prefix_len", 0) or 0)


def shared_token_init_std_from_config(config, default: float = 0.02) -> float:
    return float(getattr(config, "shared_prefix_init_std", default) or default)


def expand_shared_token_bank(
    shared_tokens: Optional[torch.Tensor],
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if shared_tokens is None:
        return None
    if shared_tokens.dim() == 2:
        shared_tokens = shared_tokens.unsqueeze(0)
    return shared_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)


def strip_leading_tokens(hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if num_tokens <= 0:
        return hidden_states
    if hidden_states.size(1) <= num_tokens:
        return hidden_states[:, :0, :]
    return hidden_states[:, num_tokens:, :]


def head_parameters(config, proj_linear, proj_ln) -> List[torch.Tensor]:
    params = list(proj_linear.parameters())
    if getattr(config, "head_use_ln", True):
        params += list(proj_ln.parameters())
    return params


def apply_head(
    hidden_states: torch.Tensor,
    config,
    proj_linear,
    proj_ln,
    head_params: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    if not getattr(config, "enable_projection_head", True):
        return hidden_states
    use_gelu = getattr(config, "head_use_gelu", True)
    use_ln = getattr(config, "head_use_ln", True)
    if head_params is None:
        delta = proj_linear(hidden_states)
        if use_gelu:
            delta = F.gelu(delta)
        if use_ln:
            delta = proj_ln(delta)
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
            delta = F.layer_norm(delta, (hidden_states.size(-1),), ln_w, ln_b, eps=1e-8)
    if getattr(config, "head_residual", False):
        return hidden_states + delta
    return delta


def _expand_batch_values(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while values.dim() < target.dim():
        values = values.unsqueeze(-1)
    return values


class ScoreCalibrationHead(nn.Module):
    """Optional scorer-side priors shared across backbones."""

    def __init__(
        self,
        num_items: int,
        max_seq_length: int,
        *,
        enable_item_bias: bool = False,
        enable_length_bias: bool = False,
        enable_length_scale: bool = False,
        bucket_size: int = 20,
    ) -> None:
        super().__init__()
        self.bucket_size = max(1, int(bucket_size or 1))
        max_seq_length = max(1, int(max_seq_length or 1))
        self.num_buckets = max(1, (max_seq_length + self.bucket_size - 1) // self.bucket_size)

        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0) if enable_item_bias else None
        self.length_bias = nn.Embedding(self.num_buckets, 1) if enable_length_bias else None
        self.length_scale_log = nn.Embedding(self.num_buckets, 1) if enable_length_scale else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)
            if self.item_bias.padding_idx is not None:
                with torch.no_grad():
                    self.item_bias.weight[self.item_bias.padding_idx].fill_(0)
        if self.length_bias is not None:
            nn.init.zeros_(self.length_bias.weight)
        if self.length_scale_log is not None:
            nn.init.zeros_(self.length_scale_log.weight)

    def _bucket_ids(self, seq_lengths: torch.Tensor) -> torch.Tensor:
        lengths = seq_lengths.to(dtype=torch.long).clamp_min(1)
        bucket_ids = torch.div(lengths - 1, self.bucket_size, rounding_mode="floor")
        return bucket_ids.clamp_max(self.num_buckets - 1)

    def forward(
        self,
        logits: torch.Tensor,
        item_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.item_bias is None and self.length_bias is None and self.length_scale_log is None:
            return logits
        bucket_ids = self._bucket_ids(seq_lengths)
        out = logits
        if self.length_scale_log is not None:
            scale = torch.exp(self.length_scale_log(bucket_ids).squeeze(-1).clamp(min=-5.0, max=5.0))
            out = out * _expand_batch_values(scale.to(dtype=out.dtype), out)
        if self.item_bias is not None:
            out = out + self.item_bias(item_ids).squeeze(-1).to(dtype=out.dtype)
        if self.length_bias is not None:
            bias = self.length_bias(bucket_ids).squeeze(-1).to(dtype=out.dtype)
            out = out + _expand_batch_values(bias, out)
        return out
