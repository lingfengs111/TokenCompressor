"""Shared helper utilities for backbones."""

from typing import List, Optional, Tuple

import torch
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
