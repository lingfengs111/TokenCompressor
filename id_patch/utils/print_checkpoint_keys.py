#!/usr/bin/env python3
"""Print all parameter names from a checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

DEFAULT_CKPT = "/home/lingfengs111/codes/soft_patch_training/id_patch/best_model.pth"


def load_checkpoint(path: str, trust_pickle: bool = True) -> Dict:
    if trust_pickle:
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            import numpy as np
            from torch.serialization import safe_globals

            with safe_globals([np.core.multiarray.scalar]):
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict or a state_dict-like object.")
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, val in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = val
        else:
            cleaned[key] = val
    return cleaned


def _format_tensor(t: torch.Tensor, max_values: int = 8) -> str:
    flat = t.detach().cpu().flatten()
    if flat.numel() == 0:
        sample = "[]"
    else:
        sample_vals = flat[:max_values].tolist()
        sample = "[" + ", ".join(f"{v:.6g}" for v in sample_vals) + (", ..." if flat.numel() > max_values else "") + "]"
    stats = ""
    if flat.numel() > 0 and flat.dtype.is_floating_point:
        stats = f", min={flat.min().item():.6g}, max={flat.max().item():.6g}, mean={flat.mean().item():.6g}"
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, sample={sample}{stats}"


def main() -> int:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_CKPT)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    ckpt = load_checkpoint(str(ckpt_path), trust_pickle=True)
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Total keys: {len(state_dict)}")
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if torch.is_tensor(val):
            print(f"{key}: {_format_tensor(val)}")
        else:
            print(f"{key}: {repr(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
