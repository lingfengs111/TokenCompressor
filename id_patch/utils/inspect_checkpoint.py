#!/usr/bin/env python3
"""Inspect a SASRec checkpoint and infer core hyper-parameters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

DEFAULT_CKPT = "/home/lingfengs111/codes/soft_patch_training/id_patch/best_model.pth"


def load_checkpoint(path: str, trust_pickle: bool = True) -> Dict:
    """Load a checkpoint with PyTorch 2.6+ weights_only safety handling."""
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


def _infer_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    inferred = {}
    if "item_emb.weight" in state_dict:
        inferred["hidden_units"] = int(state_dict["item_emb.weight"].shape[1])
        inferred["num_items"] = int(state_dict["item_emb.weight"].shape[0]) - 1
    if "pos_emb.weight" in state_dict:
        inferred["max_seq_length"] = int(state_dict["pos_emb.weight"].shape[0]) - 1
    block_indices = []
    for key in state_dict.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    if block_indices:
        inferred["num_blocks"] = max(block_indices) + 1
    return inferred


def _print_key_shapes(state_dict: Dict[str, torch.Tensor]) -> None:
    def shape_str(name: str) -> str:
        tensor = state_dict.get(name)
        if tensor is None:
            return "(missing)"
        return str(tuple(tensor.shape))

    print("Key tensor shapes:")
    print(f"- item_emb.weight: {shape_str('item_emb.weight')}")
    print(f"- pos_emb.weight: {shape_str('pos_emb.weight')}")

    attn_keys = [k for k in state_dict.keys() if "attn" in k and k.endswith("weight")]
    if attn_keys:
        for key in sorted(attn_keys)[:5]:
            print(f"- {key}: {tuple(state_dict[key].shape)}")
    else:
        print("- attn.*.weight: (not found)")

    ln_keys = [k for k in state_dict.keys() if k.endswith("ln_f.weight")]
    if ln_keys:
        for key in ln_keys:
            print(f"- {key}: {tuple(state_dict[key].shape)}")


def main() -> int:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_CKPT)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    ckpt = load_checkpoint(str(ckpt_path), trust_pickle=True)
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Total keys: {len(state_dict)}")

    _print_key_shapes(state_dict)

    inferred = _infer_config(state_dict)
    print("\nInferred config:")
    for key in ("hidden_units", "max_seq_length", "num_blocks", "num_items"):
        if key in inferred:
            print(f"- {key}: {inferred[key]}")
    print("\nNotes:")
    print("- num_heads is not encoded in weight shapes for this SASRec variant; set it manually.")
    print("- dropout_rate is not stored in the checkpoint; use your prior default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
