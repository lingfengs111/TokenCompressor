#!/usr/bin/env python3
"""Train PersRec-style backbone with learnable tokens and segment masking."""

import argparse
import math
import os
import sys
import random
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.device_manager import DeviceManager
from core.logger import setup_logger
from core.loo_dataset import LooSequenceDataset, resolve_loo_dataset, infer_loo_min_len
from core.streaming_eval import (
    finalize_eval_metrics,
    flatten_streaming_eval_metrics,
    normalize_eval_protocol,
    resolve_eval_target_positions,
    resolve_train_cutoff,
    update_rank_metrics,
)
from backbones.HSTU import HSTU
from backbones.SASRec import SASRec

logger = setup_logger("train-persrec", log_to_file=True)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def none_or_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def none_or_int(value: Optional[str]) -> Optional[int]:
    text = none_or_str(value)
    if text is None:
        return None
    return int(text)


def parse_int_list(value: Optional[str]) -> List[int]:
    text = none_or_str(value)
    if text is None:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


class _StoreProvided(argparse.Action):
    """Argparse action that records whether an option was explicitly provided."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_provided", True)

DEFAULT_LONG_CKPT = (
    "/home/lingfengs111/codes/soft_patch_training/checkpoints/sasrec_loo_standard/"
    "sasrec_taobao_loo202_seq202_dim128_L2_H1_best.pt"
)
DEFAULT_SHORT_CKPT = (
    "/home/lingfengs111/codes/soft_patch_training/checkpoints/sasrec_loo_standard/"
    "sasrec_taobao_loo202_seq50_dim128_L2_H1_best.pt"
)


@dataclass
class BaselineConfig:
    """Configuration for PersRec-style training."""

    dataset: str = "taobao_loo202"
    data_dir: Optional[Path] = None
    data_txt_path: Optional[Path] = None
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "persrec")

    # Model parameters
    backbone: str = "sasrec"  # sasrec | hstu | fmlp | linrec | bert4rec | gru4rec
    max_seq_length: Optional[int] = 200  # Long sequence length (items only)
    hidden_units: int = 128
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.2
    right_align_positions: bool = True
    sasrec_attention_norm: str = "softmax"
    use_flash_attention: bool = False
    use_gradient_checkpointing: bool = False
    hstu_linear_dim: Optional[int] = None
    hstu_attention_dim: Optional[int] = None
    hstu_linear_activation: str = "silu"
    hstu_attn_dropout: Optional[float] = None
    hstu_enable_relative_attention_bias: bool = False
    hstu_normalization: str = "rel_bias"
    hstu_concat_ua: bool = False
    hstu_epsilon: float = 1e-6

    # Head (disabled by default for PersRec)
    head_residual: bool = False
    head_zero_init: bool = True
    enable_projection_head: bool = False
    head_use_gelu: bool = False
    head_use_ln: bool = False

    # Optional PEFT adapters
    input_emb_lora_rank: int = 0
    input_emb_lora_alpha: float = 8.0
    train_input_emb_lora: bool = False
    attn_lora_rank: int = 0
    attn_lora_alpha: float = 8.0
    attn_lora_blocks: str = "all"
    train_attn_lora: bool = False

    # PersRec (learnable tokens + segment mask)
    persrec_enable: bool = True
    persrec_num_tokens: int = 10
    persrec_pretrain_len: Optional[int] = None
    persrec_recent_len: Optional[int] = None

    # Evaluation
    eval_seq_length: int = 20  # Short-view length for eval truncation
    persrec_eval_use_full_seq: bool = True  # If True, eval uses full long sequence
    persrec_train_mode: str = "full"  # full | tokens | tokens_bias_ln | tokens_bias_ln_head
    eval_protocol: str = "legacy_loo"  # legacy_loo | holdout_anchor
    last_k_eval_test: int = 10  # Used when eval_protocol=holdout_anchor
    streaming_eval_last_k: int = 0  # If >1, run extra rolling final-test eval on the last K targets

    # Device
    device: str = "cuda:2"

    # Meta-patch config (disabled for PersRec; kept for SASRec module compatibility)
    num_patches: int = 1
    patch_len: int = 0
    use_gating: bool = False
    gating_hidden_dim: int = 64
    patch_init_std: float = 0.0
    gating_init_std: float = 0.0
    gating_pool: str = "mean"
    gating_temperature: float = 1.0
    gating_noise_std: float = 0.0
    patch_routing: str = "learned"

    # Training parameters
    batch_size: int = 512
    num_epochs: int = 50
    max_learning_rate: float = 5e-5  #5e-5  
    min_learning_rate: float = 5e-6 #5e-6
    scheduler_type: str = "cosine"  # cosine | cosine_with_warmup
    warmup_steps: int = 100
    warmup_start_lr: float = 5e-7
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    steps_per_train_log: int = 100
    steps_per_val_log: int = 1000
    eval_sample_size: int = 1000
    noise_ratio: float = 0.0
    early_stop_patience: int = 5

    # Unseen item handling
    drop_unseen_items: bool = True
    inner_unk_mask_prob: float = 0.0

    # RNG
    seed: int = 2026
    deterministic: bool = False

    # Checkpoint loading
    strict_load_pretrained: bool = False
    ckpt_prefix_to_strip: Optional[str] = None
    pretrained_ckpt_path: Optional[str] = DEFAULT_LONG_CKPT

    infer_ckpt_config: bool = True
    preserve_max_seq_length: bool = True

    # Output
    save_best: bool = True
    save_item_embeddings: bool = False


    def log_config(self) -> None:
        logger.info("=== PersRec Configuration ===")
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  data_txt_path: {self.data_txt_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        logger.info("Model Parameters:")
        logger.info(f"  backbone: {self.backbone}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  hidden_units: {self.hidden_units}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")
        logger.info(f"  right_align_positions: {self.right_align_positions}")
        logger.info(f"  sasrec_attention_norm: {self.sasrec_attention_norm}")
        logger.info(f"  use_flash_attention: {self.use_flash_attention}")
        logger.info(f"  use_gradient_checkpointing: {self.use_gradient_checkpointing}")

        logger.info("Head Parameters:")
        logger.info(f"  enable_projection_head: {self.enable_projection_head}")
        logger.info(f"  head_residual: {self.head_residual}")
        logger.info(f"  head_zero_init: {self.head_zero_init}")
        logger.info(f"  head_use_gelu: {self.head_use_gelu}")
        logger.info(f"  head_use_ln: {self.head_use_ln}")

        logger.info("Adapter Parameters:")
        logger.info(f"  input_emb_lora_rank: {self.input_emb_lora_rank}")
        logger.info(f"  input_emb_lora_alpha: {self.input_emb_lora_alpha}")
        logger.info(f"  train_input_emb_lora: {self.train_input_emb_lora}")
        logger.info(f"  attn_lora_rank: {self.attn_lora_rank}")
        logger.info(f"  attn_lora_alpha: {self.attn_lora_alpha}")
        logger.info(f"  attn_lora_blocks: {self.attn_lora_blocks}")
        logger.info(f"  train_attn_lora: {self.train_attn_lora}")

        logger.info("PersRec Settings:")
        logger.info(f"  eval_seq_length: {self.eval_seq_length}")
        logger.info(f"  persrec_enable: {self.persrec_enable}")
        logger.info(f"  persrec_num_tokens: {self.persrec_num_tokens}")
        logger.info(f"  persrec_pretrain_len: {self.persrec_pretrain_len}")
        logger.info(f"  persrec_recent_len: {self.persrec_recent_len}")
        logger.info(f"  persrec_eval_use_full_seq: {self.persrec_eval_use_full_seq}")
        logger.info(f"  persrec_train_mode: {self.persrec_train_mode}")
        logger.info(f"  eval_protocol: {self.eval_protocol}")
        logger.info(f"  last_k_eval_test: {self.last_k_eval_test}")
        logger.info(f"  streaming_eval_last_k: {self.streaming_eval_last_k}")

        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  grad_clip: {self.grad_clip}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info(f"  noise_ratio: {self.noise_ratio}")
        logger.info(f"  early_stop_patience: {self.early_stop_patience}")

        logger.info("Checkpoint Settings:")
        logger.info(f"  pretrained_ckpt_path: {self.pretrained_ckpt_path}")
        logger.info(f"  strict_load_pretrained: {self.strict_load_pretrained}")
        logger.info(f"  ckpt_prefix_to_strip: {self.ckpt_prefix_to_strip}")
        logger.info(f"  infer_ckpt_config: {self.infer_ckpt_config}")
        logger.info(f"  preserve_max_seq_length: {self.preserve_max_seq_length}")

        logger.info("Device Settings:")
        logger.info(f"  device: {self.device}")
        logger.info("============================")


def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict or a state_dict-like object.")
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


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


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, val in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = val
        else:
            cleaned[key] = val
    return cleaned


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: Optional[str]) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    cleaned = {}
    for key, val in state_dict.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix) :]] = val
        else:
            cleaned[key] = val
    return cleaned


def _auto_strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["_orig_mod.", "model.", "net.", "encoder.", "sasrec.", "student."]
    total = len(state_dict)
    if total == 0:
        return state_dict
    best_prefix = None
    best_count = 0
    for prefix in prefixes:
        count = sum(1 for k in state_dict.keys() if k.startswith(prefix))
        if count > best_count:
            best_count = count
            best_prefix = prefix
    if best_prefix is not None and best_count / total >= 0.5:
        logger.info("Auto-stripping checkpoint prefix: %s", best_prefix)
        return _strip_prefix(state_dict, best_prefix)
    return state_dict


def _maybe_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: Optional[str]) -> Dict[str, torch.Tensor]:
    if prefix:
        return _strip_prefix(state_dict, prefix)
    return _auto_strip_prefix(state_dict)


def infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor], config: BaselineConfig) -> BaselineConfig:
    if any(k.endswith(".uvqk") for k in state_dict.keys()):
        config.backbone = "hstu"
    if "item_emb.weight" in state_dict:
        config.hidden_units = int(state_dict["item_emb.weight"].shape[1])
    if "pos_emb.weight" in state_dict:
        if config.max_seq_length is None or int(config.max_seq_length) <= 0:
            config.max_seq_length = int(state_dict["pos_emb.weight"].shape[0]) - 1

    block_indices = []
    for key in state_dict.keys():
        if key.startswith("blocks.") or key.startswith("encoder.layer.") or key.startswith("item_encoder.layer."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
            elif len(parts) > 3 and parts[2].isdigit():
                block_indices.append(int(parts[2]))
    if block_indices:
        config.num_blocks = max(block_indices) + 1

    if any(k.endswith(".rel_pos_bias.bias") for k in state_dict.keys()):
        config.hstu_enable_relative_attention_bias = True

    if "input_emb_lora.rows.weight" in state_dict:
        config.input_emb_lora_rank = int(state_dict["input_emb_lora.rows.weight"].shape[1])
    for key, value in state_dict.items():
        if key.endswith(".attn.c_attn_lora.down.weight") and value.ndim == 2:
            config.attn_lora_rank = int(value.shape[0])
            config.attn_lora_blocks = "all"
            break

    if config.hidden_units % config.num_heads != 0:
        logger.warning(
            "hidden_units (%s) not divisible by num_heads (%s). Forcing num_heads=1.",
            config.hidden_units,
            config.num_heads,
        )
        config.num_heads = 1
    return config


def apply_config_from_checkpoint_payload(config: BaselineConfig, ckpt_payload: Optional[Dict[str, Any]]) -> BaselineConfig:
    if not isinstance(ckpt_payload, dict):
        return config
    ckpt_config = ckpt_payload.get("config")
    if not isinstance(ckpt_config, dict):
        return config

    keys = [
        "backbone",
        "max_seq_length",
        "hidden_units",
        "num_blocks",
        "num_heads",
        "dropout_rate",
        "right_align_positions",
        "sasrec_attention_norm",
        "use_flash_attention",
        "use_gradient_checkpointing",
        "hstu_linear_dim",
        "hstu_attention_dim",
        "hstu_linear_activation",
        "hstu_attn_dropout",
        "hstu_enable_relative_attention_bias",
        "hstu_normalization",
        "hstu_concat_ua",
        "hstu_epsilon",
    ]
    applied = {}
    for key in keys:
        if key in ckpt_config and hasattr(config, key):
            setattr(config, key, ckpt_config[key])
            applied[key] = ckpt_config[key]
    if applied:
        logger.info("Applied architecture config from checkpoint: %s", applied)
    return config


def load_pretrained_backbone(
    model: nn.Module, ckpt_path: str, state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> None:
    if state_dict is None:
        if not ckpt_path:
            logger.warning("No pretrained_ckpt_path provided; skipping backbone load.")
            return
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, trust_pickle=True)
        state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
    state_dict = _maybe_strip_prefix(state_dict, getattr(model.config, "ckpt_prefix_to_strip", None))
    filtered = {}
    pos_weight = None
    rel_bias_weights: Dict[str, torch.Tensor] = {}
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k not in model_state and not k.startswith("item_emb.") and not k.startswith("pos_emb."):
            continue
        if k == "item_emb.weight":
            if hasattr(model, "item_emb") and v.shape == model.item_emb.weight.shape:
                filtered[k] = v
            elif hasattr(model, "item_emb") and v.shape[0] + 1 == model.item_emb.weight.shape[0] and v.shape[1] == model.item_emb.weight.shape[1]:
                new_weight = model.item_emb.weight.detach().clone()
                new_weight.zero_()
                new_weight[0] = v[0]
                new_weight[1] = v[1:].mean(dim=0) if v.size(0) > 1 else torch.zeros_like(v[0])
                new_weight[2:] = v[1:]
                filtered[k] = new_weight
                logger.info("Expanded item_emb.weight with UNK row (shifted by +1).")
            else:
                logger.warning(
                    "Skipped item_emb.weight due to shape mismatch (ckpt=%s, model=%s).",
                    v.shape,
                    model.item_emb.weight.shape,
                )
        elif k == "pos_emb.weight":
            pos_weight = v
            if hasattr(model, "pos_emb") and v.shape == model.pos_emb.weight.shape:
                filtered[k] = v
        elif k.endswith(".rel_pos_bias.bias") and k in model_state:
            if v.shape == model_state[k].shape:
                filtered[k] = v
            elif v.ndim == 1 and model_state[k].ndim == 1:
                rel_bias_weights[k] = v
        elif k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("Missing keys when loading backbone: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading backbone: %s", unexpected)
    if pos_weight is not None and hasattr(model, "pos_emb") and pos_weight.shape != model.pos_emb.weight.shape:
        # Right-align short positional embeddings into the tail of the longer table.
        with torch.no_grad():
            target = model.pos_emb.weight
            src_len = pos_weight.size(0)
            tgt_len = target.size(0)
            if src_len >= 2 and tgt_len > src_len:
                offset = tgt_len - src_len
                # Preserve PAD at 0; copy 1..src_len-1 into the tail positions.
                target[offset:] = pos_weight.to(target.dtype).to(target.device)
                logger.info(
                    "Expanded pos_emb.weight by right-aligning (ckpt=%s -> model=%s, offset=%s).",
                    tuple(pos_weight.shape),
                    tuple(target.shape),
                    offset,
                )
            else:
                logger.warning(
                    "Skipped pos_emb.weight due to shape mismatch (ckpt=%s, model=%s).",
                    pos_weight.shape,
                    model.pos_emb.weight.shape,
                )
    for key, src_weight in rel_bias_weights.items():
        target = model_state.get(key)
        if target is None:
            continue
        src_len = int(src_weight.numel())
        tgt_len = int(target.numel())
        if src_len >= tgt_len or (tgt_len - src_len) % 2 != 0:
            logger.warning(
                "Skipped %s due to shape mismatch (ckpt=%s, model=%s).",
                key,
                tuple(src_weight.shape),
                tuple(target.shape),
            )
            continue
        offset = (tgt_len - src_len) // 2
        with torch.no_grad():
            param = model.get_parameter(key)
            param.data[offset : offset + src_len].copy_(
                src_weight.to(device=param.device, dtype=param.dtype)
            )
        logger.info(
            "Expanded %s by center-aligning relative bias (ckpt=%s -> model=%s, offset=%s).",
            key,
            tuple(src_weight.shape),
            tuple(target.shape),
            offset,
        )


def initialize_head_as_identity(model: nn.Module) -> None:
    if not hasattr(model, "proj_linear"):
        return
    residual = bool(getattr(model.config, "head_residual", False))
    zero_init = bool(getattr(model.config, "head_zero_init", False))
    use_ln = bool(getattr(model.config, "head_use_ln", True))
    if residual and zero_init:
        logger.info("Initializing projection head as residual-zero (output starts as identity).")
    else:
        logger.info("Initializing projection head as identity-like (linear + layernorm).")
    with torch.no_grad():
        linear = model.proj_linear
        if hasattr(linear, "weight") and linear.weight is not None:
            if residual and zero_init:
                linear.weight.zero_()
            else:
                linear.weight.zero_()
                dim = min(linear.weight.size(0), linear.weight.size(1))
                linear.weight[:dim, :dim].copy_(torch.eye(dim, device=linear.weight.device, dtype=linear.weight.dtype))
        if hasattr(linear, "bias") and linear.bias is not None:
            linear.bias.zero_()
        if use_ln and hasattr(model, "proj_ln"):
            if model.proj_ln.weight is not None:
                model.proj_ln.weight.fill_(1.0)
            if model.proj_ln.bias is not None:
                model.proj_ln.bias.zero_()


def resolve_dataset_config(config: BaselineConfig) -> None:
    spec = resolve_loo_dataset(config.dataset, str(config.data_dir) if config.data_dir else None)
    config.dataset = spec.name
    config.data_dir = spec.root
    config.data_txt_path = spec.data_txt
    if config.max_seq_length is None or int(config.max_seq_length) <= 0:
        min_len = infer_loo_min_len(spec)
        if min_len is None or min_len <= 0:
            raise ValueError(f"Unable to infer max_seq_length from {spec.name}.")
        logger.info("Setting max_seq_length to %s based on %s.", min_len, spec.name)
        config.max_seq_length = int(min_len)


def resolve_persrec_config(config: BaselineConfig) -> None:
    if not config.persrec_enable:
        return
    if config.backbone.lower() not in {"sasrec", "hstu"}:
        raise ValueError("PersRec is only implemented for SASRec/HSTU in this script.")
    if config.persrec_recent_len is None:
        config.persrec_recent_len = int(config.eval_seq_length)
    if config.persrec_pretrain_len is None:
        if config.max_seq_length is None:
            raise ValueError("max_seq_length must be set before resolving PersRec lengths.")
        config.persrec_pretrain_len = int(config.max_seq_length) - int(config.persrec_recent_len)
    if config.persrec_pretrain_len <= 0 or config.persrec_recent_len <= 0:
        raise ValueError("PersRec requires positive pretrain_len and recent_len.")
    total_len = int(config.persrec_pretrain_len) + int(config.persrec_recent_len)
    if config.max_seq_length is None or int(config.max_seq_length) != total_len:
        logger.info(
            "Setting max_seq_length to %s for PersRec (pretrain=%s, recent=%s).",
            total_len,
            config.persrec_pretrain_len,
            config.persrec_recent_len,
        )
        config.max_seq_length = total_len
    if config.persrec_num_tokens <= 0:
        raise ValueError("persrec_num_tokens must be > 0 when persrec_enable=True.")


def resolve_eval_protocol_config(config: BaselineConfig) -> None:
    config.eval_protocol = normalize_eval_protocol(getattr(config, "eval_protocol", "legacy_loo"))
    config.last_k_eval_test = int(getattr(config, "last_k_eval_test", 0) or 0)
    if config.eval_protocol != "legacy_loo" and config.last_k_eval_test < 2:
        raise ValueError(
            f"holdout_anchor protocol requires last_k_eval_test >= 2, got {config.last_k_eval_test}."
        )


def resolve_eval_truncate_len(config: BaselineConfig) -> Optional[int]:
    if config.persrec_enable and config.persrec_eval_use_full_seq:
        return None
    return config.eval_seq_length


class SequentialSampler:
    """Sampler for sequential data that generates training batches."""

    def __init__(self, dataset: LooSequenceDataset, config: BaselineConfig, inject_noise: bool = False):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length
        self.max_item = dataset.max_item
        self.sample_id_stride = max(1, dataset.max_train_seq_len + 1)
        self.inject_noise = inject_noise
        self.noise_ratio = config.noise_ratio if inject_noise else 0.0

        self.valid_user_seqs = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            train_end = resolve_train_cutoff(
                len(seq),
                eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            if train_end > 1:
                self.valid_user_seqs.append((user, seq[:train_end]))

    @staticmethod
    def sample_negative_item(min_id: int, max_id_exclusive: int, seen_items: set) -> int:
        item_id = np.random.randint(min_id, max_id_exclusive)
        while item_id in seen_items:
            item_id = np.random.randint(min_id, max_id_exclusive)
        return item_id

    def __iter__(self):
        indices = np.random.permutation(len(self.valid_user_seqs))

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data = [self.valid_user_seqs[idx] for idx in batch_indices]

            actual_batch_size = len(batch_data)
            seq_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            pos_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            neg_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            sample_id_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            user_id_tensors = torch.zeros((actual_batch_size,), dtype=torch.long)

            for idx, (user, seq) in enumerate(batch_data):
                user_id_tensors[idx] = self.dataset.internal_to_user_id.get(user, user)
                seq_len = min(len(seq), self.max_seq_length)

                if seq_len < 1:
                    continue

                if len(seq) > self.max_seq_length:
                    start_idx = len(seq) - self.max_seq_length
                    seq = seq[-self.max_seq_length :]
                    seq_len = self.max_seq_length
                else:
                    start_idx = 0

                seq_tensors[idx, -seq_len:] = torch.tensor(seq[:seq_len])

                for pos in range(seq_len):
                    if pos < seq_len - 1:
                        pos_tensors[idx, -seq_len + pos] = seq[pos + 1]
                    global_pos = start_idx + pos
                    sample_id_tensors[idx, -seq_len + pos] = user * self.sample_id_stride + global_pos

                # Only use training-visible items for negative sampling.
                seen_set = {x for x in seq if x > 1}
                for pos in range(seq_len):
                    neg_item = self.sample_negative_item(2, self.max_item + 1, seen_set)
                    neg_tensors[idx, -seq_len + pos] = neg_item

            if self.noise_ratio > 0:
                noise_mask = (pos_tensors != 0) & (
                    torch.rand_like(pos_tensors, dtype=torch.float32) < self.noise_ratio
                )
                if noise_mask.any():
                    random_items = torch.randint(
                        2, self.max_item + 1, size=pos_tensors.shape, dtype=pos_tensors.dtype
                    )
                    same_mask = random_items == pos_tensors
                    if same_mask.any():
                        random_items[same_mask] = ((random_items[same_mask] - 2) % (self.max_item - 1)) + 2
                    pos_tensors = pos_tensors.clone()
                    pos_tensors[noise_mask] = random_items[noise_mask]

            yield {
                "input_ids": seq_tensors,
                "pos_ids": pos_tensors,
                "neg_ids": neg_tensors,
                "sample_ids": sample_id_tensors,
                "user_ids": user_id_tensors,
            }

    def __len__(self):
        return (len(self.valid_user_seqs) + self.batch_size - 1) // self.batch_size


# === Evaluation (sampled ranking) ===
def evaluate(
    model: SASRec,
    dataset,
    config: BaselineConfig,
    mode: str = "test",
    batch_size: int = 256,
    device: str = "cpu",
    use_head: bool = True,
    max_seq_length: Optional[int] = None,
    truncate_len: Optional[int] = None,
    streaming_last_k: int = 0,
) -> Dict[str, float]:
    model.eval()

    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_users = 0
    per_position: Dict[int, Dict[str, float]] = {}
    saw_nonfinite_scores = False

    users = dataset.users

    for batch_start in range(0, len(users), batch_size):
        batch_users = users[batch_start : batch_start + batch_size]
        batch_examples: List[Tuple[int, List[int], int, int]] = []

        for user in batch_users:
            seq = dataset.user_seq[user]
            target_positions = resolve_eval_target_positions(
                len(seq),
                mode=mode,
                streaming_last_k=streaming_last_k,
                eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            for target_idx in target_positions:
                target = seq[target_idx]
                if target == 1 and not config.drop_unseen_items:
                    continue
                batch_examples.append((user, seq[:target_idx], target, len(seq) - target_idx))

        if not batch_examples:
            continue

        batch_seqs = [input_seq for _, input_seq, _, _ in batch_examples]
        max_len = min(max(len(s) for s in batch_seqs), dataset.max_seq_length)
        if max_seq_length is not None and max_seq_length > 0:
            max_len = min(max_len, max_seq_length)
        use_len = max_len
        if truncate_len is not None and truncate_len > 0:
            use_len = min(truncate_len, max_len)

        input_tensor = torch.zeros((len(batch_examples), max_len), dtype=torch.long)

        for i, seq in enumerate(batch_seqs):
            seq_len = min(len(seq), use_len)
            input_tensor[i, -seq_len:] = torch.tensor(seq[-seq_len:])

        input_tensor = input_tensor.to(device)
        valid_input = input_tensor
        valid_targets = [target for _, _, target, _ in batch_examples]
        valid_user_ids = [user for user, _, _, _ in batch_examples]
        valid_rel_positions = [rel_from_end for _, _, _, rel_from_end in batch_examples]

        with torch.no_grad():
            sample_size = max(2, config.eval_sample_size)
            candidates_list = []
            use_fixed_neg = hasattr(dataset, "neg_item_by_user")
            for idx, user in enumerate(valid_user_ids):
                target = valid_targets[idx]
                candidates = [target]
                seen_items = {x for x in dataset.user_seq[user] if x > 1}
                if use_fixed_neg:
                    fixed_neg = dataset.neg_item_by_user.get(user)
                    if fixed_neg and fixed_neg not in seen_items and fixed_neg != target and fixed_neg > 1:
                        candidates.append(fixed_neg)
                while len(candidates) < sample_size:
                    neg_item = np.random.randint(2, dataset.max_item + 1)
                    if neg_item not in seen_items and neg_item not in candidates:
                        candidates.append(neg_item)
                candidates_list.append(torch.tensor(candidates, device=device))
            candidates_tensor = torch.stack(candidates_list, dim=0)

            scores = model.predict(valid_input, candidates_tensor, use_patch=False, use_head=use_head)
            if not torch.isfinite(scores).all():
                bad_examples = int((~torch.isfinite(scores).all(dim=1)).sum().item())
                logger.error(
                    "Non-finite PersRec scores during %s evaluation: %s/%s examples are invalid.",
                    mode,
                    f"{bad_examples:,}",
                    f"{scores.size(0):,}",
                )
                saw_nonfinite_scores = True
                break

        _, indices = torch.sort(scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1].cpu().numpy() + 1

        for rel_from_end, rank in zip(valid_rel_positions, ranks):
            valid_users += 1
            update_rank_metrics(per_position, rel_from_end, int(rank))
            if rank <= 10:
                hr_sum += 1
                ndcg_sum += 1 / np.log2(rank + 1)

    if saw_nonfinite_scores:
        return {
            "ndcg@10": float("nan"),
            "hr@10": float("nan"),
        }

    eval_entity = "examples" if int(streaming_last_k or 0) > 1 else "users"
    logger.info("Evaluated on %s %s", f"{valid_users:,}", eval_entity)

    return finalize_eval_metrics(
        ndcg_sum=ndcg_sum,
        hr_sum=hr_sum,
        num_examples=valid_users,
        per_position=per_position,
        streaming_last_k=streaming_last_k,
    )


def build_backbone(config: BaselineConfig, item_num: int) -> nn.Module:
    name = getattr(config, "backbone", "sasrec").lower()
    if name == "sasrec":
        return SASRec(config, item_num=item_num)
    if name == "hstu":
        return HSTU(config, item_num=item_num)
    raise ValueError(f"PersRec script only supports SASRec/HSTU backbone, got: {config.backbone}")


def build_checkpoint_tag(config: BaselineConfig) -> str:
    sasrec_suffix = ""
    if config.backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        sasrec_suffix = "_SM1"
    suffix = f"_PersRecT{config.persrec_num_tokens}"
    train_mode = normalize_persrec_train_mode(config.persrec_train_mode)
    if train_mode != "full":
        suffix += f"_{persrec_train_mode_tag(train_mode)}"
    if int(getattr(config, "input_emb_lora_rank", 0) or 0) > 0:
        suffix += f"_iel{int(config.input_emb_lora_rank)}"
    if int(getattr(config, "attn_lora_rank", 0) or 0) > 0:
        suffix += f"_alr{int(config.attn_lora_rank)}"
    return (
        f"{config.backbone}_{config.dataset}_seq{config.max_seq_length}_dim{config.hidden_units}"
        f"_L{config.num_blocks}_H{config.num_heads}{sasrec_suffix}{suffix}"
    )


def normalize_persrec_train_mode(mode: Optional[str]) -> str:
    text = str(mode or "full").strip().lower().replace("-", "_")
    aliases = {
        "full_ft": "full",
        "full_finetune": "full",
        "token": "tokens",
        "tokens_only": "tokens",
        "token_only": "tokens",
        "tokens_bias": "tokens_bias_ln",
        "tokens_bias_layernorm": "tokens_bias_ln",
        "tokens_bitfit": "tokens_bias_ln",
        "tokens_bias_ln_head": "tokens_bias_ln_head",
        "tokens_bitfit_head": "tokens_bias_ln_head",
    }
    text = aliases.get(text, text)
    valid = {"full", "tokens", "tokens_bias_ln", "tokens_bias_ln_head"}
    if text not in valid:
        raise ValueError(f"Unsupported persrec_train_mode: {mode}")
    return text


def persrec_train_mode_tag(mode: str) -> str:
    tags = {
        "full": "fullft",
        "tokens": "pefttok",
        "tokens_bias_ln": "peftbitfit",
        "tokens_bias_ln_head": "peftbitfithead",
    }
    return tags[normalize_persrec_train_mode(mode)]


def configure_persrec_trainable_params(model: nn.Module, config: BaselineConfig) -> str:
    mode = normalize_persrec_train_mode(config.persrec_train_mode)
    config.persrec_train_mode = mode

    if mode == "full":
        # full 模式就是把 backbone 整体当成普通微调来训。
        for p in model.parameters():
            p.requires_grad = True
        return mode

    # 非 full 模式先统一冻结，再按 train_mode 有选择地打开一小部分参数。
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "persrec_tokens") and isinstance(model.persrec_tokens, torch.nn.Parameter):
        # 不管是哪种 PEFT 模式，PersRec token 本身都必须可训练。
        model.persrec_tokens.requires_grad_(True)
    else:
        raise ValueError("persrec_train_mode requires a model with persrec_tokens.")

    if mode in {"tokens_bias_ln", "tokens_bias_ln_head"}:
        # 这两种模式等价于“token + bitfit + layernorm”。
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.requires_grad = True
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True

    if mode == "tokens_bias_ln_head":
        # 再进一步把 projection head 打开，便于 patch / PersRec 共用同一套输出头。
        if hasattr(model, "proj_linear"):
            for p in model.proj_linear.parameters():
                p.requires_grad = True
        if hasattr(model, "proj_ln"):
            for p in model.proj_ln.parameters():
                p.requires_grad = True

    # 这里是“尽力而为”的 LoRA 开关：只有 backbone 真的实现了对应模块才会生效。
    # 当前 HSTU 没有像 SASRec 那样的 LoRA 模块，所以很多 HSTU 实验实际上不是 LoRA。
    if bool(getattr(config, "train_input_emb_lora", False)) and hasattr(model, "input_emb_lora"):
        adapter = getattr(model, "input_emb_lora", None)
        if adapter is not None:
            for p in adapter.parameters():
                p.requires_grad = True

    if bool(getattr(config, "train_attn_lora", False)) and hasattr(model, "blocks"):
        for block in model.blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            for attr in ("c_attn_lora", "c_proj_lora"):
                adapter = getattr(attn, attr, None)
                if adapter is None:
                    continue
                for p in adapter.parameters():
                    p.requires_grad = True

    return mode


def save_model_checkpoint(model: nn.Module, config: BaselineConfig) -> Path:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{build_checkpoint_tag(config)}_best.pt"
    out_path = config.checkpoint_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "item_num": model.item_num,
            "config": {**config.__dict__, "checkpoint_dir": str(config.checkpoint_dir)},
        },
        out_path,
    )
    logger.info(f"Saved model checkpoint to {out_path}")
    return out_path


def save_item_embeddings(model: nn.Module, dataset: LooSequenceDataset, config: BaselineConfig) -> Path:
    emb = model.item_emb.weight.detach().cpu().numpy()
    emb = emb[2:]
    filename = f"{build_checkpoint_tag(config)}_item_embeddings_best.npy"
    out_path = config.checkpoint_dir / filename
    np.save(out_path, emb)
    logger.info(f"Saved item embeddings to {out_path}")
    logger.info("Item index mapping follows item2idx.json with +1 offset (PAD=0, UNK=1).")
    return out_path


def get_gradient_norm(model: nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


def get_parameter_grad_norm(parameter: Optional[torch.nn.Parameter]) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(torch.norm(parameter.grad.detach(), p=2).item())


def get_parameter_delta_norm(
    parameter: Optional[torch.nn.Parameter],
    reference: Optional[torch.Tensor],
) -> float:
    if parameter is None or reference is None:
        return 0.0
    return float(torch.norm(parameter.detach() - reference, p=2).item())


def train_persrec(
    model: nn.Module,
    train_dataset,
    config: BaselineConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Dict[str, float]:
    model = model.to(device)
    truncate_len = resolve_eval_truncate_len(config)

    # PersRec 的训练循环本质上仍然是标准 next-item BCE，
    # 只是前向里换成了“带 PersRec token 的 backbone”。
    train_sampler = SequentialSampler(train_dataset, config, inject_noise=config.noise_ratio > 0)
    steps_per_epoch = len(train_sampler)
    total_steps = config.num_epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.max_learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_learning_rate
        )
        logger.info(
            "Cosine annealing: %.1e -> %.1e for %s steps",
            config.max_learning_rate,
            config.min_learning_rate,
            f"{total_steps:,}",
        )
    elif config.scheduler_type == "cosine_with_warmup":
        warmup_steps = max(0, int(config.warmup_steps))
        if warmup_steps >= total_steps:
            warmup_steps = max(0, total_steps - 1)
        if warmup_steps == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=config.min_learning_rate
            )
            logger.info(
                "Cosine annealing: %.1e -> %.1e for %s steps (warmup skipped)",
                config.max_learning_rate,
                config.min_learning_rate,
                f"{total_steps:,}",
            )
        else:
            start_lr = min(config.warmup_start_lr, config.max_learning_rate)
            start_factor = start_lr / config.max_learning_rate if config.max_learning_rate > 0 else 1.0
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_steps,
            )
            cosine_steps = max(1, total_steps - warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_steps, eta_min=config.min_learning_rate
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
            logger.info(
                "Warmup: %.1e -> %.1e for %s steps",
                start_lr,
                config.max_learning_rate,
                f"{warmup_steps:,}",
            )
            logger.info(
                "Cosine annealing: %.1e -> %.1e for %s steps",
                config.max_learning_rate,
                config.min_learning_rate,
                f"{cosine_steps:,}",
            )
    elif config.scheduler_type not in {"", None}:
        logger.warning("Unknown scheduler_type '%s'; running without scheduler.", config.scheduler_type)

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_metrics = {"ndcg@10": -1.0, "hr@10": -1.0}
    no_improve_steps = 0
    stop_training = False
    global_step = 0
    pbar = tqdm(total=total_steps)
    initial_persrec_tokens = None
    if hasattr(model, "persrec_tokens") and isinstance(getattr(model, "persrec_tokens"), nn.Parameter):
        # 用初始 token 做参照，方便观察 token 是否真的在学。
        initial_persrec_tokens = getattr(model, "persrec_tokens").detach().clone()

    for epoch in range(config.num_epochs):
        model.train()

        for batch in train_sampler:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            # input_ids / pos_ids / neg_ids:
            # 常见为 [B, L]；如果 backbone 内部使用多负样本，neg_ids 也可能扩成 [B, L, K]。
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pos_ids = batch["pos_ids"].to(device, non_blocking=True)
            neg_ids = batch["neg_ids"].to(device, non_blocking=True)

            pos_logits, neg_logits, loss_mask = model.training_step(
                input_ids,
                pos_ids,
                neg_ids,
                patch_params=None,
                use_patch=False,
                return_loss_mask=True,
            )
            # PersRec 下这里拿到的张量长度不是原始 L，而是插入 token 槽位后的 L_total = L + T：
            # pos_logits: [B, L_total]
            # neg_logits: [B, L_total] 或 [B, L_total, K]
            # loss_mask: [B, L_total]

            valid_mask = loss_mask
            if valid_mask.any():
                pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
                neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
                raw_loss = pos_loss + neg_loss
                # PersRec token 槽位已经在 loss_mask 里被清零，这里只在有效 item 位置上做 BCE。
                loss = raw_loss[valid_mask].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite PersRec loss at step {global_step}: "
                        f"loss={float(loss.detach().item())}, "
                        f"finite_pos={bool(torch.isfinite(pos_logits).all())}, "
                        f"finite_neg={bool(torch.isfinite(neg_logits).all())}"
                    )
                loss.backward()
            else:
                loss = pos_logits.sum() * 0.0

            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.grad_clip,
                )

            grad_norm = get_gradient_norm(model)
            persrec_tokens = getattr(model, "persrec_tokens", None)
            persrec_token_grad_norm = get_parameter_grad_norm(persrec_tokens)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite PersRec grad norm at step {global_step}: "
                    f"grad_norm={grad_norm}, token_grad_norm={persrec_token_grad_norm}"
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_value = float(loss.item())

            # TokGrad / TokDelta 是专门给 PersRec 看的诊断量：
            # 前者看 token 当前有没有梯度，后者看 token 累积偏离初始化多少。
            persrec_token_delta_norm = get_parameter_delta_norm(persrec_tokens, initial_persrec_tokens)
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.update(1)

            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                logger.info(
                    "Step %06d | Epoch %03d/%03d | Loss: %.4f | LR: %.2e | Grad: %.2f | TokGrad: %.4f | TokDelta: %.4f",
                    global_step,
                    epoch + 1,
                    config.num_epochs,
                    loss_value,
                    current_lr,
                    grad_norm,
                    persrec_token_grad_norm,
                    persrec_token_delta_norm,
                )
                wandb.log(
                    {
                        "train/loss": loss_value,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm,
                        "train/persrec_token_grad_norm": persrec_token_grad_norm,
                        "train/persrec_token_delta_norm": persrec_token_delta_norm,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                )

            if config.steps_per_val_log > 0 and global_step % config.steps_per_val_log == 0:
                model.eval()
                val_metrics = evaluate(
                    model,
                    val_dataset if val_dataset is not None else train_dataset,
                    config=config,
                    mode="val",
                    device=device,
                    use_head=config.enable_projection_head,
                    truncate_len=truncate_len,
                )
                model.train()

                logger.info(
                    "Step %06d | Val - NDCG@10: %.4f, HR@10: %.4f",
                    global_step,
                    val_metrics["ndcg@10"],
                    val_metrics["hr@10"],
                )
                wandb.log(
                    {
                        "val/ndcg@10": val_metrics["ndcg@10"],
                        "val/hr@10": val_metrics["hr@10"],
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                )

                if val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
                    best_val_metrics = val_metrics
                    no_improve_steps = 0
                    if config.save_best:
                        save_model_checkpoint(model, config)
                        if config.save_item_embeddings:
                            save_item_embeddings(model, train_dataset, config)
                else:
                    no_improve_steps += 1
                    if config.early_stop_patience > 0 and no_improve_steps >= config.early_stop_patience:
                        logger.info(
                            "Early stopping: no improvement in NDCG@10 for %s validation checks.",
                            config.early_stop_patience,
                        )
                        stop_training = True
                        break

        if stop_training:
            break

        if config.steps_per_val_log <= 0:
            model.eval()
            val_metrics = evaluate(
                model,
                val_dataset if val_dataset is not None else train_dataset,
                config=config,
                mode="val",
                device=device,
                use_head=config.enable_projection_head,
                truncate_len=truncate_len,
            )
            model.train()
            if val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
                best_val_metrics = val_metrics
                no_improve_steps = 0
                if config.save_best:
                    save_model_checkpoint(model, config)
                    if config.save_item_embeddings:
                        save_item_embeddings(model, train_dataset, config)
            else:
                no_improve_steps += 1
                if config.early_stop_patience > 0 and no_improve_steps >= config.early_stop_patience:
                    logger.info(
                        "Early stopping: no improvement in NDCG@10 for %s validation checks.",
                        config.early_stop_patience,
                    )
                    break

    pbar.close()
    logger.info("Training completed. Best validation NDCG@10: %.4f", best_val_metrics["ndcg@10"])
    return best_val_metrics


def run_persrec_experiment(config: BaselineConfig, inferred_state: Optional[Dict[str, torch.Tensor]]) -> None:
    run = wandb.init(project=f"persrec-{config.dataset}", config=config.__dict__)
    if run is not None:
        config = apply_overrides_from_dict(config, dict(run.config))

    config.persrec_train_mode = normalize_persrec_train_mode(config.persrec_train_mode)
    resolve_dataset_config(config)
    resolve_eval_protocol_config(config)
    resolve_persrec_config(config)
    set_global_seed(config.seed, config.deterministic)

    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    persrec_suffix = (
        f"-PersRecT{config.persrec_num_tokens}"
        f"-pre{config.persrec_pretrain_len}-rec{config.persrec_recent_len}"
    )
    run_name = (
        f"{config.backbone}-persrec-"
        f"{config.dataset}-L{config.num_blocks}-H{config.hidden_units}-"
        f"long{config.max_seq_length}-short{config.eval_seq_length}{persrec_suffix}"
    )
    if config.backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        run_name += "-sm1"
    if normalize_eval_protocol(getattr(config, "eval_protocol", "legacy_loo")) != "legacy_loo":
        run_name += f"-anchork{int(getattr(config, 'last_k_eval_test', 0) or 0)}"
    if config.persrec_train_mode != "full":
        run_name += f"-{persrec_train_mode_tag(config.persrec_train_mode)}"
    if int(getattr(config, "input_emb_lora_rank", 0) or 0) > 0:
        run_name += f"-iel{int(config.input_emb_lora_rank)}"
    if int(getattr(config, "attn_lora_rank", 0) or 0) > 0:
        run_name += f"-alr{int(config.attn_lora_rank)}"
    if run is not None:
        run.name = run_name
        run.config.update(
            {
                **config.__dict__,
                "checkpoint_dir": str(config.checkpoint_dir),
                "data_dir": str(config.data_dir) if config.data_dir is not None else None,
                "data_txt_path": str(config.data_txt_path) if config.data_txt_path is not None else None,
            },
            allow_val_change=True,
        )
    config.log_config()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = LooSequenceDataset(config.data_txt_path, config, logger=logger)
    meta_valid_dataset = train_dataset
    test_dataset = train_dataset

    item_num = train_dataset.num_items
    model = build_backbone(config, item_num=item_num)
    if inferred_state is not None:
        if config.strict_load_pretrained:
            logger.info("Loading full checkpoint with strict=True...")
            model.load_state_dict(inferred_state, strict=True)
        else:
            load_pretrained_backbone(model, config.pretrained_ckpt_path, state_dict=inferred_state)

    model = model.to(device)
    initialize_head_as_identity(model)
    train_mode = configure_persrec_trainable_params(model, config)
    logger.info(f"PersRec train mode: {train_mode}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Running pre-train baseline on val (persrec)...")
    val_eval_dataset = meta_valid_dataset if meta_valid_dataset is not None else train_dataset
    truncate_len = resolve_eval_truncate_len(config)
    # 这里先测一遍“插入 PersRec token 但还没训练”的 baseline，
    # 便于和后续真正训练后的效果分开比较。
    val_baseline = evaluate(
        model,
        val_eval_dataset,
        config=config,
        mode="val",
        device=device,
        use_head=config.enable_projection_head,
        truncate_len=truncate_len,
    )
    logger.info(
        "Val PersRec - NDCG@10: %.4f, HR@10: %.4f",
        val_baseline["ndcg@10"],
        val_baseline["hr@10"],
    )
    wandb.log(
        {
            "val/pre_persrec_ndcg@10": val_baseline["ndcg@10"],
            "val/pre_persrec_hr@10": val_baseline["hr@10"],
            "progress/epoch": 0,
            "progress/step": 0,
        }
    )

    best_metrics = train_persrec(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=meta_valid_dataset,
    )

    logger.info("Running final test evaluation (persrec)...")
    truncate_len = resolve_eval_truncate_len(config)
    test_metrics = evaluate(
        model,
        test_dataset,
        config=config,
        mode="test",
        device=device,
        use_head=config.enable_projection_head,
        truncate_len=truncate_len,
    )
    logger.info(
        "Test - NDCG@10: %.4f, HR@10: %.4f",
        test_metrics["ndcg@10"],
        test_metrics["hr@10"],
    )

    wandb.log(
        {
            "test/ndcg@10": test_metrics["ndcg@10"],
            "test/hr@10": test_metrics["hr@10"],
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )

    if int(getattr(config, "streaming_eval_last_k", 0) or 0) > 1:
        stream_last_k = int(config.streaming_eval_last_k)
        logger.info(
            "Running additional streaming test evaluation over the last %s targets (persrec)...",
            stream_last_k,
        )
        streaming_metrics = evaluate(
            model,
            test_dataset,
            config=config,
            mode="test",
            device=device,
            use_head=config.enable_projection_head,
            truncate_len=truncate_len,
            streaming_last_k=stream_last_k,
        )
        wandb.log(flatten_streaming_eval_metrics("test_stream/persrec", streaming_metrics))

    wandb.finish()
    logger.info("PersRec training complete!")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PersRec-style backbone baseline.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=none_or_str, default=None)
    parser.add_argument("--checkpoint_dir", type=none_or_str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", type=str2bool, default=None)
    parser.add_argument("--backbone", type=none_or_str, default=None)

    parser.add_argument("--max_seq_length", type=none_or_int, default=None)
    parser.add_argument("--hidden_units", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--right_align_positions", type=str2bool, default=None)
    parser.add_argument("--sasrec_attention_norm", type=none_or_str, default=None)
    parser.add_argument("--use_flash_attention", type=str2bool, default=None)
    parser.add_argument("--use_gradient_checkpointing", type=str2bool, default=None)
    parser.add_argument("--hstu_linear_dim", type=int, default=None)
    parser.add_argument("--hstu_attention_dim", type=int, default=None)
    parser.add_argument("--hstu_linear_activation", type=none_or_str, default=None)
    parser.add_argument("--hstu_attn_dropout", type=float, default=None)
    parser.add_argument("--hstu_enable_relative_attention_bias", type=str2bool, default=None)
    parser.add_argument("--hstu_normalization", type=none_or_str, default=None)
    parser.add_argument("--hstu_concat_ua", type=str2bool, default=None)
    parser.add_argument("--hstu_epsilon", type=float, default=None)
    parser.add_argument("--input_emb_lora_rank", type=int, default=None)
    parser.add_argument("--input_emb_lora_alpha", type=float, default=None)
    parser.add_argument("--train_input_emb_lora", type=str2bool, default=None)
    parser.add_argument("--attn_lora_rank", type=int, default=None)
    parser.add_argument("--attn_lora_alpha", type=float, default=None)
    parser.add_argument("--attn_lora_blocks", type=none_or_str, default=None)
    parser.add_argument("--train_attn_lora", type=str2bool, default=None)

    parser.add_argument("--persrec_enable", type=str2bool, default=None)
    parser.add_argument("--persrec_num_tokens", type=int, default=None)
    parser.add_argument("--persrec_pretrain_len", type=none_or_int, default=None)
    parser.add_argument("--persrec_recent_len", type=none_or_int, default=None)
    parser.add_argument("--persrec_eval_use_full_seq", type=str2bool, default=None)
    parser.add_argument("--persrec_train_mode", type=none_or_str, default=None)
    parser.add_argument("--eval_seq_length", type=int, default=None)
    parser.add_argument("--eval_protocol", type=none_or_str, default=None)
    parser.add_argument("--last_k_eval_test", type=int, default=None)
    parser.add_argument("--streaming_eval_last_k", type=int, default=None)
    parser.add_argument("--token_sweep", type=none_or_str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_learning_rate", type=float, default=None)
    parser.add_argument("--min_learning_rate", type=float, default=None)
    parser.add_argument("--scheduler_type", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_start_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--steps_per_train_log", type=int, default=None)
    parser.add_argument("--steps_per_val_log", type=int, default=None)
    parser.add_argument("--eval_sample_size", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)

    parser.add_argument("--drop_unseen_items", type=str2bool, default=None)
    parser.add_argument("--strict_load_pretrained", type=str2bool, default=None)
    parser.add_argument("--ckpt_prefix_to_strip", type=none_or_str, default=None)
    parser.add_argument(
        "--pretrained_ckpt_path",
        type=none_or_str,
        default=None,
        action=_StoreProvided,
    )
    parser.add_argument("--infer_ckpt_config", type=str2bool, default=None)
    parser.add_argument("--preserve_max_seq_length", type=str2bool, default=None)
    parser.add_argument("--save_best", type=str2bool, default=None)
    parser.add_argument("--save_item_embeddings", type=str2bool, default=None)
    return parser


def apply_cli_overrides(config: BaselineConfig, args: argparse.Namespace) -> BaselineConfig:
    mapping = {
        "dataset": "dataset",
        "device": "device",
        "seed": "seed",
        "deterministic": "deterministic",
        "backbone": "backbone",
        "max_seq_length": "max_seq_length",
        "hidden_units": "hidden_units",
        "num_blocks": "num_blocks",
        "num_heads": "num_heads",
        "dropout_rate": "dropout_rate",
        "right_align_positions": "right_align_positions",
        "sasrec_attention_norm": "sasrec_attention_norm",
        "use_flash_attention": "use_flash_attention",
        "use_gradient_checkpointing": "use_gradient_checkpointing",
        "hstu_linear_dim": "hstu_linear_dim",
        "hstu_attention_dim": "hstu_attention_dim",
        "hstu_linear_activation": "hstu_linear_activation",
        "hstu_attn_dropout": "hstu_attn_dropout",
        "hstu_enable_relative_attention_bias": "hstu_enable_relative_attention_bias",
        "hstu_normalization": "hstu_normalization",
        "hstu_concat_ua": "hstu_concat_ua",
        "hstu_epsilon": "hstu_epsilon",
        "input_emb_lora_rank": "input_emb_lora_rank",
        "input_emb_lora_alpha": "input_emb_lora_alpha",
        "train_input_emb_lora": "train_input_emb_lora",
        "attn_lora_rank": "attn_lora_rank",
        "attn_lora_alpha": "attn_lora_alpha",
        "attn_lora_blocks": "attn_lora_blocks",
        "train_attn_lora": "train_attn_lora",
        "persrec_enable": "persrec_enable",
        "persrec_num_tokens": "persrec_num_tokens",
        "persrec_pretrain_len": "persrec_pretrain_len",
        "persrec_recent_len": "persrec_recent_len",
        "persrec_eval_use_full_seq": "persrec_eval_use_full_seq",
        "persrec_train_mode": "persrec_train_mode",
        "eval_seq_length": "eval_seq_length",
        "eval_protocol": "eval_protocol",
        "last_k_eval_test": "last_k_eval_test",
        "streaming_eval_last_k": "streaming_eval_last_k",
        "batch_size": "batch_size",
        "num_epochs": "num_epochs",
        "max_learning_rate": "max_learning_rate",
        "min_learning_rate": "min_learning_rate",
        "scheduler_type": "scheduler_type",
        "warmup_steps": "warmup_steps",
        "warmup_start_lr": "warmup_start_lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "steps_per_train_log": "steps_per_train_log",
        "steps_per_val_log": "steps_per_val_log",
        "eval_sample_size": "eval_sample_size",
        "early_stop_patience": "early_stop_patience",
        "drop_unseen_items": "drop_unseen_items",
        "strict_load_pretrained": "strict_load_pretrained",
        "ckpt_prefix_to_strip": "ckpt_prefix_to_strip",
        "pretrained_ckpt_path": "pretrained_ckpt_path",
        "infer_ckpt_config": "infer_ckpt_config",
        "preserve_max_seq_length": "preserve_max_seq_length",
        "save_best": "save_best",
        "save_item_embeddings": "save_item_embeddings",
    }
    for arg_name, attr_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, attr_name, value)

    # `--pretrained_ckpt_path none` should explicitly clear the preset default.
    if getattr(args, "_pretrained_ckpt_path_provided", False):
        config.pretrained_ckpt_path = getattr(args, "pretrained_ckpt_path", None)

    if args.data_dir is not None:
        config.data_dir = Path(args.data_dir)
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = Path(args.checkpoint_dir)
    return config


def apply_overrides_from_dict(config: BaselineConfig, values: Mapping[str, Any]) -> BaselineConfig:
    mapping = {
        "dataset": "dataset",
        "device": "device",
        "seed": "seed",
        "deterministic": "deterministic",
        "backbone": "backbone",
        "max_seq_length": "max_seq_length",
        "hidden_units": "hidden_units",
        "num_blocks": "num_blocks",
        "num_heads": "num_heads",
        "dropout_rate": "dropout_rate",
        "right_align_positions": "right_align_positions",
        "sasrec_attention_norm": "sasrec_attention_norm",
        "use_flash_attention": "use_flash_attention",
        "use_gradient_checkpointing": "use_gradient_checkpointing",
        "hstu_linear_dim": "hstu_linear_dim",
        "hstu_attention_dim": "hstu_attention_dim",
        "hstu_linear_activation": "hstu_linear_activation",
        "hstu_attn_dropout": "hstu_attn_dropout",
        "hstu_enable_relative_attention_bias": "hstu_enable_relative_attention_bias",
        "hstu_normalization": "hstu_normalization",
        "hstu_concat_ua": "hstu_concat_ua",
        "hstu_epsilon": "hstu_epsilon",
        "input_emb_lora_rank": "input_emb_lora_rank",
        "input_emb_lora_alpha": "input_emb_lora_alpha",
        "train_input_emb_lora": "train_input_emb_lora",
        "attn_lora_rank": "attn_lora_rank",
        "attn_lora_alpha": "attn_lora_alpha",
        "attn_lora_blocks": "attn_lora_blocks",
        "train_attn_lora": "train_attn_lora",
        "persrec_enable": "persrec_enable",
        "persrec_num_tokens": "persrec_num_tokens",
        "persrec_pretrain_len": "persrec_pretrain_len",
        "persrec_recent_len": "persrec_recent_len",
        "persrec_eval_use_full_seq": "persrec_eval_use_full_seq",
        "persrec_train_mode": "persrec_train_mode",
        "eval_seq_length": "eval_seq_length",
        "eval_protocol": "eval_protocol",
        "last_k_eval_test": "last_k_eval_test",
        "streaming_eval_last_k": "streaming_eval_last_k",
        "batch_size": "batch_size",
        "num_epochs": "num_epochs",
        "max_learning_rate": "max_learning_rate",
        "min_learning_rate": "min_learning_rate",
        "scheduler_type": "scheduler_type",
        "warmup_steps": "warmup_steps",
        "warmup_start_lr": "warmup_start_lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "steps_per_train_log": "steps_per_train_log",
        "steps_per_val_log": "steps_per_val_log",
        "eval_sample_size": "eval_sample_size",
        "early_stop_patience": "early_stop_patience",
        "drop_unseen_items": "drop_unseen_items",
        "strict_load_pretrained": "strict_load_pretrained",
        "ckpt_prefix_to_strip": "ckpt_prefix_to_strip",
        "pretrained_ckpt_path": "pretrained_ckpt_path",
        "infer_ckpt_config": "infer_ckpt_config",
        "preserve_max_seq_length": "preserve_max_seq_length",
        "save_best": "save_best",
        "save_item_embeddings": "save_item_embeddings",
    }
    for key, attr in mapping.items():
        value = values.get(key)
        if value is not None:
            setattr(config, attr, value)

    data_dir = values.get("data_dir")
    if data_dir is not None:
        config.data_dir = Path(data_dir)
    checkpoint_dir = values.get("checkpoint_dir")
    if checkpoint_dir is not None:
        config.checkpoint_dir = Path(checkpoint_dir)
    return config


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    base_config = BaselineConfig()
    # === PersRec preset: long-train, short-view comparison ===
    # The original PerSRec setup is an end-to-end long-context personalized model.
    # We keep warm-start optional, but default to the long checkpoint rather than
    # the seq50 short model to avoid a mismatched transfer setup.
    base_config.pretrained_ckpt_path = DEFAULT_LONG_CKPT
    # base_config.pretrained_ckpt_path = None  # closest to the original repository: train from scratch
    base_config.max_seq_length = None
    base_config.eval_seq_length = 20
    base_config.persrec_enable = True
    base_config.persrec_num_tokens = 4
    base_config.persrec_pretrain_len = None
    base_config.persrec_recent_len = None
    base_config.persrec_eval_use_full_seq = True
    base_config.persrec_train_mode = "full"
    base_config.batch_size = 512
    base_config.num_epochs = 50
    base_config.max_learning_rate = 5e-5
    base_config.min_learning_rate = 5e-6
    base_config.scheduler_type = "cosine"
    base_config.warmup_steps = 100
    base_config.weight_decay = 0.0
    base_config.grad_clip = 1.0
    base_config.eval_sample_size = 1000
    base_config.early_stop_patience = 5
    base_config.strict_load_pretrained = False
    base_config.infer_ckpt_config = True
    base_config.preserve_max_seq_length = True
    base_config.drop_unseen_items = True
    base_config = apply_cli_overrides(base_config, args)

    resolve_dataset_config(base_config)
    resolve_eval_protocol_config(base_config)
    set_global_seed(base_config.seed, base_config.deterministic)
    desired_max_seq_length = base_config.max_seq_length

    inferred_state = None
    if base_config.pretrained_ckpt_path and Path(base_config.pretrained_ckpt_path).exists():
        ckpt = load_checkpoint(base_config.pretrained_ckpt_path, trust_pickle=True)
        if base_config.infer_ckpt_config:
            base_config = apply_config_from_checkpoint_payload(base_config, ckpt)
        inferred_state = _strip_module_prefix(_extract_state_dict(ckpt))
        inferred_state = _maybe_strip_prefix(inferred_state, base_config.ckpt_prefix_to_strip)
        if base_config.infer_ckpt_config:
            base_config = infer_config_from_state_dict(inferred_state, base_config)
            base_config = apply_cli_overrides(base_config, args)
            if base_config.preserve_max_seq_length and desired_max_seq_length is not None:
                base_config.max_seq_length = desired_max_seq_length
            resolve_eval_protocol_config(base_config)
    else:
        logger.warning("Pretrained checkpoint not found; proceeding without inference.")

    token_sweep = parse_int_list(args.token_sweep)
    if not token_sweep:
        token_sweep = [base_config.persrec_num_tokens]
    for k in token_sweep:
        config = copy.deepcopy(base_config)
        config.persrec_num_tokens = k
        set_global_seed(config.seed, config.deterministic)
        run_persrec_experiment(config, inferred_state)
