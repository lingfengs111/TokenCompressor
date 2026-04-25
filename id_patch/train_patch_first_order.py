#!/usr/bin/env python3
"""Short-view patch distillation without MixFlow unroll."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

import train_gating_patch_long_short as base

logger = base.logger

FIRST_ORDER_REMOVED_ARG_DESTS = (
    "inner_steps",
    "inner_lr",
    "inner_momentum",
    "inner_grad_clip",
    "meta_truncate_steps",
    "lambda_meta",
    "inner_reset_every",
)


def _drop_parser_dest(parser: base.argparse.ArgumentParser, dest: str) -> None:
    for action in list(parser._actions):
        if action.dest != dest:
            continue
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)
        parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)


def normalize_loss_ablation_preset(preset: Optional[str]) -> str:
    if preset is None:
        return "none"
    norm = str(preset).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "tailgt": "tail_gt",
        "tailmidgt": "tail_mid_gt",
        "tailmidrelgt": "tail_midrel_gt",
        "gtfuture": "gt_patch_future",
        "gtfuturemid": "gt_patch_future_mid",
        "gtboundary": "gt_patch_boundary",
        "all": "all_losses",
        "full": "all_losses",
    }
    norm = aliases.get(norm, norm)
    allowed = {
        "none",
        "gt_only",
        "gt_patch_future",
        "gt_patch_future_mid",
        "gt_patch_boundary",
        "tail_only",
        "tail_gt",
        "tail_mid_gt",
        "tail_midrel_gt",
        "all_losses",
    }
    if norm not in allowed:
        raise ValueError(f"Unsupported loss_ablation_preset: {preset}")
    return norm


def apply_loss_ablation_preset(config: base.SASRecConfig, preset: Optional[str]) -> str:
    norm = normalize_loss_ablation_preset(preset)
    setattr(config, "loss_ablation_preset", norm)
    if norm in {"none", "all_losses"}:
        return norm

    tail_on = float(getattr(config, "outer_tail_weight", 0.0))
    mid_on = float(getattr(config, "outer_mid_weight", 0.0))
    midrel_on = float(getattr(config, "outer_mid_rel_weight", 0.0))
    gt_on = float(getattr(config, "outer_gt_weight", 0.0))
    future_on = float(getattr(config, "outer_patch_future_weight", 0.0))
    boundary_on = float(getattr(config, "outer_patch_boundary_weight", 0.0))

    config.outer_tail_weight = tail_on if norm in {"tail_only", "tail_gt", "tail_mid_gt", "tail_midrel_gt"} else 0.0
    config.outer_mid_weight = mid_on if norm in {"tail_mid_gt", "gt_patch_future_mid"} else 0.0
    config.outer_mid_rel_weight = midrel_on if norm in {"tail_midrel_gt"} else 0.0
    config.outer_gt_weight = gt_on if norm in {
        "gt_only",
        "gt_patch_future",
        "gt_patch_future_mid",
        "gt_patch_boundary",
        "tail_gt",
        "tail_mid_gt",
        "tail_midrel_gt",
    } else 0.0
    config.outer_patch_future_weight = future_on if norm in {"gt_patch_future", "gt_patch_future_mid"} else 0.0
    config.outer_patch_boundary_weight = boundary_on if norm in {"gt_patch_boundary"} else 0.0
    return norm


def build_arg_parser() -> base.argparse.ArgumentParser:
    parser = base.build_arg_parser()
    parser.description = "Train patch with short-view distillation and no MixFlow unroll."
    for dest in FIRST_ORDER_REMOVED_ARG_DESTS:
        _drop_parser_dest(parser, dest)
    parser.add_argument(
        "--train_adapter",
        "--train_theta",
        type=base._str2bool,
        dest="train_theta",
        default=None,
        help="If set, overrides train_bias_ln/train_head for first-order distillation.",
    )
    parser.add_argument("--enable_projection_head", type=base._str2bool, default=None)
    parser.add_argument("--train_bias_ln", "--inner_train_bias_ln", dest="inner_train_bias_ln", type=base._str2bool, default=None)
    parser.add_argument("--train_head", "--inner_train_head", dest="inner_train_head", type=base._str2bool, default=None)
    parser.add_argument("--short_seq_length", dest="inner_seq_length", type=int, default=None)
    parser.add_argument("--short_eval_length", dest="eval_seq_length", type=int, default=None)
    parser.add_argument("--short_drop_prefix", dest="inner_drop_prefix", type=base._str2bool, default=None)
    parser.add_argument("--short_unk_mask_prob", dest="inner_unk_mask_prob", type=float, default=None)
    parser.add_argument("--student_loss_mode", dest="inner_loss_mode", type=str, default=None)
    parser.add_argument("--distill_lr", dest="outer_lr", type=float, default=None)
    parser.add_argument("--distill_update_every", dest="outer_update_every", type=int, default=None)
    parser.add_argument("--distill_weight_decay", dest="outer_weight_decay", type=float, default=None)
    parser.add_argument("--distill_grad_clip", dest="outer_grad_clip", type=float, default=None)
    parser.add_argument("--distill_loss_mode", dest="outer_loss_mode", type=str, default=None)
    parser.add_argument("--distill_loss_decay", dest="outer_loss_decay", type=float, default=None)
    parser.add_argument("--distill_type", dest="outer_distill", type=str, default=None)
    parser.add_argument("--distill_temperature", dest="outer_distill_temperature", type=float, default=None)
    parser.add_argument("--distill_neg_samples", dest="outer_neg_samples", type=int, default=None)
    parser.add_argument("--distill_tail_weight", dest="outer_tail_weight", type=float, default=None)
    parser.add_argument("--distill_mid_weight", dest="outer_mid_weight", type=float, default=None)
    parser.add_argument("--distill_mid_samples", dest="outer_mid_samples", type=int, default=None)
    parser.add_argument("--student_gt_weight", dest="outer_gt_weight", type=float, default=None)
    parser.add_argument(
        "--student_gt_loss_type",
        "--gt_loss_type",
        dest="outer_gt_loss_type",
        choices=["bce", "sampled_softmax"],
        type=str,
        default=None,
    )
    parser.add_argument("--student_gt_num_negatives", dest="outer_gt_num_negatives", type=int, default=None)
    parser.add_argument("--student_gt_chunk_size", dest="outer_gt_chunk_size", type=int, default=None)
    parser.add_argument("--distill_mid_rel_weight", dest="outer_mid_rel_weight", type=float, default=None)
    parser.add_argument("--patch_future_weight", dest="outer_patch_future_weight", type=float, default=None)
    parser.add_argument("--patch_future_steps", dest="outer_patch_future_steps", type=int, default=None)
    parser.add_argument("--patch_boundary_weight", dest="outer_patch_boundary_weight", type=float, default=None)
    parser.add_argument("--patch_boundary_steps", dest="outer_patch_boundary_steps", type=int, default=None)
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--sampled_softmax_chunk_size", type=int, default=None)
    parser.add_argument("--eval_adapt_steps", dest="meta_test_adapt_steps", type=int, default=None)
    parser.add_argument("--eval_adapt_lr", dest="meta_test_adapt_lr", type=float, default=None)
    parser.add_argument(
        "--eval_adapt_from_trained",
        dest="meta_test_init_from_trained",
        type=base._str2bool,
        default=None,
    )
    parser.add_argument("--eval_unk_mask_prob", dest="meta_test_unk_mask_prob", type=float, default=None)
    parser.add_argument("--loss_ablation_preset", type=str, default=None)
    return parser


def _resolve_student_loss_mode(config: base.SASRecConfig) -> str:
    mode = getattr(config, "inner_loss_mode", "match_outer")
    if mode == "match_outer":
        mode = getattr(config, "outer_loss_mode", "all")
    return mode


def _build_first_order_run_name(config: base.SASRecConfig) -> str:
    run_name = (
        f"{config.backbone}-patch-distill-{config.dataset}-L{config.num_blocks}-H{config.hidden_units}"
        f"-P{config.num_patches}x{config.patch_len}"
        f"-short{config.inner_seq_length}-long{config.max_seq_length}"
    )
    suffix = []
    sweep_id = os.getenv("WANDB_SWEEP_ID")
    if sweep_id:
        suffix.append(f"sw{sweep_id[:4]}")
    suffix.append(f"route{str(config.patch_routing).lower()}")
    suffix.append(f"pref{config.prefix_len}")
    prefix_source = str(getattr(config, "prefix_source", "head") or "head").lower()
    if prefix_source != "head":
        suffix.append(f"ps{prefix_source}")
    if int(getattr(config, "shared_prefix_len", 0) or 0) > 0:
        suffix.append(f"sp{int(config.shared_prefix_len)}")
    if str(config.backbone).lower() == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        suffix.append("sm1")
    elif str(config.backbone).lower() == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax_custom":
        suffix.append("smcustom")
    if str(config.backbone).lower() == "sasrec" and bool(getattr(config, "sasrec_enable_relative_attention_bias", False)):
        suffix.append("rbias")
    if bool(getattr(config, "sasrec_use_rope", False)):
        suffix.append("rope")
    suffix.append("peT" if config.prefix_tail_positions else "peF")
    suffix.append("papT" if config.patch_after_prefix else "papF")
    if bool(getattr(config, "patch_use_position_embeddings", False)):
        suffix.append("ppos")
    suffix.append(f"tailw{base._format_run_float(float(getattr(config, 'outer_tail_weight', 1.0)))}")
    suffix.append(f"midw{base._format_run_float(float(getattr(config, 'outer_mid_weight', 0.0)))}")
    suffix.append(f"gtw{base._format_run_float(float(getattr(config, 'outer_gt_weight', 1.0)))}")
    gt_loss_type = str(getattr(config, "outer_gt_loss_type", "bce") or "bce").lower()
    if gt_loss_type != "bce":
        suffix.append(f"gt{gt_loss_type}")
    if float(getattr(config, "outer_patch_future_weight", 0.0) or 0.0) > 0:
        suffix.append(f"pfw{base._format_run_float(float(config.outer_patch_future_weight))}")
    if float(getattr(config, "outer_patch_boundary_weight", 0.0) or 0.0) > 0:
        suffix.append(f"pbw{base._format_run_float(float(config.outer_patch_boundary_weight))}")
    if getattr(config, "full_finetune", False):
        suffix.append("fullft")
    else:
        train_bias, train_layernorm = base.resolve_inner_bitfit_flags(config)
        if train_bias and train_layernorm:
            suffix.append("biasln")
        elif train_bias:
            suffix.append("bias")
        elif train_layernorm:
            suffix.append("ln")
        if bool(getattr(config, "inner_train_head", False)) and bool(getattr(config, "enable_projection_head", False)):
            suffix.append("head")
    if float(getattr(config, "gating_balance_weight", 0.0) or 0.0) > 0:
        suffix.append(f"gb{base._format_run_float(float(config.gating_balance_weight))}")
    if float(getattr(config, "patch_orth_weight", 0.0) or 0.0) > 0:
        suffix.append(f"po{base._format_run_float(float(config.patch_orth_weight))}")
    if float(getattr(config, "patch_inner_orth_weight", 0.0) or 0.0) > 0:
        suffix.append(f"pio{base._format_run_float(float(config.patch_inner_orth_weight))}")
    if int(getattr(config, "input_emb_lora_rank", 0) or 0) > 0:
        suffix.append(f"iel{int(config.input_emb_lora_rank)}")
    if int(getattr(config, "attn_lora_rank", 0) or 0) > 0:
        suffix.append(f"alr{int(config.attn_lora_rank)}")
        suffix.append(f"alb{str(getattr(config, 'attn_lora_blocks', 'all')).replace(',', '_')}")
    if base.normalize_eval_protocol(getattr(config, "eval_protocol", "legacy_loo")) != "legacy_loo":
        suffix.append(f"anchork{int(getattr(config, 'last_k_eval_test', 0) or 0)}")
    run_label = os.getenv("PATCH_RUN_LABEL")
    if run_label:
        suffix.append(str(run_label))
    return f"{run_name}-{'-'.join(suffix)}" if suffix else run_name


def _resolve_first_order_checkpoint_root(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.name == "gating_patch_long_short":
        return checkpoint_dir.parent / "patch_first_order"
    if checkpoint_dir.name != "patch_first_order":
        return checkpoint_dir / "patch_first_order"
    return checkpoint_dir


def log_first_order_config(config: base.SASRecConfig) -> None:
    logger.info("First-Order Patch Distillation:")
    logger.info("  dataset: %s | backbone: %s | device: %s", config.dataset, config.backbone, config.device)
    logger.info("  checkpoint_dir: %s", config.checkpoint_dir)
    logger.info("  pretrained_ckpt_path: %s", config.pretrained_ckpt_path)
    logger.info(
        "  short_seq_length: %s | eval_seq_length: %s | long_seq_length: %s",
        config.inner_seq_length,
        config.eval_seq_length,
        config.max_seq_length,
    )
    logger.info(
        "  batch_size: %s | val_batch_size: %s | num_epochs: %s",
        config.batch_size,
        config.val_batch_size,
        config.num_epochs,
    )
    logger.info(
        "  distill_update_every: %s | distill_lr: %s | distill_wd: %s | distill_grad_clip: %s",
        config.outer_update_every,
        config.outer_lr,
        config.outer_weight_decay,
        config.outer_grad_clip,
    )
    logger.info(
        "  distill_type: %s | distill_temperature: %s | distill_neg_samples: %s",
        config.outer_distill,
        config.outer_distill_temperature,
        config.outer_neg_samples,
    )
    logger.info(
        "  distill_weights: tail=%s | mid=%s | midrel=%s | gt=%s | future=%s | boundary=%s",
        config.outer_tail_weight,
        config.outer_mid_weight,
        config.outer_mid_rel_weight,
        config.outer_gt_weight,
        getattr(config, "outer_patch_future_weight", 0.0),
        getattr(config, "outer_patch_boundary_weight", 0.0),
    )
    logger.info(
        "  gt_loss_type: %s | gt_num_negatives: %s | gt_chunk_size: %s | train_num_negatives: %s | train_chunk_size: %s",
        getattr(config, "outer_gt_loss_type", "bce"),
        getattr(config, "outer_gt_num_negatives", 0),
        getattr(config, "outer_gt_chunk_size", 0),
        getattr(config, "num_negatives", 128),
        getattr(config, "sampled_softmax_chunk_size", 4096),
    )
    logger.info(
        "  regularizers: gating_balance=%s | patch_orth=%s | patch_inner_orth=%s",
        getattr(config, "gating_balance_weight", 0.0),
        getattr(config, "patch_orth_weight", 0.0),
        getattr(config, "patch_inner_orth_weight", 0.0),
    )
    logger.info(
        "  prefix_len: %s | prefix_source: %s | patch_after_prefix: %s | prefix_tail_positions: %s | short_drop_prefix: %s",
        config.prefix_len,
        getattr(config, "prefix_source", "head"),
        config.patch_after_prefix,
        config.prefix_tail_positions,
        config.inner_drop_prefix,
    )
    logger.info(
        "  short_unk_mask_prob: %s | supervise_prefix_targets: %s | drop_unseen_items: %s",
        config.inner_unk_mask_prob,
        config.supervise_prefix_targets,
        config.drop_unseen_items,
    )
    train_bias, train_layernorm = base.resolve_inner_bitfit_flags(config)
    logger.info(
        "  full_finetune: %s | train_bias_ln: %s | train_bias: %s | train_layernorm: %s | train_head: %s | train_input_emb_lora: %s | train_attn_lora: %s",
        config.full_finetune,
        config.inner_train_bias_ln,
        train_bias,
        train_layernorm,
        config.inner_train_head,
        config.train_input_emb_lora,
        config.train_attn_lora,
    )
    logger.info(
        "  sasrec_attention_norm: %s | sasrec_rel_bias: %s | sasrec_rope: %s",
        getattr(config, "sasrec_attention_norm", None),
        getattr(config, "sasrec_enable_relative_attention_bias", False),
        getattr(config, "sasrec_use_rope", False),
    )
    logger.info(
        "  eval_adapt_steps: %s | eval_adapt_lr: %s | eval_init_from_trained: %s | eval_unk_mask_prob: %s",
        config.meta_test_adapt_steps,
        config.meta_test_adapt_lr,
        config.meta_test_init_from_trained,
        config.meta_test_unk_mask_prob,
    )


def _call_training_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    return_gating: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if base._uses_similarity_scoring(getattr(model, "config", None)) and hasattr(model, "forward_features"):
        out = base._compute_sequence_logits_with_optional_user_ids(
            model,
            input_ids,
            pos_ids,
            neg_ids,
            user_ids=user_ids,
            use_patch=use_patch,
            use_head=True,
            patch_params=None,
            return_gating=return_gating,
        )
        if return_gating:
            return out
        pos_logits, neg_logits, _ = out
        return pos_logits, neg_logits
    kwargs = {
        "patch_params": None,
        "return_gating": return_gating,
        "use_patch": use_patch,
    }
    if user_ids is not None:
        try:
            return model.training_step(input_ids, pos_ids, neg_ids, user_ids=user_ids, **kwargs)
        except TypeError:
            return model.training_step(input_ids, pos_ids, neg_ids, **kwargs)
    return model.training_step(input_ids, pos_ids, neg_ids, **kwargs)


def _call_forward_features(
    model: nn.Module,
    input_ids: torch.Tensor,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
) -> torch.Tensor:
    kwargs = {
        "patch_params": None,
        "return_gating": False,
        "use_patch": use_patch,
    }
    if user_ids is not None:
        try:
            return model.forward_features(input_ids, user_ids=user_ids, **kwargs)
        except TypeError:
            return model.forward_features(input_ids, **kwargs)
    return model.forward_features(input_ids, **kwargs)


def _normalize_student_gt_loss_type(config: base.SASRecConfig) -> str:
    loss_type = str(getattr(config, "outer_gt_loss_type", "bce") or "bce").strip().lower()
    if loss_type not in {"bce", "sampled_softmax"}:
        raise ValueError(f"Unsupported outer_gt_loss_type: {loss_type}")
    return loss_type


def _resolve_student_gt_num_negatives(config: base.SASRecConfig) -> int:
    num_neg = int(getattr(config, "outer_gt_num_negatives", 0) or 0)
    if num_neg <= 0:
        num_neg = int(getattr(config, "num_negatives", 128) or 128)
    return max(1, num_neg)


def _resolve_student_gt_chunk_size(config: base.SASRecConfig) -> int:
    chunk_size = int(getattr(config, "outer_gt_chunk_size", 0) or 0)
    if chunk_size <= 0:
        chunk_size = int(getattr(config, "sampled_softmax_chunk_size", 4096) or 4096)
    return max(1, chunk_size)


def _sampled_softmax_gt_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    user_ids: torch.Tensor,
    train_dataset,
    config: base.SASRecConfig,
    *,
    use_patch: bool,
) -> torch.Tensor:
    projected = base._compute_projected_hidden_with_optional_user_ids(
        model,
        input_ids,
        user_ids=user_ids,
        use_patch=use_patch,
        use_head=True,
        return_gating=False,
    )
    valid_mask = pos_ids != 0
    if not valid_mask.any():
        return projected.sum() * 0.0

    neg_ids = base._build_outer_neg_ids(
        dataset=train_dataset,
        user_ids=user_ids,
        pos_ids=pos_ids,
        num_neg=_resolve_student_gt_num_negatives(config),
        device=pos_ids.device,
    )
    chunk_size = _resolve_student_gt_chunk_size(config)
    item_emb = base._resolve_item_embedding(model)

    proj_flat = projected[valid_mask]
    pos_ids_flat = pos_ids[valid_mask]
    neg_ids_flat = neg_ids[valid_mask]

    pos_embs_flat = item_emb(pos_ids_flat).to(proj_flat.dtype)
    pos_logits_flat = base._apply_similarity_logits(proj_flat, pos_embs_flat, config)

    total_loss = proj_flat.sum() * 0.0
    total_count = 0
    for start in range(0, proj_flat.size(0), chunk_size):
        end = min(start + chunk_size, proj_flat.size(0))
        proj_chunk = proj_flat[start:end]
        pos_logits_chunk = pos_logits_flat[start:end]
        neg_ids_chunk = neg_ids_flat[start:end]
        neg_embs_chunk = item_emb(neg_ids_chunk).to(proj_chunk.dtype)
        neg_logits_chunk = base._apply_similarity_logits(proj_chunk, neg_embs_chunk, config)
        logits_chunk = torch.cat([pos_logits_chunk.unsqueeze(1), neg_logits_chunk], dim=1)
        labels_chunk = torch.zeros(logits_chunk.size(0), dtype=torch.long, device=logits_chunk.device)
        total_loss = total_loss + F.cross_entropy(logits_chunk, labels_chunk, reduction="sum")
        total_count += logits_chunk.size(0)
    return total_loss / max(total_count, 1)


def _select_patch_future_targets(
    input_ids: torch.Tensor,
    prefix_len: int,
    patch_after_prefix: bool,
    num_targets: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_ids = input_ids.new_zeros((input_ids.size(0), num_targets))
    target_mask = torch.zeros((input_ids.size(0), num_targets), dtype=torch.bool, device=input_ids.device)
    if num_targets <= 0:
        return target_ids, target_mask

    suffix_start = prefix_len if patch_after_prefix and prefix_len > 0 else 0
    suffix = input_ids[:, suffix_start:]
    for row_idx in range(suffix.size(0)):
        valid = suffix[row_idx][suffix[row_idx] > 1]
        take = min(num_targets, int(valid.numel()))
        if take <= 0:
            continue
        target_ids[row_idx, :take] = valid[:take]
        target_mask[row_idx, :take] = True
    return target_ids, target_mask


def _select_patch_boundary_targets(
    long_input_ids: torch.Tensor,
    prefix_len: int,
    tail_len: int,
    num_targets: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_ids = long_input_ids.new_zeros((long_input_ids.size(0), num_targets))
    target_mask = torch.zeros((long_input_ids.size(0), num_targets), dtype=torch.bool, device=long_input_ids.device)
    if num_targets <= 0:
        return target_ids, target_mask

    batch_size, seq_width = long_input_ids.size()
    seq_lens = (long_input_ids != 0).sum(dim=1)
    for row_idx in range(batch_size):
        length = int(seq_lens[row_idx].item())
        if length <= 0:
            continue
        prefix_eff = min(prefix_len, length)
        tail_eff = min(tail_len, max(0, length - prefix_eff))
        mid_len = length - prefix_eff - tail_eff
        if mid_len <= 0:
            continue
        take = min(num_targets, mid_len)
        start = seq_width - length + prefix_eff + (mid_len - take)
        end = start + take
        vals = long_input_ids[row_idx, start:end]
        valid = vals > 1
        count = int(valid.sum().item())
        if count <= 0:
            continue
        target_ids[row_idx, :count] = vals[valid]
        target_mask[row_idx, :count] = True
    return target_ids, target_mask


def train_sasrec_first_order(
    model: nn.Module,
    train_dataset,
    config: base.SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Tuple[Dict[str, float], Optional[Path]]:
    """Short-view distillation: short+patch student matches a long-view teacher signal."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.train()

    short_seq_length = int(config.inner_seq_length)
    eval_seq_length = int(config.eval_seq_length)
    student_drop_prefix = bool(getattr(config, "inner_drop_prefix", False))
    short_unk_mask_prob = float(getattr(config, "inner_unk_mask_prob", 0.0) or 0.0)
    distill_update_every = int(getattr(config, "outer_update_every", 0) or 0)
    distill_grad_clip = float(getattr(config, "outer_grad_clip", 0.0) or 0.0)

    if short_seq_length > config.max_seq_length:
        logger.warning(
            "short_seq_length (%s) > max_seq_length (%s); clamping to max_seq_length.",
            short_seq_length,
            config.max_seq_length,
        )
        short_seq_length = config.max_seq_length
        config.inner_seq_length = short_seq_length
    if eval_seq_length > config.max_seq_length:
        logger.warning(
            "eval_seq_length (%s) > max_seq_length (%s); clamping to max_seq_length.",
            eval_seq_length,
            config.max_seq_length,
        )
        eval_seq_length = config.max_seq_length
        config.eval_seq_length = eval_seq_length

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    last_distill_loss = None
    last_tail_distill_loss = None
    last_mid_distill_loss = None
    last_mid_rel_distill_loss = None
    last_short_gt_loss = None
    last_patch_future_loss = None
    last_patch_boundary_loss = None
    last_gating_balance_loss = None
    last_patch_orth_loss = None
    last_patch_inner_orth_loss = None

    def _compute_distill_loss(
        short_input_ids: torch.Tensor,
        short_pos_ids: torch.Tensor,
        short_neg_ids: torch.Tensor,
        long_input_ids: torch.Tensor,
        long_pos_ids: torch.Tensor,
        long_neg_ids: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal last_mid_distill_loss
        nonlocal last_tail_distill_loss
        nonlocal last_mid_rel_distill_loss
        nonlocal last_short_gt_loss
        nonlocal last_patch_future_loss
        nonlocal last_patch_boundary_loss
        nonlocal last_gating_balance_loss
        nonlocal last_patch_orth_loss
        nonlocal last_patch_inner_orth_loss

        # 这个函数是一阶 patch 的核心：
        # short+patch 作为 student，long view 作为 teacher，
        # 然后把 tail / middle / future / boundary 等信号加权合起来。
        # 常见形状：
        # short_input_ids / short_pos_ids / short_neg_ids: [B, Ls] 或 [B, Ls, K]
        # long_input_ids / long_pos_ids / long_neg_ids: [B, Ll] 或 [B, Ll, K]
        distill_neg_samples = max(1, int(getattr(config, "outer_neg_samples", 1)))
        if distill_neg_samples > 1:
            neg_long_ids = base._build_outer_neg_ids(
                dataset=train_dataset,
                user_ids=user_ids,
                pos_ids=long_pos_ids,
                num_neg=distill_neg_samples,
                device=long_pos_ids.device,
            )
            neg_short_ids = neg_long_ids[:, -short_pos_ids.size(1) :, :]
        else:
            neg_short_ids = short_neg_ids
            neg_long_ids = long_neg_ids

        # student 看 short view + patch；teacher 看完整 long view。
        pos_short, neg_short, gating_short = _call_training_step(
            model,
            short_input_ids,
            short_pos_ids,
            neg_short_ids,
            user_ids=user_ids,
            use_patch=True,
            return_gating=True,
        )
        pos_long_full, neg_long_full = _call_training_step(
            model,
            long_input_ids,
            long_pos_ids,
            neg_long_ids,
            user_ids=user_ids,
            use_patch=False,
        )
        # pos_short: [B, Ls]
        # neg_short: [B, Ls] 或 [B, Ls, K]
        # pos_long_full: [B, Ll]
        # neg_long_full: [B, Ll] 或 [B, Ll, K]

        pos_long_tail = pos_long_full
        neg_long_tail = neg_long_full
        if pos_long_tail.size(1) != pos_short.size(1):
            # teacher 最终只拿和 short student 对齐的尾部位置来蒸馏。
            pos_long_tail = pos_long_tail[:, -pos_short.size(1) :]
            neg_long_tail = neg_long_tail[:, -pos_short.size(1) :]
        pos_long_tail = pos_long_tail.detach()
        neg_long_tail = neg_long_tail.detach()

        distill_type = getattr(config, "outer_distill", "kl")
        distill_temperature = float(getattr(config, "outer_distill_temperature", 1.0))
        if distill_temperature <= 0:
            distill_temperature = 1.0

        def _distill_logits(
            pos_student: torch.Tensor,
            neg_student: torch.Tensor,
            pos_teacher: torch.Tensor,
            neg_teacher: torch.Tensor,
        ) -> torch.Tensor:
            # 这里统一封装不同蒸馏形式，便于把 tail/mid 都复用同一套逻辑。
            if distill_type == "mse":
                if neg_student.dim() == 3:
                    neg_term = (neg_student - neg_teacher).pow(2).mean(dim=-1)
                else:
                    neg_term = (neg_student - neg_teacher).pow(2)
                return (pos_student - pos_teacher).pow(2) + neg_term
            if distill_type == "soft_bce":
                pos_targets = torch.sigmoid(pos_teacher / distill_temperature)
                neg_targets = torch.sigmoid(neg_teacher / distill_temperature)
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_student / distill_temperature,
                    pos_targets,
                    reduction="none",
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_student / distill_temperature,
                    neg_targets,
                    reduction="none",
                )
                if neg_loss.dim() == 3:
                    neg_loss = neg_loss.mean(dim=-1)
                return (pos_loss + neg_loss) * (distill_temperature**2)
            if distill_type == "kl":
                if neg_student.dim() == 3:
                    logits_student = torch.cat([pos_student.unsqueeze(-1), neg_student], dim=-1)
                    logits_teacher = torch.cat([pos_teacher.unsqueeze(-1), neg_teacher], dim=-1)
                else:
                    logits_student = torch.stack([pos_student, neg_student], dim=-1)
                    logits_teacher = torch.stack([pos_teacher, neg_teacher], dim=-1)
                logp_student = F.log_softmax(logits_student / distill_temperature, dim=-1)
                logp_teacher = F.log_softmax(logits_teacher / distill_temperature, dim=-1)
                p_teacher = logp_teacher.exp()
                return (p_teacher * (logp_teacher - logp_student)).sum(dim=-1) * (distill_temperature**2)
            raise ValueError(f"Unknown outer_distill: {distill_type}")

        raw_loss = _distill_logits(pos_short, neg_short, pos_long_tail, neg_long_tail)
        valid_mask = short_pos_ids != 0
        distill_loss_mode = getattr(config, "outer_loss_mode", "all")
        # tail distill 是 patch 线最核心的监督：让 short+patch 在尾部行为上模仿 long teacher。
        # raw_loss / valid_mask: [B, Ls]
        tail_loss = base._reduce_loss(raw_loss, valid_mask, distill_loss_mode, config.outer_loss_decay)
        last_tail_distill_loss = tail_loss.item() if isinstance(tail_loss, torch.Tensor) else float(tail_loss)
        tail_weight = float(getattr(config, "outer_tail_weight", 1.0))
        if tail_weight < 0:
            tail_weight = 0.0
        loss = tail_weight * tail_loss

        last_mid_distill_loss = None
        last_mid_rel_distill_loss = None
        last_short_gt_loss = None
        last_patch_future_loss = None
        last_patch_boundary_loss = None
        last_gating_balance_loss = None
        last_patch_orth_loss = None
        last_patch_inner_orth_loss = None
        mid_weight = float(getattr(config, "outer_mid_weight", 0.0))
        mid_rel_weight = float(getattr(config, "outer_mid_rel_weight", 0.0))
        patch_future_weight = float(getattr(config, "outer_patch_future_weight", 0.0))
        patch_boundary_weight = float(getattr(config, "outer_patch_boundary_weight", 0.0))
        patch_hidden_full: Optional[torch.Tensor] = None
        patch_start = 0
        patch_len = int(getattr(config, "patch_len", 0) or 0)
        if patch_len > 0 and (mid_weight > 0 or patch_future_weight != 0.0 or patch_boundary_weight != 0.0):
            # 只有需要直接监督 patch token 时，才额外取出 patch hidden。
            hidden_states = _call_forward_features(
                model,
                short_input_ids,
                user_ids=user_ids,
                use_patch=True,
            )
            patch_start = base.get_patch_token_start(config, short_input_ids.size(1))
            # hidden_states: [B, Ls_total, D]
            # patch_hidden_full: [B, P, D]
            patch_hidden_full = hidden_states[:, patch_start : patch_start + patch_len, :]

        if mid_weight > 0 and patch_len > 0:
            mid_samples = int(getattr(config, "outer_mid_samples", 0))
            if mid_samples <= 0:
                mid_samples = patch_len
            mid_samples = min(mid_samples, patch_len)
            if mid_samples > 0:
                # 从 long 序列中挑出被 short view 丢掉的中间位置，让 patch token 去拟合这部分 teacher 信号。
                mid_idx, mid_mask = base._select_middle_positions(
                    long_input_ids,
                    int(getattr(config, "prefix_len", 0) or 0),
                    short_seq_length,
                    mid_samples,
                )
                if mid_mask.any():
                    # mid_idx / mid_mask: [B, M]
                    mid_pos_ids = long_pos_ids.gather(1, mid_idx)
                    if neg_long_ids.dim() == 3:
                        idx_exp = mid_idx.unsqueeze(-1).expand(-1, -1, neg_long_ids.size(-1))
                        mid_neg_ids = neg_long_ids.gather(1, idx_exp)
                    else:
                        mid_neg_ids = neg_long_ids.gather(1, mid_idx)

                    mid_pos_ids = mid_pos_ids.masked_fill(~mid_mask, 0)
                    if mid_neg_ids.dim() == 3:
                        mid_neg_ids = mid_neg_ids.masked_fill(~mid_mask.unsqueeze(-1), 0)
                    else:
                        mid_neg_ids = mid_neg_ids.masked_fill(~mid_mask, 0)

                    mid_teacher_pos = pos_long_full.gather(1, mid_idx).detach()
                    if neg_long_full.dim() == 3:
                        mid_teacher_neg = neg_long_full.gather(1, idx_exp).detach()
                    else:
                        mid_teacher_neg = neg_long_full.gather(1, mid_idx).detach()

                    if patch_hidden_full is not None and patch_hidden_full.size(1) >= mid_samples:
                        patch_hidden = patch_hidden_full[:, :mid_samples, :]
                        patch_proj = model.apply_head(patch_hidden)
                        # patch_hidden / patch_proj: [B, M, D]
                        # mid_pos_ids: [B, M]
                        # mid_neg_ids: [B, M] 或 [B, M, K]

                        item_weight = model.item_emb.weight
                        pos_emb = F.embedding(mid_pos_ids, item_weight)
                        pos_mid = base._apply_similarity_logits(patch_proj, pos_emb, config)
                        if mid_neg_ids.dim() == 3:
                            neg_emb = F.embedding(mid_neg_ids, item_weight)
                            neg_mid = base._apply_similarity_logits(patch_proj, neg_emb, config)
                        else:
                            neg_emb = F.embedding(mid_neg_ids, item_weight)
                            neg_mid = base._apply_similarity_logits(patch_proj, neg_emb, config)

                        mid_raw = _distill_logits(pos_mid, neg_mid, mid_teacher_pos, mid_teacher_neg)
                        mid_valid = mid_mask & (mid_pos_ids != 0)
                        if mid_valid.any():
                            mid_loss = mid_raw[mid_valid].mean()
                            loss = loss + mid_weight * mid_loss
                            last_mid_distill_loss = mid_loss.item()

                        if mid_rel_weight != 0.0 and mid_mask.any():
                            # 不只对齐单点 logits，还要求 patch hidden 的相对结构接近 teacher middle hidden。
                            with torch.no_grad():
                                teacher_hidden = _call_forward_features(
                                    model,
                                    long_input_ids,
                                    user_ids=user_ids,
                                    use_patch=False,
                                )
                            teacher_mid = teacher_hidden.gather(
                                1, mid_idx.unsqueeze(-1).expand(-1, -1, teacher_hidden.size(-1))
                            )
                            valid_rows = mid_mask.sum(dim=1) == mid_samples
                            if valid_rows.any():
                                stud = patch_hidden[valid_rows]
                                teach = teacher_mid[valid_rows]
                                stud = F.normalize(stud, dim=-1)
                                teach = F.normalize(teach, dim=-1)
                                s_rel = stud @ stud.transpose(1, 2)
                                t_rel = teach @ teach.transpose(1, 2)
                                rel_loss = F.mse_loss(s_rel, t_rel)
                                loss = loss + mid_rel_weight * rel_loss
                                last_mid_rel_distill_loss = rel_loss.item()

        if patch_future_weight != 0.0 and patch_hidden_full is not None and patch_hidden_full.size(1) > 0:
            future_steps = int(getattr(config, "outer_patch_future_steps", 0) or 0)
            if future_steps <= 0:
                future_steps = patch_hidden_full.size(1)
            future_steps = min(future_steps, patch_hidden_full.size(1))
            if future_steps > 0:
                # 让 patch token 直接预测 short 输入里能看到的未来 item，强化 patch 的“摘要”能力。
                future_pos_ids, future_mask = _select_patch_future_targets(
                    short_input_ids,
                    prefix_len=int(getattr(config, "prefix_len", 0) or 0),
                    patch_after_prefix=bool(getattr(config, "patch_after_prefix", False)),
                    num_targets=future_steps,
                )
                if future_mask.any():
                    future_neg_ids = base._build_outer_neg_ids(
                        dataset=train_dataset,
                        user_ids=user_ids,
                        pos_ids=future_pos_ids,
                        num_neg=distill_neg_samples,
                        device=future_pos_ids.device,
                    )
                    future_neg_ids = future_neg_ids.masked_fill(~future_mask.unsqueeze(-1), 0)
                    patch_future_hidden = patch_hidden_full[:, :future_steps, :]
                    patch_future_proj = model.apply_head(patch_future_hidden)
                    # future_pos_ids / future_mask: [B, F]
                    # future_neg_ids: [B, F, K]
                    # patch_future_proj: [B, F, D]
                    item_weight = model.item_emb.weight
                    future_pos_emb = F.embedding(future_pos_ids, item_weight)
                    future_pos_logits = base._apply_similarity_logits(patch_future_proj, future_pos_emb, config)
                    future_neg_emb = F.embedding(future_neg_ids, item_weight)
                    future_neg_logits = base._apply_similarity_logits(patch_future_proj, future_neg_emb, config)

                    future_pos_loss = bce_criterion(future_pos_logits, torch.ones_like(future_pos_logits))
                    future_neg_loss = bce_criterion(future_neg_logits, torch.zeros_like(future_neg_logits)).mean(dim=-1)
                    future_raw = future_pos_loss + future_neg_loss
                    future_loss = base._reduce_loss(
                        future_raw,
                        future_mask,
                        distill_loss_mode,
                        config.outer_loss_decay,
                    )
                    loss = loss + patch_future_weight * future_loss
                    if future_mask.any():
                        last_patch_future_loss = future_loss.item()

        if patch_boundary_weight != 0.0 and patch_hidden_full is not None and patch_hidden_full.size(1) > 0:
            boundary_steps = int(getattr(config, "outer_patch_boundary_steps", 0) or 0)
            if boundary_steps <= 0:
                boundary_steps = patch_hidden_full.size(1)
            boundary_steps = min(boundary_steps, patch_hidden_full.size(1))
            if boundary_steps > 0:
                # boundary loss 专门盯住“刚好被 short 视图裁掉的那一小段”。
                boundary_pos_ids, boundary_mask = _select_patch_boundary_targets(
                    long_input_ids,
                    prefix_len=int(getattr(config, "prefix_len", 0) or 0),
                    tail_len=short_seq_length,
                    num_targets=boundary_steps,
                )
                if boundary_mask.any():
                    boundary_neg_ids = base._build_outer_neg_ids(
                        dataset=train_dataset,
                        user_ids=user_ids,
                        pos_ids=boundary_pos_ids,
                        num_neg=distill_neg_samples,
                        device=boundary_pos_ids.device,
                    )
                    boundary_neg_ids = boundary_neg_ids.masked_fill(~boundary_mask.unsqueeze(-1), 0)
                    patch_boundary_hidden = patch_hidden_full[:, :boundary_steps, :]
                    patch_boundary_proj = model.apply_head(patch_boundary_hidden)
                    # boundary_pos_ids / boundary_mask: [B, F]
                    # boundary_neg_ids: [B, F, K]
                    # patch_boundary_proj: [B, F, D]
                    item_weight = model.item_emb.weight
                    boundary_pos_emb = F.embedding(boundary_pos_ids, item_weight)
                    boundary_pos_logits = base._apply_similarity_logits(patch_boundary_proj, boundary_pos_emb, config)
                    boundary_neg_emb = F.embedding(boundary_neg_ids, item_weight)
                    boundary_neg_logits = base._apply_similarity_logits(patch_boundary_proj, boundary_neg_emb, config)

                    boundary_pos_loss = bce_criterion(boundary_pos_logits, torch.ones_like(boundary_pos_logits))
                    boundary_neg_loss = bce_criterion(boundary_neg_logits, torch.zeros_like(boundary_neg_logits)).mean(dim=-1)
                    boundary_raw = boundary_pos_loss + boundary_neg_loss
                    boundary_loss = base._reduce_loss(
                        boundary_raw,
                        boundary_mask,
                        distill_loss_mode,
                        config.outer_loss_decay,
                    )
                    loss = loss + patch_boundary_weight * boundary_loss
                    if boundary_mask.any():
                        last_patch_boundary_loss = boundary_loss.item()

        gt_weight = float(getattr(config, "outer_gt_weight", 1.0))
        if gt_weight != 0.0:
            gt_valid = short_pos_ids != 0
            gt_loss_type = _normalize_student_gt_loss_type(config)
            if gt_loss_type == "sampled_softmax":
                gt_loss = _sampled_softmax_gt_loss(
                    model=model,
                    input_ids=short_input_ids,
                    pos_ids=short_pos_ids,
                    user_ids=user_ids,
                    train_dataset=train_dataset,
                    config=config,
                    use_patch=True,
                )
            else:
                # 即便没有 teacher，student 自己也可以直接吃真实 label 的 BCE。
                pos_loss = bce_criterion(pos_short, torch.ones_like(pos_short))
                neg_loss = bce_criterion(neg_short, torch.zeros_like(neg_short))
                if neg_loss.dim() == 3:
                    neg_loss = neg_loss.mean(dim=-1)
                gt_raw = pos_loss + neg_loss
                gt_loss = base._reduce_loss(gt_raw, gt_valid, distill_loss_mode, config.outer_loss_decay)
            if isinstance(gt_loss, torch.Tensor):
                if gt_valid.any():
                    last_short_gt_loss = gt_loss.item()
            loss = loss + gt_weight * gt_loss

        gating_balance_weight = float(getattr(config, "gating_balance_weight", 0.0))
        if gating_balance_weight != 0.0 and gating_short is not None and gating_short.numel() > 0:
            gating_balance_loss = base._gating_balance_loss(gating_short)
            loss = loss + gating_balance_weight * gating_balance_loss
            last_gating_balance_loss = gating_balance_loss.item()

        patch_orth_weight = float(getattr(config, "patch_orth_weight", 0.0))
        if patch_orth_weight != 0.0 and int(getattr(config, "patch_len", 0) or 0) > 0:
            patch_orth_loss = base._patch_orthogonality_loss(model)
            loss = loss + patch_orth_weight * patch_orth_loss
            last_patch_orth_loss = patch_orth_loss.item()

        patch_inner_orth_weight = float(getattr(config, "patch_inner_orth_weight", 0.0))
        if patch_inner_orth_weight != 0.0 and int(getattr(config, "patch_len", 0) or 0) > 1:
            patch_inner_orth_loss = base._patch_inner_orthogonality_loss(model)
            loss = loss + patch_inner_orth_weight * patch_inner_orth_loss
            last_patch_inner_orth_loss = patch_inner_orth_loss.item()

        return loss

    train_cfg = replace(config, max_seq_length=config.max_seq_length, batch_size=config.batch_size)
    train_data = base.LooTrainDataset(train_dataset, train_cfg)
    collate_fn = base.build_train_collate_fn(train_data, train_cfg)
    num_workers = max(0, int(config.num_workers))
    loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(config.pin_memory),
        "collate_fn": collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(config.persistent_workers)
    train_loader = DataLoader(train_data, **loader_kwargs)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    distill_opt = torch.optim.AdamW(
        trainable_params,
        lr=config.outer_lr,
        weight_decay=config.outer_weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps)

    best_val_metrics = {"ndcg@10": -1.0, "hr@10": -1.0}
    best_ckpt_path: Optional[Path] = None
    best_val_epoch = 0
    no_improve_evals = 0
    early_stop_patience = max(0, int(getattr(config, "early_stop_patience", 0) or 0))
    early_stop_min_epochs = max(0, int(getattr(config, "early_stop_min_epochs", 0) or 0))
    early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 0.0) or 0.0)
    global_step = 0
    stop_training = False

    for epoch in range(config.num_epochs):
        model.train()
        for batch_long in train_loader:
            global_step += 1

            batch_long = base._move_batch_to_device(batch_long, device_obj)
            if config.prefix_len and config.prefix_len > 0:
                if student_drop_prefix:
                    batch_short = base._drop_prefix_from_batch(
                        batch_long,
                        config.prefix_len,
                        short_seq_length,
                        train_dataset,
                    )
                else:
                    batch_short = base._build_prefix_tail_batch(
                        batch_long,
                        config.prefix_len,
                        short_seq_length,
                        train_dataset,
                        supervise_prefix_targets=bool(getattr(config, "supervise_prefix_targets", False)),
                        prefix_source=str(getattr(config, "prefix_source", "head") or "head"),
                    )
            else:
                # 无 prefix 时，short view 就是简单地截最后一段 tail。
                batch_short = base._slice_batch_tail(batch_long, short_seq_length)

            if (not config.drop_unseen_items) and short_unk_mask_prob > 0:
                batch_short = {
                    **batch_short,
                    "input_ids": base._mask_inputs_with_unk(
                        batch_short["input_ids"], short_unk_mask_prob
                    ),
                }

            do_distill_step = distill_update_every <= 0 or global_step % distill_update_every == 0
            if do_distill_step:
                distill_opt.zero_grad(set_to_none=True)
                # 这里没有二阶展开；每次只做一次 first-order 外层更新。
                model.eval()
                loss_distill = _compute_distill_loss(
                    batch_short["input_ids"],
                    batch_short["pos_ids"],
                    batch_short["neg_ids"],
                    batch_long["input_ids"],
                    batch_long["pos_ids"],
                    batch_long["neg_ids"],
                    batch_long["internal_user_ids"],
                )
                model.train()
                loss_distill.backward()
                if distill_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, distill_grad_clip)
                distill_opt.step()
                last_distill_loss = loss_distill.item()

            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                with torch.no_grad():
                    pos_logits, neg_logits, gating = _call_training_step(
                        model,
                        batch_short["input_ids"],
                        batch_short["pos_ids"],
                        batch_short["neg_ids"],
                        user_ids=batch_short.get("internal_user_ids"),
                        use_patch=True,
                        return_gating=True,
                    )
                    pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
                    neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
                    raw_loss = pos_loss + neg_loss
                    valid_mask = batch_short["pos_ids"] != 0
                    student_loss_mode = _resolve_student_loss_mode(config)
                    student_loss = base._reduce_loss(raw_loss, valid_mask, student_loss_mode, config.outer_loss_decay)
                    if valid_mask.any():
                        base.log_metrics(
                            {
                                "train/student_bce": student_loss.item(),
                                "meta/student_bce": student_loss.item(),
                            }
                        )

                    if gating is not None and gating.numel() > 0:
                        weights = gating.detach().float().cpu()
                        avg_weights = weights.mean(dim=0)
                        log_dict = {
                            f"gating/avg_weight_{i}": avg_weights[i].item()
                            for i in range(avg_weights.numel())
                        }
                        top1 = weights.argmax(dim=1)
                        for i in range(avg_weights.numel()):
                            log_dict[f"gating/top1_frac_{i}"] = (top1 == i).float().mean().item()

                        lengths = (batch_short["input_ids"] != 0).sum(dim=1).detach().cpu()
                        if lengths.numel() > 0:
                            median = lengths.median()
                            short_mask = lengths <= median
                            long_mask = lengths > median
                            max_log = min(avg_weights.numel(), 8)
                            if short_mask.any():
                                short_avg = weights[short_mask].mean(dim=0)
                                for i in range(max_log):
                                    log_dict[f"gating/avg_weight_short_{i}"] = short_avg[i].item()
                            if long_mask.any():
                                long_avg = weights[long_mask].mean(dim=0)
                                for i in range(max_log):
                                    log_dict[f"gating/avg_weight_long_{i}"] = long_avg[i].item()

                        base.log_metrics(log_dict)
                        base.log_metrics({"gating/weight_hist": wandb.Histogram(weights.numpy())})

                if last_distill_loss is not None:
                    log_dict = {
                        "train/distill_loss": last_distill_loss,
                        "meta/outer_loss": last_distill_loss,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                    if last_tail_distill_loss is not None:
                        log_dict["train/distill_tail_loss"] = last_tail_distill_loss
                        log_dict["meta/outer_tail_loss"] = last_tail_distill_loss
                    if last_mid_distill_loss is not None:
                        log_dict["train/distill_mid_loss"] = last_mid_distill_loss
                        log_dict["meta/outer_mid_loss"] = last_mid_distill_loss
                    if last_mid_rel_distill_loss is not None:
                        log_dict["train/distill_mid_rel_loss"] = last_mid_rel_distill_loss
                        log_dict["meta/outer_mid_rel_loss"] = last_mid_rel_distill_loss
                    if last_short_gt_loss is not None:
                        log_dict["train/short_gt_loss"] = last_short_gt_loss
                        log_dict["meta/outer_gt_loss"] = last_short_gt_loss
                    if last_patch_future_loss is not None:
                        log_dict["train/patch_future_loss"] = last_patch_future_loss
                        log_dict["meta/outer_patch_future_loss"] = last_patch_future_loss
                    if last_patch_boundary_loss is not None:
                        log_dict["train/patch_boundary_loss"] = last_patch_boundary_loss
                        log_dict["meta/outer_patch_boundary_loss"] = last_patch_boundary_loss
                    if last_gating_balance_loss is not None:
                        log_dict["train/gating_balance_loss"] = last_gating_balance_loss
                        log_dict["meta/outer_gating_balance_loss"] = last_gating_balance_loss
                    if last_patch_orth_loss is not None:
                        log_dict["train/patch_orth_loss"] = last_patch_orth_loss
                        log_dict["meta/outer_patch_orth_loss"] = last_patch_orth_loss
                    if last_patch_inner_orth_loss is not None:
                        log_dict["train/patch_inner_orth_loss"] = last_patch_inner_orth_loss
                        log_dict["meta/outer_patch_inner_orth_loss"] = last_patch_inner_orth_loss
                    base.log_metrics(log_dict)
                    logger.info(
                        "Step %06d | Epoch %03d/%03d | DistillLoss: %.4f",
                        global_step,
                        epoch + 1,
                        config.num_epochs,
                        last_distill_loss,
                    )

            pbar.update(1)

        if (
            val_dataset is not None
            and config.val_eval_every_epochs > 0
            and (epoch + 1) % config.val_eval_every_epochs == 0
        ):
            model.eval()
            val_metrics = base.evaluate(
                model,
                val_dataset,
                config=config,
                mode="val",
                device=str(device_obj),
                use_patch=True,
                use_head=True,
                max_seq_length=config.max_seq_length,
                truncate_len=config.eval_seq_length,
            )
            model.train()
            base.log_metrics(
                {
                    "val/short_patch_ndcg@10": val_metrics["ndcg@10"],
                    "val/short_patch_hr@10": val_metrics["hr@10"],
                    "val/meta_patch_ndcg@10": val_metrics["ndcg@10"],
                    "val/meta_patch_hr@10": val_metrics["hr@10"],
                    "progress/epoch": epoch + 1,
                }
            )
            logger.info(
                "Epoch %03d | Val Short+Patch - NDCG@10: %.4f, HR@10: %.4f",
                epoch + 1,
                val_metrics["ndcg@10"],
                val_metrics["hr@10"],
            )
            improved = val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"] + early_stop_min_delta
            if improved:
                best_val_metrics = val_metrics
                best_val_epoch = epoch + 1
                no_improve_evals = 0
                if config.save_best_model:
                    best_ckpt_path = base.save_model_checkpoint(model, config)
                    logger.info("Saved best val checkpoint to %s", best_ckpt_path)
            elif early_stop_patience > 0 and (epoch + 1) >= early_stop_min_epochs:
                no_improve_evals += 1
                logger.info(
                    "No val improvement for %d/%d evals (best epoch %03d | best NDCG@10 %.4f)",
                    no_improve_evals,
                    early_stop_patience,
                    best_val_epoch,
                    best_val_metrics["ndcg@10"],
                )
                if no_improve_evals >= early_stop_patience:
                    logger.info(
                        "Early stopping at epoch %03d after %d evals without improvement.",
                        epoch + 1,
                        no_improve_evals,
                    )
                    stop_training = True

        if stop_training:
            break

    pbar.close()

    if val_dataset is not None and config.eval_after_train:
        logger.info("Running validation evaluation after training...")
        metrics = base.evaluate(
            model,
            val_dataset,
            config=config,
            mode="val",
            device=str(device_obj),
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=eval_seq_length,
        )
        if metrics["ndcg@10"] > best_val_metrics["ndcg@10"] + early_stop_min_delta:
            best_val_metrics = metrics
            best_val_epoch = config.num_epochs if not stop_training else epoch + 1
            if config.save_best_model:
                best_ckpt_path = base.save_model_checkpoint(model, config)
                logger.info("Saved best val checkpoint to %s", best_ckpt_path)
    if best_val_metrics["ndcg@10"] < 0:
        best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}

    return best_val_metrics, best_ckpt_path


def main() -> None:
    config = base.SASRecConfig()

    parser = build_arg_parser()
    args = parser.parse_args()
    base.apply_overrides_from_args(config, args)
    loss_preset = normalize_loss_ablation_preset(args.loss_ablation_preset)
    setattr(config, "loss_ablation_preset", loss_preset)
    if args.train_theta is not None:
        config.inner_train_bias_ln = bool(args.train_theta)
        config.inner_train_bias = bool(args.train_theta)
        config.inner_train_layernorm = bool(args.train_theta)
        config.inner_train_head = bool(args.train_theta)

    project_name = os.getenv("WANDB_PROJECT") or f"patch_first_order-{config.dataset}"
    run = wandb.init(project=project_name, config=config.__dict__)
    if run is not None:
        base.apply_overrides_from_dict(config, dict(run.config))

    base.resolve_dataset_config(config)
    base.resolve_eval_protocol_config(config)
    base.resolve_shared_prefix_config(config)
    base.set_global_seed(config.seed, config.deterministic)

    inferred_state = None
    if config.pretrained_ckpt_path and Path(config.pretrained_ckpt_path).exists():
        ckpt = base.load_checkpoint(config.pretrained_ckpt_path, trust_pickle=True)
        config = base.apply_config_from_checkpoint_payload(config, ckpt)
        inferred_state = base._strip_module_prefix(base._extract_state_dict(ckpt))
        inferred_state = base._maybe_strip_prefix(inferred_state, config.ckpt_prefix_to_strip)
        config = base.infer_config_from_state_dict(inferred_state, config)
        base.apply_overrides_from_args(config, args)
        base.resolve_eval_protocol_config(config)
        base.resolve_shared_prefix_config(config)
    else:
        logger.warning("Pretrained checkpoint not found; proceeding without inference.")

    if run is not None and "loss_ablation_preset" in run.config:
        loss_preset = normalize_loss_ablation_preset(run.config.get("loss_ablation_preset"))
    loss_preset = apply_loss_ablation_preset(config, loss_preset)
    logger.info(
        "Loss ablation preset: %s | tail=%.3f | mid=%.3f | midrel=%.3f | gt=%.3f | patch_future=%.3f | patch_boundary=%.3f",
        loss_preset,
        float(config.outer_tail_weight),
        float(config.outer_mid_weight),
        float(config.outer_mid_rel_weight),
        float(config.outer_gt_weight),
        float(getattr(config, "outer_patch_future_weight", 0.0)),
        float(getattr(config, "outer_patch_boundary_weight", 0.0)),
    )

    if run is not None:
        run.config.update(config.__dict__, allow_val_change=True)

    if str(getattr(config, "checkpoint_mode", "full")).lower() == "delta":
        if not (config.pretrained_ckpt_path and Path(config.pretrained_ckpt_path).exists()):
            raise FileNotFoundError(
                "checkpoint_mode=delta requires a valid pretrained_ckpt_path, but got: "
                f"{config.pretrained_ckpt_path}"
            )

    device_manager = base.DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_name = _build_first_order_run_name(config)
    if loss_preset != "none":
        run_name = f"{run_name}-{loss_preset}"
    if run is not None:
        run.name = run_name

    base_ckpt_dir = _resolve_first_order_checkpoint_root(Path(config.checkpoint_dir))
    config.run_tag = base._build_run_tag(config, run)
    config.checkpoint_dir = base_ckpt_dir / str(config.run_tag)
    if run is not None:
        run.config.update(
            {"checkpoint_dir": str(config.checkpoint_dir), "run_tag": config.run_tag},
            allow_val_change=True,
        )

    base.LOCAL_METRICS_LOGGER = base.LocalMetricsLogger(
        log_dir=str(config.checkpoint_dir / "logs"),
        run_name=(run.name if run is not None else run_name),
    )
    log_first_order_config(config)
    if base.LOCAL_METRICS_LOGGER is not None and base.LOCAL_METRICS_LOGGER.jsonl_path is not None:
        logger.info("Local metrics JSONL: %s", base.LOCAL_METRICS_LOGGER.jsonl_path)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base.save_run_config(config, run_name, sys.argv)

    train_dataset = base.LooSequenceDataset(config.data_txt_path, config, logger=logger)
    val_dataset = train_dataset
    test_dataset = train_dataset
    item_num = train_dataset.num_items

    model = base.build_backbone(config, item_num=item_num)
    if inferred_state is not None:
        has_optional_adapters = (
            int(getattr(config, "input_emb_lora_rank", 0) or 0) > 0
            or int(getattr(config, "attn_lora_rank", 0) or 0) > 0
        )
        if config.strict_load_pretrained and not has_optional_adapters:
            logger.info("Loading full checkpoint with strict=True...")
            model.load_state_dict(inferred_state, strict=True)
        else:
            if config.strict_load_pretrained and has_optional_adapters:
                logger.info("Optional adapters enabled; falling back to non-strict backbone loading.")
            base.load_pretrained_backbone(model, config.pretrained_ckpt_path, state_dict=inferred_state)
    model = model.to(device)

    if config.patch_routing in {"kmeans", "user_table"}:
        centers = base.build_kmeans_centers(train_dataset, model, config)
        model.meta_patch.set_kmeans_centers(centers)
        logger.info("KMeans routing centers set: %s patches.", centers.size(0))
        if config.patch_routing == "user_table":
            user_to_patch = base.build_user_patch_table(train_dataset, model, centers, config)
            model.meta_patch.set_user_table(user_to_patch)
            logger.info("User routing table set for %s users.", len(user_to_patch))

    base.initialize_head_as_identity(model)
    train_bias, train_layernorm = base.resolve_inner_bitfit_flags(config)
    adapt_param_names = base.build_bitfit_param_names(
        model,
        enable_bias=train_bias,
        enable_layernorm=train_layernorm,
        enable_head=config.inner_train_head and config.enable_projection_head,
        enable_score_head=bool(getattr(config, "inner_train_score_head", True)),
    )
    adapt_init_state = base._snapshot_params_by_name(model, adapt_param_names)
    if getattr(config, "full_finetune", False):
        for p in model.parameters():
            p.requires_grad = True
        logger.info("Full fine-tuning enabled for first-order patch distillation.")
    else:
        base.apply_bitfit_freeze(
            model,
            enable_bias=train_bias,
            enable_layernorm=train_layernorm,
            enable_head=config.inner_train_head and config.enable_projection_head,
            enable_score_head=bool(getattr(config, "inner_train_score_head", True)),
            enable_input_emb_lora=config.train_input_emb_lora,
            enable_attn_lora=config.train_attn_lora,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    if config.eval_before_train:
        logger.info("Running pre-train backbone baseline on val (pure pretrained backbone, direct eval)...")
        val_baseline = base.evaluate_pure_backbone_baseline(
            val_dataset,
            config=config,
            device=device,
            state_dict=inferred_state,
            mode="val",
        )
        logger.info(
            "Val Backbone Baseline - NDCG@10: %.4f, HR@10: %.4f",
            val_baseline["ndcg@10"],
            val_baseline["hr@10"],
        )
        base.log_metrics(
            {
                "val/baseline_ndcg@10": val_baseline["ndcg@10"],
                "val/baseline_hr@10": val_baseline["hr@10"],
                "progress/epoch": 0,
                "progress/step": 0,
            }
        )

        logger.info("Running pre-train short+patch evaluation on val...")
        val_short_patch = base.evaluate(
            model,
            val_dataset,
            config=config,
            mode="val",
            device=device,
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=adapt_param_names,
            bitfit_init_state=adapt_init_state,
        )
        logger.info(
            "Val Short+Patch (pre-train) - NDCG@10: %.4f, HR@10: %.4f",
            val_short_patch["ndcg@10"],
            val_short_patch["hr@10"],
        )
        base.log_metrics(
            {
                "val/pre_short_patch_ndcg@10": val_short_patch["ndcg@10"],
                "val/pre_short_patch_hr@10": val_short_patch["hr@10"],
                "val/pre_meta_patch_ndcg@10": val_short_patch["ndcg@10"],
                "val/pre_meta_patch_hr@10": val_short_patch["hr@10"],
                "progress/epoch": 0,
                "progress/step": 0,
            }
        )

    best_metrics, best_ckpt_path = train_sasrec_first_order(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=val_dataset,
    )

    if best_ckpt_path is not None and Path(best_ckpt_path).exists():
        logger.info("Loading best val checkpoint for test: %s", best_ckpt_path)
        best_ckpt = base.load_checkpoint(str(best_ckpt_path), trust_pickle=True)
        best_state = base._strip_module_prefix(base._extract_state_dict(best_ckpt))
        ckpt_mode = str(best_ckpt.get("checkpoint_mode", "full")).lower() if isinstance(best_ckpt, dict) else "full"
        if ckpt_mode == "delta":
            base_path = best_ckpt.get("base_ckpt_path") if isinstance(best_ckpt, dict) else None
            if base_path and config.pretrained_ckpt_path and str(base_path) != str(config.pretrained_ckpt_path):
                logger.warning(
                    "Delta checkpoint base_ckpt_path (%s) differs from current pretrained_ckpt_path (%s).",
                    base_path,
                    config.pretrained_ckpt_path,
                )
            model.load_state_dict(best_state, strict=False)
        else:
            model.load_state_dict(best_state, strict=True)

    logger.info("Running final test evaluation (baseline: pure pretrained backbone, direct test)...")
    baseline_metrics = base.evaluate_pure_backbone_baseline(
        test_dataset,
        config=config,
        device=device,
        state_dict=inferred_state,
        mode="test",
    )
    logger.info(
        "Backbone Baseline Test - NDCG@10: %.4f, HR@10: %.4f",
        baseline_metrics["ndcg@10"],
        baseline_metrics["hr@10"],
    )

    logger.info("Running final adapted short-view test evaluation (no patch)...")
    trained_short_metrics = base.evaluate(
        model,
        test_dataset,
        config=config,
        mode="meta-test",
        device=device,
        use_patch=False,
        use_head=True,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
        theta_names=adapt_param_names,
        bitfit_init_state=adapt_init_state,
    )
    logger.info(
        "Trained Short No-Patch Test - NDCG@10: %.4f, HR@10: %.4f",
        trained_short_metrics["ndcg@10"],
        trained_short_metrics["hr@10"],
    )

    logger.info("Running final adapted short+patch test evaluation...")
    short_patch_metrics = base.evaluate(
        model,
        test_dataset,
        config=config,
        mode="meta-test",
        device=device,
        use_patch=True,
        use_head=True,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
        theta_names=adapt_param_names,
        bitfit_init_state=adapt_init_state,
    )
    logger.info(
        "Short+Patch Test - NDCG@10: %.4f, HR@10: %.4f",
        short_patch_metrics["ndcg@10"],
        short_patch_metrics["hr@10"],
    )

    base.log_metrics(
        {
            "test/baseline_ndcg@10": baseline_metrics["ndcg@10"],
            "test/baseline_hr@10": baseline_metrics["hr@10"],
            "test/trained_short_no_patch_ndcg@10": trained_short_metrics["ndcg@10"],
            "test/trained_short_no_patch_hr@10": trained_short_metrics["hr@10"],
            "test/short_patch_ndcg@10": short_patch_metrics["ndcg@10"],
            "test/short_patch_hr@10": short_patch_metrics["hr@10"],
            "test/meta_patch_ndcg@10": short_patch_metrics["ndcg@10"],
            "test/meta_patch_hr@10": short_patch_metrics["hr@10"],
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )

    if int(getattr(config, "streaming_eval_last_k", 0) or 0) > 1:
        stream_last_k = int(config.streaming_eval_last_k)
        logger.info(
            "Running additional streaming test evaluation over the last %s targets...",
            stream_last_k,
        )
        baseline_stream_metrics = base.evaluate_pure_backbone_baseline(
            test_dataset,
            config=config,
            device=device,
            state_dict=inferred_state,
            mode="test",
            streaming_last_k=stream_last_k,
        )
        trained_short_stream_metrics = base.evaluate(
            model,
            test_dataset,
            config=config,
            mode="meta-test",
            device=device,
            use_patch=False,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=adapt_param_names,
            bitfit_init_state=adapt_init_state,
            streaming_last_k=stream_last_k,
        )
        short_patch_stream_metrics = base.evaluate(
            model,
            test_dataset,
            config=config,
            mode="meta-test",
            device=device,
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=adapt_param_names,
            bitfit_init_state=adapt_init_state,
            streaming_last_k=stream_last_k,
        )
        base.log_metrics(
            {
                **base.flatten_streaming_eval_metrics("test_stream/baseline", baseline_stream_metrics),
                **base.flatten_streaming_eval_metrics("test_stream/trained_short_no_patch", trained_short_stream_metrics),
                **base.flatten_streaming_eval_metrics("test_stream/short_patch", short_patch_stream_metrics),
                **base.flatten_streaming_eval_metrics("test_stream/meta_patch", short_patch_stream_metrics),
                **base.flatten_streaming_eval_test_aliases("baseline", baseline_stream_metrics),
                **base.flatten_streaming_eval_test_aliases(
                    "trained_short_no_patch",
                    trained_short_stream_metrics,
                ),
                **base.flatten_streaming_eval_test_aliases("short_patch", short_patch_stream_metrics),
                **base.flatten_streaming_eval_test_aliases("meta_patch", short_patch_stream_metrics),
            }
        )

    if run is not None and best_ckpt_path is not None and Path(best_ckpt_path).exists():
        run.save(str(best_ckpt_path))

    if config.save_item_embeddings:
        emb_path = base.save_item_embeddings(model, train_dataset, config)
        if run is not None and emb_path is not None:
            run.save(str(emb_path))

    metrics_jsonl = None
    metrics_csv = None
    if base.LOCAL_METRICS_LOGGER is not None:
        metrics_jsonl = base.LOCAL_METRICS_LOGGER.jsonl_path
        csv_path = base.LOCAL_METRICS_LOGGER.export_csv()
        if csv_path is not None:
            metrics_csv = csv_path
            logger.info("Local metrics CSV: %s", csv_path)
        base.LOCAL_METRICS_LOGGER.close()

    base.save_run_summary(
        config,
        run_name,
        best_metrics,
        baseline_metrics,
        short_patch_metrics,
        best_ckpt_path,
        metrics_jsonl=metrics_jsonl,
        metrics_csv=metrics_csv,
    )

    wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
