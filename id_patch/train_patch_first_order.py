#!/usr/bin/env python3
"""First-order (no MixFlow) patch distillation: short+patch -> long teacher."""

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


def build_arg_parser() -> base.argparse.ArgumentParser:
    parser = base.build_arg_parser()
    parser.description = "Train patch with first-order outer distillation (no MixFlow)."
    parser.add_argument(
        "--train_theta",
        type=base._str2bool,
        default=None,
        help="If set, overrides inner_train_bias_ln/head for outer training.",
    )
    # Expose BitFit toggles (not in the base parser)
    parser.add_argument("--inner_train_bias_ln", type=base._str2bool, default=None)
    parser.add_argument("--inner_train_head", type=base._str2bool, default=None)
    return parser


def _call_training_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    return_gating: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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


def train_sasrec_first_order(
    model: nn.Module,
    train_dataset,
    config: base.SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Tuple[Dict[str, float], Optional[Path]]:
    """First-order outer distillation: short+patch student matches long teacher."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.train()

    if config.inner_seq_length > config.max_seq_length:
        logger.warning(
            "inner_seq_length (%s) > max_seq_length (%s); clamping to max_seq_length.",
            config.inner_seq_length,
            config.max_seq_length,
        )
        config.inner_seq_length = config.max_seq_length
    if config.eval_seq_length > config.max_seq_length:
        logger.warning(
            "eval_seq_length (%s) > max_seq_length (%s); clamping to max_seq_length.",
            config.eval_seq_length,
            config.max_seq_length,
        )
        config.eval_seq_length = config.max_seq_length

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    last_outer_loss = None
    last_outer_tail_loss = None
    last_outer_mid_loss = None

    def _outer_loss(
        short_input_ids: torch.Tensor,
        short_pos_ids: torch.Tensor,
        short_neg_ids: torch.Tensor,
        long_input_ids: torch.Tensor,
        long_pos_ids: torch.Tensor,
        long_neg_ids: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal last_outer_mid_loss
        nonlocal last_outer_tail_loss

        num_neg = max(1, int(getattr(config, "outer_neg_samples", 1)))
        if num_neg > 1:
            neg_long_ids = base._build_outer_neg_ids(
                dataset=train_dataset,
                user_ids=user_ids,
                pos_ids=long_pos_ids,
                num_neg=num_neg,
                device=long_pos_ids.device,
            )
            neg_short_ids = neg_long_ids[:, -short_pos_ids.size(1) :, :]
        else:
            neg_short_ids = short_neg_ids
            neg_long_ids = long_neg_ids

        pos_short, neg_short = _call_training_step(
            model,
            short_input_ids,
            short_pos_ids,
            neg_short_ids,
            user_ids=user_ids,
            use_patch=True,
        )
        pos_long_full, neg_long_full = _call_training_step(
            model,
            long_input_ids,
            long_pos_ids,
            neg_long_ids,
            user_ids=user_ids,
            use_patch=False,
        )

        pos_long_tail = pos_long_full
        neg_long_tail = neg_long_full
        if pos_long_tail.size(1) != pos_short.size(1):
            pos_long_tail = pos_long_tail[:, -pos_short.size(1) :]
            neg_long_tail = neg_long_tail[:, -pos_short.size(1) :]
        pos_long_tail = pos_long_tail.detach()
        neg_long_tail = neg_long_tail.detach()

        distill = getattr(config, "outer_distill", "kl")
        temp = float(getattr(config, "outer_distill_temperature", 1.0))
        if temp <= 0:
            temp = 1.0

        def _distill_logits(
            pos_student: torch.Tensor,
            neg_student: torch.Tensor,
            pos_teacher: torch.Tensor,
            neg_teacher: torch.Tensor,
        ) -> torch.Tensor:
            if distill == "mse":
                if neg_student.dim() == 3:
                    neg_term = (neg_student - neg_teacher).pow(2).mean(dim=-1)
                else:
                    neg_term = (neg_student - neg_teacher).pow(2)
                return (pos_student - pos_teacher).pow(2) + neg_term
            if distill == "soft_bce":
                pos_targets = torch.sigmoid(pos_teacher / temp)
                neg_targets = torch.sigmoid(neg_teacher / temp)
                pos_loss = F.binary_cross_entropy_with_logits(pos_student / temp, pos_targets, reduction="none")
                neg_loss = F.binary_cross_entropy_with_logits(neg_student / temp, neg_targets, reduction="none")
                if neg_loss.dim() == 3:
                    neg_loss = neg_loss.mean(dim=-1)
                return (pos_loss + neg_loss) * (temp**2)
            if distill == "kl":
                if neg_student.dim() == 3:
                    logits_student = torch.cat([pos_student.unsqueeze(-1), neg_student], dim=-1)
                    logits_teacher = torch.cat([pos_teacher.unsqueeze(-1), neg_teacher], dim=-1)
                else:
                    logits_student = torch.stack([pos_student, neg_student], dim=-1)
                    logits_teacher = torch.stack([pos_teacher, neg_teacher], dim=-1)
                logp_student = F.log_softmax(logits_student / temp, dim=-1)
                logp_teacher = F.log_softmax(logits_teacher / temp, dim=-1)
                p_teacher = logp_teacher.exp()
                return (p_teacher * (logp_teacher - logp_student)).sum(dim=-1) * (temp**2)
            raise ValueError(f"Unknown outer_distill: {distill}")

        raw_loss = _distill_logits(pos_short, neg_short, pos_long_tail, neg_long_tail)
        valid_mask = short_pos_ids != 0
        mode = getattr(config, "outer_loss_mode", "all")
        tail_loss = base._reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
        last_outer_tail_loss = tail_loss.item() if isinstance(tail_loss, torch.Tensor) else float(tail_loss)
        tail_weight = float(getattr(config, "outer_tail_weight", 1.0))
        if tail_weight < 0:
            tail_weight = 0.0
        loss = tail_weight * tail_loss

        last_outer_mid_loss = None
        mid_weight = float(getattr(config, "outer_mid_weight", 0.0))
        if mid_weight > 0 and config.patch_len > 0:
            mid_samples = int(getattr(config, "outer_mid_samples", 0))
            if mid_samples <= 0:
                mid_samples = int(config.patch_len)
            mid_samples = min(mid_samples, int(config.patch_len))
            if mid_samples > 0:
                mid_idx, mid_mask = base._select_middle_positions(
                    long_input_ids,
                    int(getattr(config, "prefix_len", 0) or 0),
                    int(getattr(config, "inner_seq_length", 0) or 0),
                    mid_samples,
                )
                if mid_mask.any():
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

                    hidden_states = _call_forward_features(
                        model,
                        short_input_ids,
                        user_ids=user_ids,
                        use_patch=True,
                    )
                    patch_hidden = hidden_states[:, :mid_samples, :]
                    patch_proj = model.apply_head(patch_hidden)

                    item_weight = model.item_emb.weight
                    pos_emb = F.embedding(mid_pos_ids, item_weight)
                    pos_mid = (patch_proj * pos_emb).sum(dim=-1)
                    if mid_neg_ids.dim() == 3:
                        neg_emb = F.embedding(mid_neg_ids, item_weight)
                        neg_mid = (patch_proj.unsqueeze(2) * neg_emb).sum(dim=-1)
                    else:
                        neg_emb = F.embedding(mid_neg_ids, item_weight)
                        neg_mid = (patch_proj * neg_emb).sum(dim=-1)

                    mid_raw = _distill_logits(pos_mid, neg_mid, mid_teacher_pos, mid_teacher_neg)
                    mid_valid = mid_mask & (mid_pos_ids != 0)
                    if mid_valid.any():
                        mid_loss = mid_raw[mid_valid].mean()
                        loss = loss + mid_weight * mid_loss
                        last_outer_mid_loss = mid_loss.item()

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
    outer_opt = torch.optim.AdamW(
        trainable_params,
        lr=config.outer_lr,
        weight_decay=config.outer_weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps)

    best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}
    best_ckpt_path: Optional[Path] = None
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        for batch_long in train_loader:
            global_step += 1

            batch_long = base._move_batch_to_device(batch_long, device_obj)
            if config.prefix_len and config.prefix_len > 0:
                if config.inner_drop_prefix:
                    batch_short = base._drop_prefix_from_batch(
                        batch_long,
                        config.prefix_len,
                        config.inner_seq_length,
                        train_dataset,
                    )
                else:
                    batch_short = base._build_prefix_tail_batch(
                        batch_long,
                        config.prefix_len,
                        config.inner_seq_length,
                        train_dataset,
                    )
            else:
                batch_short = base._slice_batch_tail(batch_long, config.inner_seq_length)

            if (not config.drop_unseen_items) and config.inner_unk_mask_prob > 0:
                batch_short = {
                    **batch_short,
                    "input_ids": base._mask_inputs_with_unk(
                        batch_short["input_ids"], config.inner_unk_mask_prob
                    ),
                }

            do_outer = config.outer_update_every <= 0 or global_step % config.outer_update_every == 0
            if do_outer:
                outer_opt.zero_grad(set_to_none=True)
                model.eval()
                loss_outer = _outer_loss(
                    batch_short["input_ids"],
                    batch_short["pos_ids"],
                    batch_short["neg_ids"],
                    batch_long["input_ids"],
                    batch_long["pos_ids"],
                    batch_long["neg_ids"],
                    batch_long["internal_user_ids"],
                )
                model.train()
                loss_outer.backward()
                if config.outer_grad_clip and config.outer_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config.outer_grad_clip)
                outer_opt.step()
                last_outer_loss = loss_outer.item()

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
                    mode = getattr(config, "inner_loss_mode", "match_outer")
                    if mode == "match_outer":
                        mode = getattr(config, "outer_loss_mode", "all")
                    student_loss = base._reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
                    if valid_mask.any():
                        base.log_metrics({"meta/student_bce": student_loss.item()})

                    if gating is not None and gating.numel() > 0:
                        weights = gating.detach().cpu()
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

                if last_outer_loss is not None:
                    log_dict = {
                        "meta/outer_loss": last_outer_loss,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                    if last_outer_tail_loss is not None:
                        log_dict["meta/outer_tail_loss"] = last_outer_tail_loss
                    if last_outer_mid_loss is not None:
                        log_dict["meta/outer_mid_loss"] = last_outer_mid_loss
                    base.log_metrics(log_dict)
                    logger.info(
                        "Step %06d | Epoch %03d/%03d | OuterLoss: %.4f",
                        global_step,
                        epoch + 1,
                        config.num_epochs,
                        last_outer_loss,
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
                    "val/meta_patch_ndcg@10": val_metrics["ndcg@10"],
                    "val/meta_patch_hr@10": val_metrics["hr@10"],
                    "progress/epoch": epoch + 1,
                }
            )
            logger.info(
                "Epoch %03d | Val Meta-Patch - NDCG@10: %.4f, HR@10: %.4f",
                epoch + 1,
                val_metrics["ndcg@10"],
                val_metrics["hr@10"],
            )
            if val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
                best_val_metrics = val_metrics
                if config.save_best_model:
                    best_ckpt_path = base.save_model_checkpoint(model, config)
                    logger.info("Saved best val checkpoint to %s", best_ckpt_path)

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
            truncate_len=config.eval_seq_length,
        )
        if metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
            best_val_metrics = metrics
            if config.save_best_model:
                best_ckpt_path = base.save_model_checkpoint(model, config)
                logger.info("Saved best val checkpoint to %s", best_ckpt_path)
    else:
        best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}

    return best_val_metrics, best_ckpt_path


def main() -> None:
    config = base.SASRecConfig()

    parser = build_arg_parser()
    args = parser.parse_args()
    base.apply_overrides_from_args(config, args)
    if args.train_theta is not None:
        config.inner_train_bias_ln = bool(args.train_theta)
        config.inner_train_head = bool(args.train_theta)

    project_name = os.getenv("WANDB_PROJECT") or f"patch_first_order-{config.dataset}"
    run = wandb.init(project=project_name, config=config.__dict__)
    if run is not None:
        base.apply_overrides_from_dict(config, dict(run.config))

    base.resolve_dataset_config(config)
    base.set_global_seed(config.seed, config.deterministic)

    inferred_state = None
    if config.pretrained_ckpt_path and Path(config.pretrained_ckpt_path).exists():
        ckpt = base.load_checkpoint(config.pretrained_ckpt_path, trust_pickle=True)
        inferred_state = base._strip_module_prefix(base._extract_state_dict(ckpt))
        inferred_state = base._maybe_strip_prefix(inferred_state, config.ckpt_prefix_to_strip)
        config = base.infer_config_from_state_dict(inferred_state, config)
    else:
        logger.warning("Pretrained checkpoint not found; proceeding without inference.")

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

    run_name = base._build_run_name(config)
    if run is not None:
        run.name = run_name

    base_ckpt_dir = Path(config.checkpoint_dir)
    if base_ckpt_dir.name != "gating_patch_long_short":
        base_ckpt_dir = base_ckpt_dir / "gating_patch_long_short"
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
    config.log_config()
    if base.LOCAL_METRICS_LOGGER is not None and base.LOCAL_METRICS_LOGGER.jsonl_path is not None:
        logger.info("Local metrics JSONL: %s", base.LOCAL_METRICS_LOGGER.jsonl_path)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base.save_run_config(config, run_name, sys.argv)

    train_dataset = base.LooSequenceDataset(config.data_txt_path, config, logger=logger)
    meta_valid_dataset = train_dataset
    test_dataset = train_dataset
    item_num = train_dataset.num_items

    model = base.build_backbone(config, item_num=item_num)
    if inferred_state is not None:
        if config.strict_load_pretrained:
            logger.info("Loading full checkpoint with strict=True...")
            model.load_state_dict(inferred_state, strict=True)
        else:
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
    base.apply_bitfit_freeze(
        model,
        enable_bias_ln=config.inner_train_bias_ln,
        enable_head=config.inner_train_head and config.enable_projection_head,
    )
    theta_names = base.build_bitfit_param_names(
        model,
        enable_bias_ln=config.inner_train_bias_ln,
        enable_head=config.inner_train_head and config.enable_projection_head,
    )
    bitfit_init_state = base._snapshot_params_by_name(model, theta_names)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    if config.eval_before_train:
        logger.info("Running pre-train baseline on val (short seq, no patch)...")
        val_baseline = base.evaluate(
            model,
            meta_valid_dataset,
            config=config,
            mode="val",
            device=device,
            use_patch=False,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=theta_names,
            bitfit_init_state=bitfit_init_state,
        )
        logger.info(
            "Val Baseline - NDCG@10: %.4f, HR@10: %.4f",
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

        logger.info("Running pre-train meta-patch on val (patch + head)...")
        val_meta_patch = base.evaluate(
            model,
            meta_valid_dataset,
            config=config,
            mode="val",
            device=device,
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=theta_names,
            bitfit_init_state=bitfit_init_state,
        )
        logger.info(
            "Val Meta-Patch (pre-train) - NDCG@10: %.4f, HR@10: %.4f",
            val_meta_patch["ndcg@10"],
            val_meta_patch["hr@10"],
        )
        base.log_metrics(
            {
                "val/pre_meta_patch_ndcg@10": val_meta_patch["ndcg@10"],
                "val/pre_meta_patch_hr@10": val_meta_patch["hr@10"],
                "progress/epoch": 0,
                "progress/step": 0,
            }
        )

    best_metrics, best_ckpt_path = train_sasrec_first_order(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=meta_valid_dataset,
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

    logger.info("Running final test evaluation (baseline: short seq, no patch, meta-test)...")
    baseline_metrics = base.evaluate(
        model,
        test_dataset,
        config=config,
        mode="meta-test",
        device=device,
        use_patch=False,
        use_head=True,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
        theta_names=theta_names,
        bitfit_init_state=bitfit_init_state,
    )
    logger.info(
        "Baseline Test - NDCG@10: %.4f, HR@10: %.4f",
        baseline_metrics["ndcg@10"],
        baseline_metrics["hr@10"],
    )

    logger.info("Running final test evaluation (short seq + patch, meta-test)...")
    meta_metrics = base.evaluate(
        model,
        test_dataset,
        config=config,
        mode="meta-test",
        device=device,
        use_patch=True,
        use_head=True,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
        theta_names=theta_names,
        bitfit_init_state=bitfit_init_state,
    )
    logger.info(
        "Meta-Patch Test - NDCG@10: %.4f, HR@10: %.4f",
        meta_metrics["ndcg@10"],
        meta_metrics["hr@10"],
    )

    base.log_metrics(
        {
            "test/baseline_ndcg@10": baseline_metrics["ndcg@10"],
            "test/baseline_hr@10": baseline_metrics["hr@10"],
            "test/meta_patch_ndcg@10": meta_metrics["ndcg@10"],
            "test/meta_patch_hr@10": meta_metrics["hr@10"],
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
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
        meta_metrics,
        best_ckpt_path,
        metrics_jsonl=metrics_jsonl,
        metrics_csv=metrics_csv,
    )

    wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
