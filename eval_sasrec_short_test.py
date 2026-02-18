#!/usr/bin/env python3
"""Evaluate a pretrained SASRec backbone on short sequences (test set)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional

import wandb

ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.device_manager import DeviceManager
from core.logger import setup_logger
from core.loo_dataset import LooSequenceDataset
from id_patch.train_gating_patch_long_short import (
    SASRecConfig,
    SASRec,
    load_checkpoint,
    _extract_state_dict,
    _strip_module_prefix,
    _maybe_strip_prefix,
    infer_config_from_state_dict,
    load_pretrained_backbone,
    resolve_dataset_config,
    evaluate,
)

logger = setup_logger("eval-sasrec-short-test", log_to_file=True)

DEFAULT_SHORT_CKPT = (
    "/home/lingfengs111/codes/soft_patch_training/checkpoints/sasrec_loo_standard/"
    "sasrec_taobao_loo202_seq50_dim128_L2_H1_best.pt"
)
DEFAULT_LONG_CKPT = (
    "/home/lingfengs111/codes/soft_patch_training/checkpoints/sasrec_loo_standard/"
    "sasrec_taobao_loo202_seq202_dim128_L2_H1_best.pt"
)


@dataclass
class EvalConfig(SASRecConfig):
    """Configuration for short-seq evaluation."""

    # Match backbone-only checkpoints (no patch usage)
    num_patches: int = 1
    patch_len: int = 0
    use_gating: bool = False

    pretrained_ckpt_path: str = DEFAULT_LONG_CKPT
    eval_seq_length: int = 10
    batch_size: int = 1024
    eval_sample_size: int = 1000
    device: str = "cuda:3"
    strict_load_pretrained: bool = True
    drop_unseen_items: bool = True

    log_to_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None

    def log_config(self) -> None:
        logger.info("=== Eval Configuration ===")
        logger.info("Data Settings:")
        logger.info("  dataset: %s", self.dataset)
        logger.info("  data_dir: %s", self.data_dir)
        logger.info("  data_txt_path: %s", self.data_txt_path)
        logger.info("Model Parameters:")
        logger.info("  backbone: %s", self.backbone)
        logger.info("  max_seq_length: %s", self.max_seq_length)
        logger.info("  hidden_units: %s", self.hidden_units)
        logger.info("  num_blocks: %s", self.num_blocks)
        logger.info("  num_heads: %s", self.num_heads)
        logger.info("  dropout_rate: %s", self.dropout_rate)
        logger.info("  right_align_positions: %s", self.right_align_positions)
        logger.info("Eval Settings:")
        logger.info("  pretrained_ckpt_path: %s", self.pretrained_ckpt_path)
        logger.info("  eval_seq_length: %s", self.eval_seq_length)
        logger.info("  batch_size: %s", self.batch_size)
        logger.info("  eval_sample_size: %s", self.eval_sample_size)
        logger.info("  drop_unseen_items: %s", self.drop_unseen_items)
        logger.info("  strict_load_pretrained: %s", self.strict_load_pretrained)
        logger.info("Device Settings:")
        logger.info("  device: %s", self.device)
        logger.info("Wandb Settings:")
        logger.info("  log_to_wandb: %s", self.log_to_wandb)
        if self.log_to_wandb:
            project = self.wandb_project or f"baselines_long_short-{self.dataset}"
            logger.info("  wandb_project: %s", project)
            logger.info("  wandb_group: %s", self.wandb_group)
            logger.info("  wandb_name: %s", self.wandb_name)
            logger.info("  wandb_tags: %s", self.wandb_tags)
            logger.info("  wandb_notes: %s", self.wandb_notes)
        logger.info("==========================")


def _init_wandb(config: EvalConfig) -> Optional[wandb.sdk.wandb_run.Run]:
    if not config.log_to_wandb:
        return None
    project = config.wandb_project or f"baselines_long_short-{config.dataset}"
    ckpt_tag = Path(config.pretrained_ckpt_path).stem
    ckpt_label = _label_ckpt(config.pretrained_ckpt_path, ckpt_tag)
    test_label = _label_test(config.eval_seq_length)
    run_name = config.wandb_name or (
        f"{ckpt_label}-{test_label}-{config.backbone}-{config.dataset}"
        f"-L{config.num_blocks}-H{config.hidden_units}"
        f"-ckpt{config.max_seq_length}-eval{config.eval_seq_length}"
    )
    wandb_kwargs = {
        "project": project,
        "name": run_name,
        "config": config.__dict__,
        "job_type": "eval",
    }
    if config.wandb_group:
        wandb_kwargs["group"] = config.wandb_group
    if config.wandb_tags:
        wandb_kwargs["tags"] = config.wandb_tags
    if config.wandb_notes:
        wandb_kwargs["notes"] = config.wandb_notes
    return wandb.init(**wandb_kwargs)


def _label_ckpt(path: str, ckpt_tag: Optional[str] = None) -> str:
    resolved = str(Path(path).resolve())
    if Path(DEFAULT_LONG_CKPT).resolve() == Path(resolved):
        return "long_ckpt"
    if Path(DEFAULT_SHORT_CKPT).resolve() == Path(resolved):
        return "short_ckpt"
    stem = ckpt_tag or Path(path).stem
    match = re.search(r"seq(\\d+)", stem)
    if match:
        seq_len = int(match.group(1))
        return "long_ckpt" if seq_len >= 200 else "short_ckpt"
    return "ckpt"


def _label_test(eval_seq_length: int) -> str:
    if eval_seq_length >= 200:
        return "long_test"
    if eval_seq_length <= 50:
        return "short_test"
    return f"test{eval_seq_length}"


def main() -> None:
    config = EvalConfig()
    # Example override:
    # config.pretrained_ckpt_path = DEFAULT_LONG_CKPT

    if not Path(config.pretrained_ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.pretrained_ckpt_path}")

    ckpt = load_checkpoint(config.pretrained_ckpt_path, trust_pickle=True)
    inferred_state = _strip_module_prefix(_extract_state_dict(ckpt))
    inferred_state = _maybe_strip_prefix(inferred_state, config.ckpt_prefix_to_strip)
    config = infer_config_from_state_dict(inferred_state, config)
    resolve_dataset_config(config)
    if config.eval_seq_length > config.max_seq_length:
        logger.warning(
            "eval_seq_length (%s) > max_seq_length (%s); will be truncated.",
            config.eval_seq_length,
            config.max_seq_length,
        )

    run = _init_wandb(config)
    config.log_config()

    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    test_dataset = LooSequenceDataset(config.data_txt_path, config, logger=logger)

    model = SASRec(config, item_num=test_dataset.num_items)
    if config.strict_load_pretrained:
        logger.info("Loading full checkpoint with strict=True...")
        model.load_state_dict(inferred_state, strict=True)
    else:
        load_pretrained_backbone(model, config.pretrained_ckpt_path, state_dict=inferred_state)

    model = model.to(device)
    model.eval()

    logger.info(
        "Evaluating short test (eval_seq_length=%s) on %s users...",
        config.eval_seq_length,
        len(test_dataset.users),
    )

    metrics = evaluate(
        model,
        test_dataset,
        config=config,
        mode="test",
        batch_size=config.batch_size,
        device=device,
        use_patch=False,
        use_head=False,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
    )

    logger.info(
        "Short-seq Test - NDCG@10: %.4f, HR@10: %.4f",
        metrics["ndcg@10"],
        metrics["hr@10"],
    )

    if run is not None:
        wandb.log(
            {
                "test/ndcg@10": metrics["ndcg@10"],
                "test/hr@10": metrics["hr@10"],
                "eval/seq_length": config.eval_seq_length,
                "eval/ckpt": Path(config.pretrained_ckpt_path).stem,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
