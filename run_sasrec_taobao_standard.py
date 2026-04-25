#!/usr/bin/env python3
"""Launch SASRec backbone training on Taobao with the standard pipeline."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import wandb

import train_backbone_standard as tbs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="taobao_loo202")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--max-learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--steps-per-train-log", type=int, default=50)
    parser.add_argument("--steps-per-val-log", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--eval-sample-size", type=int, default=1000)
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "cosine_with_warmup"])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-8)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-flash-attention", action="store_true")
    parser.add_argument("--sasrec-attention-norm", type=str, default="softmax", choices=["softmax", "softmax_custom", "softmax1"])
    parser.add_argument("--sasrec-enable-relative-attention-bias", action="store_true")
    parser.add_argument("--user-embedding-norm", type=str, default="none", choices=["none", "l2_norm", "layer_norm"])
    parser.add_argument("--item-l2-norm", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--l2-norm-eps", type=float, default=1e-6)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--shared-token-len", dest="shared_prefix_len", type=int, default=0)
    parser.add_argument("--shared-token-init-std", dest="shared_prefix_init_std", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config = tbs.SASRecConfig()
    config.backbone = "sasrec"
    config.dataset = args.dataset
    config.device = args.device
    if args.max_seq_length is not None and int(args.max_seq_length) > 0:
        config.max_seq_length = int(args.max_seq_length)
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.hidden_units = args.hidden_units
    config.num_blocks = args.num_blocks
    config.num_heads = args.num_heads
    config.dropout_rate = args.dropout_rate
    config.max_learning_rate = args.max_learning_rate
    config.min_learning_rate = args.min_learning_rate
    config.weight_decay = args.weight_decay
    config.grad_clip = args.grad_clip
    config.steps_per_train_log = args.steps_per_train_log
    config.steps_per_val_log = args.steps_per_val_log
    config.early_stop_patience = args.early_stop_patience
    config.eval_sample_size = args.eval_sample_size
    config.scheduler_type = args.scheduler_type
    config.warmup_steps = args.warmup_steps
    config.warmup_start_lr = args.warmup_start_lr
    config.use_gradient_checkpointing = not args.disable_gradient_checkpointing
    config.use_flash_attention = not args.disable_flash_attention
    config.sasrec_attention_norm = str(args.sasrec_attention_norm).lower()
    config.sasrec_enable_relative_attention_bias = bool(args.sasrec_enable_relative_attention_bias)
    config.user_embedding_norm = args.user_embedding_norm
    config.item_l2_norm = bool(args.item_l2_norm)
    config.temperature = args.temperature
    config.l2_norm_eps = args.l2_norm_eps
    config.shared_prefix_len = int(args.shared_prefix_len or 0)
    config.shared_prefix_init_std = float(args.shared_prefix_init_std or 0.02)

    config.backbone = config.backbone.lower()
    config.apply_backbone_overrides()
    tbs.resolve_dataset_config(config)

    device_manager = tbs.DeviceManager(tbs.logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    es_suffix = "noes" if config.early_stop_patience <= 0 else f"es{config.early_stop_patience}"
    run_name = (
        f"{config.backbone}-{config.dataset}-{config.num_blocks}b-{config.num_heads}h-"
        f"{config.hidden_units}-{config.max_seq_length}_{es_suffix}-standard"
    )
    if config.shared_prefix_len > 0:
        run_name += f"-sp{config.shared_prefix_len}"
    if config.sasrec_attention_norm == "softmax1":
        run_name += "-sm1"
    elif config.sasrec_attention_norm == "softmax_custom":
        run_name += "-smcustom"
    if config.sasrec_enable_relative_attention_bias:
        run_name += "-rbias"
    if tbs._uses_similarity_scoring(config):
        if str(config.user_embedding_norm).lower() == "l2_norm":
            run_name += "-uq"
        elif str(config.user_embedding_norm).lower() == "layer_norm":
            run_name += "-uln"
        if bool(config.item_l2_norm):
            run_name += "-iq"
        if abs(float(config.temperature) - 1.0) > 1e-12:
            run_name += f"-t{str(config.temperature).replace('.', 'p')}"
    run = wandb.init(
        project=f"backbone-standard-{config.dataset}",
        name=run_name,
        config=config.__dict__,
        mode=args.wandb_mode,
    )

    base_ckpt_dir = config.checkpoint_dir / f"{config.backbone}_loo_standard"
    if args.run_tag:
        config.run_tag = args.run_tag
    elif not config.run_tag:
        run_id = getattr(run, "id", None) or getattr(wandb.run, "id", None)
        tag = time.strftime("%Y%m%d_%H%M%S")
        if run_id:
            tag = f"{tag}-{run_id}"
        config.run_tag = tag
    config.checkpoint_dir = base_ckpt_dir / str(config.run_tag)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_config()

    train_dataset = tbs.LooSequenceDataset(config.data_txt_path, config, logger=tbs.logger)
    val_dataset = train_dataset
    test_dataset = train_dataset
    item_num = train_dataset.num_items

    model = tbs.build_backbone(config, item_num=item_num)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tbs.logger.info(f"Total parameters: {total_params:,}")
    tbs.logger.info(f"Trainable parameters: {trainable_params:,}")

    best_metrics = tbs.train_sasrec(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=val_dataset,
    )

    tbs.logger.info("Running final test evaluation...")
    test_metrics = tbs.evaluate(model, test_dataset, config=config, mode="test", device=device)
    tbs.logger.info(f"Test Results - NDCG@10: {test_metrics['ndcg@10']:.4f}, HR@10: {test_metrics['hr@10']:.4f}")

    wandb.log({"test/ndcg@10": test_metrics["ndcg@10"], "test/hr@10": test_metrics["hr@10"]})
    wandb.log(
        {
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )
    wandb.finish()
    tbs.logger.info("Training complete!")


if __name__ == "__main__":
    main()
