#!/usr/bin/env python3
"""Launch LONGER backbone training on LOO datasets with the sampled-softmax pipeline."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import wandb

import train_backbone_sample_softmax as tbs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="taobao_loo202")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-seq-length", type=int, default=202)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--longer-global-tokens", type=int, default=4)
    parser.add_argument("--longer-merge-size", type=int, default=4)
    parser.add_argument("--longer-merge-pool", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--longer-inner-num-layers", type=int, default=1)
    parser.add_argument("--max-learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-negatives", type=int, default=128)
    parser.add_argument("--sampled-softmax-chunk-size", type=int, default=1024)
    parser.add_argument("--steps-per-train-log", type=int, default=20)
    parser.add_argument("--steps-per-val-log", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--eval-sample-size", type=int, default=1000)
    parser.add_argument("--eval-protocol", type=str, default="legacy_loo", choices=["legacy_loo", "holdout_anchor"])
    parser.add_argument("--last-k-eval-test", type=int, default=10)
    parser.add_argument("--streaming-eval-last-k", type=int, default=0)
    parser.add_argument("--selection-metric", type=str, default="ndcg@10", choices=["ndcg@10", "hr@10"])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--user-embedding-norm", type=str, default="l2_norm")
    parser.add_argument("--disable-item-l2-norm", action="store_true")
    parser.add_argument("--l2-norm-eps", type=float, default=1e-6)
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "cosine_with_warmup"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-start-lr", type=float, default=5e-7)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run-tag", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config = tbs.SASRecConfig()
    config.backbone = "longer"
    config.dataset = args.dataset
    config.device = args.device
    config.max_seq_length = int(args.max_seq_length)
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.seed = args.seed
    config.eval_seed = args.eval_seed
    config.deterministic = bool(args.deterministic)
    if config.eval_seed is None:
        config.eval_seed = config.seed
    config.hidden_units = args.hidden_units
    config.num_blocks = args.num_blocks
    config.num_heads = args.num_heads
    config.dropout_rate = args.dropout_rate
    config.longer_global_tokens = args.longer_global_tokens
    config.longer_merge_size = args.longer_merge_size
    config.longer_merge_pool = args.longer_merge_pool
    config.longer_inner_num_layers = args.longer_inner_num_layers
    config.max_learning_rate = args.max_learning_rate
    config.min_learning_rate = args.min_learning_rate
    config.weight_decay = args.weight_decay
    config.grad_clip = args.grad_clip
    config.num_negatives = args.num_negatives
    config.sampled_softmax_chunk_size = args.sampled_softmax_chunk_size
    config.steps_per_train_log = args.steps_per_train_log
    config.steps_per_val_log = args.steps_per_val_log
    config.early_stop_patience = args.early_stop_patience
    config.selection_metric = args.selection_metric
    config.num_workers = args.num_workers
    config.prefetch_factor = args.prefetch_factor
    config.eval_sample_size = args.eval_sample_size
    config.eval_protocol = args.eval_protocol
    config.last_k_eval_test = args.last_k_eval_test
    config.streaming_eval_last_k = args.streaming_eval_last_k
    config.temperature = args.temperature
    config.user_embedding_norm = str(args.user_embedding_norm).lower()
    config.item_l2_norm = not args.disable_item_l2_norm
    config.l2_norm_eps = args.l2_norm_eps
    config.scheduler_type = args.scheduler_type
    config.warmup_steps = args.warmup_steps
    config.warmup_start_lr = args.warmup_start_lr
    config.use_gradient_checkpointing = not args.disable_gradient_checkpointing
    config.use_amp = not args.disable_amp
    config.amp_dtype = args.amp_dtype
    config.use_torch_compile = args.enable_torch_compile
    config.enable_projection_head = False

    config.backbone = config.backbone.lower()
    config.apply_backbone_overrides()
    tbs.resolve_dataset_config(config)
    tbs.resolve_eval_protocol_config(config)
    tbs.set_global_seed(config.seed, config.deterministic)

    device_manager = tbs.DeviceManager(tbs.logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    es_suffix = "noes" if config.early_stop_patience <= 0 else f"es{config.early_stop_patience}"
    metric_suffix = config.selection_metric.replace("@", "").replace("/", "_")
    run_name = (
        f"{config.backbone}-{config.dataset}-{config.num_blocks}b-{config.num_heads}h-"
        f"{config.hidden_units}-{config.max_seq_length}_{es_suffix}-{metric_suffix}-sample_softmax"
    )
    run_name += tbs.build_protocol_run_suffix(config)
    run = wandb.init(
        project=f"backbone-standard-{config.dataset}",
        name=run_name,
        config=config.__dict__,
        mode=args.wandb_mode,
    )

    base_ckpt_dir = config.checkpoint_dir / f"{config.backbone}_loo_sample_softmax"
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
    if int(config.streaming_eval_last_k or 0) > 1:
        stream_last_k = int(config.streaming_eval_last_k)
        tbs.logger.info("Running additional streaming test evaluation over the last %s targets...", stream_last_k)
        stream_metrics = tbs.evaluate(
            model,
            test_dataset,
            config=config,
            mode="test",
            device=device,
            streaming_last_k=stream_last_k,
        )
        wandb.log(
            {
                **tbs.flatten_streaming_eval_metrics("test_stream/backbone", stream_metrics),
                **tbs.flatten_streaming_eval_test_aliases("backbone", stream_metrics),
            }
        )
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
