#!/usr/bin/env python3
"""
Training script for semantic code decoder.
Predicts next semantic codes given a sequence of previous codes.
Complete end-to-end training with validation, early stopping, and wandb logging.
Includes Hit Rate and NDCG evaluation metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import time
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from model import SemanticCodeDecoder, SemanticCodeDecoderOnly
from data import create_dataloaders_from_txt
from evaluate import evaluate

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train-decoder")


@dataclass
class TrainConfig:
    """Configuration for training semantic code decoder."""

    # Data settings
    data_txt_path: Optional[str] = 'data/movielens1m/proc/data.txt'  # Path to data.txt file
    semantic_codes_path: Optional[str] = 'data/movielens1m/codes/item_semantic_codes.pt'  # Path to semantic codes
    flatten_codes: bool = False # ??这到底干嘛的

    # Model parameters
    model_arch: str = "enc-dec"  # "enc-dec" or "decoder-only"
    codebook_size: int = 256
    num_levels: int = 3
    hidden_dim: int = 512
    num_layers: int = 6  # total layers; for enc-dec split evenly unless overridden
    encoder_layers: Optional[int] = None
    decoder_layers: Optional[int] = None
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.2
    attn_dropout: float = 0.1
    max_seq_len: int = 50
    carry_decoder_state: bool = True  # If True, carry decoder hidden across levels (instead of overwriting with code embedding)

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    num_epochs: int = 30
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 5e-3

    # Evaluation and checkpointing
    eval_steps: int = 300
    checkpoint_dir: str = "checkpoints/decoder"
    gpu_id: int = 1  # Select which CUDA device to use
    use_cuda: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Generation/eval settings
    eval_use_beam: bool = True
    eval_num_beams: int = 30
    eval_num_return_sequences: int = 30
    eval_item_ranking: bool = False  # item-level ranking via semantic codes (assumes codes uniquely map to items)

    # Early stopping
    early_stopping_patience: int = 5  # Stop if no improvement for N evals
    early_stopping_min_delta: float = 1e-4  # Minimum improvement threshold

    # Logging
    wandb_enabled: bool = True
    wandb_project: str = "semantic-code-decoder"
    num_workers: int = 0

    def __post_init__(self):
        """Validate configuration."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # Allow overriding CUDA usage via config and GPU index selection
        if not self.use_cuda or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            if self.gpu_id < 0 or self.gpu_id >= torch.cuda.device_count():
                logger.warning(
                    f"gpu_id {self.gpu_id} is out of range (0-{torch.cuda.device_count()-1}), "
                    "falling back to default CUDA device."
                )
                self.device = "cuda"
            else:
                self.device = f"cuda:{self.gpu_id}"

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=" * 80)
        logger.info("SEMANTIC CODE DECODER - Configuration")
        logger.info("=" * 80)

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  data_txt_path: {self.data_txt_path}")
        logger.info(f"  semantic_codes_path: {self.semantic_codes_path}")
        logger.info(f"  flatten_codes: {self.flatten_codes}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  model_arch: {self.model_arch}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  num_levels: {self.num_levels}")
        logger.info(f"  hidden_dim: {self.hidden_dim}")
        logger.info(f"  num_layers: {self.num_layers}")
        logger.info(f"  encoder_layers: {self.encoder_layers}")
        logger.info(f"  decoder_layers: {self.decoder_layers}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  ffn_dim: {self.ffn_dim}")
        logger.info(f"  dropout: {self.dropout}")
        logger.info(f"  carry_decoder_state: {self.carry_decoder_state}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  learning_rate: {self.learning_rate}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  use_cuda: {self.use_cuda}")
        logger.info(f"  gpu_id: {self.gpu_id}")
        logger.info(f"  device: {self.device}")
        logger.info(f"  eval_use_beam: {self.eval_use_beam}")
        logger.info(f"  eval_num_beams: {self.eval_num_beams}")
        logger.info(f"  eval_num_return_sequences: {self.eval_num_return_sequences}")
        logger.info(f"  eval_item_ranking: {self.eval_item_ranking}")

        # Early stopping
        logger.info("Early Stopping:")
        logger.info(f"  patience: {self.early_stopping_patience}")
        logger.info(f"  min_delta: {self.early_stopping_min_delta}")

        logger.info("=" * 80)


class EarlyStoppingMonitor:
    """Monitor validation loss and trigger early stopping."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        """
        Update monitor with validation loss.

        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = val_loss
            self.counter = 0
            logger.info(f"✓ Val loss improved to {val_loss:.4f}")
            return False
        else:
            # No improvement
            self.counter += 1
            logger.info(
                f"✗ No improvement ({self.counter}/{self.patience}). "
                f"Best loss: {self.best_loss:.4f}, Current: {val_loss:.4f}"
            )

            if self.counter >= self.patience:
                logger.warning(f"Early stopping triggered after {self.counter} evals without improvement")
                self.should_stop = True
                return True

            return False


def create_model(config: TrainConfig) -> SemanticCodeDecoder:
    """Create model from config."""
    if config.model_arch == "decoder-only":
        return SemanticCodeDecoderOnly(
            codebook_size=config.codebook_size,
            num_levels=config.num_levels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            max_seq_len=config.max_seq_len,
            carry_decoder_state=config.carry_decoder_state,
        )
    else:
        enc_layers = config.encoder_layers if config.encoder_layers is not None else config.num_layers // 2
        dec_layers = config.decoder_layers if config.decoder_layers is not None else config.num_layers // 2
        return SemanticCodeDecoder(
            codebook_size=config.codebook_size,
            num_levels=config.num_levels,
            hidden_dim=config.hidden_dim,
            encoder_layers=enc_layers,
            decoder_layers=dec_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            max_seq_len=config.max_seq_len,
            carry_decoder_state=config.carry_decoder_state,
        )


def create_dataloaders(config: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders from real data."""
    if not config.data_txt_path or not config.semantic_codes_path:
        raise ValueError(
            "Must provide data_txt_path and semantic_codes_path"
        )
    
    logger.info(f"Loading data from {config.data_txt_path}...")
    train_dataloader, val_dataloader, _ = create_dataloaders_from_txt(
        data_txt_path=config.data_txt_path,
        semantic_codes_path=config.semantic_codes_path,
        batch_size=config.batch_size,
        num_levels=config.num_levels,
        flatten_codes=config.flatten_codes,
        max_seq_len=config.max_seq_len,
    )
    logger.info("✓ Data loaded successfully")

    return train_dataloader, val_dataloader


def train(config: TrainConfig):
    """Main training loop."""
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # Initialize wandb
    if config.wandb_enabled:
        run_name = f"decoder-L{config.num_levels}-C{config.codebook_size}-H{config.hidden_dim}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.__dict__,
        )
        logger.info(f"WandB enabled: {run_name}")

    # Log config
    config.log_config()

    # Create model and dataloaders
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {num_params:,} parameters")

    train_dataloader, val_dataloader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # Setup training
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = config.warmup_steps
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=max(1, warmup_steps),
    )
    
    # Cosine annealing scheduler
    cosine_steps = max(1, total_steps - warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-5,
    )
    
    # Combined scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # CrossEntropyLoss for all predictions
    # Use ignore_index for padding tokens to avoid training on padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 is the pad_token_id
    early_stopping = EarlyStoppingMonitor(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    # Training variables
    global_step = 0
    best_val_loss = float("inf")
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    start_time = time.time()

    # Main training loop
    for epoch in range(1, config.num_epochs + 1):
        # ===== Training phase =====
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{config.num_epochs}",
            total=len(train_dataloader),
        )

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch.get("target_ids")
            attention_mask = batch["attention_mask"].to(device)

            if target_ids is None:
                logger.warning("Batch has no target IDs, skipping...")
                continue

            target_ids = target_ids.to(device)

            # Forward pass with teacher forcing
            logits, hidden = model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=target_ids,
            )

            # Compute loss - flatten for CrossEntropyLoss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_ids_flat = target_ids.reshape(-1)
            loss = loss_fn(logits_flat, target_ids_flat)
            loss = loss / config.gradient_accumulation_steps

            # Backward pass
            loss.backward()
            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            # Optimizer step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    "loss": total_loss / num_batches,
                    "lr": optimizer.param_groups[0]["lr"],
                })

        # ===== Epoch-end evaluation =====
        if val_dataloader is not None:
            metrics = evaluate(model, val_dataloader, device, loss_fn, config)
            val_loss = metrics["loss"]
            logger.info(f"Epoch {epoch} - Train loss: {total_loss / max(num_batches, 1):.4f}, Val loss: {val_loss:.4f}")
            logger.info(f"  Hit Rate @5/@10: {metrics['hit_rate@5']:.4f} / {metrics['hit_rate@10']:.4f}")
            logger.info(f"  NDCG @5/@10: {metrics['ndcg@5']:.4f} / {metrics['ndcg@10']:.4f}")

            if config.wandb_enabled:
                log_dict = {
                    "epoch/train_loss": total_loss / max(num_batches, 1),
                    "epoch/val_loss": val_loss,
                    "epoch": epoch,
                }
                # Add metric logs
                for key, value in metrics.items():
                    if key != "loss":
                        log_dict[f"epoch/{key}"] = value
                wandb.log(log_dict)

            # Checkpoint saving at epoch end
            checkpoint_path = checkpoint_dir / f"checkpoin33_epoch_{epoch}.pt"
            torch.save({
                "step": global_step,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics,
                "config": config.__dict__,
            }, checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "checkpoint33_best.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "config": config.__dict__,
                }, best_path)
                logger.info(f"[Epoch {epoch}] New best model saved with val_loss={val_loss:.4f}")
                logger.info(f"  Hit Rate @5/@10: {metrics['hit_rate@5']:.4f} / {metrics['hit_rate@10']:.4f}")
                logger.info(f"  NDCG @5/@10: {metrics['ndcg@5']:.4f} / {metrics['ndcg@10']:.4f}")

            # Early stopping check at epoch end only
            if early_stopping.update(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Training complete
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Training time: {elapsed_time / 3600:.2f} hours")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info("=" * 80)

    if config.wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    # Simple entry: edit TrainConfig defaults above or modify here before running.
    cfg = TrainConfig()
    train(cfg)
