#!/usr/bin/env python3
"""Train backbone models on LOO datasets (leave-two-out)."""

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

from core.device_manager import DeviceManager
from core.logger import setup_logger
from core.loo_dataset import LooSequenceDataset, resolve_loo_dataset, infer_loo_min_len
from backbones.FMLP import FMLP
from backbones.LinRec import LinRec
from backbones.LRU import LRU
from backbones.Mamba4Rec import Mamba4Rec
from backbones.Bert4rec import Bert4Rec
from backbones.GRU4Rec import GRU4Rec
from backbones.HSTU import HSTU
from backbones.HSTUOfficialish import HSTUOfficialish
from backbones.HSTUResearchAligned import HSTUResearchAligned
from backbones.LONGER import LONGER
from backbones.SASRec import SASRec as SASRecBackbone

logger = setup_logger("train-backbone-standard", log_to_file=True)


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class SASRecConfig:
    """Configuration for backbone training (LOO datasets)."""

    dataset: str = "taobao_loo202"
    data_dir: Optional[Path] = None
    data_txt_path: Optional[Path] = None
    checkpoint_dir: Path = REPO_ROOT / "checkpoints"

    # Model (shared)
    backbone: str = "hstu"  # sasrec | hstu | longer | fmlp | linrec | bert4rec | gru4rec | lru | mamba4rec
    max_seq_length: Optional[int] = None
    hidden_units: int = 128
    num_blocks: int = 2
    num_heads: int = 1
    dropout_rate: float = 0.1
    right_align_positions: bool = True
    initializer_range: float = 0.02
    shared_prefix_len: int = 0
    shared_prefix_init_std: float = 0.02

    # SASRec-specific
    use_flash_attention: Optional[bool] = True
    sasrec_attention_norm: str = "softmax"
    sasrec_enable_relative_attention_bias: bool = False
    use_gradient_checkpointing: bool = False

    # HSTU-specific
    hstu_linear_dim: Optional[int] = None
    hstu_attention_dim: Optional[int] = None
    hstu_linear_activation: str = "silu"
    hstu_attn_dropout: Optional[float] = None
    hstu_enable_relative_attention_bias: bool = False
    hstu_concat_ua: bool = False
    hstu_epsilon: float = 1e-6

    # LONGER-specific
    longer_global_tokens: int = 4
    longer_merge_size: int = 4
    longer_merge_pool: str = "last"  # last | mean
    longer_inner_num_layers: int = 1

    # FMLP-specific
    fmlp_num_layers: Optional[int] = 2
    fmlp_num_heads: Optional[int] = 2
    fmlp_hidden_dropout: Optional[float] = None
    fmlp_attn_dropout: Optional[float] = None
    fmlp_hidden_act: Optional[str] = None
    fmlp_no_filters: bool = False

    # LinRec-specific
    linrec_num_layers: Optional[int] = None
    linrec_num_heads: Optional[int] = None
    linrec_inner_size: Optional[int] = None
    linrec_hidden_dropout: Optional[float] = None
    linrec_attn_dropout: Optional[float] = None
    linrec_hidden_act: Optional[str] = None
    linrec_layer_norm_eps: float = 1e-12
    # LinRec tuning overrides (applied when backbone == "linrec")
    linrec_hidden_dropout_override: Optional[float] = 0.15
    linrec_attn_dropout_override: Optional[float] = 0.15
    linrec_max_learning_rate: Optional[float] = 5e-4
    linrec_min_learning_rate: Optional[float] = 5e-6
    linrec_weight_decay: Optional[float] = 5e-3
    linrec_grad_clip: Optional[float] = 1.0
    linrec_early_stop_patience: Optional[int] = 10

    # Bert4Rec-specific
    bert_num_layers: Optional[int] = 2
    bert_num_heads: Optional[int] = 1
    bert_dropout: Optional[float] = None

    # GRU4Rec-specific
    gru_embedding_size: Optional[int] = None
    gru_num_layers: Optional[int] = 1
    gru_dropout: Optional[float] = None

    # LRU-specific
    lru_num_blocks: Optional[int] = 2
    lru_dropout: Optional[float] = None
    lru_attn_dropout: Optional[float] = None

    # Mamba4Rec-specific
    mamba_num_layers: Optional[int] = None
    mamba_d_state: int = 32
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: Optional[float] = None

    # Head (shared)
    enable_projection_head: bool = False
    head_use_gelu: bool = False
    head_use_ln: bool = True
    head_residual: bool = False
    user_embedding_norm: str = "none"  # none | l2_norm | layer_norm
    item_l2_norm: bool = False
    temperature: float = 1.0
    l2_norm_eps: float = 1e-6

    # Training parameters
    batch_size: int = 1024  # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    max_learning_rate: float = 5e-4  # Maximum learning rate (start of cosine)
    min_learning_rate: float = 5e-6  # Minimum learning rate (end of cosine)
    weight_decay: float = 1e-4  # AdamW weight decay
    grad_clip: float = 1.0  # Global grad clip (0 disables)

    # Training settings
    scheduler_type: str = "cosine"  # Learning rate scheduler type ("cosine" or "cosine_with_warmup")
    warmup_steps: int = 100  # Number of warmup steps (only for cosine_with_warmup)
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    steps_per_train_log: int = 50  # Log training progress every N steps
    steps_per_val_log: int = 50  # Validate and checkpoint every N steps
    # ⚠️之前eval用的是100导致结果看起来很高
    eval_sample_size: int = 1000  # Total candidates per user when eval_mode="sampled" (includes target)
    early_stop_patience: int = 10  # Stop after N validations without improvement (<=0 disables)

    # Output settings
    save_item_embeddings: bool = True  # Save item embeddings on best validation

    # Run tagging (for unique checkpoint folders)
    run_tag: Optional[str] = None

    # Device settings
    device: str = "cuda:1"  # e.g., "cuda:1", "cpu", "mps"

    def __post_init__(self):
        if self.use_flash_attention is None:
            self.use_flash_attention = True
        self.sasrec_attention_norm = str(self.sasrec_attention_norm or "softmax").strip().lower()
        if self.sasrec_attention_norm not in {"softmax", "softmax_custom", "softmax1"}:
            raise ValueError(f"Unsupported sasrec_attention_norm: {self.sasrec_attention_norm}")
        if self.hstu_linear_dim is None:
            self.hstu_linear_dim = max(1, self.hidden_units // max(self.num_heads, 1))
        if self.hstu_attention_dim is None:
            self.hstu_attention_dim = max(1, self.hidden_units // max(self.num_heads, 1))
        if self.hstu_attn_dropout is None:
            self.hstu_attn_dropout = self.dropout_rate
        self.longer_merge_pool = str(self.longer_merge_pool).lower()
        if self.longer_merge_pool not in {"last", "mean"}:
            raise ValueError(f"Unsupported longer_merge_pool: {self.longer_merge_pool}")
        if int(self.longer_merge_size) <= 0:
            raise ValueError("longer_merge_size must be > 0.")
        if int(self.longer_global_tokens) < 0:
            raise ValueError("longer_global_tokens must be >= 0.")
        if int(self.longer_inner_num_layers) <= 0:
            raise ValueError("longer_inner_num_layers must be > 0.")

        if self.fmlp_num_layers is None:
            self.fmlp_num_layers = self.num_blocks
        if self.fmlp_num_heads is None:
            self.fmlp_num_heads = self.num_heads
        if self.fmlp_hidden_dropout is None:
            self.fmlp_hidden_dropout = self.dropout_rate
        if self.fmlp_attn_dropout is None:
            self.fmlp_attn_dropout = self.dropout_rate
        if self.fmlp_hidden_act is None:
            self.fmlp_hidden_act = "gelu"

        if self.linrec_num_layers is None:
            self.linrec_num_layers = self.num_blocks
        if self.linrec_num_heads is None:
            self.linrec_num_heads = self.num_heads
        if self.linrec_inner_size is None:
            self.linrec_inner_size = self.hidden_units * 4
        if self.linrec_hidden_dropout is None:
            self.linrec_hidden_dropout = self.dropout_rate
        if self.linrec_attn_dropout is None:
            self.linrec_attn_dropout = self.dropout_rate
        if self.linrec_hidden_act is None:
            self.linrec_hidden_act = "gelu"

        if self.bert_num_layers is None:
            self.bert_num_layers = self.num_blocks
        if self.bert_num_heads is None:
            self.bert_num_heads = self.num_heads
        if self.bert_dropout is None:
            self.bert_dropout = self.dropout_rate

        if self.gru_embedding_size is None:
            self.gru_embedding_size = self.hidden_units
        if self.gru_num_layers is None:
            self.gru_num_layers = 1
        if self.gru_dropout is None:
            self.gru_dropout = self.dropout_rate

        if self.lru_num_blocks is None:
            self.lru_num_blocks = self.num_blocks
        if self.lru_dropout is None:
            self.lru_dropout = self.dropout_rate
        if self.lru_attn_dropout is None:
            self.lru_attn_dropout = self.dropout_rate
        if self.mamba_num_layers is None:
            self.mamba_num_layers = self.num_blocks
        if self.mamba_dropout is None:
            self.mamba_dropout = self.dropout_rate

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== Backbone Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  data_txt_path: {self.data_txt_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  backbone: {self.backbone}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  hidden_units: {self.hidden_units}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")
        logger.info(f"  right_align_positions: {self.right_align_positions}")
        logger.info(f"  initializer_range: {self.initializer_range}")
        logger.info(f"  shared_prefix_len: {self.shared_prefix_len}")
        logger.info(f"  shared_prefix_init_std: {self.shared_prefix_init_std}")

        # Backbone-specific parameters
        backbone = self.backbone.lower()
        if backbone == "sasrec":
            logger.info("SASRec Parameters:")
            logger.info(f"  use_flash_attention: {self.use_flash_attention}")
            logger.info(f"  sasrec_attention_norm: {self.sasrec_attention_norm}")
            logger.info(f"  sasrec_enable_relative_attention_bias: {self.sasrec_enable_relative_attention_bias}")
        elif backbone in {
            "hstu",
            "hstu_officialish",
            "hstu_official",
            "hstu_orig",
            "hstu_research_aligned",
            "hstu_research",
            "hstu_ra",
        }:
            logger.info("HSTU Parameters:")
            logger.info(f"  hstu_linear_dim: {self.hstu_linear_dim}")
            logger.info(f"  hstu_attention_dim: {self.hstu_attention_dim}")
            logger.info(f"  hstu_linear_activation: {self.hstu_linear_activation}")
            logger.info(f"  hstu_attn_dropout: {self.hstu_attn_dropout}")
            logger.info(f"  hstu_enable_relative_attention_bias: {self.hstu_enable_relative_attention_bias}")
            logger.info(f"  hstu_concat_ua: {self.hstu_concat_ua}")
            logger.info(f"  hstu_epsilon: {self.hstu_epsilon}")
        elif backbone == "longer":
            logger.info("LONGER Parameters:")
            logger.info(f"  longer_global_tokens: {self.longer_global_tokens}")
            logger.info(f"  longer_merge_size: {self.longer_merge_size}")
            logger.info(f"  longer_merge_pool: {self.longer_merge_pool}")
            logger.info(f"  longer_inner_num_layers: {self.longer_inner_num_layers}")
        elif backbone == "fmlp":
            logger.info("FMLP Parameters:")
            logger.info(f"  fmlp_num_layers: {self.fmlp_num_layers}")
            logger.info(f"  fmlp_num_heads: {self.fmlp_num_heads}")
            logger.info(f"  fmlp_hidden_dropout: {self.fmlp_hidden_dropout}")
            logger.info(f"  fmlp_attn_dropout: {self.fmlp_attn_dropout}")
            logger.info(f"  fmlp_hidden_act: {self.fmlp_hidden_act}")
            logger.info(f"  fmlp_no_filters: {self.fmlp_no_filters}")
        elif backbone == "linrec":
            logger.info("LinRec Parameters:")
            logger.info(f"  linrec_num_layers: {self.linrec_num_layers}")
            logger.info(f"  linrec_num_heads: {self.linrec_num_heads}")
            logger.info(f"  linrec_inner_size: {self.linrec_inner_size}")
            logger.info(f"  linrec_hidden_dropout: {self.linrec_hidden_dropout}")
            logger.info(f"  linrec_attn_dropout: {self.linrec_attn_dropout}")
            logger.info(f"  linrec_hidden_act: {self.linrec_hidden_act}")
            logger.info(f"  linrec_layer_norm_eps: {self.linrec_layer_norm_eps}")
        elif backbone == "bert4rec":
            logger.info("Bert4Rec Parameters:")
            logger.info(f"  bert_num_layers: {self.bert_num_layers}")
            logger.info(f"  bert_num_heads: {self.bert_num_heads}")
            logger.info(f"  bert_dropout: {self.bert_dropout}")
        elif backbone == "gru4rec":
            logger.info("GRU4Rec Parameters:")
            logger.info(f"  gru_embedding_size: {self.gru_embedding_size}")
            logger.info(f"  gru_num_layers: {self.gru_num_layers}")
            logger.info(f"  gru_dropout: {self.gru_dropout}")
        elif backbone == "lru":
            logger.info("LRU Parameters:")
            logger.info(f"  lru_num_blocks: {self.lru_num_blocks}")
            logger.info(f"  lru_dropout: {self.lru_dropout}")
            logger.info(f"  lru_attn_dropout: {self.lru_attn_dropout}")
        elif backbone in {"mamba4rec", "mamba"}:
            logger.info("Mamba4Rec Parameters:")
            logger.info(f"  mamba_num_layers: {self.mamba_num_layers}")
            logger.info(f"  mamba_d_state: {self.mamba_d_state}")
            logger.info(f"  mamba_d_conv: {self.mamba_d_conv}")
            logger.info(f"  mamba_expand: {self.mamba_expand}")
            logger.info(f"  mamba_dropout: {self.mamba_dropout}")
        else:
            logger.info("Backbone Parameters: (unknown backbone)")

        # Head parameters
        logger.info("Head Parameters:")
        logger.info(f"  enable_projection_head: {self.enable_projection_head}")
        logger.info(f"  head_use_gelu: {self.head_use_gelu}")
        logger.info(f"  head_use_ln: {self.head_use_ln}")
        logger.info(f"  head_residual: {self.head_residual}")
        logger.info(f"  user_embedding_norm: {self.user_embedding_norm}")
        logger.info(f"  item_l2_norm: {self.item_l2_norm}")
        logger.info(f"  temperature: {self.temperature}")
        logger.info(f"  l2_norm_eps: {self.l2_norm_eps}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  grad_clip: {self.grad_clip}")
        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info(f"  early_stop_patience: {self.early_stop_patience}")
        logger.info("Output Settings:")
        logger.info(f"  save_item_embeddings: {self.save_item_embeddings}")
        logger.info(f"  run_tag: {self.run_tag}")

        logger.info("Device Settings:")
        logger.info(f"  device: {self.device}")
        logger.info("===========================")

    def apply_backbone_overrides(self) -> None:
        """Apply backbone-specific tuning overrides."""
        backbone = self.backbone.lower()
        self.sasrec_attention_norm = str(self.sasrec_attention_norm or "softmax").strip().lower()
        if self.sasrec_attention_norm not in {"softmax", "softmax_custom", "softmax1"}:
            raise ValueError(f"Unsupported sasrec_attention_norm: {self.sasrec_attention_norm}")
        if backbone == "linrec":
            if self.linrec_max_learning_rate is not None:
                self.max_learning_rate = float(self.linrec_max_learning_rate)
            if self.linrec_min_learning_rate is not None:
                self.min_learning_rate = float(self.linrec_min_learning_rate)
            if self.linrec_weight_decay is not None:
                self.weight_decay = float(self.linrec_weight_decay)
            if self.linrec_grad_clip is not None:
                self.grad_clip = float(self.linrec_grad_clip)
            if self.linrec_early_stop_patience is not None:
                self.early_stop_patience = int(self.linrec_early_stop_patience)
            if self.linrec_hidden_dropout_override is not None:
                self.linrec_hidden_dropout = float(self.linrec_hidden_dropout_override)
            if self.linrec_attn_dropout_override is not None:
                self.linrec_attn_dropout = float(self.linrec_attn_dropout_override)


def resolve_dataset_config(config: SASRecConfig) -> None:
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


class SequentialSampler:
    """Sampler for sequential data that generates training batches."""

    def __init__(self, dataset: LooSequenceDataset, config: SASRecConfig):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length
        self.max_item = dataset.max_item
        self.min_item_id = dataset.min_item_id

        # Pre-compute which sequences are valid for training
        self.valid_user_seqs = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            if len(seq) > 3:  # Need at least 4 items to keep at least one target
                # Leave-last-two-out for training.
                self.valid_user_seqs.append((user, seq[:-2]))

    @staticmethod
    def sample_negative_item(min_id: int, max_id_exclusive: int, seen_items: set) -> int:
        """Sample a random item ID that is not in seen_items."""
        item_id = np.random.randint(min_id, max_id_exclusive)
        while item_id in seen_items:
            item_id = np.random.randint(min_id, max_id_exclusive)
        return item_id

    def __iter__(self):
        # Shuffle at the beginning of each epoch
        indices = np.random.permutation(len(self.valid_user_seqs))
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data = [self.valid_user_seqs[idx] for idx in batch_indices]

            # Generate batch tensors
            actual_batch_size = len(batch_data)
            seq_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            pos_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            neg_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)

            for idx, (user, seq) in enumerate(batch_data):
                # For each training step, we predict all positions in the sequence
                seq_len = min(len(seq), self.max_seq_length)

                if seq_len < 1:
                    continue

                # If sequence is longer than max_seq_length, take the most recent items
                if len(seq) > self.max_seq_length:
                    start_idx = len(seq) - self.max_seq_length
                    seq = seq[-self.max_seq_length :]
                    seq_len = self.max_seq_length
                else:
                    start_idx = 0

                # Input sequence: all items in the sequence
                seq_tensors[idx, -seq_len:] = torch.tensor(seq[:seq_len])

                # Positive items: for each position i, predict item at position i+1.
                # The last position has no next item, so target stays 0.
                for pos in range(seq_len):
                    if pos < seq_len - 1:
                        pos_tensors[idx, -seq_len + pos] = seq[pos + 1]

                # Sample negative items for each position
                seen_set = set(self.dataset.user_seq[user])  # Use full sequence for negative sampling
                for pos in range(seq_len):
                    neg_item = self.sample_negative_item(self.min_item_id, self.max_item + 1, seen_set)
                    neg_tensors[idx, -seq_len + pos] = neg_item

            yield {
                "input_ids": seq_tensors,
                "pos_ids": pos_tensors,
                "neg_ids": neg_tensors,
            }

    def __len__(self):
        return (len(self.valid_user_seqs) + self.batch_size - 1) // self.batch_size


# === Evaluation (sampled ranking) ===
def evaluate(
    model: nn.Module,
    dataset,
    config: SASRecConfig,
    mode: str = "test",
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate on LOO split using sampled negatives."""
    model.eval()

    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_users = 0

    users = dataset.users
    min_item_id = getattr(dataset, "min_item_id", 1)

    for batch_start in range(0, len(users), batch_size):
        batch_users = users[batch_start : batch_start + batch_size]
        batch_seqs = []
        batch_targets = []
        batch_valid_mask = []

        for user in batch_users:
            seq = dataset.user_seq[user]

            if mode == "val":
                if len(seq) < 3:
                    batch_valid_mask.append(False)
                    batch_seqs.append([])
                    batch_targets.append(0)
                    continue
                input_seq = seq[:-2]
                target = seq[-2]
            else:  # test
                if len(seq) < 2:
                    batch_valid_mask.append(False)
                    batch_seqs.append([])
                    batch_targets.append(0)
                    continue
                input_seq = seq[:-1]
                target = seq[-1]

            batch_valid_mask.append(True)
            batch_seqs.append(input_seq)
            batch_targets.append(target)

        if not any(batch_valid_mask):
            continue

        max_len = min(max(len(s) for s in batch_seqs if s), dataset.max_seq_length)
        input_tensor = torch.zeros((len(batch_users), max_len), dtype=torch.long)

        for i, seq in enumerate(batch_seqs):
            if seq and batch_valid_mask[i]:
                seq_len = min(len(seq), max_len)
                input_tensor[i, -seq_len:] = torch.tensor(seq[-seq_len:])

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            valid_indices = [i for i, valid in enumerate(batch_valid_mask) if valid]
            if not valid_indices:
                continue

            valid_input = input_tensor[valid_indices]
            valid_targets = [batch_targets[i] for i in valid_indices]

            sample_size = max(2, config.eval_sample_size)
            candidates_list = []
            use_fixed_neg = hasattr(dataset, "neg_item_by_user")
            for idx, user in enumerate([batch_users[i] for i in valid_indices]):
                target = valid_targets[idx]
                candidates = [target]
                seen_items = set(dataset.user_seq[user])
                if use_fixed_neg:
                    fixed_neg = dataset.neg_item_by_user.get(user)
                    if fixed_neg and fixed_neg not in seen_items and fixed_neg != target and fixed_neg >= min_item_id:
                        candidates.append(fixed_neg)
                while len(candidates) < sample_size:
                    neg_item = np.random.randint(min_item_id, dataset.max_item + 1)
                    if neg_item not in seen_items and neg_item not in candidates:
                        candidates.append(neg_item)
                candidates_list.append(torch.tensor(candidates, device=device))
            candidates_tensor = torch.stack(candidates_list, dim=0)

            if _uses_similarity_scoring(config):
                scores = _predict_with_similarity(model, valid_input, candidates_tensor, config)
            else:
                scores = model.predict(valid_input, candidates_tensor)

        _, indices = torch.sort(scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1].cpu().numpy() + 1  # 1-indexed ranks

        for rank in ranks:
            valid_users += 1
            if rank <= 10:
                hr_sum += 1
                ndcg_sum += 1 / np.log2(rank + 1)

    ndcg_10 = ndcg_sum / valid_users if valid_users > 0 else 0.0
    hr_10 = hr_sum / valid_users if valid_users > 0 else 0.0

    logger.info(f"Evaluated on {valid_users:,} users")

    return {"ndcg@10": ndcg_10, "hr@10": hr_10}


def build_backbone(config: SASRecConfig, item_num: int) -> nn.Module:
    _ensure_patch_defaults(config)
    backbone_name = config.backbone.lower()
    if backbone_name == "sasrec":
        return SASRecBackbone(config, item_num=item_num)
    if backbone_name == "hstu":
        return HSTU(config, item_num=item_num)
    if backbone_name in {"hstu_officialish", "hstu_official", "hstu_orig"}:
        return HSTUOfficialish(config, item_num=item_num)
    if backbone_name in {"hstu_research_aligned", "hstu_research", "hstu_ra"}:
        return HSTUResearchAligned(config, item_num=item_num)
    if backbone_name == "longer":
        return LONGER(config, item_num=item_num)
    if backbone_name == "fmlp":
        return FMLP(config, item_num=item_num)
    if backbone_name == "linrec":
        return LinRec(config, item_num=item_num)
    if backbone_name in {"bert4rec", "bert"}:
        return Bert4Rec(config, item_num=item_num)
    if backbone_name in {"gru4rec", "gru"}:
        return GRU4Rec(config, item_num=item_num)
    if backbone_name == "lru":
        return LRU(config, item_num=item_num)
    if backbone_name in {"mamba4rec", "mamba"}:
        return Mamba4Rec(config, item_num=item_num)
    raise ValueError(
        f"Unknown backbone '{config.backbone}'. Expected one of: sasrec, hstu, hstu_officialish, hstu_research_aligned, longer, fmlp, linrec, bert4rec, gru4rec, lru, mamba4rec."
    )


def _ensure_patch_defaults(config: SASRecConfig) -> None:
    defaults = {
        "num_patches": 1,
        "patch_len": 0,
        "use_gating": False,
        "gating_hidden_dim": 64,
        "patch_init_std": 0.0,
        "gating_init_std": 0.0,
        "gating_pool": "last",
        "gating_temperature": 1.0,
        "gating_noise_std": 0.0,
        "patch_routing": "learned",
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            setattr(config, name, value)


def _uses_similarity_scoring(config: Optional[SASRecConfig]) -> bool:
    if config is None:
        return False
    norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
    item_l2 = bool(getattr(config, "item_l2_norm", False))
    temperature = float(getattr(config, "temperature", 1.0) or 1.0)
    return norm != "none" or item_l2 or abs(temperature - 1.0) > 1e-12


def _resolve_item_embedding(model: nn.Module) -> nn.Module:
    if hasattr(model, "item_emb"):
        return model.item_emb
    if hasattr(model, "embedding") and hasattr(model.embedding, "token"):
        return model.embedding.token
    raise AttributeError("Unable to locate item embedding on model (expected item_emb or embedding.token).")


def _strip_hidden_states_for_scoring(model: nn.Module, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    _ = input_ids
    if hasattr(model, "_strip_patch_tokens"):
        try:
            hidden_states = model._strip_patch_tokens(hidden_states)
        except TypeError:
            patch_len = int(getattr(model, "patch_len", 0) or 0)
            if patch_len > 0:
                hidden_states = hidden_states[:, patch_len:, :]
    if hasattr(model, "strip_patch_tokens"):
        try:
            hidden_states = model.strip_patch_tokens(hidden_states)
        except TypeError:
            patch_len = int(getattr(model, "patch_len", 0) or 0)
            if patch_len > 0:
                hidden_states = hidden_states[:, patch_len:, :]
    if hasattr(model, "_strip_shared_tokens"):
        hidden_states = model._strip_shared_tokens(hidden_states)
    return hidden_states


def _normalize_user_embeddings(
    projected: torch.Tensor,
    config: SASRecConfig,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    if model is not None and hasattr(model, "postprocess_query_embeddings"):
        return model.postprocess_query_embeddings(projected)
    norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
    eps = float(getattr(config, "l2_norm_eps", 1e-6) or 1e-6)
    if norm == "none":
        return projected
    if norm == "l2_norm":
        return F.normalize(projected, p=2, dim=-1, eps=eps)
    if norm == "layer_norm":
        return F.layer_norm(projected, normalized_shape=(projected.size(-1),), eps=eps)
    raise ValueError(f"Unsupported user_embedding_norm: {norm}")


def _normalize_item_embeddings(
    item_embeddings: torch.Tensor,
    config: SASRecConfig,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    if model is not None and hasattr(model, "postprocess_item_embeddings"):
        return model.postprocess_item_embeddings(item_embeddings)
    if not bool(getattr(config, "item_l2_norm", False)):
        return item_embeddings
    eps = float(getattr(config, "l2_norm_eps", 1e-6) or 1e-6)
    return F.normalize(item_embeddings, p=2, dim=-1, eps=eps)


def _apply_similarity_logits(
    projected: torch.Tensor,
    item_embeddings: torch.Tensor,
    config: SASRecConfig,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    projected = _normalize_user_embeddings(projected, config, model=model)
    item_embeddings = _normalize_item_embeddings(item_embeddings.to(projected.dtype), config, model=model)
    if item_embeddings.dim() == projected.dim():
        logits = (projected * item_embeddings).sum(dim=-1)
    elif item_embeddings.dim() == projected.dim() + 1:
        logits = (projected.unsqueeze(-2) * item_embeddings).sum(dim=-1)
    elif projected.dim() == 2 and item_embeddings.dim() == 3:
        logits = torch.einsum("bd,bnd->bn", projected, item_embeddings)
    else:
        raise ValueError(
            f"Unsupported shapes for similarity logits: projected={tuple(projected.shape)}, "
            f"item_embeddings={tuple(item_embeddings.shape)}"
        )
    temperature = float(getattr(config, "temperature", 1.0) or 1.0)
    if temperature != 1.0:
        logits = logits / temperature
    return logits


def _training_logits_with_similarity(
    model: nn.Module,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    config: SASRecConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = model.forward_features(input_ids)
    hidden_states = _strip_hidden_states_for_scoring(model, hidden_states, input_ids)
    projected = model.apply_head(hidden_states)
    item_emb = _resolve_item_embedding(model)
    pos_embs = item_emb(pos_ids).to(projected.dtype)
    neg_embs = item_emb(neg_ids).to(projected.dtype)
    pos_logits = _apply_similarity_logits(projected, pos_embs, config, model=model)
    neg_logits = _apply_similarity_logits(projected, neg_embs, config, model=model)
    return pos_logits, neg_logits


def _predict_with_similarity(
    model: nn.Module,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: SASRecConfig,
) -> torch.Tensor:
    hidden_states = model.forward_features(input_ids)
    hidden_states = _strip_hidden_states_for_scoring(model, hidden_states, input_ids)
    final_hidden = model.apply_head(hidden_states[:, -1, :])
    candidate_embs = _resolve_item_embedding(model)(candidate_ids).to(final_hidden.dtype)
    return _apply_similarity_logits(final_hidden, candidate_embs, config, model=model)


def build_checkpoint_tag(config: SASRecConfig) -> str:
    backbone_name = config.backbone.lower()
    num_blocks = config.lru_num_blocks if backbone_name == "lru" else config.num_blocks
    tag = (
        f"{backbone_name}_{config.dataset}_seq{config.max_seq_length}_dim{config.hidden_units}"
        f"_L{num_blocks}_H{config.num_heads}"
    )
    if int(getattr(config, "shared_prefix_len", 0) or 0) > 0:
        tag += f"_SP{int(config.shared_prefix_len)}"
    if backbone_name == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        tag += "_SM1"
    elif backbone_name == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax_custom":
        tag += "_SMCUSTOM"
    if backbone_name == "sasrec" and bool(getattr(config, "sasrec_enable_relative_attention_bias", False)):
        tag += "_RB"
    if _uses_similarity_scoring(config):
        norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
        if norm == "l2_norm":
            tag += "_UQ"
        elif norm == "layer_norm":
            tag += "_ULN"
        if bool(getattr(config, "item_l2_norm", False)):
            tag += "_IQ"
        temperature = float(getattr(config, "temperature", 1.0) or 1.0)
        if abs(temperature - 1.0) > 1e-12:
            tag += f"_T{str(temperature).replace('.', 'p')}"
    return tag


def save_model_checkpoint(model: nn.Module, config: SASRecConfig) -> Path:
    """Save model state for the current best validation (overwrite)."""
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


def save_item_embeddings(model: nn.Module, config: SASRecConfig) -> Path:
    """Save item embedding matrix (excluding padding idx=0, overwrite)."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "item_emb"):
        emb_weight = model.item_emb.weight
    elif hasattr(model, "embedding") and hasattr(model.embedding, "token"):
        emb_weight = model.embedding.token.weight
    else:
        raise AttributeError("Unable to locate item embedding weights on model (expected item_emb or embedding.token).")
    emb = emb_weight.detach().cpu().numpy()
    emb = emb[1:]  # drop padding row
    filename = f"{build_checkpoint_tag(config)}_item_embeddings_best.npy"
    out_path = config.checkpoint_dir / filename
    np.save(out_path, emb)
    logger.info(f"Saved item embeddings to {out_path}")
    return out_path


def get_gradient_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of gradients across all model parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    # Compute norm without materializing concatenated tensor
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


def train_sasrec(
    model: nn.Module,
    train_dataset,
    config: SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Dict[str, float]:
    """
    Train backbone model.

    Args:
        model: backbone model to train
        train_dataset: Training dataset
        config: Training configuration
        device: Device to train on

    Returns:
        Dictionary with best validation metrics
    """
    model = model.to(device)

    # Apply torch.compile for faster training (CUDA only, not MPS)
    if device.startswith("cuda"):
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    logger.info("Sampling strategy: sequential (full-seq, per-position) with 1 negative/position")
    train_sampler = SequentialSampler(train_dataset, config)

    steps_per_epoch = len(train_sampler)
    total_steps = config.num_epochs * steps_per_epoch

    logger.info(f"Training for {config.num_epochs} epochs, {steps_per_epoch} steps per epoch")
    logger.info(f"Total training steps: {total_steps:,}")

    # Optimizer with fused support
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    has_complex = any(p.is_complex() for p in model.parameters())
    use_fused = fused_available and device.startswith("cuda") and (not has_complex)
    if fused_available and device.startswith("cuda") and has_complex:
        logger.info("Detected complex parameters; disabling fused AdamW.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
        fused=use_fused,
    )

    # Learning rate scheduler
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_learning_rate
        )
        logger.info(
            f"Cosine annealing: {config.max_learning_rate:.1e} -> {config.min_learning_rate:.1e} for {total_steps:,} steps"
        )
    elif config.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.max_learning_rate,
            total_iters=config.warmup_steps,
        )
        logger.info(
            f"Warmup: {config.warmup_start_lr:.1e} -> {config.max_learning_rate:.1e} for {config.warmup_steps:,} steps"
        )

        cosine_steps = total_steps - config.warmup_steps
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=config.min_learning_rate
        )
        logger.info(
            f"Cosine annealing: {config.max_learning_rate:.1e} -> {config.min_learning_rate:.1e} for {cosine_steps:,} steps"
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
        )
    else:
        scheduler = None

    # Loss function
    bce_criterion = nn.BCEWithLogitsLoss()

    # Track best model
    best_val_metrics = {"ndcg@10": -1.0, "hr@10": -1.0}
    no_improve_steps = 0
    stop_training = False
    global_step = 0

    # Create progress bar for entire training (all epochs)
    pbar = tqdm(total=total_steps)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_sampler):
            t0 = time.time()
            optimizer.zero_grad()

            # Initialize loss values for logging
            loss_value = 0.0
            pos_loss_value = 0.0
            neg_loss_value = 0.0
            pos_logit_stats = None
            neg_logit_stats = None

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pos_ids = batch["pos_ids"].to(device, non_blocking=True)
            neg_ids = batch["neg_ids"].to(device, non_blocking=True)

            # Forward pass
            if _uses_similarity_scoring(config):
                pos_logits, neg_logits = _training_logits_with_similarity(model, input_ids, pos_ids, neg_ids, config)
            else:
                pos_logits, neg_logits = model.training_step(input_ids, pos_ids, neg_ids)

            valid_mask = pos_ids != 0

            if valid_mask.any():
                pos_labels = torch.ones_like(pos_logits)[valid_mask]
                neg_labels = torch.zeros_like(neg_logits)[valid_mask]

                pos_loss = bce_criterion(pos_logits[valid_mask], pos_labels)
                neg_loss = bce_criterion(neg_logits[valid_mask], neg_labels)
                loss = pos_loss + neg_loss
                loss_value = loss.item()
                pos_loss_value = pos_loss.item()
                neg_loss_value = neg_loss.item()

                with torch.no_grad():
                    pos_logit_stats = (
                        pos_logits[valid_mask].mean().item(),
                        pos_logits[valid_mask].min().item(),
                        pos_logits[valid_mask].max().item(),
                    )
                    neg_logit_stats = (
                        neg_logits[valid_mask].mean().item(),
                        neg_logits[valid_mask].min().item(),
                        neg_logits[valid_mask].max().item(),
                    )

                loss.backward()
                if config.grad_clip and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Get gradient norm for logging
            grad_norm = get_gradient_norm(model)

            # Optimizer step
            optimizer.step()

            # Step the learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]["lr"]

            # Update metrics
            epoch_loss += loss_value
            epoch_steps += 1
            global_step += 1
            pbar.update(1)  # Update progress bar by one step

            # Time measurement
            t1 = time.time()
            batch_time_ms = (t1 - t0) * 1000
            batch_size_actual = input_ids.size(0)
            samples_per_second = batch_size_actual / (t1 - t0)

            # Logging
            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                log_str = (
                    f"Step {global_step:06d} | Epoch {epoch + 1:03d}/{config.num_epochs:03d} | "
                    f"Loss: {loss_value:.4f} (pos: {pos_loss_value:.4f}, neg: {neg_loss_value:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Grad: {grad_norm:.2f}"
                )
                if pos_logit_stats is not None and neg_logit_stats is not None:
                    pl_mean, pl_min, pl_max = pos_logit_stats
                    nl_mean, nl_min, nl_max = neg_logit_stats
                    log_str += (
                        f" | Logits pos(m/min/max): {pl_mean:.2f}/{pl_min:.2f}/{pl_max:.2f}"
                        f" neg(m/min/max): {nl_mean:.2f}/{nl_min:.2f}/{nl_max:.2f}"
                    )
                log_str += f" | Time: {batch_time_ms:.0f}ms | Samples/s: {samples_per_second:,.0f}"
                logger.info(log_str)

                # Log to W&B
                wandb.log(
                    {
                        # Losses
                        "loss/total": loss_value,
                        "train/learning_rate": current_lr,
                        "train/gradient_norm": grad_norm,
                        "train/batch_time_ms": batch_time_ms,
                        "train/samples_per_second": samples_per_second,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                )

            # Validation
            if global_step % config.steps_per_val_log == 0:
                logger.info("Running validation...")
                eval_dataset = val_dataset if val_dataset is not None else train_dataset
                eval_mode = "val" if val_dataset is not None else "test"
                val_metrics = evaluate(model, eval_dataset, config=config, mode=eval_mode, device=device)
                model.train()

                logger.info(
                    f"Step {global_step:06d} | Validation - NDCG@10: {val_metrics['ndcg@10']:.4f}, HR@10: {val_metrics['hr@10']:.4f}"
                )

                # Log to W&B
                wandb.log(
                    {
                        "val/ndcg@10": val_metrics["ndcg@10"],
                        "val/hr@10": val_metrics["hr@10"],
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                )

                # Save best model
                if val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
                    best_val_metrics = val_metrics
                    no_improve_steps = 0
                    logger.info(
                        f"New best validation - NDCG@10: {val_metrics['ndcg@10']:.4f}, HR@10: {val_metrics['hr@10']:.4f}"
                    )
                    save_model_checkpoint(model, config)
                    if config.save_item_embeddings:
                        save_item_embeddings(model, config)
                else:
                    no_improve_steps += 1
                    if config.early_stop_patience > 0 and no_improve_steps >= config.early_stop_patience:
                        logger.info(
                            "Early stopping: no improvement in NDCG@10 for "
                            f"{config.early_stop_patience} validation checks."
                        )
                        stop_training = True
                        break

        if stop_training:
            break

    pbar.close()  # Close the progress bar
    logger.info(f"Training completed. Best validation NDCG@10: {best_val_metrics['ndcg@10']:.4f}")

    return best_val_metrics


if __name__ == "__main__":
    # Adjust hyperparameters by editing the SASRecConfig defaults above or overriding them here.
    config = SASRecConfig()
    config.backbone = config.backbone.lower()
    config.apply_backbone_overrides()
    resolve_dataset_config(config)
    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_blocks = config.lru_num_blocks if config.backbone == "lru" else config.num_blocks
    es_suffix = "noes" if config.early_stop_patience <= 0 else f"es{config.early_stop_patience}"
    run_name = (
        f"{config.backbone}-{config.dataset}-{run_blocks}b-{config.num_heads}h-"
        f"{config.hidden_units}-{config.max_seq_length}_{es_suffix}-standard"
    )
    if int(getattr(config, "shared_prefix_len", 0) or 0) > 0:
        run_name += f"-sp{int(config.shared_prefix_len)}"
    if config.backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        run_name += "-sm1"
    elif config.backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax_custom":
        run_name += "-smcustom"
    if config.backbone == "sasrec" and bool(getattr(config, "sasrec_enable_relative_attention_bias", False)):
        run_name += "-rbias"
    if _uses_similarity_scoring(config):
        norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
        if norm == "l2_norm":
            run_name += "-uq"
        elif norm == "layer_norm":
            run_name += "-uln"
        if bool(getattr(config, "item_l2_norm", False)):
            run_name += "-iq"
        temperature = float(getattr(config, "temperature", 1.0) or 1.0)
        if abs(temperature - 1.0) > 1e-12:
            run_name += f"-t{str(temperature).replace('.', 'p')}"
    run = wandb.init(project=f"backbone-standard-{config.dataset}", name=run_name, config=config.__dict__)
    base_ckpt_dir = config.checkpoint_dir / f"{config.backbone}_loo_standard"
    if not config.run_tag:
        run_id = getattr(run, "id", None) or getattr(wandb.run, "id", None)
        tag = time.strftime("%Y%m%d_%H%M%S")
        if run_id:
            tag = f"{tag}-{run_id}"
        config.run_tag = tag
    config.checkpoint_dir = base_ckpt_dir / str(config.run_tag)
    config.log_config()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = LooSequenceDataset(config.data_txt_path, config, logger=logger)
    val_dataset = train_dataset
    test_dataset = train_dataset
    # PAD=0; real items are offset by +1.
    item_num = train_dataset.num_items

    model = build_backbone(config, item_num=item_num)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    best_metrics = train_sasrec(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=val_dataset,
    )

    logger.info("Running final test evaluation...")
    test_metrics = evaluate(model, test_dataset, config=config, mode="test", device=device)
    logger.info(f"Test Results - NDCG@10: {test_metrics['ndcg@10']:.4f}, HR@10: {test_metrics['hr@10']:.4f}")

    wandb.log({"test/ndcg@10": test_metrics["ndcg@10"], "test/hr@10": test_metrics["hr@10"]})
    wandb.log(
        {
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )

    wandb.finish()
    logger.info("Training complete!")
