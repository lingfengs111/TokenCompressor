#!/usr/bin/env python3
"""Train backbone models on protocol-aware LOO datasets."""

import inspect
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import wandb

from core.device_manager import DeviceManager
from core.logger import setup_logger
from core.loo_dataset import LooSequenceDataset, resolve_loo_dataset, infer_loo_min_len
from core.streaming_eval import (
    LEGACY_LOO_PROTOCOL,
    finalize_eval_metrics,
    flatten_streaming_eval_metrics,
    flatten_streaming_eval_test_aliases,
    normalize_eval_protocol,
    resolve_eval_target_positions,
    resolve_train_cutoff,
    update_rank_metrics,
)
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


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python/NumPy/PyTorch RNG state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_eval_seed(config: "SASRecConfig", mode: str, streaming_last_k: int = 0) -> int:
    base_seed = getattr(config, "eval_seed", None)
    if base_seed is None:
        base_seed = getattr(config, "seed", 2026)
    offset = 0 if str(mode).lower() == "val" else 1_000_003
    offset += int(streaming_last_k or 0) * 7_919
    return int(base_seed) + offset


def _build_eval_rng(config: "SASRecConfig", mode: str, streaming_last_k: int = 0) -> np.random.RandomState:
    return np.random.RandomState(_resolve_eval_seed(config, mode, streaming_last_k))


@dataclass
class SASRecConfig:
    """Configuration for backbone training (protocol-aware LOO datasets)."""

    dataset: str = "taobao_loo202"
    data_dir: Optional[Path] = None
    data_txt_path: Optional[Path] = None
    checkpoint_dir: Path = Path("/home/lingfengs111/codes/soft_patch_training/checkpoints")

    # Model (shared)
    backbone: str = "sasrec"  # sasrec | hstu | longer | fmlp | linrec | bert4rec | gru4rec | lru | mamba4rec
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
    use_gradient_checkpointing: bool = False
    use_torch_compile: bool = True
    use_amp: bool = False
    amp_dtype: str = "bf16"

    # HSTU-specific
    hstu_linear_dim: Optional[int] = None
    hstu_attention_dim: Optional[int] = None
    hstu_linear_activation: str = "silu"
    hstu_attn_dropout: Optional[float] = None
    hstu_enable_relative_attention_bias: bool = False
    hstu_normalization: str = "rel_bias"
    hstu_concat_ua: bool = False
    hstu_epsilon: float = 1e-6
    hstu_parametric_block_norm: bool = False

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
    linrec_hidden_dropout_override: Optional[float] = 0.2
    linrec_attn_dropout_override: Optional[float] = 0.2
    linrec_max_learning_rate: Optional[float] = 5e-4
    linrec_min_learning_rate: Optional[float] = 5e-6
    linrec_weight_decay: Optional[float] = 1e-2
    linrec_grad_clip: Optional[float] = 1.0
    linrec_early_stop_patience: Optional[int] = 5

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

    # Training parameters
    batch_size: int = 1024  # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    seed: int = 2026  # Global RNG seed
    eval_seed: Optional[int] = None  # Fixed RNG for sampled evaluation negatives
    deterministic: bool = False  # Enable deterministic ops (slower)
    max_learning_rate: float = 5e-4  # Maximum learning rate (start of cosine)
    min_learning_rate: float = 5e-6  # Minimum learning rate (end of cosine)
    weight_decay: float = 0.0  # AdamW weight decay
    grad_clip: float = 0.0  # Global grad clip (0 disables)
    num_negatives: int = 128  # Number of negatives per positive for sampled softmax
    sampled_softmax_chunk_size: int = 4096  # Chunk size for sampled softmax logits
    user_embedding_norm: str = "none"  # none | l2_norm | layer_norm
    item_l2_norm: bool = False
    temperature: float = 1.0
    l2_norm_eps: float = 1e-6
    enable_score_item_bias: bool = False
    enable_score_length_bias: bool = False
    enable_score_length_scale: bool = False
    score_length_bucket_size: int = 20

    # Training settings
    scheduler_type: str = "cosine_with_warmup"  # Learning rate scheduler type ("cosine" or "cosine_with_warmup")
    warmup_steps: int = 200  # Number of warmup steps (only for cosine_with_warmup)
    warmup_start_lr: float = 5e-7  # Starting learning rate for warmup (only for cosine_with_warmup)
    steps_per_train_log: int = 100  # Log training progress every N steps
    steps_per_val_log: int = 300  # Validate and checkpoint every N steps
    # ⚠️之前eval用的是100导致结果看起来很高
    eval_sample_size: int = 1000  # Total candidates per user when eval_mode="sampled" (includes target)
    selection_metric: str = "ndcg@10"  # Validation metric used for checkpoint selection / early stopping
    early_stop_patience: int = 0  # Stop after N validations without improvement (<=0 disables)
    eval_protocol: str = "legacy_loo"  # legacy_loo | holdout_anchor
    last_k_eval_test: int = 10  # Used when eval_protocol=holdout_anchor
    streaming_eval_last_k: int = 0  # If >1, run extra rolling final-test eval on the last K targets

    # DataLoader settings
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True

    # Output settings
    save_item_embeddings: bool = True  # Save item embeddings on best validation

    # Run tagging (for unique checkpoint folders)
    run_tag: Optional[str] = None

    # Device settings
    device: str = "cuda:2"  # e.g., "cuda:1", "cpu", "mps"

    def __post_init__(self):
        if self.use_flash_attention is None:
            self.use_flash_attention = True
        self.sasrec_attention_norm = str(self.sasrec_attention_norm or "softmax").strip().lower()
        if self.sasrec_attention_norm not in {"softmax", "softmax1"}:
            raise ValueError(f"Unsupported sasrec_attention_norm: {self.sasrec_attention_norm}")
        self.amp_dtype = str(self.amp_dtype).lower()
        if self.hstu_linear_dim is None:
            self.hstu_linear_dim = max(1, self.hidden_units // max(self.num_heads, 1))
        if self.hstu_attention_dim is None:
            self.hstu_attention_dim = max(1, self.hidden_units // max(self.num_heads, 1))
        if self.hstu_attn_dropout is None:
            self.hstu_attn_dropout = self.dropout_rate
        self.hstu_normalization = str(self.hstu_normalization).lower()
        self.user_embedding_norm = str(self.user_embedding_norm).lower()
        self.selection_metric = str(self.selection_metric).lower()
        self.eval_protocol = normalize_eval_protocol(getattr(self, "eval_protocol", LEGACY_LOO_PROTOCOL))
        self.last_k_eval_test = int(getattr(self, "last_k_eval_test", 0) or 0)
        if self.eval_protocol == LEGACY_LOO_PROTOCOL:
            self.last_k_eval_test = 0
        self.streaming_eval_last_k = int(getattr(self, "streaming_eval_last_k", 0) or 0)
        if self.hstu_normalization not in {"rel_bias", "hstu_rel_bias", "softmax_rel_bias", "softmax1_rel_bias"}:
            raise ValueError(f"Unsupported hstu_normalization: {self.hstu_normalization}")
        self.longer_merge_pool = str(self.longer_merge_pool).lower()
        if self.longer_merge_pool not in {"last", "mean"}:
            raise ValueError(f"Unsupported longer_merge_pool: {self.longer_merge_pool}")
        if int(self.longer_merge_size) <= 0:
            raise ValueError("longer_merge_size must be > 0.")
        if int(self.longer_global_tokens) < 0:
            raise ValueError("longer_global_tokens must be >= 0.")
        if int(self.longer_inner_num_layers) <= 0:
            raise ValueError("longer_inner_num_layers must be > 0.")
        if self.user_embedding_norm not in {"none", "l2_norm", "layer_norm"}:
            raise ValueError(f"Unsupported user_embedding_norm: {self.user_embedding_norm}")
        if self.selection_metric not in {"ndcg@10", "hr@10"}:
            raise ValueError(f"Unsupported selection_metric: {self.selection_metric}")
        if self.eval_protocol != LEGACY_LOO_PROTOCOL and self.last_k_eval_test < 2:
            raise ValueError(
                f"holdout_anchor protocol requires last_k_eval_test >= 2, got {self.last_k_eval_test}."
            )
        if self.streaming_eval_last_k < 0:
            raise ValueError("streaming_eval_last_k must be >= 0.")
        if self.eval_seed is None:
            self.eval_seed = int(self.seed)
        if float(self.temperature) <= 0:
            raise ValueError("temperature must be > 0.")
        self.score_length_bucket_size = max(1, int(self.score_length_bucket_size or 1))
        if self.amp_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype}")

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
            logger.info(f"  use_gradient_checkpointing: {self.use_gradient_checkpointing}")
            logger.info(f"  use_torch_compile: {self.use_torch_compile}")
            logger.info(f"  use_amp: {self.use_amp}")
            logger.info(f"  amp_dtype: {self.amp_dtype}")
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
            logger.info(f"  hstu_normalization: {self.hstu_normalization}")
            logger.info(f"  hstu_concat_ua: {self.hstu_concat_ua}")
            logger.info(f"  hstu_epsilon: {self.hstu_epsilon}")
            logger.info(f"  hstu_parametric_block_norm: {self.hstu_parametric_block_norm}")
            logger.info(f"  use_gradient_checkpointing: {self.use_gradient_checkpointing}")
            logger.info(f"  use_torch_compile: {self.use_torch_compile}")
            logger.info(f"  use_amp: {self.use_amp}")
            logger.info(f"  amp_dtype: {self.amp_dtype}")
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

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  seed: {self.seed}")
        logger.info(f"  eval_seed: {self.eval_seed}")
        logger.info(f"  deterministic: {self.deterministic}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  grad_clip: {self.grad_clip}")
        logger.info(f"  num_negatives: {self.num_negatives}")
        logger.info(f"  sampled_softmax_chunk_size: {self.sampled_softmax_chunk_size}")
        logger.info("Similarity Parameters:")
        logger.info(f"  user_embedding_norm: {self.user_embedding_norm}")
        logger.info(f"  item_l2_norm: {self.item_l2_norm}")
        logger.info(f"  temperature: {self.temperature}")
        logger.info(f"  l2_norm_eps: {self.l2_norm_eps}")
        logger.info(f"  enable_score_item_bias: {self.enable_score_item_bias}")
        logger.info(f"  enable_score_length_bias: {self.enable_score_length_bias}")
        logger.info(f"  enable_score_length_scale: {self.enable_score_length_scale}")
        logger.info(f"  score_length_bucket_size: {self.score_length_bucket_size}")
        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info(f"  selection_metric: {self.selection_metric}")
        logger.info(f"  early_stop_patience: {self.early_stop_patience}")
        logger.info(f"  eval_protocol: {self.eval_protocol}")
        logger.info(f"  last_k_eval_test: {self.last_k_eval_test}")
        logger.info(f"  streaming_eval_last_k: {self.streaming_eval_last_k}")
        logger.info("DataLoader Settings:")
        logger.info(f"  num_workers: {self.num_workers}")
        logger.info(f"  prefetch_factor: {self.prefetch_factor}")
        logger.info(f"  persistent_workers: {self.persistent_workers}")
        logger.info(f"  pin_memory: {self.pin_memory}")
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
        if self.sasrec_attention_norm not in {"softmax", "softmax1"}:
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


def resolve_eval_protocol_config(config: SASRecConfig) -> None:
    config.eval_protocol = normalize_eval_protocol(getattr(config, "eval_protocol", LEGACY_LOO_PROTOCOL))
    config.last_k_eval_test = int(getattr(config, "last_k_eval_test", 0) or 0)
    if config.eval_protocol == LEGACY_LOO_PROTOCOL and config.last_k_eval_test != 0:
        logger.warning(
            "Ignoring last_k_eval_test=%s because eval_protocol=legacy_loo uses the final item target.",
            config.last_k_eval_test,
        )
        config.last_k_eval_test = 0
    config.streaming_eval_last_k = int(getattr(config, "streaming_eval_last_k", 0) or 0)
    if config.eval_protocol != LEGACY_LOO_PROTOCOL and config.last_k_eval_test < 2:
        raise ValueError(
            f"holdout_anchor protocol requires last_k_eval_test >= 2, got {config.last_k_eval_test}."
        )


def build_protocol_run_suffix(config: SASRecConfig) -> str:
    suffix = []
    if normalize_eval_protocol(getattr(config, "eval_protocol", LEGACY_LOO_PROTOCOL)) != LEGACY_LOO_PROTOCOL:
        suffix.append(f"anchork{int(getattr(config, 'last_k_eval_test', 0) or 0)}")
    if int(getattr(config, "streaming_eval_last_k", 0) or 0) > 1:
        suffix.append(f"stream{int(getattr(config, 'streaming_eval_last_k', 0) or 0)}")
    return "".join(f"-{part}" for part in suffix)


class LooTrainDataset(Dataset):
    """Training dataset for protocol-aware holdout sequences."""

    def __init__(self, dataset: LooSequenceDataset, config: SASRecConfig):
        self.dataset = dataset
        self.max_seq_length = config.max_seq_length
        self.samples = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            train_end = resolve_train_cutoff(
                len(seq),
                eval_protocol=getattr(config, "eval_protocol", LEGACY_LOO_PROTOCOL),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            if train_end > 1:
                self.samples.append((user, seq[:train_end]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _resample_negatives(
    neg_row: torch.Tensor,
    seen_tensor: torch.Tensor,
    min_item_id: int,
    max_item: int,
    valid_mask: Optional[torch.Tensor] = None,
    max_tries: int = 10,
) -> None:
    if seen_tensor.numel() == 0:
        return
    mask = torch.isin(neg_row, seen_tensor)
    if valid_mask is not None:
        if valid_mask.ndim < mask.ndim:
            for _ in range(mask.ndim - valid_mask.ndim):
                valid_mask = valid_mask.unsqueeze(-1)
        mask = mask & valid_mask
    tries = 0
    while mask.any() and tries < max_tries:
        neg_row[mask] = torch.randint(
            min_item_id,
            max_item + 1,
            size=(int(mask.sum().item()),),
            dtype=neg_row.dtype,
        )
        mask = torch.isin(neg_row, seen_tensor)
        if valid_mask is not None:
            mask = mask & valid_mask
        tries += 1


def build_train_collate_fn(train_data: LooTrainDataset, config: SASRecConfig):
    base_dataset = train_data.dataset
    max_seq_length = config.max_seq_length
    min_item_id = base_dataset.min_item_id
    max_item = base_dataset.max_item
    num_negatives = max(1, int(config.num_negatives))
    def collate(batch):
        batch_size = len(batch)
        seq_tensors = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        pos_tensors = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        seen_sequences = []

        for i, (user, seq) in enumerate(batch):
            seen_sequences.append([x for x in seq if x >= min_item_id])
            seq_len = min(len(seq), max_seq_length)
            if seq_len < 1:
                continue
            if len(seq) > max_seq_length:
                seq = seq[-max_seq_length:]
                seq_len = max_seq_length
            seq_tensors[i, -seq_len:] = torch.as_tensor(seq[:seq_len], dtype=torch.long)
            if seq_len > 1:
                pos_tensors[i, -seq_len:-1] = torch.as_tensor(seq[1:seq_len], dtype=torch.long)
        neg_tensors = torch.randint(
            min_item_id,
            max_item + 1,
            size=(batch_size, max_seq_length, num_negatives),
            dtype=torch.long,
        )
        valid_mask = pos_tensors != 0
        for i, seen in enumerate(seen_sequences):
            if not seen:
                continue
            seen_tensor = torch.as_tensor(seen, dtype=torch.long)
            _resample_negatives(neg_tensors[i], seen_tensor, min_item_id, max_item, valid_mask[i])
        return {
            "input_ids": seq_tensors,
            "pos_ids": pos_tensors,
            "neg_ids": neg_tensors,
        }

    return collate


def _resolve_item_embedding(model: nn.Module) -> nn.Module:
    if hasattr(model, "item_emb"):
        return model.item_emb
    if hasattr(model, "embedding") and hasattr(model.embedding, "token"):
        return model.embedding.token
    raise AttributeError("Unable to locate item embedding on model (expected item_emb or embedding.token).")


def _compute_projected_hidden(
    model: nn.Module,
    input_ids: torch.Tensor,
    use_patch: bool = True,
) -> torch.Tensor:
    hidden_states = model.forward_features(input_ids, use_patch=use_patch)
    if use_patch and hasattr(model, "strip_patch_tokens"):
        hidden_states = model.strip_patch_tokens(hidden_states)
    else:
        patch_len = getattr(model, "patch_len", 0)
        if use_patch and patch_len and patch_len > 0:
            hidden_states = hidden_states[:, patch_len:, :]
    return model.apply_head(hidden_states)


def _sequence_lengths_from_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return (input_ids != 0).sum(dim=1)


def _normalize_user_embeddings(
    projected: torch.Tensor,
    config: SASRecConfig,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    if model is not None and hasattr(model, "postprocess_query_embeddings"):
        return model.postprocess_query_embeddings(projected)
    norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
    eps = float(getattr(config, "l2_norm_eps", 1e-6))
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
    eps = float(getattr(config, "l2_norm_eps", 1e-6))
    return F.normalize(item_embeddings, p=2, dim=-1, eps=eps)


def _apply_similarity_logits(
    projected: torch.Tensor,
    item_embeddings: torch.Tensor,
    config: SASRecConfig,
    model: Optional[nn.Module] = None,
    *,
    item_ids: Optional[torch.Tensor] = None,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    projected = _normalize_user_embeddings(projected, config, model=model)
    item_embeddings = _normalize_item_embeddings(item_embeddings.to(projected.dtype), config, model=model)
    if item_embeddings.dim() == 2:
        logits = (projected * item_embeddings).sum(dim=-1)
    elif item_embeddings.dim() == 3:
        logits = torch.einsum("bd,bnd->bn", projected, item_embeddings)
    else:
        raise ValueError(f"Unsupported item embedding rank: {item_embeddings.dim()}")
    temperature = float(getattr(config, "temperature", 1.0))
    if temperature != 1.0:
        logits = logits / temperature
    if (
        model is not None
        and hasattr(model, "score_calibration")
        and item_ids is not None
        and seq_lengths is not None
    ):
        logits = model.score_calibration(logits, item_ids, seq_lengths)
    return logits


def _predict_with_similarity(
    model: nn.Module,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: SASRecConfig,
    use_patch: bool = True,
) -> torch.Tensor:
    hidden_states = _compute_projected_hidden(model, input_ids, use_patch=use_patch)
    final_hidden = hidden_states[:, -1, :]
    candidate_embs = _resolve_item_embedding(model)(candidate_ids).to(final_hidden.dtype)
    seq_lengths = _sequence_lengths_from_input_ids(input_ids)
    return _apply_similarity_logits(
        final_hidden,
        candidate_embs,
        config,
        model=model,
        item_ids=candidate_ids,
        seq_lengths=seq_lengths,
    )


def _resolve_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if amp_dtype.lower() == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        logger.warning("AMP bf16 requested but not supported; falling back to fp16.")
        return torch.float16
    return torch.float16


def _amp_context(use_amp: bool, amp_dtype: torch.dtype, device: torch.device):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _sampled_softmax_loss_chunked(
    model: nn.Module,
    projected: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    item_emb: nn.Module,
    chunk_size: int,
    config: SASRecConfig,
    seq_lengths: torch.Tensor,
) -> tuple[torch.Tensor, Optional[tuple[float, float, float]], Optional[tuple[float, float, float]]]:
    valid_mask = pos_ids != 0
    if not valid_mask.any():
        zero = projected.sum() * 0.0
        return zero, None, None

    proj_flat = projected[valid_mask]
    pos_ids_flat = pos_ids[valid_mask]
    neg_ids_flat = neg_ids[valid_mask]
    seq_lengths_flat = seq_lengths.unsqueeze(1).expand_as(pos_ids)[valid_mask]

    pos_embs_flat = item_emb(pos_ids_flat).to(proj_flat.dtype)
    pos_logits_flat = _apply_similarity_logits(
        proj_flat,
        pos_embs_flat,
        config,
        model=model,
        item_ids=pos_ids_flat,
        seq_lengths=seq_lengths_flat,
    )

    pos_stats = (
        pos_logits_flat.mean().item(),
        pos_logits_flat.min().item(),
        pos_logits_flat.max().item(),
    )

    total_loss = proj_flat.sum() * 0.0
    total_count = 0
    neg_sum = 0.0
    neg_count = 0
    neg_min = None
    neg_max = None

    for start in range(0, proj_flat.size(0), chunk_size):
        end = min(start + chunk_size, proj_flat.size(0))
        proj_c = proj_flat[start:end]
        pos_logits_c = pos_logits_flat[start:end]
        neg_ids_c = neg_ids_flat[start:end]
        seq_lengths_c = seq_lengths_flat[start:end]
        neg_embs_c = item_emb(neg_ids_c).to(proj_c.dtype)
        neg_logits_c = _apply_similarity_logits(
            proj_c,
            neg_embs_c,
            config,
            model=model,
            item_ids=neg_ids_c,
            seq_lengths=seq_lengths_c,
        )

        neg_sum += float(neg_logits_c.sum().item())
        neg_count += int(neg_logits_c.numel())
        cur_min = float(neg_logits_c.min().item())
        cur_max = float(neg_logits_c.max().item())
        neg_min = cur_min if neg_min is None else min(neg_min, cur_min)
        neg_max = cur_max if neg_max is None else max(neg_max, cur_max)

        logits_c = torch.cat([pos_logits_c.unsqueeze(1), neg_logits_c], dim=1)
        labels_c = torch.zeros(logits_c.size(0), dtype=torch.long, device=logits_c.device)
        total_loss = total_loss + F.cross_entropy(logits_c, labels_c, reduction="sum")
        total_count += logits_c.size(0)

    loss = total_loss / max(total_count, 1)
    neg_stats = None
    if neg_count > 0 and neg_min is not None and neg_max is not None:
        neg_stats = (neg_sum / neg_count, neg_min, neg_max)
    return loss, pos_stats, neg_stats


# === Evaluation (sampled ranking) ===
def evaluate(
    model: nn.Module,
    dataset,
    config: SASRecConfig,
    mode: str = "test",
    batch_size: int = 256,
    device: str = "cpu",
    streaming_last_k: int = 0,
) -> Dict[str, Any]:
    """Evaluate on protocol-aware LOO split using sampled negatives."""
    model.eval()
    device_obj = torch.device(device)
    rng = _build_eval_rng(config, mode=mode, streaming_last_k=streaming_last_k)
    use_amp = bool(getattr(config, "use_amp", False)) and device_obj.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(getattr(config, "amp_dtype", "bf16")), device_obj)

    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_examples = 0
    per_position: Dict[int, Dict[str, float]] = {}

    users = dataset.users
    min_item_id = getattr(dataset, "min_item_id", 1)

    for batch_start in range(0, len(users), batch_size):
        batch_users = users[batch_start : batch_start + batch_size]
        batch_examples = []

        for user in batch_users:
            seq = dataset.user_seq[user]
            target_positions = resolve_eval_target_positions(
                len(seq),
                mode=mode,
                streaming_last_k=streaming_last_k,
                eval_protocol=getattr(config, "eval_protocol", LEGACY_LOO_PROTOCOL),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            for target_idx in target_positions:
                batch_examples.append((user, seq[:target_idx], seq[target_idx], len(seq) - target_idx))

        if not batch_examples:
            continue

        batch_seqs = [input_seq for _, input_seq, _, _ in batch_examples]
        max_len = min(max(len(s) for s in batch_seqs), dataset.max_seq_length)
        input_tensor = torch.zeros((len(batch_examples), max_len), dtype=torch.long)

        for i, seq in enumerate(batch_seqs):
            seq_len = min(len(seq), max_len)
            input_tensor[i, -seq_len:] = torch.tensor(seq[-seq_len:])

        input_tensor = input_tensor.to(device_obj)
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
                seen_items = {x for x in batch_seqs[idx] if x >= min_item_id}
                if use_fixed_neg:
                    fixed_neg = dataset.neg_item_by_user.get(user)
                    if fixed_neg and fixed_neg not in seen_items and fixed_neg != target and fixed_neg >= min_item_id:
                        candidates.append(fixed_neg)
                while len(candidates) < sample_size:
                    neg_item = rng.randint(min_item_id, dataset.max_item + 1)
                    if neg_item not in seen_items and neg_item not in candidates:
                        candidates.append(neg_item)
                candidates_list.append(torch.tensor(candidates, device=device_obj))
            candidates_tensor = torch.stack(candidates_list, dim=0)

            with _amp_context(use_amp, amp_dtype, device_obj):
                scores = _predict_with_similarity(model, input_tensor, candidates_tensor, config=config, use_patch=True)

        _, indices = torch.sort(scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1].cpu().numpy() + 1  # 1-indexed ranks

        for rel_from_end, rank in zip(valid_rel_positions, ranks):
            valid_examples += 1
            update_rank_metrics(per_position, rel_from_end, int(rank))
            if rank <= 10:
                hr_sum += 1
                ndcg_sum += 1 / np.log2(rank + 1)

    eval_entity = "examples" if int(streaming_last_k or 0) > 1 else "users"
    logger.info("Evaluated on %s %s", f"{valid_examples:,}", eval_entity)

    return finalize_eval_metrics(
        ndcg_sum=ndcg_sum,
        hr_sum=hr_sum,
        num_examples=valid_examples,
        per_position=per_position,
        streaming_last_k=streaming_last_k,
    )


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
    if normalize_eval_protocol(getattr(config, "eval_protocol", LEGACY_LOO_PROTOCOL)) != LEGACY_LOO_PROTOCOL:
        tag += f"_ANCHORk{int(getattr(config, 'last_k_eval_test', 0) or 0)}"
    if int(getattr(config, "streaming_eval_last_k", 0) or 0) > 1:
        tag += f"_STREAM{int(getattr(config, 'streaming_eval_last_k', 0) or 0)}"
    return tag


def get_model_checkpoint_path(config: SASRecConfig) -> Path:
    return config.checkpoint_dir / f"{build_checkpoint_tag(config)}_best.pt"


def save_model_checkpoint(model: nn.Module, config: SASRecConfig) -> Path:
    """Save model state for the current best validation (overwrite)."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = get_model_checkpoint_path(config)
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


def load_model_checkpoint(model: nn.Module, config: SASRecConfig, device: torch.device | str) -> Path:
    out_path = get_model_checkpoint_path(config)
    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded best model checkpoint from {out_path}")
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
    device_obj = torch.device(device)
    model = model.to(device_obj)

    use_amp = bool(getattr(config, "use_amp", False)) and device_obj.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(getattr(config, "amp_dtype", "bf16")), device_obj)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Apply torch.compile for faster training (CUDA only, not MPS)
    if bool(getattr(config, "use_torch_compile", True)) and device_obj.type == "cuda":
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    logger.info(
        "Sampling strategy: local uniform negatives (K=%s) with DataLoader + sampled softmax",
        config.num_negatives,
    )
    train_data = LooTrainDataset(train_dataset, config)
    collate_fn = build_train_collate_fn(train_data, config)
    num_workers = max(0, int(config.num_workers))
    loader_kwargs = {
        "batch_size": config.batch_size,
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

    steps_per_epoch = len(train_loader)
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

    # Track best model
    best_val_metrics = {"ndcg@10": -1.0, "hr@10": -1.0}
    best_selection_value = float("-inf")
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

        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            optimizer.zero_grad()

            # Initialize loss values for logging
            loss_value = 0.0
            pos_logit_stats = None
            neg_logit_stats = None

            input_ids = batch["input_ids"].to(device_obj, non_blocking=True)
            pos_ids = batch["pos_ids"].to(device_obj, non_blocking=True)
            neg_ids = batch["neg_ids"].to(device_obj, non_blocking=True)
            seq_lengths = _sequence_lengths_from_input_ids(input_ids)

            with _amp_context(use_amp, amp_dtype, device_obj):
                projected = _compute_projected_hidden(model, input_ids, use_patch=True)
                item_emb = _resolve_item_embedding(model)
                loss, pos_logit_stats, neg_logit_stats = _sampled_softmax_loss_chunked(
                    model=model,
                    projected=projected,
                    pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    item_emb=item_emb,
                    chunk_size=int(config.sampled_softmax_chunk_size),
                    config=config,
                    seq_lengths=seq_lengths,
                )
            loss_value = float(loss.detach().item())

            if pos_logit_stats is not None and neg_logit_stats is not None:
                with torch.no_grad():
                    pos_logit_stats = tuple(float(x) for x in pos_logit_stats)
                    neg_logit_stats = tuple(float(x) for x in neg_logit_stats)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if use_scaler:
                scaler.unscale_(optimizer)
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Get gradient norm for logging
            grad_norm = get_gradient_norm(model)

            # Optimizer step
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
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
                    f"Loss: {loss_value:.4f} | "
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
                current_selection_value = float(val_metrics[config.selection_metric])
                if current_selection_value > best_selection_value:
                    best_val_metrics = dict(val_metrics)
                    best_selection_value = current_selection_value
                    no_improve_steps = 0
                    logger.info(
                        f"New best validation - {config.selection_metric}: {current_selection_value:.4f} "
                        f"(NDCG@10: {val_metrics['ndcg@10']:.4f}, HR@10: {val_metrics['hr@10']:.4f})"
                    )
                    save_model_checkpoint(model, config)
                    if config.save_item_embeddings:
                        save_item_embeddings(model, config)
                else:
                    no_improve_steps += 1
                    if config.early_stop_patience > 0 and no_improve_steps >= config.early_stop_patience:
                        logger.info(
                            f"Early stopping: no improvement in {config.selection_metric} for "
                            f"{config.early_stop_patience} validation checks."
                        )
                        stop_training = True
                        break

        if stop_training:
            break

    pbar.close()  # Close the progress bar
    if get_model_checkpoint_path(config).exists():
        load_model_checkpoint(model, config, device)
    logger.info(
        f"Training completed. Best validation {config.selection_metric}: {best_selection_value:.4f} "
        f"(NDCG@10: {best_val_metrics['ndcg@10']:.4f}, HR@10: {best_val_metrics['hr@10']:.4f})"
    )

    return best_val_metrics


if __name__ == "__main__":
    # Adjust hyperparameters by editing the SASRecConfig defaults above or overriding them here.
    config = SASRecConfig()
    config.backbone = config.backbone.lower()
    config.apply_backbone_overrides()
    resolve_dataset_config(config)
    resolve_eval_protocol_config(config)
    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_blocks = config.lru_num_blocks if config.backbone == "lru" else config.num_blocks
    es_suffix = "noes" if config.early_stop_patience <= 0 else f"es{config.early_stop_patience}"
    run_name = (
        f"{config.backbone}-{config.dataset}-{run_blocks}b-{config.num_heads}h-"
        f"{config.hidden_units}-{config.max_seq_length}_{es_suffix}-sample_softmax"
    )
    if int(getattr(config, "shared_prefix_len", 0) or 0) > 0:
        run_name += f"-sp{int(config.shared_prefix_len)}"
    if config.backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        run_name += "-sm1"
    run_name += build_protocol_run_suffix(config)
    run = wandb.init(project=f"backbone-standard-{config.dataset}", name=run_name, config=config.__dict__)
    base_ckpt_dir = config.checkpoint_dir / f"{config.backbone}_loo_sample_softmax"
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
    if int(config.streaming_eval_last_k or 0) > 1:
        stream_last_k = int(config.streaming_eval_last_k)
        logger.info("Running additional streaming test evaluation over the last %s targets...", stream_last_k)
        stream_metrics = evaluate(
            model,
            test_dataset,
            config=config,
            mode="test",
            device=device,
            streaming_last_k=stream_last_k,
        )
        wandb.log(
            {
                **flatten_streaming_eval_metrics("test_stream/backbone", stream_metrics),
                **flatten_streaming_eval_test_aliases("backbone", stream_metrics),
            }
        )
    wandb.log(
        {
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )

    wandb.finish()
    logger.info("Training complete!")
