#!/usr/bin/env python3
"""Train SASRec on the xlong dataset (pair-format only)."""

import os
import sys
import random
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable

import numpy as np
try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    KMeans = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.device_manager import DeviceManager
from core.mixflow import MomentumInner, get_fwdrev_grad_fn_eta
from core.logger import setup_logger
logger = setup_logger("train-sasrec-meta-patch", log_to_file=True)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class SASRecConfig:
    """Configuration for SASRec training (xlong only)."""

    dataset: str = "xlong2018"
    xlong_train_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/train_corpus_total_dual.txt"
    )
    xlong_meta_valid_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_sub2_1.txt"
    )
    xlong_test_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_sub2_2.txt"
    )
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "sasrec")

    # Model parameters
    max_seq_length: int = 100  # Maximum sequence length
    hidden_units: int = 128  # Hidden dimension size
    num_blocks: int = 2  # Number of transformer blocks
    num_heads: int = 1  # Number of attention heads
    dropout_rate: float = 0.2  # Dropout rate

    # Meta-patch parameters
    num_patches: int = 4  # Number of patches in the bank
    patch_len: int = 10  # Length of each patch (soft prompt tokens)
    use_gating: bool = True  # Use gating network to mix patches
    gating_hidden_dim: int = 64  # Hidden size for gating MLP
    patch_init_std: float = 0.0  # Init std for patch/gating params
    head_residual: bool = True  # Use residual head: output = x + head(x)
    head_zero_init: bool = True  # Zero-init head for identity start when residual
    enable_projection_head: bool = True  # If False, skip projection head entirely
    head_use_gelu: bool = False  # Keep head lightweight by default
    head_use_ln: bool = False  # Keep head lightweight by default
    gating_pool: str = "mean"  # mean | last
    gating_init_std: float = 0.2  # Init std for gating params
    gating_temperature: float = 0.5  # Softmax temperature (<1 sharpens)
    gating_noise_std: float = 0.01  # Logit noise for symmetry breaking
    patch_routing: str = "learned"  # learned | kmeans | random | single
    kmeans_max_iters: int = 25
    kmeans_seed: int = 2026
    kmeans_max_samples: int = 20000

    # Training parameters
    batch_size: int = 1024  # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    seed: int = 2026  # Global RNG seed
    deterministic: bool = False  # Enable deterministic ops (slower)

    # Meta-learning (MixFlow) parameters
    inner_steps: int = 1  # Inner updates per outer update block
    inner_lr: float = 5e-5  # Inner (head) learning rate
    inner_momentum: float = 0.0  # Inner momentum (0 disables)
    inner_grad_clip: float = 0.0  # Clip inner gradients (0 disables)
    outer_update_every: int = 20  # Perform outer update every N inner steps
    outer_lr: float = 1e-4  # Outer (patch) learning rate
    outer_weight_decay: float = 0.0  # Outer weight decay
    outer_grad_clip: float = 1.0  # Clip patch gradients (0 disables)
    meta_truncate_steps: int = 1  # Truncated unroll steps for meta-gradients
    lambda_meta: float = 1.0  # Scaling for meta-gradients
    val_batch_size: int = 1024  # Batch size for outer loop (clean val)
    val_eval_every_epochs: int = 4  # Run meta-patch eval on val every N epochs
    outer_loss_mode: str = "decay"  # all | last | decay
    outer_loss_decay: float = 0.9  # decay factor for outer loss when mode=decay
    inner_loss_mode: str = "all"  # match_outer | all | last | decay

    # Data noise (train only)
    noise_ratio: float = 0.3  # Probability of corrupting training labels

    # Training settings
    steps_per_train_log: int = 100  # Log training progress every N steps
    # ⚠️之前eval用的是100导致结果看起来很高
    eval_sample_size: int = 1000  # Total candidates per user when eval_mode="sampled" (includes target)
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing (memory fallback)
    use_flash_attention: bool = False  # Disable flash attention (use math kernel)

    # Checkpoint loading
    strict_load_pretrained: bool = False  # If True, load full model with strict=True
    ckpt_prefix_to_strip: Optional[str] = None  # Optional prefix to strip from checkpoint keys

    # Output settings
    save_item_embeddings: bool = False  # Save item embeddings after training

    # Device settings
    device: str = "cuda:0"  # e.g., "cuda:1", "cpu", "mps"

    # Pretrained checkpoint
    pretrained_ckpt_path: str = "/home/lingfengs111/codes/soft_patch_training/checkpoints/sasrec_new/sasrec_xlong2018_seq100_dim128_L2_H1_best.pt"

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== SASRec Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  xlong_train_path: {self.xlong_train_path}")
        logger.info(f"  xlong_meta_valid_path: {self.xlong_meta_valid_path}")
        logger.info(f"  xlong_test_path: {self.xlong_test_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  hidden_units: {self.hidden_units}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")
        logger.info("Patch Parameters:")
        logger.info(f"  num_patches: {self.num_patches}")
        logger.info(f"  patch_len: {self.patch_len}")
        logger.info(f"  use_gating: {self.use_gating}")
        logger.info(f"  gating_hidden_dim: {self.gating_hidden_dim}")
        logger.info(f"  patch_init_std: {self.patch_init_std}")
        logger.info(f"  head_residual: {self.head_residual}")
        logger.info(f"  head_zero_init: {self.head_zero_init}")
        logger.info(f"  enable_projection_head: {self.enable_projection_head}")
        logger.info(f"  head_use_gelu: {self.head_use_gelu}")
        logger.info(f"  head_use_ln: {self.head_use_ln}")
        logger.info(f"  gating_pool: {self.gating_pool}")
        logger.info(f"  gating_init_std: {self.gating_init_std}")
        logger.info(f"  gating_temperature: {self.gating_temperature}")
        logger.info(f"  gating_noise_std: {self.gating_noise_std}")
        logger.info(f"  patch_routing: {self.patch_routing}")
        logger.info(f"  kmeans_max_iters: {self.kmeans_max_iters}")
        logger.info(f"  kmeans_seed: {self.kmeans_seed}")
        logger.info(f"  kmeans_max_samples: {self.kmeans_max_samples}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  seed: {self.seed}")
        logger.info(f"  deterministic: {self.deterministic}")
        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info("Meta-learning Settings:")
        logger.info(
            "  inner_steps: %s | inner_lr: %s | inner_momentum: %s | inner_grad_clip: %s",
            self.inner_steps,
            self.inner_lr,
            self.inner_momentum,
            self.inner_grad_clip,
        )
        logger.info(
            "  outer_update_every: %s | outer_lr: %s | outer_wd: %s | outer_grad_clip: %s",
            self.outer_update_every,
            self.outer_lr,
            self.outer_weight_decay,
            self.outer_grad_clip,
        )
        logger.info(f"  meta_truncate_steps: {self.meta_truncate_steps} | lambda_meta: {self.lambda_meta}")
        logger.info(f"  val_batch_size: {self.val_batch_size}")
        logger.info(f"  val_eval_every_epochs: {self.val_eval_every_epochs}")
        logger.info(f"  outer_loss_mode: {self.outer_loss_mode}")
        logger.info(f"  outer_loss_decay: {self.outer_loss_decay}")
        logger.info(f"  inner_loss_mode: {self.inner_loss_mode}")
        logger.info(f"  noise_ratio: {self.noise_ratio}")
        logger.info(f"  use_gradient_checkpointing: {self.use_gradient_checkpointing}")
        logger.info(f"  use_flash_attention: {self.use_flash_attention}")
        logger.info(f"  strict_load_pretrained: {self.strict_load_pretrained}")
        logger.info(f"  ckpt_prefix_to_strip: {self.ckpt_prefix_to_strip}")

        logger.info("Output Settings:")
        logger.info(f"  save_item_embeddings: {self.save_item_embeddings}")

        logger.info("Device Settings:")
        logger.info(f"  device: {self.device}")
        logger.info(f"  pretrained_ckpt_path: {self.pretrained_ckpt_path}")
        logger.info("===========================")


def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict or a state_dict-like object.")
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


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


def infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor], config: SASRecConfig) -> SASRecConfig:
    if "item_emb.weight" in state_dict:
        config.hidden_units = int(state_dict["item_emb.weight"].shape[1])
    if "pos_emb.weight" in state_dict:
        config.max_seq_length = int(state_dict["pos_emb.weight"].shape[0]) - 1
    block_indices = []
    for key in state_dict.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    if block_indices:
        config.num_blocks = max(block_indices) + 1
    if config.hidden_units % config.num_heads != 0:
        logger.warning(
            "hidden_units (%s) not divisible by num_heads (%s). Forcing num_heads=1.",
            config.hidden_units,
            config.num_heads,
        )
        config.num_heads = 1
    return config


def load_pretrained_backbone(
    model: "SASRec", ckpt_path: str, state_dict: Optional[Dict[str, torch.Tensor]] = None
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
    backbone_keys = ("item_emb.", "pos_emb.", "blocks.", "ln_f.")
    filtered = {}
    pos_weight = None
    for k, v in state_dict.items():
        if not k.startswith(backbone_keys):
            continue
        if k == "item_emb.weight":
            if v.shape == model.item_emb.weight.shape:
                filtered[k] = v
            else:
                logger.warning(
                    "Skipped item_emb.weight due to shape mismatch (ckpt=%s, model=%s).",
                    v.shape,
                    model.item_emb.weight.shape,
                )
        elif k == "pos_emb.weight":
            pos_weight = v
            if v.shape == model.pos_emb.weight.shape:
                filtered[k] = v
        else:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("Missing keys when loading backbone: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading backbone: %s", unexpected)
    if pos_weight is not None and pos_weight.shape != model.pos_emb.weight.shape:
        logger.warning(
            "Skipped pos_emb.weight due to shape mismatch (ckpt=%s, model=%s).",
            pos_weight.shape,
            model.pos_emb.weight.shape,
        )


def initialize_head_as_identity(model: nn.Module) -> None:
    """Initialize projection head to preserve pretrained features at start."""
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


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float, use_flash_attention: bool = True):
        super().__init__()
        assert hidden_units % num_heads == 0

        # Combined QKV projection
        self.c_attn = nn.Linear(hidden_units, 3 * hidden_units)
        # Output projection
        self.c_proj = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.use_flash_attention = use_flash_attention
        self._flash_fallback_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, hidden units

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_units, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)

        # Causal self-attention (prefer flash when available; disable under forward AD)
        try:
            from torch.autograd import forward_ad

            fwd_ad_enabled = forward_ad.is_enabled()
        except Exception:
            fwd_ad_enabled = False

        if q.is_cuda and self.use_flash_attention and not fwd_ad_enabled:
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
                )
            except RuntimeError:
                if not self._flash_fallback_warned:
                    logger.warning("Flash attention failed; falling back to math kernel.")
                    self._flash_fallback_warned = True
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                ):
                    y = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
                    )
        elif q.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
                )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
            )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network)."""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-LN architecture."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float, use_flash_attention: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = CausalSelfAttention(hidden_units, num_heads, dropout_rate, use_flash_attention=use_flash_attention)
        self.ln_2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = MLP(hidden_units, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LayerNorm -> Sub-layer -> Residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MetaPatch(nn.Module):
    """Learnable patch bank + gating network flattened into a single eta tensor."""

    def __init__(self, config: SASRecConfig):
        super().__init__()
        self.config = config
        self.num_patches = max(1, config.num_patches)
        self.patch_len = max(0, config.patch_len)
        self.hidden_units = config.hidden_units
        self.use_gating = config.use_gating
        self.gating_hidden_dim = config.gating_hidden_dim

        self.patch_param_size = self.num_patches * self.patch_len * self.hidden_units
        if self.use_gating:
            self.gate_param_size = (
                self.gating_hidden_dim * self.hidden_units
                + self.gating_hidden_dim
                + self.num_patches * self.gating_hidden_dim
                + self.num_patches
            )
        else:
            self.gate_param_size = self.num_patches

        total = self.patch_param_size + self.gate_param_size
        patch_init = torch.randn(self.patch_param_size) * config.patch_init_std
        gate_init = torch.randn(self.gate_param_size) * config.gating_init_std
        self.eta = nn.Parameter(torch.cat([patch_init, gate_init], dim=0))
        self.register_buffer("kmeans_centers", torch.empty(0), persistent=False)

    def set_kmeans_centers(self, centers: torch.Tensor) -> None:
        if centers.dim() != 2 or centers.size(1) != self.hidden_units:
            raise ValueError("kmeans centers must have shape [num_patches, hidden_units].")
        if centers.size(0) != self.num_patches:
            if centers.size(0) < self.num_patches:
                repeats = self.num_patches - centers.size(0)
                extra = centers[torch.arange(repeats) % centers.size(0)].clone()
                centers = torch.cat([centers, extra], dim=0)
            else:
                centers = centers[: self.num_patches]
        self.kmeans_centers = centers.to(device=self.eta.device, dtype=self.eta.dtype)

    def _split_eta(self, eta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_flat = eta[: self.patch_param_size]
        gate_flat = eta[self.patch_param_size :]
        return patch_flat, gate_flat

    def forward(self, seq_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_with_eta(seq_emb, self.eta)

    def forward_with_eta(self, seq_emb: torch.Tensor, eta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.patch_len == 0:
            empty_patch = seq_emb.new_zeros((seq_emb.size(0), 0, self.hidden_units))
            empty_weights = seq_emb.new_zeros((seq_emb.size(0), self.num_patches))
            return empty_patch, empty_weights

        patch_flat, gate_flat = self._split_eta(eta)
        patch_bank = patch_flat.view(self.num_patches, self.patch_len, self.hidden_units)

        routing = getattr(self.config, "patch_routing", "learned")
        if routing != "learned":
            if routing == "single" or self.num_patches == 1:
                weights = seq_emb.new_zeros((seq_emb.size(0), self.num_patches))
                weights[:, 0] = 1.0
            elif routing == "random":
                idx = torch.randint(0, self.num_patches, (seq_emb.size(0),), device=seq_emb.device)
                weights = F.one_hot(idx, num_classes=self.num_patches).to(seq_emb.dtype)
            elif routing == "kmeans":
                if self.kmeans_centers.numel() == 0:
                    raise RuntimeError("KMeans routing requested but centers are not set.")
                centers = self.kmeans_centers
                if centers.device != seq_emb.device:
                    centers = centers.to(seq_emb.device)
                dist = torch.cdist(seq_emb, centers)
                idx = dist.argmin(dim=1)
                weights = F.one_hot(idx, num_classes=self.num_patches).to(seq_emb.dtype)
            else:
                raise ValueError(f"Unknown patch_routing: {routing}")
            patch = torch.einsum("bp,plh->blh", weights, patch_bank)
            return patch, weights

        if self.use_gating:
            idx = 0
            w1 = gate_flat[idx : idx + self.gating_hidden_dim * self.hidden_units].view(
                self.gating_hidden_dim, self.hidden_units
            )
            idx += self.gating_hidden_dim * self.hidden_units
            b1 = gate_flat[idx : idx + self.gating_hidden_dim]
            idx += self.gating_hidden_dim
            w2 = gate_flat[idx : idx + self.num_patches * self.gating_hidden_dim].view(
                self.num_patches, self.gating_hidden_dim
            )
            idx += self.num_patches * self.gating_hidden_dim
            b2 = gate_flat[idx : idx + self.num_patches]

            hidden = F.relu(F.linear(seq_emb, w1, b1))
            logits = F.linear(hidden, w2, b2)
        else:
            logits = gate_flat.view(1, -1).expand(seq_emb.size(0), -1)

        temp = float(getattr(self.config, "gating_temperature", 1.0))
        if temp <= 0:
            temp = 1e-6
        logits = logits / temp
        noise_std = float(getattr(self.config, "gating_noise_std", 0.0))
        if noise_std > 0 and self.training:
            logits = logits + noise_std * torch.randn_like(logits)
        weights = torch.softmax(logits, dim=-1)
        patch = torch.einsum("bp,plh->blh", weights, patch_bank)
        return patch, weights

    def zero_eta(self) -> torch.Tensor:
        return torch.zeros_like(self.eta)


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model."""

    def __init__(self, config: SASRecConfig, item_num: int):
        super().__init__()

        self.config = config  # Store config for later use
        self.item_num = item_num
        self.max_seq_length = config.max_seq_length
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len

        # Embedding layers
        self.item_emb = nn.Embedding(item_num + 1, config.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_seq_length + 1, config.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_units,
                    config.num_heads,
                    config.dropout_rate,
                    use_flash_attention=config.use_flash_attention,
                )
                for _ in range(config.num_blocks)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_units, eps=1e-8)

        # Trainable projection head (only head is adapted in inner loop)
        self.proj_linear = nn.Linear(config.hidden_units, config.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(config.hidden_units, eps=1e-8)

        # Meta-patch module (outer loop)
        self.meta_patch = MetaPatch(config)

        # Initialize weights
        self.apply(self._init_weights)
        self._freeze_backbone()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Don't initialize padding_idx
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def _freeze_backbone(self) -> None:
        """Freeze the transformer body (backbone)."""
        for p in self.item_emb.parameters():
            p.requires_grad = False
        for p in self.pos_emb.parameters():
            p.requires_grad = False
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.ln_f.parameters():
            p.requires_grad = False

    def _sequence_summary(self, item_embs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return last-item embedding and sequence lengths."""
        valid_mask = input_ids != 0
        lengths = valid_mask.sum(dim=1)
        pool = getattr(self.config, "gating_pool", "last")
        if pool == "mean":
            denom = lengths.clamp_min(1).unsqueeze(1).to(item_embs.dtype)
            mask = valid_mask.unsqueeze(-1).to(item_embs.dtype)
            mean_emb = (item_embs * mask).sum(dim=1) / denom
            return mean_emb, lengths
        seq_len = input_ids.size(1)
        if seq_len == 0:
            last_idx = torch.zeros_like(lengths)
        else:
            last_from_end = valid_mask.flip(1).float().argmax(dim=1)
            last_idx = (seq_len - 1) - last_from_end
        batch_idx = torch.arange(item_embs.size(0), device=item_embs.device)
        last_emb = item_embs[batch_idx, last_idx]
        return last_emb, lengths

    def head_parameters(self) -> List[torch.Tensor]:
        params = list(self.proj_linear.parameters())
        if getattr(self.config, "head_use_ln", True):
            params += list(self.proj_ln.parameters())
        return params

    def apply_head(self, hidden_states: torch.Tensor, head_params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if not getattr(self.config, "enable_projection_head", True):
            return hidden_states
        use_gelu = getattr(self.config, "head_use_gelu", True)
        use_ln = getattr(self.config, "head_use_ln", True)
        if head_params is None:
            delta = self.proj_linear(hidden_states)
            if use_gelu:
                delta = F.gelu(delta)
            if use_ln:
                delta = self.proj_ln(delta)
        else:
            needed = 2 + (2 if use_ln else 0)
            if len(head_params) < needed:
                raise ValueError("head_params length does not match head configuration")
            w, b = head_params[0], head_params[1]
            delta = F.linear(hidden_states, w, b)
            if use_gelu:
                delta = F.gelu(delta)
            if use_ln:
                ln_w, ln_b = head_params[2], head_params[3]
                delta = F.layer_norm(delta, (self.hidden_units,), ln_w, ln_b, eps=1e-8)
        if getattr(self.config, "head_residual", False):
            return hidden_states + delta
        return delta

    def forward_features(
        self,
        input_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            input_ids: Item sequences [batch_size, seq_length]

        Returns:
            Hidden states [batch_size, seq_length, hidden_units]
        """
        batch_size, seq_length = input_ids.size()

        # Item embeddings
        item_embs = self.item_emb(input_ids)
        item_embs *= self.hidden_units**0.5  # Scale by sqrt(d) as in Transformer

        # Patch embeddings (meta-patch)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
        else:
            patch_emb = item_embs.new_zeros((batch_size, 0, self.hidden_units))
            gating_weights = None

        # Positional embeddings
        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        positions = positions * (input_ids != 0).long()

        if use_patch and self.patch_len > 0:
            pos_embs_item = self.pos_emb(positions)
            hidden_states = torch.cat([patch_emb, item_embs + pos_embs_item], dim=1)
        else:
            pos_embs = self.pos_emb(positions)
            hidden_states = item_embs + pos_embs

        hidden_states = self.emb_dropout(hidden_states)

        # Pass through transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = gradient_checkpoint(block, hidden_states)
            else:
                hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        if return_gating:
            return hidden_states, gating_weights
        return hidden_states

    def get_gating_weights(
        self, input_ids: torch.Tensor, patch_params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        item_embs = self.item_emb(input_ids)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if patch_params is None:
            _, weights = self.meta_patch(seq_summary)
        else:
            _, weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
        return weights

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_features(input_ids)

    def predict(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        use_patch: bool = True,
        use_head: bool = True,
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            candidate_ids: Candidate items to score [batch_size, num_candidates]

        Returns:
            Scores for each candidate [batch_size, num_candidates]
        """
        hidden_states = self.forward_features(input_ids, patch_params=patch_params, use_patch=use_patch)
        if use_patch and self.patch_len > 0:
            hidden_states = hidden_states[:, self.patch_len :, :]
        final_hidden = hidden_states[:, -1, :]  # [B, H]
        if use_head:
            final_hidden = self.apply_head(final_hidden, head_params=head_params)
        candidate_embs = self.item_emb(candidate_ids)  # [B, C, H]
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)  # [B, C]
        return scores

    def training_step(
        self,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Training step with positive and negative items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            pos_ids: Positive items (next items) [batch_size, seq_length]
            neg_ids: Negative items (sampled) [batch_size, seq_length]

        Returns:
            pos_logits: Scores for positive items [batch_size, seq_length]
            neg_logits: Scores for negative items [batch_size, seq_length]
        """
        hidden_states, gating_weights = self.forward_features(
            input_ids, patch_params=patch_params, return_gating=True, use_patch=use_patch
        )
        if use_patch and self.patch_len > 0:
            hidden_states = hidden_states[:, self.patch_len :, :]

        projected = self.apply_head(hidden_states, head_params=head_params)
        pos_embs = self.item_emb(pos_ids)  # [B, T, H]
        neg_embs = self.item_emb(neg_ids)  # [B, T, H]

        pos_logits = (projected * pos_embs).sum(dim=-1)  # [B, T]
        neg_logits = (projected * neg_embs).sum(dim=-1)  # [B, T]

        if return_gating:
            return pos_logits, neg_logits, gating_weights
        return pos_logits, neg_logits


def sasrec_training_step_stateless(
    model: SASRec,
    theta_list: List[torch.Tensor],
    eta: torch.Tensor,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    return_gating: bool = False,
    use_patch: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    head_params = theta_list if getattr(model.config, "enable_projection_head", True) else None
    pos_logits, neg_logits, gating_weights = model.training_step(
        input_ids,
        pos_ids,
        neg_ids,
        patch_params=eta,
        head_params=head_params,
        return_gating=True,
        use_patch=use_patch,
    )
    if return_gating:
        return pos_logits, neg_logits, gating_weights
    return pos_logits, neg_logits, None


def _mean_loss_from_logits(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_ids: torch.Tensor,
    bce_criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = pos_ids != 0
    pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
    neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
    raw_loss = pos_loss + neg_loss
    if not valid_mask.any():
        zero = pos_logits.sum() * 0.0
        return zero, valid_mask
    return raw_loss[valid_mask].mean(), valid_mask


def _last_position_loss(
    raw_loss: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if not valid_mask.any():
        return raw_loss.sum() * 0.0
    lengths = valid_mask.sum(dim=1)
    valid_rows = lengths > 0
    if not valid_rows.any():
        return raw_loss.sum() * 0.0
    last_idx = (lengths - 1).clamp_min(0)
    row_indices = torch.arange(raw_loss.size(0), device=raw_loss.device)
    last_loss = raw_loss[row_indices, last_idx]
    return last_loss[valid_rows].mean()


def _decay_position_loss(
    raw_loss: torch.Tensor, valid_mask: torch.Tensor, decay: float
) -> torch.Tensor:
    if not valid_mask.any():
        return raw_loss.sum() * 0.0
    lengths = valid_mask.sum(dim=1)
    max_len = raw_loss.size(1)
    pos_idx = torch.arange(max_len, device=raw_loss.device).view(1, -1)
    dist = (lengths.unsqueeze(1) - 1) - pos_idx
    dist = dist.clamp(min=0)
    gamma = float(decay)
    if gamma <= 0:
        gamma = 1e-6
    if gamma > 1:
        gamma = 1.0
    weights = (gamma ** dist) * valid_mask.float()
    denom = weights.sum()
    if denom <= 0:
        return raw_loss.sum() * 0.0
    return (raw_loss * weights).sum() / denom


def _reduce_loss(
    raw_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
    decay: float,
) -> torch.Tensor:
    if mode == "all":
        if not valid_mask.any():
            return raw_loss.sum() * 0.0
        return raw_loss[valid_mask].mean()
    if mode == "last":
        return _last_position_loss(raw_loss, valid_mask)
    if mode == "decay":
        return _decay_position_loss(raw_loss, valid_mask, decay)
    raise ValueError(f"Unknown loss mode: {mode}")

def load_xlong_samples(data_path: str):
    """Load xlong rows and append pos_item to the end of each sequence."""
    data_path = Path(data_path)
    samples = []

    with data_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                raise ValueError(f"{data_path} line {line_num}: expected 7 fields, got {len(parts)}")
            idx = int(parts[0])
            user_id = int(parts[1])
            item_seq = [int(x) for x in parts[2].split(",") if x]
            pos_item = int(parts[3])
            neg_item = int(parts[4])
            full_seq = item_seq + [pos_item]
            samples.append((idx, user_id, full_seq, neg_item))

    return samples


class XLongSequenceDataset(Dataset):
    """Sequence dataset derived from xlong rows (pos_item appended)."""

    def __init__(self, samples, config: SASRecConfig, item_to_id: Optional[Dict[int, int]] = None):
        self.config = config
        self.max_seq_length = config.max_seq_length

        dropped = 0
        if item_to_id is None:
            item_set = set()
            for _, _, seq, neg_item in samples:
                item_set.update(seq)
                item_set.add(neg_item)
            item_list = sorted(item_set)
            self.item_to_id = {item_id: i + 1 for i, item_id in enumerate(item_list)}
            self.id_to_item = {i + 1: item_id for i, item_id in enumerate(item_list)}
        else:
            self.item_to_id = item_to_id
            self.id_to_item = {v: k for k, v in item_to_id.items()}

        self.user_seq = {}
        self.users = []
        self.neg_item_by_user = {}
        self.internal_to_user_id = {}
        internal_id = 0
        max_train_len = 0

        for _, user_id, seq, neg_item in samples:
            if item_to_id is not None and (
                any(x not in self.item_to_id for x in seq) or neg_item not in self.item_to_id
            ):
                dropped += 1
                continue
            mapped = [self.item_to_id[x] for x in seq]
            self.user_seq[internal_id] = mapped
            self.users.append(internal_id)
            self.neg_item_by_user[internal_id] = self.item_to_id[neg_item]
            self.internal_to_user_id[internal_id] = user_id
            train_len = max(len(mapped) - 2, 0)
            if train_len > max_train_len:
                max_train_len = train_len
            internal_id += 1

        self.num_users = len(self.users)
        self.num_items = len(self.item_to_id)
        self.max_item = self.num_items
        self.max_train_seq_len = max_train_len

        avg_seq_len = np.mean([len(self.user_seq[u]) for u in self.users]) if self.users else 0.0
        logger.info(f"Loaded {self.num_users:,} sequences, {self.num_items:,} items")
        logger.info(f"Average sequence length: {avg_seq_len:.2f}")
        if dropped:
            logger.info(f"Dropped {dropped:,} samples with unseen items")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_seq[user]
        return user, seq


def _run_kmeans(data: torch.Tensor, num_clusters: int, max_iters: int, seed: int) -> torch.Tensor:
    if data.numel() == 0 or num_clusters <= 0:
        raise ValueError("KMeans requires non-empty data and num_clusters > 0.")
    if KMeans is not None:
        arr = data.cpu().numpy()
        try:
            km = KMeans(n_clusters=num_clusters, n_init="auto", max_iter=max_iters, random_state=seed)
        except TypeError:
            km = KMeans(n_clusters=num_clusters, n_init=10, max_iter=max_iters, random_state=seed)
        km.fit(arr)
        centers = torch.from_numpy(km.cluster_centers_).to(data.dtype)
        return centers

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.size(0), generator=g)
    centers = data[perm[:num_clusters]].clone()
    assignments = torch.zeros(data.size(0), dtype=torch.long)
    for _ in range(max_iters):
        dist = torch.cdist(data, centers)
        new_assign = dist.argmin(dim=1)
        if torch.equal(new_assign, assignments):
            break
        assignments = new_assign
        for ci in range(num_clusters):
            mask = assignments == ci
            if mask.any():
                centers[ci] = data[mask].mean(dim=0)
    return centers


def build_kmeans_centers(
    dataset: "XLongSequenceDataset",
    model: "SASRec",
    config: SASRecConfig,
) -> torch.Tensor:
    users = list(dataset.users)
    total_users = len(users)
    if config.kmeans_max_samples > 0 and total_users > config.kmeans_max_samples:
        rng = np.random.RandomState(config.kmeans_seed)
        idx = rng.choice(total_users, size=config.kmeans_max_samples, replace=False)
        users = [users[i] for i in idx]
    logger.info(
        "Building kmeans routing with %s/%s sequences (num_patches=%s)...",
        len(users),
        total_users,
        config.num_patches,
    )

    weight = model.item_emb.weight.detach()
    emb_list = []
    with torch.no_grad():
        for user in users:
            seq = dataset.user_seq.get(user, [])
            if not seq:
                continue
            if len(seq) > config.max_seq_length:
                seq = seq[-config.max_seq_length :]
            ids = torch.tensor(seq, dtype=torch.long, device=weight.device)
            emb = weight.index_select(0, ids)
            if emb.numel() == 0:
                continue
            seq_emb = emb.mean(dim=0).float().cpu()
            emb_list.append(seq_emb)

    if not emb_list:
        raise RuntimeError("No sequence embeddings available for kmeans routing.")

    data = torch.stack(emb_list, dim=0)
    k = min(config.num_patches, data.size(0))
    centers = _run_kmeans(data, k, config.kmeans_max_iters, config.kmeans_seed)
    if centers.size(0) < config.num_patches:
        repeats = config.num_patches - centers.size(0)
        extra = centers[torch.arange(repeats) % centers.size(0)].clone()
        centers = torch.cat([centers, extra], dim=0)
        logger.warning("KMeans centers < num_patches; padding centers to %s.", config.num_patches)
    return centers


class SequentialSampler:
    """
    Sampler for sequential data that generates training batches.
    Each batch contains user sequences with positive and negative samples.
    """

    def __init__(self, dataset: XLongSequenceDataset, config: SASRecConfig, inject_noise: bool = False):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length
        self.max_item = dataset.max_item
        self.sample_id_stride = max(1, dataset.max_train_seq_len + 1)
        self.inject_noise = inject_noise
        self.noise_ratio = config.noise_ratio if inject_noise else 0.0

        # Pre-compute which sequences are valid for training
        self.valid_user_seqs = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            if len(seq) > 2:  # Need at least 3 items
                # Store (user, seq[:-2]) for training (exclude last 2 for val/test)
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
            sample_id_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            user_id_tensors = torch.zeros((actual_batch_size,), dtype=torch.long)

            for idx, (user, seq) in enumerate(batch_data):
                user_id_tensors[idx] = self.dataset.internal_to_user_id.get(user, user)
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

                # Positive items: for each position i, predict item at position i+1
                # For the last item, we predict the next item (from the held-out part)
                for pos in range(seq_len):
                    if pos < seq_len - 1:
                        pos_tensors[idx, -seq_len + pos] = seq[pos + 1]
                    else:
                        full_seq = self.dataset.user_seq[user]
                        next_idx = len(seq)
                        if next_idx < len(full_seq):
                            pos_tensors[idx, -1] = full_seq[next_idx]
                    # Sample_ID for (user, position) in the full training sequence
                    global_pos = start_idx + pos
                    sample_id_tensors[idx, -seq_len + pos] = user * self.sample_id_stride + global_pos

                # Sample negative items for each position
                seen_set = set(self.dataset.user_seq[user])  # Use full sequence for negative sampling
                for pos in range(seq_len):
                    neg_item = self.sample_negative_item(1, self.max_item + 1, seen_set)
                    neg_tensors[idx, -seq_len + pos] = neg_item

            if self.noise_ratio > 0:
                noise_mask = (pos_tensors != 0) & (
                    torch.rand_like(pos_tensors, dtype=torch.float32) < self.noise_ratio
                )
                if noise_mask.any():
                    random_items = torch.randint(
                        1, self.max_item + 1, size=pos_tensors.shape, dtype=pos_tensors.dtype
                    )
                    same_mask = random_items == pos_tensors
                    if same_mask.any():
                        random_items[same_mask] = (random_items[same_mask] % self.max_item) + 1
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
    config: SASRecConfig,
    mode: str = "test",
    batch_size: int = 256,
    device: str = "cpu",
    use_patch: bool = True,
    use_head: bool = True,
) -> Dict[str, float]:
    """Evaluate on xlong split using sampled negatives."""
    model.eval()

    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_users = 0

    users = dataset.users

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
                    if fixed_neg and fixed_neg not in seen_items and fixed_neg != target:
                        candidates.append(fixed_neg)
                while len(candidates) < sample_size:
                    neg_item = np.random.randint(1, dataset.max_item + 1)
                    if neg_item not in seen_items and neg_item not in candidates:
                        candidates.append(neg_item)
                candidates_list.append(torch.tensor(candidates, device=device))
            candidates_tensor = torch.stack(candidates_list, dim=0)

            scores = model.predict(valid_input, candidates_tensor, use_patch=use_patch, use_head=use_head)

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


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_sasrec_meta(
    model: SASRec,
    train_dataset,
    config: SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Dict[str, float]:
    """Bi-level optimization with meta-patches (inner: head, outer: patch bank)."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.train()

    theta: List[torch.Tensor] = []
    inner_opt: Optional[MomentumInner] = None
    if config.enable_projection_head:
        theta = [p.detach().clone().requires_grad_(True) for p in model.head_parameters()]
        inner_opt = MomentumInner(theta, lr=config.inner_lr, momentum=config.inner_momentum)
    eta = model.meta_patch.eta

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def _inner_loss(
        theta_list: List[torch.Tensor],
        eta_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        pos_logits, neg_logits, _ = sasrec_training_step_stateless(
            model,
            theta_list,
            eta_tensor,
            input_ids,
            pos_ids,
            neg_ids,
        )
        pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
        neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
        raw_loss = pos_loss + neg_loss
        valid_mask = pos_ids != 0
        mode = getattr(config, "inner_loss_mode", "match_outer")
        if mode == "match_outer":
            mode = getattr(config, "outer_loss_mode", "all")
        loss = _reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
        if not valid_mask.any():
            zero = eta_tensor.sum() * 0.0
            for t in theta_list:
                zero = zero + t.sum() * 0.0
            return zero
        return loss

    def _inner_loss_meta(
        theta_list: List[torch.Tensor],
        eta_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        return config.lambda_meta * _inner_loss(theta_list, eta_tensor, input_ids, pos_ids, neg_ids)

    def _outer_loss(
        theta_list: List[torch.Tensor],
        eta_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        pos_logits, neg_logits, _ = sasrec_training_step_stateless(
            model,
            theta_list,
            eta_tensor,
            input_ids,
            pos_ids,
            neg_ids,
        )
        pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
        neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
        raw_loss = pos_loss + neg_loss
        valid_mask = pos_ids != 0
        mode = getattr(config, "outer_loss_mode", "all")
        return _reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)

    grad_fn = get_fwdrev_grad_fn_eta(_inner_loss)
    grad_fn_meta = get_fwdrev_grad_fn_eta(_inner_loss_meta)

    outer_opt = torch.optim.AdamW([eta], lr=config.outer_lr, weight_decay=config.outer_weight_decay)

    train_sampler = SequentialSampler(train_dataset, config, inject_noise=True)
    val_config = replace(config, batch_size=config.val_batch_size)
    val_sampler = SequentialSampler(val_dataset, val_config) if val_dataset is not None else train_sampler
    val_iter = iter(val_sampler)
    extra_iter = iter(train_sampler)

    steps_per_epoch = len(train_sampler)
    total_steps = config.num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps)

    recent_steps = deque(maxlen=config.meta_truncate_steps) if config.meta_truncate_steps > 0 else None

    global_step = 0
    last_outer_loss = None
    best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}

    for epoch in range(config.num_epochs):
        model.train()

        for batch in train_sampler:
            global_step += 1
            batch = _move_batch_to_device(batch, device_obj)
            step_batches = [batch]
            for _ in range(config.inner_steps - 1):
                try:
                    extra_batch = next(extra_iter)
                except StopIteration:
                    extra_iter = iter(train_sampler)
                    extra_batch = next(extra_iter)
                step_batches.append(_move_batch_to_device(extra_batch, device_obj))

            for step_batch in step_batches:
                if recent_steps is not None:
                    if inner_opt is not None:
                        w_state, m_state = inner_opt.snapshot()
                    else:
                        w_state, m_state = [], []
                    recent_steps.append(
                        (
                            w_state,
                            m_state,
                            {
                                "input_ids": step_batch["input_ids"],
                                "pos_ids": step_batch["pos_ids"],
                                "neg_ids": step_batch["neg_ids"],
                            },
                        )
                    )

                if inner_opt is not None:
                    gflat = grad_fn(
                        theta,
                        eta,
                        step_batch["input_ids"],
                        step_batch["pos_ids"],
                        step_batch["neg_ids"],
                    )
                    if config.inner_grad_clip and config.inner_grad_clip > 0:
                        gflat = torch.clamp(gflat, min=-config.inner_grad_clip, max=config.inner_grad_clip)
                    inner_opt.step(gflat)

            do_outer = (
                config.outer_update_every > 0
                and global_step % config.outer_update_every == 0
                and (recent_steps is None or len(recent_steps) >= config.meta_truncate_steps)
            )
            if do_outer:
                try:
                    batch_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_sampler)
                    batch_val = next(val_iter)
                batch_val = _move_batch_to_device(batch_val, device_obj)

                latest_w, latest_m = (inner_opt.snapshot() if inner_opt is not None else ([], []))
                if inner_opt is not None and recent_steps is not None and config.meta_truncate_steps > 0:
                    start_w, start_m, _ = recent_steps[0]
                    inner_opt.restore(start_w, start_m)
                    for _, _, step_batch in recent_steps:
                        gflat = grad_fn_meta(
                            theta,
                            eta,
                            step_batch["input_ids"],
                            step_batch["pos_ids"],
                            step_batch["neg_ids"],
                        )
                        if config.inner_grad_clip and config.inner_grad_clip > 0:
                            gflat = torch.clamp(gflat, min=-config.inner_grad_clip, max=config.inner_grad_clip)
                        inner_opt.step(gflat)

                outer_opt.zero_grad(set_to_none=True)
                model.eval()
                loss_outer = _outer_loss(
                    theta,
                    eta,
                    batch_val["input_ids"],
                    batch_val["pos_ids"],
                    batch_val["neg_ids"],
                )
                model.train()
                loss_outer.backward()
                if config.outer_grad_clip and config.outer_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([eta], config.outer_grad_clip)
                outer_opt.step()
                last_outer_loss = loss_outer.item()

                if inner_opt is not None:
                    inner_opt.restore(latest_w, latest_m)

            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                log_batch = step_batches[-1]
                avg_weights_list = None
                with torch.no_grad():
                    pos_logits, neg_logits, gating = sasrec_training_step_stateless(
                        model,
                        theta,
                        eta,
                        log_batch["input_ids"],
                        log_batch["pos_ids"],
                        log_batch["neg_ids"],
                        return_gating=True,
                    )
                    pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
                    neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
                    raw_loss = pos_loss + neg_loss
                    valid_mask = log_batch["pos_ids"] != 0
                    mode = getattr(config, "inner_loss_mode", "match_outer")
                    if mode == "match_outer":
                        mode = getattr(config, "outer_loss_mode", "all")
                    inner_loss = _reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
                    if valid_mask.any():
                        wandb.log({"meta/inner_loss": inner_loss.item()})
                    if gating is not None and gating.numel() > 0:
                        weights = gating.detach().cpu()
                        avg_weights = weights.mean(dim=0)
                        avg_weights_list = avg_weights.tolist()
                        log_dict = {
                            f"gating/avg_weight_{i}": avg_weights[i].item()
                            for i in range(avg_weights.numel())
                        }
                        top1 = weights.argmax(dim=1)
                        for i in range(avg_weights.numel()):
                            log_dict[f"gating/top1_frac_{i}"] = (top1 == i).float().mean().item()

                        lengths = (log_batch["input_ids"] != 0).sum(dim=1).detach().cpu()
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

                        wandb.log(log_dict)
                        wandb.log({"gating/weight_hist": wandb.Histogram(weights.numpy())})

                if last_outer_loss is not None:
                    gating_summary = ""
                    if avg_weights_list is not None:
                        preview = ", ".join(f"{w:.3f}" for w in avg_weights_list[: min(8, len(avg_weights_list))])
                        gating_summary = f" | GatingAvg[:8]: {preview}"
                    logger.info(
                        "Step %06d | Epoch %03d/%03d | InnerLoss: %.4f | OuterLoss: %.4f%s",
                        global_step,
                        epoch + 1,
                        config.num_epochs,
                        inner_loss.item() if valid_mask.any() else 0.0,
                        last_outer_loss,
                        gating_summary,
                    )
                    wandb.log(
                        {
                            "meta/outer_loss": last_outer_loss,
                            "progress/epoch": epoch + 1,
                            "progress/step": global_step,
                        }
                    )

            pbar.update(1)

        if (
            val_dataset is not None
            and config.val_eval_every_epochs > 0
            and (epoch + 1) % config.val_eval_every_epochs == 0
        ):
            model.eval()
            val_metrics = evaluate(
                model,
                val_dataset,
                config=config,
                mode="test",
                device=str(device_obj),
                use_patch=True,
                use_head=True,
            )
            model.train()
            wandb.log(
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

    pbar.close()

    # Copy learned theta back into the model for downstream evaluation.
    with torch.no_grad():
        for p, t in zip(model.head_parameters(), theta):
            p.copy_(t.detach())

    if val_dataset is not None:
        logger.info("Running validation evaluation after meta-training...")
        metrics = evaluate(
            model,
            val_dataset,
            config=config,
            mode="test",
            device=str(device_obj),
            use_patch=True,
            use_head=True,
        )
        if metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
            best_val_metrics = metrics
    else:
        best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}

    return best_val_metrics


def save_item_embeddings(model: SASRec, dataset, config: SASRecConfig) -> Path:
    """Save item embedding matrix (excluding padding idx=0)."""
    emb = model.item_emb.weight.detach().cpu().numpy()
    emb = emb[1:]  # drop padding row
    filename = f"item_embeddings_dim{config.hidden_units}_seq{config.max_seq_length}.npy"
    out_path = config.checkpoint_dir / filename
    np.save(out_path, emb)
    logger.info(f"Saved item embeddings to {out_path}")
    logger.info("Item index mapping is in item2idx.json (original -> idx).")
    return out_path


if __name__ == "__main__":
    # Adjust hyperparameters by editing the SASRecConfig defaults above or overriding them here.
    config = SASRecConfig()
    set_global_seed(config.seed, config.deterministic)

    inferred_state = None
    if config.pretrained_ckpt_path and Path(config.pretrained_ckpt_path).exists():
        ckpt = load_checkpoint(config.pretrained_ckpt_path, trust_pickle=True)
        inferred_state = _strip_module_prefix(_extract_state_dict(ckpt))
        inferred_state = _maybe_strip_prefix(inferred_state, config.ckpt_prefix_to_strip)
        config = infer_config_from_state_dict(inferred_state, config)
    else:
        logger.warning("Pretrained checkpoint not found; proceeding without inference.")

    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_name = (
        f"sasrec-meta-patch-{config.dataset}-L{config.num_blocks}-H{config.hidden_units}"
        f"-P{config.num_patches}x{config.patch_len}"
    )
    run = wandb.init(project="sasrec-experiments", name=run_name, config=config.__dict__)
    config.log_config()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_xlong_samples(config.xlong_train_path)
    train_dataset = XLongSequenceDataset(train_samples, config)
    meta_valid_samples = load_xlong_samples(config.xlong_meta_valid_path)
    meta_valid_dataset = XLongSequenceDataset(
        meta_valid_samples, config, item_to_id=train_dataset.item_to_id
    )

    test_samples = load_xlong_samples(config.xlong_test_path)
    test_dataset = XLongSequenceDataset(
        test_samples, config, item_to_id=train_dataset.item_to_id
    )
    item_num = train_dataset.max_item

    model = SASRec(config, item_num=item_num)
    if inferred_state is not None:
        if config.strict_load_pretrained:
            logger.info("Loading full checkpoint with strict=True...")
            model.load_state_dict(inferred_state, strict=True)
        else:
            load_pretrained_backbone(model, config.pretrained_ckpt_path, state_dict=inferred_state)
    model = model.to(device)
    if config.patch_routing == "kmeans":
        centers = build_kmeans_centers(train_dataset, model, config)
        model.meta_patch.set_kmeans_centers(centers)
        logger.info("KMeans routing centers set: %s patches.", centers.size(0))
    initialize_head_as_identity(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("Running pre-train baseline on val (no patch, no head)...")
    val_baseline = evaluate(
        model,
        meta_valid_dataset,
        config=config,
        mode="test",
        device=device,
        use_patch=False,
        use_head=False,
    )
    logger.info(
        "Val Baseline - NDCG@10: %.4f, HR@10: %.4f",
        val_baseline["ndcg@10"],
        val_baseline["hr@10"],
    )
    wandb.log(
        {
            "val/baseline_ndcg@10": val_baseline["ndcg@10"],
            "val/baseline_hr@10": val_baseline["hr@10"],
            "progress/epoch": 0,
            "progress/step": 0,
        }
    )

    logger.info("Running pre-train meta-patch on val (patch + head)...")
    val_meta_patch = evaluate(
        model,
        meta_valid_dataset,
        config=config,
        mode="test",
        device=device,
        use_patch=True,
        use_head=True,
    )
    logger.info(
        "Val Meta-Patch (pre-train) - NDCG@10: %.4f, HR@10: %.4f",
        val_meta_patch["ndcg@10"],
        val_meta_patch["hr@10"],
    )
    wandb.log(
        {
            "val/pre_meta_patch_ndcg@10": val_meta_patch["ndcg@10"],
            "val/pre_meta_patch_hr@10": val_meta_patch["hr@10"],
            "progress/epoch": 0,
            "progress/step": 0,
        }
    )

    best_metrics = train_sasrec_meta(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=meta_valid_dataset,
    )

    logger.info("Running final test evaluation (baseline: no patch, no head)...")
    baseline_metrics = evaluate(
        model, test_dataset, config=config, mode="test", device=device, use_patch=False, use_head=False
    )
    logger.info(
        "Baseline Test - NDCG@10: %.4f, HR@10: %.4f",
        baseline_metrics["ndcg@10"],
        baseline_metrics["hr@10"],
    )

    logger.info("Running final test evaluation (meta-patch + projection head)...")
    meta_metrics = evaluate(
        model, test_dataset, config=config, mode="test", device=device, use_patch=True, use_head=True
    )
    logger.info(
        "Meta-Patch Test - NDCG@10: %.4f, HR@10: %.4f",
        meta_metrics["ndcg@10"],
        meta_metrics["hr@10"],
    )

    wandb.log(
        {
            "test/baseline_ndcg@10": baseline_metrics["ndcg@10"],
            "test/baseline_hr@10": baseline_metrics["hr@10"],
            "test/meta_patch_ndcg@10": meta_metrics["ndcg@10"],
            "test/meta_patch_hr@10": meta_metrics["hr@10"],
        }
    )
    wandb.log(
        {
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )

    if config.save_item_embeddings:
        save_item_embeddings(model, train_dataset, config)

    wandb.finish()
    logger.info("Training complete!")
