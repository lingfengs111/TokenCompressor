#!/usr/bin/env python3
"""Train SASRec on the xlong dataset (pair-format only)."""

import inspect
import time
from collections import deque
from dataclasses import dataclass, field, replace
from contextlib import nullcontext, contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb

from core.device_manager import DeviceManager
from core.logger import setup_logger
from core.mixflow import MomentumInner, get_fwdrev_grad_fn_eta

logger = setup_logger("train-sasrec-new", log_to_file=True)


@dataclass
class SASRecConfig:
    """Configuration for SASRec training (xlong only)."""

    dataset: str = "xlong2018"
    xlong_train_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/train_corpus_total_dual.txt"
    )
    xlong_meta_valid_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_sub1.txt"
    )
    xlong_test_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_sub2.txt"
    )
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "sasrec")

    # Model parameters
    max_seq_length: int = 50  # Maximum sequence length
    hidden_units: int = 128  # Hidden dimension size
    num_blocks: int = 2  # Number of transformer blocks
    num_heads: int = 1  # Number of attention heads
    dropout_rate: float = 0.2  # Dropout rate
    train_item_embeddings: bool = False  # Train item embeddings (memory heavy)
    train_pos_embeddings: bool = True  # Train positional embeddings
    pretrained_item_embeddings_path: Optional[str] = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/item_embeddings_sas128_len500.npy"
    )

    # Training parameters
    batch_size: int = 128 # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    max_learning_rate: float = 1e-3  # Maximum learning rate (start of cosine)
    min_learning_rate: float = 1e-5  # Minimum learning rate (end of cosine)

    # Memory optimizations
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    use_amp: bool = True  # Enable automatic mixed precision
    amp_dtype: str = "bf16"  # "bf16" or "fp16" (bf16 preferred if supported)

    # Meta-learning (DataRater) parameters
    use_meta_learning: bool = True  # Enable bi-level optimization with DataRater
    datarater_hidden_dim: int = 128  # Hidden size for DataRater MLP
    datarater_init_std: float = 0.02  # Init std for DataRater params
    inner_steps: int = 2 # Inner updates per outer update block
    inner_lr: float = 1e-4  # Inner (SASRec) learning rate
    inner_momentum: float = 0.0  # Inner momentum (0 disables)
    inner_grad_clip: float = 0.0  # Clip inner gradients (0 disables)
    outer_update_every: int = 10  # Perform outer update every N inner steps
    outer_lr: float = 1e-4  # Outer (DataRater) learning rate
    outer_weight_decay: float = 0.0  # Outer weight decay
    outer_grad_clip: float = 1.0  # Clip DataRater gradients (0 disables)
    meta_truncate_steps: int = 2  # Truncated unroll steps for meta-gradients
    lambda_meta: float = 1.0  # Scaling for meta-gradients
    log_sample_weights: bool = True  # Record DataRater weights per Sample_ID
    sample_weight_log_dir: str = "logs/datarater_weights"
    val_batch_size: int = 2  # Batch size for outer loop (clean val)

    # Training settings
    scheduler_type: str = "cosine"  # Learning rate scheduler type ("cosine" or "cosine_with_warmup")
    warmup_steps: int = 100  # Number of warmup steps (only for cosine_with_warmup)
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    steps_per_train_log: int = 100  # Log training progress every N steps
    steps_per_val_log: int = 500  # Validate and checkpoint every N steps
    # ⚠️之前eval用的是100导致结果看起来很高
    eval_sample_size: int = 1000  # Total candidates per user when eval_mode="sampled" (includes target)

    # Output settings
    save_item_embeddings: bool = True  # Save item embeddings after training

    # Device settings
    device: str = "cuda:3"  # e.g., "cuda:1", "cpu", "mps"

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
        logger.info(f"  train_item_embeddings: {self.train_item_embeddings}")
        logger.info(f"  train_pos_embeddings: {self.train_pos_embeddings}")
        logger.info(f"  pretrained_item_embeddings_path: {self.pretrained_item_embeddings_path}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")
        logger.info("Memory Optimizations:")
        logger.info(f"  use_gradient_checkpointing: {self.use_gradient_checkpointing}")
        logger.info(f"  use_amp: {self.use_amp}")
        logger.info(f"  amp_dtype: {self.amp_dtype}")
        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info("Meta-learning Settings:")
        logger.info(f"  use_meta_learning: {self.use_meta_learning}")
        logger.info(f"  datarater_hidden_dim: {self.datarater_hidden_dim}")
        logger.info(f"  datarater_init_std: {self.datarater_init_std}")
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
        logger.info(f"  log_sample_weights: {self.log_sample_weights}")
        logger.info(f"  sample_weight_log_dir: {self.sample_weight_log_dir}")
        logger.info(f"  val_batch_size: {self.val_batch_size}")

        logger.info("Output Settings:")
        logger.info(f"  save_item_embeddings: {self.save_item_embeddings}")

        logger.info("Device Settings:")
        logger.info(f"  device: {self.device}")
        logger.info("===========================")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, hidden units

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_units, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # (B, nh, T, hs)

        # Causal self-attention (force math kernel to support forward AD in meta-gradients)
        if q.is_cuda:
            with _sdp_kernel_context():
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

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = CausalSelfAttention(hidden_units, num_heads, dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = MLP(hidden_units, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LayerNorm -> Sub-layer -> Residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model."""

    def __init__(self, config: SASRecConfig, item_num: int):
        super().__init__()

        self.config = config  # Store config for later use
        self.item_num = item_num
        self.max_seq_length = config.max_seq_length
        self.hidden_units = config.hidden_units

        # Embedding layers
        self.item_emb = nn.Embedding(item_num + 1, config.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_seq_length + 1, config.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.hidden_units, config.num_heads, config.dropout_rate)
                for _ in range(config.num_blocks)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_units, eps=1e-8)

        # Initialize weights
        self.apply(self._init_weights)

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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            input_ids: Item sequences [batch_size, seq_length]

        Returns:
            Hidden states [batch_size, seq_length, hidden_units]
        """
        batch_size, seq_length = input_ids.size()

        # Get item embeddings
        item_embs = self.item_emb(input_ids)
        item_embs *= self.hidden_units**0.5  # Scale by sqrt(d) as in Transformer

        # Add positional embeddings
        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        # Mask positions where input is padding
        positions = positions * (input_ids != 0).long()
        pos_embs = self.pos_emb(positions)

        # Combine embeddings
        hidden_states = self.emb_dropout(item_embs + pos_embs)

        # Pass through transformer blocks
        for block in self.blocks:
            if self.training and self.config.use_gradient_checkpointing:
                hidden_states = _checkpoint_block(block, hidden_states)
            else:
                hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def predict(self, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for candidate items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            candidate_ids: Candidate items to score [batch_size, num_candidates]

        Returns:
            Scores for each candidate [batch_size, num_candidates]
        """
        # Get sequence representations
        hidden_states = self.forward(input_ids)  # [B, T, H]

        # Use only the last hidden state for prediction
        final_hidden = hidden_states[:, -1, :]  # [B, H]

        # Get candidate embeddings
        candidate_embs = self.item_emb(candidate_ids)  # [B, C, H]

        # Compute scores via dot product
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)  # [B, C]

        return scores

    def training_step(
        self, input_ids: torch.Tensor, pos_ids: torch.Tensor, neg_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        hidden_states = self.forward(input_ids)  # [B, T, H]

        pos_embs = self.item_emb(pos_ids)  # [B, T, H]
        neg_embs = self.item_emb(neg_ids)  # [B, T, H]

        pos_logits = (hidden_states * pos_embs).sum(dim=-1)  # [B, T]
        neg_logits = (hidden_states * neg_embs).sum(dim=-1)  # [B, T]

        return pos_logits, neg_logits


class DataRater(nn.Module):
    """MLP-based data rater using a single flattened parameter vector (eta)."""

    def __init__(self, input_dim: int, hidden_dim: int, init_std: float = 0.02):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_params = hidden_dim * input_dim + hidden_dim + hidden_dim + 1
        self.eta = nn.Parameter(torch.randn(self.num_params) * init_std)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return datarater_forward(state, self.eta, self.input_dim, self.hidden_dim)


def datarater_forward(
    state: torch.Tensor, eta: torch.Tensor, input_dim: int, hidden_dim: int
) -> torch.Tensor:
    """Forward pass for DataRater with flattened parameters."""
    idx = 0
    w1 = eta[idx : idx + hidden_dim * input_dim].view(hidden_dim, input_dim)
    idx += hidden_dim * input_dim
    b1 = eta[idx : idx + hidden_dim]
    idx += hidden_dim
    w2 = eta[idx : idx + hidden_dim].view(1, hidden_dim)
    idx += hidden_dim
    b2 = eta[idx : idx + 1]

    h = F.relu(state @ w1.t() + b1)
    out = torch.sigmoid(h @ w2.t() + b2)
    return out.squeeze(-1)


def _build_param_dict(names: list[str], tensors: list[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: t for name, t in zip(names, tensors)}


def _merge_params(
    base_params: Dict[str, torch.Tensor],
    trainable_names: list[str],
    theta_list: list[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    params = dict(base_params)
    for name, t in zip(trainable_names, theta_list):
        params[name] = t
    return params


def _checkpoint_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    try:
        return torch_checkpoint(block, x, use_reentrant=False)
    except TypeError:
        return torch_checkpoint(block, x)


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


def _sdp_kernel_context():
    try:
        fn = torch.nn.attention.sdpa_kernel
    except AttributeError:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
    try:
        return fn(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except TypeError:
        try:
            backend_enum = torch.nn.attention.SDPBackend
            try:
                return fn(backend_enum.MATH)
            except TypeError:
                return fn(backends=[backend_enum.MATH])
        except Exception:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )


@contextmanager
def _checkpointing_override(model: nn.Module, enabled: bool):
    if not hasattr(model, "config") or not hasattr(model.config, "use_gradient_checkpointing"):
        yield
        return
    prev = model.config.use_gradient_checkpointing
    model.config.use_gradient_checkpointing = enabled
    try:
        yield
    finally:
        model.config.use_gradient_checkpointing = prev


def sasrec_forward_stateless(
    model: SASRec,
    base_params: Dict[str, torch.Tensor],
    trainable_names: list[str],
    theta_list: list[torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    params = _merge_params(base_params, trainable_names, theta_list)
    params_and_buffers = {**params, **buffers}
    with _checkpointing_override(model, False):
        return functional_call(model, params_and_buffers, (input_ids,))


def sasrec_training_step_stateless(
    model: SASRec,
    base_params: Dict[str, torch.Tensor],
    trainable_names: list[str],
    theta_list: list[torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    params = _merge_params(base_params, trainable_names, theta_list)
    params_and_buffers = {**params, **buffers}
    with _checkpointing_override(model, False):
        hidden_states = functional_call(model, params_and_buffers, (input_ids,))
    item_weight = params["item_emb.weight"]
    pos_embs = F.embedding(pos_ids, item_weight, padding_idx=0)
    neg_embs = F.embedding(neg_ids, item_weight, padding_idx=0)
    pos_logits = (hidden_states * pos_embs).sum(dim=-1)
    neg_logits = (hidden_states * neg_embs).sum(dim=-1)
    return pos_logits, neg_logits, hidden_states, pos_embs


def sasrec_loss_and_state(
    model: SASRec,
    base_params: Dict[str, torch.Tensor],
    trainable_names: list[str],
    theta_list: list[torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    bce_criterion: nn.Module,
    include_aux: bool,
    return_state: bool = True,
    force_fp32_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    pos_logits, neg_logits, hidden_states, pos_embs = sasrec_training_step_stateless(
        model,
        base_params,
        trainable_names,
        theta_list,
        buffers,
        input_ids,
        pos_ids,
        neg_ids,
    )
    if force_fp32_loss:
        pos_logits = pos_logits.float()
        neg_logits = neg_logits.float()
    pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
    neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
    raw_loss = pos_loss + neg_loss
    valid_mask = pos_ids != 0
    if not return_state:
        return raw_loss, None, valid_mask

    state = torch.cat([hidden_states, pos_embs], dim=-1)
    if include_aux:
        loss_feat = raw_loss.detach().unsqueeze(-1)
        seq_len = pos_ids.size(1)
        positions = torch.arange(
            1, seq_len + 1, device=pos_ids.device, dtype=raw_loss.dtype
        ).view(1, seq_len, 1)
        pos_feat = positions / float(seq_len)
        pos_feat = pos_feat.expand(pos_ids.size(0), -1, -1)
        state = torch.cat([state, loss_feat, pos_feat], dim=-1)

    return raw_loss, state, valid_mask

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


class SequentialSampler:
    """
    Sampler for sequential data that generates training batches.
    Each batch contains user sequences with positive and negative samples.
    """

    def __init__(self, dataset: XLongSequenceDataset, config: SASRecConfig):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length
        self.max_item = dataset.max_item
        self.sample_id_stride = max(1, dataset.max_train_seq_len + 1)

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




def get_gradient_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of gradients across all model parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    # Compute norm without materializing concatenated tensor
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


class WeightLogger:
    """Stream DataRater weights to a CSV file per epoch."""

    def __init__(
        self,
        output_path: Path,
        id_to_item: Optional[Dict[int, int]] = None,
        flush_every: int = 5000,
    ):
        self.output_path = output_path
        self.flush_every = flush_every
        self.buffer: list[str] = []
        self.id_to_item = id_to_item or {}
        self.file = self.output_path.open("w", encoding="utf-8")
        self.file.write("sample_id,user_id,target_item,original_item_id,weight\n")

    def log_batch(
        self,
        sample_ids: torch.Tensor,
        user_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if not valid_mask.any():
            return
        user_expanded = user_ids.view(-1, 1).expand_as(pos_ids)
        sample_flat = sample_ids[valid_mask].detach().cpu().tolist()
        user_flat = user_expanded[valid_mask].detach().cpu().tolist()
        target_flat = pos_ids[valid_mask].detach().cpu().tolist()
        orig_flat = [self.id_to_item.get(int(tid), int(tid)) for tid in target_flat]
        weight_flat = weights[valid_mask].detach().cpu().tolist()
        for sid, uid, tid, oid, w in zip(sample_flat, user_flat, target_flat, orig_flat, weight_flat):
            self.buffer.append(f"{sid},{uid},{tid},{oid},{w:.6f}\n")
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.file.writelines(self.buffer)
        self.buffer.clear()
        self.file.flush()

    def close(self) -> None:
        self.flush()
        self.file.close()


def train_sasrec(
    model: SASRec,
    train_dataset,
    config: SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Dict[str, float]:
    """
    Train SASRec model.

    Args:
        model: SASRec model to train
        train_dataset: Training dataset
        config: Training configuration
        device: Device to train on

    Returns:
        Dictionary with best validation metrics
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    if not config.train_item_embeddings:
        model.item_emb.weight.requires_grad_(False)
    if not config.train_pos_embeddings:
        model.pos_emb.weight.requires_grad_(False)
    use_amp = config.use_amp and device_obj.type == "cuda"
    amp_dtype = _resolve_amp_dtype(config.amp_dtype, device_obj)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Apply torch.compile for faster training (CUDA only, not MPS)
    if device_obj.type == "cuda":
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
    use_fused = fused_available and device_obj.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_learning_rate, betas=(0.9, 0.98), fused=use_fused)

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
    best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}
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
            optimizer.zero_grad(set_to_none=True)

            # Initialize loss values for logging
            loss_value = 0.0
            pos_loss_value = 0.0
            neg_loss_value = 0.0
            pos_logit_stats = None
            neg_logit_stats = None

            input_ids = batch["input_ids"].to(device_obj, non_blocking=True)
            pos_ids = batch["pos_ids"].to(device_obj, non_blocking=True)
            neg_ids = batch["neg_ids"].to(device_obj, non_blocking=True)

            # Forward pass (AMP optional)
            with _amp_context(use_amp, amp_dtype, device_obj):
                pos_logits, neg_logits = model.training_step(input_ids, pos_ids, neg_ids)

            valid_mask = pos_ids != 0
            did_backward = False

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

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                did_backward = True

            if use_amp and did_backward:
                scaler.unscale_(optimizer)
            # Get gradient norm for logging
            grad_norm = get_gradient_norm(model)

            # Optimizer step
            if use_amp:
                if did_backward:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
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
                val_metrics = evaluate(model, eval_dataset, config=config, mode="test", device=str(device_obj))
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
                    logger.info(
                        f"New best validation - NDCG@10: {val_metrics['ndcg@10']:.4f}, HR@10: {val_metrics['hr@10']:.4f}"
                    )

    pbar.close()  # Close the progress bar
    logger.info(f"Training completed. Best validation NDCG@10: {best_val_metrics['ndcg@10']:.4f}")

    return best_val_metrics


def train_sasrec_meta(
    model: SASRec,
    datarater: DataRater,
    train_dataset,
    config: SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Dict[str, float]:
    """Bi-level optimization with DataRater (inner: SASRec, outer: DataRater)."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    datarater = datarater.to(device_obj)
    model.train()
    use_amp = config.use_amp and device_obj.type == "cuda"
    amp_dtype = _resolve_amp_dtype(config.amp_dtype, device_obj)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if not config.train_item_embeddings:
        model.item_emb.weight.requires_grad_(False)
    if not config.train_pos_embeddings:
        model.pos_emb.weight.requires_grad_(False)

    base_params = dict(model.named_parameters())
    trainable_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_names = [name for name, _ in trainable_named]
    buffers = dict(model.named_buffers())
    theta = [p.detach().clone().requires_grad_(True) for _, p in trainable_named]
    inner_opt = MomentumInner(theta, lr=config.inner_lr, momentum=config.inner_momentum)

    datarater_input_dim = 2 * config.hidden_units + 2
    if datarater.input_dim != datarater_input_dim:
        raise ValueError(
            f"DataRater input_dim={datarater.input_dim} does not match expected {datarater_input_dim}"
        )
    if datarater.hidden_dim != config.datarater_hidden_dim:
        raise ValueError(
            f"DataRater hidden_dim={datarater.hidden_dim} does not match config {config.datarater_hidden_dim}"
        )

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def _inner_loss(
        theta_list: list[torch.Tensor],
        eta: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Disable AMP inside torch.func.grad path to avoid dtype mismatch in autograd.
        raw_loss, state, valid_mask = sasrec_loss_and_state(
            model,
            base_params,
            trainable_names,
            theta_list,
            buffers,
            input_ids,
            pos_ids,
            neg_ids,
            bce_criterion,
            include_aux=True,
            return_state=True,
            force_fp32_loss=True,
        )
        if not valid_mask.any():
            zero = torch.zeros((), device=input_ids.device)
            for t in theta_list:
                zero = zero + t.sum() * 0.0
            zero = zero + eta.sum() * 0.0
            return zero
        weights = datarater_forward(
            state.reshape(-1, state.size(-1)).float(),
            eta.float(),
            datarater_input_dim,
            config.datarater_hidden_dim,
        )
        weights = weights.view_as(raw_loss)
        return (raw_loss[valid_mask] * weights[valid_mask]).mean()

    def _inner_loss_meta(
        theta_list: list[torch.Tensor],
        eta: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        return config.lambda_meta * _inner_loss(theta_list, eta, input_ids, pos_ids, neg_ids)

    def _outer_loss(
        theta_list: list[torch.Tensor],
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        with _amp_context(use_amp, amp_dtype, device_obj):
            raw_loss, _, valid_mask = sasrec_loss_and_state(
                model,
                base_params,
                trainable_names,
                theta_list,
                buffers,
                input_ids,
                pos_ids,
                neg_ids,
                bce_criterion,
                include_aux=False,
                return_state=False,
                force_fp32_loss=True,
            )
        if not valid_mask.any():
            zero = torch.zeros((), device=input_ids.device)
            for t in theta_list:
                zero = zero + t.sum() * 0.0
            return zero
        lengths = valid_mask.sum(dim=1)
        valid_rows = lengths > 0
        if not valid_rows.any():
            return raw_loss.sum() * 0.0
        last_idx = (lengths - 1).clamp_min(0)
        row_indices = torch.arange(raw_loss.size(0), device=raw_loss.device)
        last_loss = raw_loss[row_indices, last_idx]
        return last_loss[valid_rows].mean()

    grad_fn = get_fwdrev_grad_fn_eta(_inner_loss)
    grad_fn_meta = get_fwdrev_grad_fn_eta(_inner_loss_meta)

    outer_opt = torch.optim.AdamW([datarater.eta], lr=config.outer_lr, weight_decay=config.outer_weight_decay)

    train_sampler = SequentialSampler(train_dataset, config)
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

    for epoch in range(config.num_epochs):
        model.train()
        weight_logger = None
        if config.log_sample_weights:
            log_dir = Path(config.sample_weight_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"datarater_weights_epoch{epoch + 1:03d}.csv"
            weight_logger = WeightLogger(log_path, id_to_item=getattr(train_dataset, "id_to_item", None))

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
                    w_state, m_state = inner_opt.snapshot()
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

                need_weights = weight_logger is not None or (
                    global_step == 1 or global_step % config.steps_per_train_log == 0
                )
                weights = None
                valid_mask = None
                if need_weights:
                    with torch.no_grad():
                        with _amp_context(use_amp, amp_dtype, device_obj):
                            _, state, valid_mask = sasrec_loss_and_state(
                                model,
                                base_params,
                                trainable_names,
                                theta,
                                buffers,
                                step_batch["input_ids"],
                                step_batch["pos_ids"],
                                step_batch["neg_ids"],
                                bce_criterion,
                                include_aux=True,
                                return_state=True,
                            )
                        weights = datarater(state.reshape(-1, state.size(-1))).view_as(step_batch["pos_ids"])
                if weight_logger is not None and weights is not None and valid_mask is not None:
                    weight_logger.log_batch(
                        step_batch["sample_ids"],
                        step_batch["user_ids"],
                        step_batch["pos_ids"],
                        weights,
                        valid_mask,
                    )
                if (
                    weights is not None
                    and valid_mask is not None
                    and (global_step == 1 or global_step % config.steps_per_train_log == 0)
                ):
                    if valid_mask.any():
                        wandb.log(
                            {"meta/weight_hist": wandb.Histogram(weights[valid_mask].detach().cpu().numpy())}
                        )

                gflat = grad_fn(
                    theta,
                    datarater.eta,
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

                latest_w, latest_m = inner_opt.snapshot()
                if recent_steps is not None and config.meta_truncate_steps > 0:
                    start_w, start_m, _ = recent_steps[0]
                    inner_opt.restore(start_w, start_m)
                    for _, _, step_batch in recent_steps:
                        gflat = grad_fn_meta(
                            theta,
                            datarater.eta,
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
                    batch_val["input_ids"],
                    batch_val["pos_ids"],
                    batch_val["neg_ids"],
                )
                model.train()
                if use_amp:
                    scaler.scale(loss_outer).backward()
                    if config.outer_grad_clip and config.outer_grad_clip > 0:
                        scaler.unscale_(outer_opt)
                        torch.nn.utils.clip_grad_norm_([datarater.eta], config.outer_grad_clip)
                    scaler.step(outer_opt)
                    scaler.update()
                else:
                    loss_outer.backward()
                    if config.outer_grad_clip and config.outer_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_([datarater.eta], config.outer_grad_clip)
                    outer_opt.step()
                last_outer_loss = loss_outer.item()

                inner_opt.restore(latest_w, latest_m)

            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                if last_outer_loss is not None:
                    logger.info(
                        "Step %06d | Epoch %03d/%03d | OuterLoss: %.4f",
                        global_step,
                        epoch + 1,
                        config.num_epochs,
                        last_outer_loss,
                    )
                    wandb.log(
                        {
                            "meta/outer_loss": last_outer_loss,
                            "progress/epoch": epoch + 1,
                            "progress/step": global_step,
                        }
                    )

            pbar.update(1)

        if weight_logger is not None:
            weight_logger.close()

    pbar.close()

    # Copy learned theta back into the model for downstream evaluation.
    with torch.no_grad():
        for (_, p), t in zip(model.named_parameters(), theta):
            p.copy_(t.detach())

    if val_dataset is not None:
        logger.info("Running validation evaluation after meta-training...")
        metrics = evaluate(model, val_dataset, config=config, mode="test", device=str(device_obj))
    else:
        metrics = {"ndcg@10": 0.0, "hr@10": 0.0}

    return metrics


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

    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_name = f"sasrec-{config.dataset}-L{config.num_blocks}-H{config.hidden_units}"
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
    if config.pretrained_item_embeddings_path:
        emb_path = Path(config.pretrained_item_embeddings_path)
        if emb_path.exists():
            emb = np.load(emb_path)
            if emb.ndim == 2 and emb.shape[1] == config.hidden_units:
                with torch.no_grad():
                    if emb.shape[0] == item_num + 1:
                        model.item_emb.weight.copy_(torch.tensor(emb, dtype=model.item_emb.weight.dtype))
                        logger.info("Loaded pretrained item embeddings (with pad row) from %s", emb_path)
                    elif emb.shape[0] == item_num:
                        model.item_emb.weight.zero_()
                        model.item_emb.weight[1:].copy_(torch.tensor(emb, dtype=model.item_emb.weight.dtype))
                        logger.info("Loaded pretrained item embeddings (no pad row) from %s", emb_path)
                    else:
                        logger.warning(
                            "Pretrained embeddings size mismatch: expected %s or %s rows, got %s",
                            item_num,
                            item_num + 1,
                            emb.shape[0],
                        )
            else:
                logger.warning("Pretrained embeddings shape mismatch: got %s", emb.shape)
        else:
            logger.warning("Pretrained embeddings path not found: %s", emb_path)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if config.use_meta_learning:
        datarater = DataRater(
            input_dim=2 * config.hidden_units + 2,
            hidden_dim=config.datarater_hidden_dim,
            init_std=config.datarater_init_std,
        )
        best_metrics = train_sasrec_meta(
            model=model,
            datarater=datarater,
            train_dataset=train_dataset,
            config=config,
            device=device,
            val_dataset=meta_valid_dataset,
        )
    else:
        best_metrics = train_sasrec(
            model=model,
            train_dataset=train_dataset,
            config=config,
            device=device,
            val_dataset=None,
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

    if config.save_item_embeddings:
        save_item_embeddings(model, train_dataset, config)

    wandb.finish()
    logger.info("Training complete!")
