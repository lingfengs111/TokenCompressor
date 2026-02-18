#!/usr/bin/env python3
"""Train SASRec on the xlong dataset (pair-format only)."""

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

from core.device_manager import DeviceManager
from core.logger import setup_logger

logger = setup_logger("train-sasrec-new", log_to_file=True)


@dataclass
class SASRecConfig:
    """Configuration for SASRec training (xlong only)."""

    dataset: str = "xlong2018"
    xlong_train_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/train_corpus_total_dual.txt"
    )
    xlong_test_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_corpus_total_dual.txt"
    )
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "sasrec")

    # Model parameters
    max_seq_length: int = 500  # Maximum sequence length
    hidden_units: int = 128  # Hidden dimension size
    num_blocks: int = 2  # Number of transformer blocks
    num_heads: int = 1  # Number of attention heads
    dropout_rate: float = 0.2  # Dropout rate

    # Training parameters
    batch_size: int = 1024*10  # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    max_learning_rate: float = 1e-3  # Maximum learning rate (start of cosine)
    min_learning_rate: float = 1e-5  # Minimum learning rate (end of cosine)
    num_negatives: int = 1  # Negatives per target during training (pairwise BCE)

    # Training settings
    scheduler_type: str = "cosine"  # Learning rate scheduler type ("cosine" or "cosine_with_warmup")
    warmup_steps: int = 100  # Number of warmup steps (only for cosine_with_warmup)
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    steps_per_train_log: int = 100  # Log training progress every N steps
    steps_per_val_log: int = 500  # Validate and checkpoint every N steps
    # ⚠️之前eval用的是100导致结果看起来很高
    eval_sample_size: int = 1000  # Total candidates per user when eval_mode="sampled" (includes target)
    train_stride: int = 1  # Step size for selecting target positions (stride-based sampling)

    # Output settings
    save_item_embeddings: bool = True  # Save item embeddings after training

    # Device settings
    device: str = "cuda:1"  # e.g., "cuda:1", "cpu", "mps"

    # Dataloader settings
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== SASRec Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  xlong_train_path: {self.xlong_train_path}")
        logger.info(f"  xlong_test_path: {self.xlong_test_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  hidden_units: {self.hidden_units}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")
        logger.info(f"  num_negatives: {self.num_negatives}")

        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info(f"  train_stride: {self.train_stride}")

        logger.info("Output Settings:")
        logger.info(f"  save_item_embeddings: {self.save_item_embeddings}")

        logger.info("Device Settings:")
        logger.info(f"  device: {self.device}")
        logger.info("Dataloader Settings:")
        logger.info(f"  num_workers: {self.num_workers}")
        logger.info(f"  prefetch_factor: {self.prefetch_factor}")
        logger.info(f"  persistent_workers: {self.persistent_workers}")

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
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention with flash attention
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
        internal_id = 0

        for _, _, seq, neg_item in samples:
            if item_to_id is not None and (
                any(x not in self.item_to_id for x in seq) or neg_item not in self.item_to_id
            ):
                dropped += 1
                continue
            mapped = [self.item_to_id[x] for x in seq]
            self.user_seq[internal_id] = mapped
            self.users.append(internal_id)
            self.neg_item_by_user[internal_id] = self.item_to_id[neg_item]
            internal_id += 1

        self.num_users = len(self.users)
        self.num_items = len(self.item_to_id)
        self.max_item = self.num_items

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

            for idx, (user, seq) in enumerate(batch_data):
                # For each training step, we predict all positions in the sequence
                seq_len = min(len(seq), self.max_seq_length)

                if seq_len < 1:
                    continue

                # If sequence is longer than max_seq_length, take the most recent items
                if len(seq) > self.max_seq_length:
                    seq = seq[-self.max_seq_length :]
                    seq_len = self.max_seq_length

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

                # Sample negative items for each position
                seen_set = set(self.dataset.user_seq[user])  # Use full sequence for negative sampling
                for pos in range(seq_len):
                    neg_item = self.sample_negative_item(1, self.max_item + 1, seen_set)
                    neg_tensors[idx, -seq_len + pos] = neg_item

            yield {"input_ids": seq_tensors, "pos_ids": pos_tensors, "neg_ids": neg_tensors}

    def __len__(self):
        return (len(self.valid_user_seqs) + self.batch_size - 1) // self.batch_size


class StrideTrainingDataset(Dataset):
    """Samples (history, target) pairs with a stride from pre-built user sequences."""

    def __init__(self, dataset, config: SASRecConfig):
        self.dataset = dataset
        self.max_seq_length = config.max_seq_length
        self.train_stride = max(1, config.train_stride)

        self.samples = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq), self.train_stride):
                self.samples.append((user, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, pos = self.samples[idx]
        seq = self.dataset.user_seq[user]

        history = seq[:pos]
        target = seq[pos]

        max_len = self.max_seq_length
        seq_tensor = torch.zeros(max_len, dtype=torch.long)
        seq_len = min(len(history), max_len)
        if seq_len > 0:
            seq_tensor[-seq_len:] = torch.tensor(history[-seq_len:], dtype=torch.long)

        return {
            "input_ids": seq_tensor,
            "target_id": torch.tensor(target, dtype=torch.long),
            "user_id": torch.tensor(user, dtype=torch.long),
        }


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


def sample_negatives_vectorized(
    targets: torch.Tensor,
    histories: torch.Tensor,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    oversample: int = 3,
    max_trials: int = 5,
) -> torch.Tensor:
    """
    Per-sample negative sampling that avoids each sample's history and target.
    Uses a per-row seen_mask to prevent cross-sample contamination and limits
    resampling rounds to avoid pathological while-loops.

    Notes:
    - For very large vocabularies the dense seen_mask can be memory heavy; we
      fall back to per-row rejection sampling when the estimated mask size
      would exceed `max_mask_bytes`.
    """
    if num_negatives <= 0:
        return torch.empty(targets.size(0), 0, device=device, dtype=torch.long)

    B = targets.size(0)
    targets = targets.to(device)
    histories = histories.to(device)

    # Heuristic guard for mask memory usage
    max_mask_bytes = 512 * 1024 * 1024  # 512MB
    mask_bytes = (B * (num_items + 1))  # bool ~1 byte
    use_dense_mask = mask_bytes <= max_mask_bytes

    if use_dense_mask:
        seen_mask = torch.zeros((B, num_items + 1), device=device, dtype=torch.bool)
    else:
        seen_mask = None

    if use_dense_mask and histories.numel() > 0:
        hist_mask = histories > 0  # ignore padding 0
        if hist_mask.any():
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(histories)
            seen_mask[batch_idx[hist_mask], histories[hist_mask]] = True

    batch_idx = torch.arange(B, device=device)
    if use_dense_mask:
        seen_mask[batch_idx, targets] = True

    negs = torch.empty((B, num_negatives), device=device, dtype=torch.long)
    filled = torch.zeros(B, device=device, dtype=torch.long)

    if use_dense_mask:
        trials = 0
        while (filled < num_negatives).any() and trials < max_trials:
            remaining = num_negatives - filled
            max_need = remaining.max()
            sample = torch.randint(1, num_items + 1, (B, int(max_need) * oversample), device=device)
            valid = ~seen_mask.gather(1, sample)

            for b in range(B):
                need = int(remaining[b])
                if need == 0:
                    continue
                row_valid = sample[b][valid[b]]
                if row_valid.numel() == 0:
                    continue
                take = min(need, row_valid.numel())
                start = int(filled[b])
                chosen = row_valid[:take]
                negs[b, start : start + take] = chosen
                if take > 0:
                    seen_mask[b, chosen] = True  # prevent duplicates within the row
                filled[b] += take

            trials += 1
    else:
        # Memory-safe fallback: per-row rejection sampling
        for b in range(B):
            history = histories[b]
            seen = set(history[history > 0].tolist())
            seen.add(int(targets[b]))
            pos = 0
            while pos < num_negatives:
                candidate = torch.randint(1, num_items + 1, (1,), device=device).item()
                if candidate in seen:
                    continue
                negs[b, pos] = candidate
                seen.add(candidate)
                pos += 1

    # Fallback for any rows still short (rare; ensures termination)
    if use_dense_mask:
        for b in range(B):
            need = int(num_negatives - filled[b])
            if need <= 0:
                continue
            seen_row = seen_mask[b]
            fill_pos = int(filled[b])
            guard = 0
            while need > 0 and guard < 1000:
                candidate = torch.randint(1, num_items + 1, (1,), device=device)
                cval = int(candidate)
                if not seen_row[cval]:
                    negs[b, fill_pos] = candidate
                    seen_row[cval] = True
                    fill_pos += 1
                    need -= 1
                guard += 1

    return negs


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
    model = model.to(device)

    # Apply torch.compile for faster training (CUDA only, not MPS)
    if device.startswith("cuda"):
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    # DataLoader with worker-based batch preparation to reduce CPU bottlenecks
    pin_memory = device.startswith("cuda")
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loader_kwargs["persistent_workers"] = config.persistent_workers

    logger.info(
        "Sampling strategy: stride_vectorized (stride=%d, %d negatives/target, device-side sampling)",
        config.train_stride,
        config.num_negatives,
    )
    train_loader = DataLoader(StrideTrainingDataset(train_dataset, config), **loader_kwargs)

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch

    logger.info(f"Training for {config.num_epochs} epochs, {steps_per_epoch} steps per epoch")
    logger.info(f"Total training steps: {total_steps:,}")

    # Optimizer with fused support
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.startswith("cuda")
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

    # Item vocabulary size for vectorized negative sampling
    item_num = getattr(train_dataset, "max_item", None) or getattr(train_dataset, "num_items", None)
    if item_num is None:
        raise ValueError("Could not infer item vocabulary size from train_dataset")

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
            pos_loss_value = 0.0
            neg_loss_value = 0.0
            pos_logit_stats = None
            neg_logit_stats = None

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_id"].to(device, non_blocking=True)

            # Forward pass
            hidden_states = model.forward(input_ids)
            final_hidden = hidden_states[:, -1, :]

            neg_ids = sample_negatives_vectorized(
                targets=target_ids,
                histories=input_ids,
                num_items=item_num,
                num_negatives=config.num_negatives,
                device=input_ids.device,
            )

            pos_embs = model.item_emb(target_ids)
            pos_logits = (final_hidden * pos_embs).sum(dim=-1)

            neg_embs = model.item_emb(neg_ids)
            neg_logits = torch.einsum("bd,bnd->bn", final_hidden, neg_embs)

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            pos_loss = bce_criterion(pos_logits, pos_labels)
            neg_loss = bce_criterion(neg_logits, neg_labels)
            loss = pos_loss + neg_loss
            loss_value = loss.item()
            pos_loss_value = pos_loss.item()
            neg_loss_value = neg_loss.item()

            with torch.no_grad():
                pos_logit_stats = (
                    pos_logits.mean().item(),
                    pos_logits.min().item(),
                    pos_logits.max().item(),
                )
                neg_logit_stats = (
                    neg_logits.mean().item(),
                    neg_logits.min().item(),
                    neg_logits.max().item(),
                )

            loss.backward()

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
                val_metrics = evaluate(model, eval_dataset, config=config, mode="test", device=device)
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

    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_name = f"sasrec-{config.dataset}-L{config.num_blocks}-H{config.hidden_units}"
    run = wandb.init(project="sasrec-experiments", name=run_name, config=config.__dict__)
    config.log_config()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_xlong_samples(config.xlong_train_path)
    train_dataset = XLongSequenceDataset(train_samples, config)
    val_dataset = None

    test_samples = load_xlong_samples(config.xlong_test_path)
    test_dataset = XLongSequenceDataset(test_samples, config, item_to_id=train_dataset.item_to_id)
    item_num = train_dataset.max_item

    model = SASRec(config, item_num=item_num)
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

    if config.save_item_embeddings:
        save_item_embeddings(model, train_dataset, config)

    wandb.finish()
    logger.info("Training complete!")
