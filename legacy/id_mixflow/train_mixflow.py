#!/usr/bin/env python3
"""
MixFlow training scaffold for ID-based recommender.

Design:
  - Inner loop: train encoder/decoder on short sequences + soft patch.
  - Outer loop: update patch using longer real sequences; optionally inject patch (outer_use_patch).
  - Meta-gradients: reuse core/mixflow.py (eta-aware gradients).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime
import math
import random
import os
import sys

import numpy as np
try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    KMeans = None
import torch
import torch.nn as nn
from torch.func import functional_call
from tqdm import tqdm

import wandb

from data import (
    _build_xlong_item_map,
    _map_xlong_sequences,
    load_item_embeddings_from_indexed_npz,
    load_xlong_samples,
)
from evaluate import EarlyStopper, FMEConfig, run_eval_online_fme
from model import IDRecModel, ItemEmbeddingTable, load_item_embeddings
from train import SequenceDataset, SequentialSampler

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from core.mixflow import get_fwdrev_grad_fn_eta, MomentumInner  # type: ignore
from core.logger import setup_logger

logger = setup_logger("train-mixflow-id", log_to_file=True)

# Fixed defaults (previously configurable)
DEFAULT_EMBEDDINGS_HAVE_PAD = False
DETERMINISTIC_TRAINING = False
DEBUG_NAN_CHECKS = True
KEEP_BEST_K = 1
LOAD_BEST_FOR_TEST = True
RUN_FINAL_TEST_EVAL = True


@dataclass
class MixFlowIDConfig:
    
    # Adjustable hyperparameters
    L_recent: int = 50
    L_soft: int = 100
    outer_use_patch: bool = True # 外循环是否用patch

    run_fme_short_only: bool = False
    run_fme_long_only: bool = False
    run_main_training: bool = True

    inner_reset_every_outer: int = 0  # resets theta/momentum after this many outer updates (>0 enables)
    inner_reset_every_inner: int = 0  # resets theta/momentum after this many inner steps (>0 enables)
    #### END of adjustable hyperparameters ####

    # Data
    data_format: str = "xlong_pair"  # only xlong_pair supported for SASRec-style flow
    xlong_train_path: str = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/train_corpus_total_dual.txt"
    xlong_test_path: str = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_corpus_total_dual.txt"
    item_embeddings_path: Optional[str] = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/item_embeddings_sas128_len500.npy"
    L_full: int = 500
    train_stride: int = 5
    inner_batch_size: int = 256
    outer_batch_size: int = 256

    # Model
    hidden_dim: int = 128
    encoder_layers: int = 3
    decoder_layers: int = 3
    num_heads: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    attn_dropout: float = 0.1
    model_mode: str = "decoder_only" # "encoder_decoder"  # decoder_only | encoder_decoder
    patch_location: str = "decoder"  # encoder | decoder
    train_item_embeddings: bool = False

    # Patch

    num_patches: int = 30
    patch_routing: str = "kmeans_per_user"  # single | random_per_user | kmeans_per_user
    patch_seed: Optional[int] = None
    kmeans_max_iters: int = 25
    user_embedding_method: str = "mean"  # mean | exp_decay
    user_embedding_decay: float = 1.0

    # Reproducibility
    seed: int = 2026

    # Training
    device: str = "auto"  # "auto", "cpu", or explicit "cuda:{id}"
    inner_steps: int = 5
    inner_lr: float = 1e-4
    inner_momentum: float = 0.9
    inner_grad_clip: float = 0.0
    
    outer_max_lr: float = 1e-4
    outer_min_lr: float = 1e-5
    outer_scheduler_type: str = "cosine_with_warmup"  # "cosine" | "cosine_with_warmup"
    outer_warmup_steps: int = 30 # very short
    outer_warmup_start_lr: float = 1e-7
    outer_grad_clip: float = 0.1
    weight_decay: float = 0.0
    meta_truncate_steps: int = 4  # inner steps, 0 disables meta unroll
    outer_update_every: int = 3
    max_iters: int = 5000
    log_every: int = 50
    # Gradient weighting
    lambda_direct: float = 1.0  # scales direct outer loss gradients wrt patch
    lambda_meta: float = 1.0    # scales meta-gradients wrt patch

    # Logging
    wandb_enabled: bool = True
    wandb_project: str = "mixflow-id3"
    wandb_run_name: Optional[str] = None

    # Checkpointing
    checkpoint_dir: str = os.path.join(ROOT_DIR, "checkpoints", "mixflow_id")

    # Evaluation (FME)
    fme_eval_every: int = 700
    fme_lr: float = 1e-4
    fme_weight_decay: float = 0.0
    fme_patience: int = 3
    early_stop_patience: int = 4
    # FME eval/train settings (cheap during training)
    fme_train_num_negatives: int = 200
    fme_train_epochs: int = 10
    # Final test settings (more expensive)
    fme_test_num_negatives: int = 400
    fme_test_epochs: int = 30
    fme_long_epochs: Optional[int] = 45  # None -> use fme_test_epochs
    fme_batch_size: int = 256
    fme_max_train_batches: Optional[int] = 200
    fme_max_val_batches: Optional[int] = 50


    # FME progress display
    fme_show_progress: bool = True
    # Compute
    tf32_enabled: bool = False             # enable TF32 matmul on Ampere+ for speed
    # Data pipeline
    vectorized_neg_sampling: bool = True  # use batched negative sampler (vs original per-position)

    def log_config(self):
        logger.info("=== MixFlow ID Config ===")
        logger.info("data_format: %s", self.data_format)
        logger.info("xlong_train_path: %s", self.xlong_train_path)
        logger.info("xlong_test_path: %s", self.xlong_test_path)
        logger.info("item_embeddings_path: %s", self.item_embeddings_path)
        logger.info("embeddings_have_pad: auto | default=%s", DEFAULT_EMBEDDINGS_HAVE_PAD)
        logger.info("L_recent: %s | L_full: %s", self.L_recent, self.L_full)
        logger.info("train_stride: %s", self.train_stride)
        logger.info("inner_batch_size: %s | outer_batch_size: %s", self.inner_batch_size, self.outer_batch_size)
        logger.info("hidden_dim: %s | heads: %s", self.hidden_dim, self.num_heads)
        logger.info("encoder_layers: %s | decoder_layers: %s", self.encoder_layers, self.decoder_layers)
        logger.info("ffn_dim: %s | dropout: %s | attn_dropout: %s", self.ffn_dim, self.dropout, self.attn_dropout)
        logger.info("model_mode: %s | patch_location: %s", self.model_mode, self.patch_location)
        logger.info("train_item_embeddings: %s", self.train_item_embeddings)
        logger.info(
            "L_soft: %s | num_patches: %s | routing: %s | user_emb: %s | decay: %s",
            self.L_soft,
            self.num_patches,
            self.patch_routing,
            self.user_embedding_method,
            self.user_embedding_decay,
        )
        logger.info("patch_seed: %s", self.patch_seed)
        logger.info(
            "device: %s | inner_steps: %s | inner_lr: %s | outer_max_lr: %s | outer_min_lr: %s | wd: %s | truncate: %s | meta_every: %s | warmup: %s | max_iters: %s",
            self.device,
            self.inner_steps,
            self.inner_lr,
            self.outer_max_lr,
            self.outer_min_lr,
            self.weight_decay,
            self.meta_truncate_steps,
            self.outer_update_every,
            self.outer_warmup_steps,
            self.max_iters,
        )
        logger.info(
            "inner_reset_every_outer: %s | inner_reset_every_inner: %s",
            self.inner_reset_every_outer,
            self.inner_reset_every_inner,
        )
        logger.info("inner_grad_clip: %s | outer_grad_clip: %s", self.inner_grad_clip, self.outer_grad_clip)
        logger.info("wandb_enabled: %s | project: %s | run_name: %s", self.wandb_enabled, self.wandb_project, self.wandb_run_name)
        logger.info(
            "fme_eval_every: %s | fme_train_epochs: %s | fme_lr: %s | fme_patience: %s",
            self.fme_eval_every,
            self.fme_train_epochs,
            self.fme_lr,
            self.fme_patience,
        )
        logger.info(
            "fme_batch_size: %s | fme_train_neg: %s | fme_test_epochs: %s | fme_long_epochs: %s | fme_test_neg: %s | fme_show_progress: %s",
            self.fme_batch_size,
            self.fme_train_num_negatives,
            self.fme_test_epochs,
            self.fme_long_epochs,
            self.fme_test_num_negatives,
            self.fme_show_progress,
        )
        logger.info(
            "vectorized_neg_sampling: %s | sampler_device: %s (auto) | tf32_enabled: %s",
            self.vectorized_neg_sampling,
            self.device,
            self.tf32_enabled,
        )
        logger.info("keep_best_k: %s | load_best_for_test: %s", KEEP_BEST_K, LOAD_BEST_FOR_TEST)
        logger.info("final_test_eval: %s | run_main_training: %s", RUN_FINAL_TEST_EVAL, self.run_main_training)
        logger.info("run_fme_short_only: %s | run_fme_long_only: %s", self.run_fme_short_only, self.run_fme_long_only)
        logger.info("outer_use_patch: %s", self.outer_use_patch)
        logger.info("early_stop_patience: %s", self.early_stop_patience)
        logger.info("seed: %s | deterministic: %s", self.seed, DETERMINISTIC_TRAINING)
        logger.info("=========================")


def load_xlong_datasets(cfg: MixFlowIDConfig) -> Tuple[SequenceDataset, SequenceDataset, Optional[SequenceDataset], Optional[SequenceDataset], Dict[str, int]]:
    train_samples = load_xlong_samples(cfg.xlong_train_path)
    item_to_id = _build_xlong_item_map(train_samples)
    train_sequences, train_user_ids, dropped_train = _map_xlong_sequences(train_samples, item_to_id)

    test_samples = load_xlong_samples(cfg.xlong_test_path) if cfg.xlong_test_path else []
    test_sequences, test_user_ids, dropped_test = _map_xlong_sequences(test_samples, item_to_id) if test_samples else ([], [], 0)

    ds_short = SequenceDataset(train_sequences, train_user_ids, cfg.L_recent, item_to_id)
    ds_long = SequenceDataset(train_sequences, train_user_ids, cfg.L_full, item_to_id)
    ds_test_short = SequenceDataset(test_sequences, test_user_ids, cfg.L_recent, item_to_id) if test_sequences else None
    ds_test_long = SequenceDataset(test_sequences, test_user_ids, cfg.L_full, item_to_id) if test_sequences else None
    meta = {
        "num_items": len(item_to_id),
        "dropped_train": dropped_train,
        "dropped_test": dropped_test,
    }
    return ds_short, ds_long, ds_test_short, ds_test_long, meta


class SoftPatch(nn.Module):
    """Learnable (possibly multi) soft patch applied after item embeddings."""

    def __init__(self, L_soft: int, hidden_dim: int, num_patches: int = 1):
        super().__init__()
        self.num_patches = max(1, num_patches)
        self.phi = nn.Parameter(torch.randn(self.num_patches, L_soft, hidden_dim) * 0.02)

    def select(self, patch_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Return patch embeddings for given patch ids."""
        if self.num_patches == 1:
            return self.phi[0]
        if patch_ids is None:
            return self.phi[0]
        patch_ids = patch_ids.to(self.phi.device).long()
        return self.phi[patch_ids]


def collect_base_params_and_buffers(module: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params = {n: p for n, p in module.named_parameters()}
    buffers = {n: b for n, b in module.named_buffers()}
    return params, buffers



def build_override(names: List[str], tensors: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {n: t for n, t in zip(names, tensors)}


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _auto_select_device(device_pref: str) -> torch.device:
    """Pick a device; when 'auto', choose GPU with most free mem, else fallback."""
    if device_pref != "auto":
        return torch.device(device_pref)
    if not torch.cuda.is_available():
        logger.info("Auto device: CUDA not available, using CPU")
        return torch.device("cpu")
    best_idx = None
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(idx)
        except Exception:
            free = -1
        if free > best_free:
            best_free = free
            best_idx = idx
    if best_idx is None:
        logger.info("Auto device: mem query failed, defaulting to cuda:0")
        return torch.device("cuda:0")
    logger.info("Auto device: picked cuda:%d (free %.1f GB)", best_idx, best_free / (1024**3))
    return torch.device(f"cuda:{best_idx}")


def _default_run_name(cfg: "MixFlowIDConfig") -> str:
    """Build a concise run name with key hyperparameters."""
    parts = [
        f"Lr{cfg.L_recent}",
        f"Lsoft{cfg.L_soft}",
        f"rst{cfg.inner_reset_every_outer}",
        f"patch{int(cfg.outer_use_patch)}",
        f"in{cfg.inner_batch_size}",
        f"out{cfg.outer_batch_size}",
    ]
    return "_".join(parts)

def _init_wandb(cfg: MixFlowIDConfig):
    if not cfg.wandb_enabled:
        return None
    run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
    if run is not None:
        wandb.define_metric("iter")
        wandb.define_metric("epoch")
        wandb.define_metric("loss/*", step_metric="iter")
        wandb.define_metric("fme/*", step_metric="iter")
        wandb.define_metric("test/*", step_metric="iter")
    return run


_wandb_last_step = -1


def _wandb_log(run, metrics: Dict[str, float], step: int, commit: bool = True):
    if run is not None:
        global _wandb_last_step
        if step <= _wandb_last_step:
            step = _wandb_last_step + 1  # keep wandb step monotonically increasing
        _wandb_last_step = step
        run.log(metrics, step=step, commit=commit)


def _build_run_checkpoint_dir(base_dir: str, run_name: Optional[str]) -> str:
    """Create a unique subdir per training run to avoid ckpt collisions."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = (run_name or "mixflow_id").replace(os.path.sep, "-").replace(" ", "_")
    return os.path.join(base_dir, f"{safe_name}_ts{ts}_pid{os.getpid()}")


def _tensor_has_pad_row(embeddings: torch.Tensor) -> bool:
    """Heuristic: treat embeddings[0] == 0 as a provided pad row."""
    return embeddings.numel() > 0 and torch.count_nonzero(embeddings[0]).item() == 0


def _save_patch_checkpoint(path: str, patch: SoftPatch, cfg: MixFlowIDConfig, step: int, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "patch_state_dict": patch.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "metrics": metrics,
    }
    torch.save(payload, path)
    logger.info("Saved best patch checkpoint to %s", path)


class PatchAssigner:
    """Static mapping from user id to patch id."""

    def __init__(self, num_patches: int, mapping: Dict[int, int]):
        self.num_patches = max(1, num_patches)
        self.mapping = mapping

    def _fallback(self, user_id: int) -> int:
        return hash(int(user_id)) % self.num_patches

    def get_patch_ids(self, user_ids: torch.Tensor) -> torch.Tensor:
        if self.num_patches == 1 or user_ids is None:
            return torch.zeros(user_ids.size(0), dtype=torch.long, device=user_ids.device)
        ids = user_ids.detach().cpu().tolist()
        patch_ids = [self.mapping.get(int(uid), self._fallback(uid)) for uid in ids]
        return torch.tensor(patch_ids, dtype=torch.long, device=user_ids.device)


def _build_user_sequence_map(dataset) -> Dict[int, List[List[int]]]:
    """Collect raw sequences grouped by user id from a dataset."""
    user_to_seqs: Dict[int, List[List[int]]] = defaultdict(list)
    if hasattr(dataset, "user_seq") and hasattr(dataset, "users"):
        for uid in dataset.users:
            user_to_seqs[int(uid)].append(dataset.user_seq[uid])
    elif hasattr(dataset, "sequences") and hasattr(dataset, "user_ids"):
        sequences = getattr(dataset, "sequences")
        user_ids = getattr(dataset, "user_ids")
        for seq, uid in zip(sequences, user_ids):
            user_to_seqs[int(uid)].append(seq)
    elif hasattr(dataset, "histories"):
        histories = getattr(dataset, "histories")
        user_ids = getattr(dataset, "user_ids", list(range(len(histories))))
        offset = getattr(dataset, "item_id_offset", 0)
        for seq, uid in zip(histories, user_ids):
            seq_list = [int(x) + offset for x in np.asarray(seq).tolist() if int(x) != 0]
            user_to_seqs[int(uid)].append(seq_list)
    return user_to_seqs


def _compute_user_embeddings(
    user_to_seqs: Dict[int, List[List[int]]],
    item_table: ItemEmbeddingTable,
    weighting: str,
    decay: float,
    item_id_offset: int = 0,
) -> Dict[int, torch.Tensor]:
    """Compute user-level embeddings by averaging item embeddings."""
    if not user_to_seqs:
        return {}
    table = item_table.weight.detach().cpu()
    dim = table.size(1)
    user_embs: Dict[int, torch.Tensor] = {}
    for uid, seqs in user_to_seqs.items():
        total = torch.zeros(dim)
        count = 0
        for seq in seqs:
            if not seq:
                continue
            ids = torch.tensor(seq, dtype=torch.long)
            if item_id_offset != 0:
                ids = ids + item_id_offset
            ids = ids[(ids >= 0) & (ids < table.size(0))]
            if ids.numel() == 0:
                continue
            emb = table[ids]
            if weighting == "exp_decay" and decay != 1.0:
                weights = torch.pow(
                    torch.full((emb.size(0),), decay, dtype=emb.dtype),
                    torch.arange(emb.size(0) - 1, -1, -1, dtype=emb.dtype),
                )
                weights = weights / weights.sum()
                seq_emb = (emb * weights.unsqueeze(1)).sum(dim=0)
            else:
                seq_emb = emb.mean(dim=0)
            total += seq_emb
            count += 1
        if count > 0:
            user_embs[uid] = total / count
    return user_embs


def _random_patch_assignments(user_ids: List[int], num_patches: int, seed: int) -> Dict[int, int]:
    if not user_ids:
        return {}
    g = torch.Generator().manual_seed(seed)
    rand = torch.randint(0, num_patches, (len(user_ids),), generator=g)
    return {uid: int(pid) for uid, pid in zip(user_ids, rand.tolist())}


def _kmeans_assignments(
    user_embs: Dict[int, torch.Tensor],
    num_patches: int,
    max_iters: int,
    seed: int,
) -> Dict[int, int]:
    if not user_embs:
        return {}
    user_ids = list(user_embs.keys())
    data = torch.stack([v for v in user_embs.values()])
    k = min(num_patches, data.size(0))
    if KMeans is not None:
        arr = data.cpu().numpy()
        try:
            km = KMeans(n_clusters=k, n_init="auto", max_iter=max_iters, random_state=seed)
        except TypeError:
            km = KMeans(n_clusters=k, n_init=10, max_iter=max_iters, random_state=seed)
        labels = km.fit_predict(arr)
        return {uid: int(labels[i]) for i, uid in enumerate(user_ids)}

    # Fallback torch kmeans
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.size(0), generator=g)
    centers = data[perm[:k]].clone()
    assignments = torch.zeros(data.size(0), dtype=torch.long)
    for _ in range(max_iters):
        dist = torch.cdist(data, centers)
        new_assign = dist.argmin(dim=1)
        if torch.equal(new_assign, assignments):
            break
        assignments = new_assign
        for ci in range(k):
            mask = assignments == ci
            if mask.any():
                centers[ci] = data[mask].mean(dim=0)
    return {uid: int(assignments[i].item()) for i, uid in enumerate(user_ids)}


def build_patch_assigner(cfg: MixFlowIDConfig, dataset, item_table: ItemEmbeddingTable) -> PatchAssigner:
    patch_seed = cfg.patch_seed if cfg.patch_seed is not None else cfg.seed
    if cfg.num_patches <= 1 or cfg.patch_routing == "single":
        logger.info("Using single patch (no routing).")
        return PatchAssigner(1, {})
    user_to_seqs = _build_user_sequence_map(dataset)
    if not user_to_seqs:
        logger.warning("No user sequences found; falling back to single patch.")
        return PatchAssigner(1, {})
    logger.info("Building patch routing for %s users (strategy=%s)...", len(user_to_seqs), cfg.patch_routing)
    if cfg.patch_routing == "random_per_user":
        mapping = _random_patch_assignments(list(user_to_seqs.keys()), cfg.num_patches, patch_seed)
    elif cfg.patch_routing == "kmeans_per_user":
        user_embs = _compute_user_embeddings(
            user_to_seqs,
            item_table,
            weighting=cfg.user_embedding_method,
            decay=cfg.user_embedding_decay,
            item_id_offset=getattr(dataset, "item_id_offset", 0),
        )
        if not user_embs:
            logger.warning("User embeddings are empty; falling back to random assignment.")
            mapping = _random_patch_assignments(list(user_to_seqs.keys()), cfg.num_patches, patch_seed)
        else:
            mapping = _kmeans_assignments(user_embs, cfg.num_patches, cfg.kmeans_max_iters, patch_seed)
    else:
        raise ValueError(f"Unknown patch_routing: {cfg.patch_routing}")
    logger.info("Routing built: %s patches, %s users covered.", cfg.num_patches, len(mapping))
    return PatchAssigner(cfg.num_patches, mapping)


def select_patch_for_batch(
    patch: SoftPatch,
    assigner: PatchAssigner,
    batch: dict,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve a patch tensor for the given batch (optionally filtered by mask)."""
    if assigner is None or patch.num_patches == 1:
        return patch.select(None)
    user_ids = batch.get("user_ids", None)
    if user_ids is None:
        return patch.select(None)
    if mask is not None:
        mask = mask.to(user_ids.device)
        user_ids = user_ids[mask]
    if user_ids.numel() == 0:
        return patch.select(None)
    patch_ids = assigner.get_patch_ids(user_ids)
    return patch.select(patch_ids)


def get_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_pos = attention_mask.long().sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos, :]


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor is None:
        return
    finite = torch.isfinite(tensor)
    if finite.all():
        return
    with torch.no_grad():
        safe = tensor.detach()
        safe = safe[finite] if finite.any() else tensor.detach().reshape(-1)
        if safe.numel() == 0:
            stats = "no_finite_values"
        else:
            stats = f"min={safe.min().item():.4g} max={safe.max().item():.4g} mean={safe.mean().item():.4g}"
    logger.error("Non-finite detected at %s (%s)", name, stats)
    raise RuntimeError(f"Non-finite detected at {name}")


def _clip_flat_grad(gflat: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return gflat
    norm = gflat.norm()
    if norm <= max_norm:
        return gflat
    scale = max_norm / (norm + 1e-6)
    return gflat * scale


def run_encoder_stateless(
    encoder: nn.Module,
    base_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    override_params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_emb: Optional[torch.Tensor],
):
    params = {**base_params, **override_params, **base_buffers}
    return functional_call(
        encoder,
        params,
        args=(),
        kwargs={"input_ids": input_ids, "attention_mask": attention_mask, "patch_emb": patch_emb},
    )


def run_decoder_stateless(
    decoder: nn.Module,
    base_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    override_params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_output: Optional[torch.Tensor],
    encoder_mask: Optional[torch.Tensor],
    patch_emb: Optional[torch.Tensor],
):
    params = {**base_params, **override_params, **base_buffers}
    return functional_call(
        decoder,
        params,
        args=(),
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_output": encoder_output,
            "encoder_mask": encoder_mask,
            "patch_emb": patch_emb,
        },
    )


def build_item_table(cfg: MixFlowIDConfig, num_items: Optional[int] = None) -> ItemEmbeddingTable:
    embeddings = None
    has_pad = DEFAULT_EMBEDDINGS_HAVE_PAD
    if cfg.item_embeddings_path:
        if cfg.item_embeddings_path.endswith(".npz"):
            emb_arr = load_item_embeddings_from_indexed_npz(cfg.item_embeddings_path)
            if emb_arr is not None:
                embeddings = torch.from_numpy(emb_arr).float()
                has_pad = True
        else:
            embeddings = load_item_embeddings(cfg.item_embeddings_path)
            if embeddings is not None and _tensor_has_pad_row(embeddings):
                has_pad = True
    if embeddings is None:
        if num_items is None:
            raise ValueError("num_items is required when building fresh embeddings")
        emb_dim = cfg.hidden_dim
        return ItemEmbeddingTable(num_items=num_items, embedding_dim=emb_dim, trainable=True)
    if DEBUG_NAN_CHECKS:
        _check_finite("item_embeddings", embeddings)

    if has_pad:
        table = ItemEmbeddingTable(
            num_items=embeddings.size(0) - 1,
            embedding_dim=embeddings.size(1),
            trainable=cfg.train_item_embeddings,
        )
        with torch.no_grad():
            table.embedding.weight.copy_(embeddings)
    else:
        table = ItemEmbeddingTable.from_pretrained(
            embeddings,
            trainable=cfg.train_item_embeddings,
            pad_zero=True,
        )
    return table


def main(cfg: MixFlowIDConfig) -> None:
    cfg.log_config()
    device = _auto_select_device(cfg.device)
    cfg.device = str(device)
    logger.info("Using device: %s", device)
    if cfg.tf32_enabled and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    set_global_seed(cfg.seed, DETERMINISTIC_TRAINING)
    keep_best_k = KEEP_BEST_K
    load_best_for_test = LOAD_BEST_FOR_TEST
    run_final_test_eval = RUN_FINAL_TEST_EVAL
    run_name = cfg.wandb_run_name or _default_run_name(cfg)
    cfg.wandb_run_name = run_name
    checkpoint_dir = _build_run_checkpoint_dir(cfg.checkpoint_dir, run_name)
    logger.info("Run checkpoint directory: %s", checkpoint_dir)

    # Sampler now always follows the main device for simplicity.
    sampler_device = device

    def _move_batch_to_device(batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    if cfg.model_mode == "encoder_decoder" and cfg.patch_location != "encoder":
        logger.warning("encoder_decoder uses encoder-side patch; overriding patch_location=encoder")
        cfg.patch_location = "encoder"
    if cfg.model_mode == "decoder_only" and cfg.patch_location != "decoder":
        logger.warning("decoder_only uses decoder-side patch; overriding patch_location=decoder")
        cfg.patch_location = "decoder"

    if cfg.data_format != "xlong_pair":
        raise ValueError("MixFlow now supports only data_format='xlong_pair' for SASRec-style flow")
    ds_short, ds_long, ds_test_short, ds_test_long, xlong_meta = load_xlong_datasets(cfg)
    logger.info(
        "xlong meta: num_items=%s dropped_train=%s dropped_test=%s",
        xlong_meta["num_items"],
        xlong_meta["dropped_train"],
        xlong_meta["dropped_test"],
    )
    sampler_short = SequentialSampler(
        ds_short,
        batch_size=cfg.inner_batch_size,
        max_seq_length=cfg.L_recent,
        train_stride=cfg.train_stride,
        vectorized_neg_sampling=cfg.vectorized_neg_sampling,
        device=sampler_device,
    )
    sampler_long = SequentialSampler(
        ds_long,
        batch_size=cfg.outer_batch_size,
        max_seq_length=cfg.L_full,
        train_stride=cfg.train_stride,
        vectorized_neg_sampling=cfg.vectorized_neg_sampling,
        device=sampler_device,
    )
    iter_short_extra = iter(sampler_short)
    iter_long = iter(sampler_long)

    item_table = build_item_table(cfg, num_items=xlong_meta["num_items"]).to(device)
    patch_assigner = build_patch_assigner(cfg, ds_short, item_table)
    if xlong_meta is not None:
        expected_items = xlong_meta["num_items"]
        actual_items = item_table.weight.size(0) - 1
        if expected_items != actual_items:
            logger.warning(
                "Embedding item count mismatch: expected=%s actual=%s (train vocab vs embeddings)",
                expected_items,
                actual_items,
            )

    model = IDRecModel(
        item_table=item_table,
        hidden_dim=cfg.hidden_dim,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        mode=cfg.model_mode,
    ).to(device)
    # IMPORTANT: training uses stateless overrides (run_*_stateless); avoid calling model(...) directly during training.

    for p in item_table.parameters():
        p.requires_grad = cfg.train_item_embeddings

    enc_trainable, dec_trainable = [], []
    embedding_added = False
    if model.encoder is not None:
        for n, p in model.encoder.named_parameters():
            if n.startswith("item_embedding"):
                p.requires_grad = cfg.train_item_embeddings
                if cfg.train_item_embeddings and not embedding_added:
                    enc_trainable.append(n)
                    embedding_added = True
            else:
                p.requires_grad = True
                enc_trainable.append(n)
    if model.decoder is not None:
        for n, p in model.decoder.named_parameters():
            if n.startswith("item_embedding"):
                p.requires_grad = cfg.train_item_embeddings
                if cfg.train_item_embeddings and not embedding_added:
                    dec_trainable.append(n)
                    embedding_added = True
            else:
                p.requires_grad = True
                dec_trainable.append(n)

    theta = []
    if model.encoder is not None:
        theta += [dict(model.encoder.named_parameters())[n] for n in enc_trainable]
    if model.decoder is not None:
        theta += [dict(model.decoder.named_parameters())[n] for n in dec_trainable]

    theta_splits = {
        "enc": len(enc_trainable),
        "dec": len(enc_trainable) + len(dec_trainable),
    }

    base_params_enc, base_buffers_enc = ({} , {})
    base_params_dec, base_buffers_dec = ({} , {})
    if model.encoder is not None:
        base_params_enc, base_buffers_enc = collect_base_params_and_buffers(model.encoder)
    if model.decoder is not None:
        base_params_dec, base_buffers_dec = collect_base_params_and_buffers(model.decoder)

    patch = SoftPatch(cfg.L_soft, cfg.hidden_dim, cfg.num_patches).to(device)
    def patch_fn(batch: dict, _: torch.device, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return select_patch_for_batch(patch, patch_assigner, batch, mask)
    outer_params = list(patch.parameters())
    opt_eta = torch.optim.AdamW(outer_params, lr=cfg.outer_max_lr, weight_decay=cfg.weight_decay)
    best_patch_metric = -float("inf")
    best_patch_ckpt: Optional[str] = None
    saved_ckpts: deque[str] = deque()
    if keep_best_k > 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    outer_updates = max(1, math.ceil(cfg.max_iters / max(cfg.outer_update_every, 1)))
    steps_total = outer_updates
    if cfg.outer_scheduler_type == "cosine":
        warmup_scheduler = None
        scheduler_eta = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_eta,
            T_max=steps_total,
            eta_min=cfg.outer_min_lr,
        )
    elif cfg.outer_scheduler_type == "cosine_with_warmup":
        warmup_iters = min(cfg.outer_warmup_steps, steps_total)
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt_eta,
            start_factor=cfg.outer_warmup_start_lr / cfg.outer_max_lr,
            total_iters=warmup_iters,
        )
        cosine_steps = max(steps_total - warmup_iters, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_eta,
            T_max=cosine_steps,
            eta_min=cfg.outer_min_lr,
        )
        scheduler_eta = torch.optim.lr_scheduler.SequentialLR(
            opt_eta,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
        )
        warmup_scheduler = None
    else:
        scheduler_eta = None
        warmup_scheduler = None
    inner_opt = MomentumInner(theta, lr=cfg.inner_lr, momentum=cfg.inner_momentum)
    bce_loss = nn.BCEWithLogitsLoss()

    def inner_loss(
        theta_list: List[torch.Tensor],
        patch_emb: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_mask = input_ids != 0
        valid_batch = attention_mask.any(dim=1)
        if not valid_batch.any():
            zero = torch.zeros((), device=input_ids.device)
            for t in theta_list:
                zero = zero + t.sum() * 0.0
            if patch_emb is not None:
                zero = zero + patch_emb.sum() * 0.0
            return zero
        if not valid_batch.all():
            input_ids = input_ids[valid_batch]
            pos_ids = pos_ids[valid_batch]
            neg_ids = neg_ids[valid_batch]
            attention_mask = attention_mask[valid_batch]
            if user_ids is not None:
                user_ids = user_ids[valid_batch]
        enc_theta = theta_list[: theta_splits["enc"]]
        dec_theta = theta_list[theta_splits["enc"] : theta_splits["dec"]]
        override_enc = build_override(enc_trainable, enc_theta)
        override_dec = build_override(dec_trainable, dec_theta)

        patch_for_enc = patch_emb if cfg.patch_location == "encoder" else None
        patch_for_dec = patch_emb if cfg.patch_location == "decoder" else None

        if cfg.model_mode == "decoder_only":
            dec_out = run_decoder_stateless(
                decoder=model.decoder,
                base_params=base_params_dec,
                base_buffers=base_buffers_dec,
                override_params=override_dec,
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_output=None,
                encoder_mask=None,
                patch_emb=patch_for_dec,
            )
            hidden, attn = dec_out.hidden, dec_out.attention_mask
        elif cfg.model_mode == "encoder_decoder":
            enc_out = run_encoder_stateless(
                encoder=model.encoder,
                base_params=base_params_enc,
                base_buffers=base_buffers_enc,
                override_params=override_enc,
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_emb=patch_for_enc,
            )
            dec_out = run_decoder_stateless(
                decoder=model.decoder,
                base_params=base_params_dec,
                base_buffers=base_buffers_dec,
                override_params=override_dec,
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_output=enc_out.hidden,
                encoder_mask=enc_out.attention_mask,
                patch_emb=patch_for_dec,
            )
            hidden, attn = dec_out.hidden, dec_out.attention_mask
        else:
            raise ValueError(f"Unknown model_mode: {cfg.model_mode}")

        # Align to semantic-ids SASRec baseline: multi-position loss over all valid targets.
        pos_embs = item_table(pos_ids)
        neg_embs = item_table(neg_ids)
        if hidden.size(1) != pos_ids.size(1):
            hidden = hidden[:, -pos_ids.size(1) :, :]
        pos_logits = (hidden * pos_embs).sum(dim=-1)
        neg_logits = (hidden * neg_embs).sum(dim=-1)

        valid_mask = pos_ids != 0
        if not valid_mask.any():
            return pos_logits.sum() * 0.0

        pos_loss = bce_loss(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
        neg_loss = bce_loss(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))
        loss = pos_loss + neg_loss
        if DEBUG_NAN_CHECKS:
            _check_finite("inner/loss", loss)
        return loss

    def inner_loss_meta(
        theta_list: List[torch.Tensor],
        patch_emb: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return cfg.lambda_meta * inner_loss(
            theta_list, patch_emb, input_ids, pos_ids, neg_ids, user_ids
        )

    grad_fn = get_fwdrev_grad_fn_eta(inner_loss)
    grad_fn_meta = get_fwdrev_grad_fn_eta(inner_loss_meta)

    wandb_run = _init_wandb(cfg)

    # Baseline evaluation before any training to gauge performance with random patch.
    if cfg.run_fme_short_only and ds_short is not None and len(ds_short.users) > 0:
        # Align baseline config/datasets with final test for fair comparison (no patch).
        fme_cfg = FMEConfig(
            num_negatives=cfg.fme_test_num_negatives,
            ks=(10, 20),
            epochs=cfg.fme_test_epochs,
            lr=cfg.fme_lr,
            weight_decay=cfg.fme_weight_decay,
            patience=cfg.fme_patience,
            max_train_batches=cfg.fme_max_train_batches,
            max_val_batches=cfg.fme_max_val_batches,
            batch_size=cfg.fme_batch_size,
            train_stride=cfg.train_stride,
            show_progress=cfg.fme_show_progress,
        )

        def _builder():
            return IDRecModel(
                item_table=item_table,
                hidden_dim=cfg.hidden_dim,
                encoder_layers=cfg.encoder_layers,
                decoder_layers=cfg.decoder_layers,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                mode=cfg.model_mode,
            )

        init_metrics = run_eval_online_fme(
            patch=None,
            patch_fn=None,
            model_builder=_builder,
            item_table=item_table,
            train_dataset=ds_short,
            val_dataset=ds_test_short if ds_test_short is not None else ds_short,
            device=device,
            fme_cfg=fme_cfg,
        )
        msg = " ".join(f"{k}={v:.4f}" for k, v in init_metrics.items())
        logger.info("[FME short-only] %s", msg)
        _wandb_log(wandb_run, {f"fme_short_only/{k}": v for k, v in init_metrics.items()}, step=0)

    # Long-sequence upper bound baseline: same config as short-only, but train on ds_long.
    if cfg.run_fme_long_only and ds_long is not None and len(ds_long.users) > 0:
        fme_cfg = FMEConfig(
            num_negatives=cfg.fme_test_num_negatives,
            ks=(10, 20),
            epochs=cfg.fme_long_epochs if cfg.fme_long_epochs is not None else cfg.fme_test_epochs,
            lr=cfg.fme_lr,
            weight_decay=cfg.fme_weight_decay,
            patience=cfg.fme_patience,
            max_train_batches=cfg.fme_max_train_batches,
            max_val_batches=cfg.fme_max_val_batches,
            batch_size=cfg.fme_batch_size,
            train_stride=cfg.train_stride,
            show_progress=cfg.fme_show_progress,
        )

        def _builder():
            return IDRecModel(
                item_table=item_table,
                hidden_dim=cfg.hidden_dim,
                encoder_layers=cfg.encoder_layers,
                decoder_layers=cfg.decoder_layers,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                mode=cfg.model_mode,
            )

        long_metrics = run_eval_online_fme(
            patch=None,
            patch_fn=None,
            model_builder=_builder,
            item_table=item_table,
            train_dataset=ds_long,
            val_dataset=ds_test_long if ds_test_long is not None else ds_long,
            device=device,
            fme_cfg=fme_cfg,
        )
        msg = " ".join(f"{k}={v:.4f}" for k, v in long_metrics.items())
        logger.info("[FME long-only] %s", msg)
        _wandb_log(wandb_run, {f"fme_long_only/{k}": v for k, v in long_metrics.items()}, step=0)

    if not cfg.run_main_training:
        logger.info("run_main_training=False; skipping MixFlow training after baselines.")
        if wandb_run is not None:
            wandb_run.finish()
        return

    truncate_steps = max(0, cfg.meta_truncate_steps)
    recent_steps = deque(maxlen=truncate_steps) if truncate_steps > 0 else None
    if truncate_steps == 0 and cfg.outer_use_patch:
        logger.info("Meta truncate disabled (0); outer updates will train patch directly via outer_use_patch=True.")
    def _build_fresh_theta(reset_idx: int) -> List[torch.Tensor]:
        """Create a freshly initialized theta list (encoder/decoder params only)."""
        fresh_model = IDRecModel(
            item_table=item_table,
            hidden_dim=cfg.hidden_dim,
            encoder_layers=cfg.encoder_layers,
            decoder_layers=cfg.decoder_layers,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            mode=cfg.model_mode,
        ).to(device)
        if fresh_model.encoder is not None:
            for n, p in fresh_model.encoder.named_parameters():
                if n.startswith("item_embedding"):
                    p.requires_grad = cfg.train_item_embeddings
                else:
                    p.requires_grad = True
        if fresh_model.decoder is not None:
            for n, p in fresh_model.decoder.named_parameters():
                if n.startswith("item_embedding"):
                    p.requires_grad = cfg.train_item_embeddings
                else:
                    p.requires_grad = True

        theta_new: List[torch.Tensor] = []
        if fresh_model.encoder is not None:
            enc_params_fresh = dict(fresh_model.encoder.named_parameters())
            theta_new += [enc_params_fresh[n] for n in enc_trainable]
        if fresh_model.decoder is not None:
            dec_params_fresh = dict(fresh_model.decoder.named_parameters())
            theta_new += [dec_params_fresh[n] for n in dec_trainable]
        return theta_new

    inner_reset_count = 0

    def _reset_inner_state(reason: str) -> Tuple[List[torch.Tensor], MomentumInner]:
        """Reset theta and momentum state; keep patch/outer optimizers intact."""
        nonlocal inner_reset_count
        inner_reset_count += 1
        reset_seed = cfg.seed + inner_reset_count
        devices_to_fork = [device.index] if device.type == "cuda" and device.index is not None else None
        with torch.random.fork_rng(devices=devices_to_fork, enabled=True):
            torch.manual_seed(reset_seed)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(reset_seed)
            new_theta = _build_fresh_theta(reset_idx=inner_reset_count)
        new_inner_opt = MomentumInner(new_theta, lr=cfg.inner_lr, momentum=cfg.inner_momentum)
        if recent_steps is not None:
            recent_steps.clear()
        logger.info("Reset inner loop state (%s) with seed=%s (reset_idx=%s)", reason, reset_seed, inner_reset_count)
        return new_theta, new_inner_opt

    inner_steps_since_reset = 0
    outer_updates_since_reset = 0
    inner_step_count = 0

    it = 0
    last_outer_loss = None
    early_stopper = EarlyStopper(patience=cfg.early_stop_patience)
    stop_training = False
    for epoch in range(10**6):
        for batch in tqdm(sampler_short, desc=f"Epoch {epoch}", leave=False):
            it += 1
            if it > cfg.max_iters:
                break
            batch = _move_batch_to_device(batch)

            step_batches = [batch]
            for _ in range(cfg.inner_steps - 1):
                try:
                    extra_batch = next(iter_short_extra)
                except StopIteration:
                    iter_short_extra = iter(sampler_short)
                    extra_batch = next(iter_short_extra)
                step_batches.append(_move_batch_to_device(extra_batch))

            for step_batch in step_batches:
                patch_emb = patch_fn(step_batch, device)
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
                                "user_ids": step_batch.get("user_ids"),
                            },
                        )
                    )
                gflat = grad_fn(
                    theta,
                    patch_emb,
                    step_batch["input_ids"],
                    step_batch["pos_ids"],
                    step_batch["neg_ids"],
                    step_batch.get("user_ids") if step_batch.get("user_ids") is not None else None,
                )
                gflat = _clip_flat_grad(gflat, cfg.inner_grad_clip)
                if DEBUG_NAN_CHECKS:
                    _check_finite("inner/gflat", gflat)
                inner_opt.step(gflat)
                inner_step_count += 1
                inner_steps_since_reset += 1

            enough_history = True if recent_steps is None else len(recent_steps) >= truncate_steps
            do_outer = cfg.outer_update_every > 0 and (inner_step_count % cfg.outer_update_every == 0) and enough_history
            if do_outer:
                try:
                    batch_long = next(iter_long)
                except StopIteration:
                    iter_long = iter(sampler_long)
                    batch_long = next(iter_long)
                batch_long = _move_batch_to_device(batch_long)

                latest_w, latest_m = inner_opt.snapshot()
                if recent_steps is not None and truncate_steps > 0:
                    start_w, start_m, _ = recent_steps[0]
                    inner_opt.restore(start_w, start_m)

                    for _, _, step_batch in recent_steps:
                        patch_emb = patch_fn(step_batch, device)
                        gflat = grad_fn_meta(
                            theta,
                            patch_emb,
                            step_batch["input_ids"],
                            step_batch["pos_ids"],
                            step_batch["neg_ids"],
                            step_batch.get("user_ids") if step_batch.get("user_ids") is not None else None,
                        )
                        gflat = _clip_flat_grad(gflat, cfg.inner_grad_clip)
                        if DEBUG_NAN_CHECKS:
                            _check_finite("outer/gflat", gflat)
                        inner_opt.step(gflat)

                opt_eta.zero_grad(set_to_none=True)
                patch_for_outer = patch_fn(batch_long, device) if cfg.outer_use_patch else None
                loss_outer = cfg.lambda_direct * inner_loss(
                    theta,
                    patch_emb=patch_for_outer,
                    input_ids=batch_long["input_ids"],
                    pos_ids=batch_long["pos_ids"],
                    neg_ids=batch_long["neg_ids"],
                    user_ids=batch_long.get("user_ids") if batch_long.get("user_ids") is not None else None,
                )
                if DEBUG_NAN_CHECKS:
                    _check_finite("outer/loss", loss_outer)
                loss_outer.backward()
                if cfg.outer_grad_clip and cfg.outer_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(outer_params, cfg.outer_grad_clip)
                opt_eta.step()
                if scheduler_eta is not None:
                    scheduler_eta.step()

                outer_updates_since_reset += 1
                last_outer_loss = loss_outer.item()
                inner_opt.restore(latest_w, latest_m)

            reset_due_to_inner = cfg.inner_reset_every_inner > 0 and inner_steps_since_reset >= cfg.inner_reset_every_inner
            reset_due_to_outer = cfg.inner_reset_every_outer > 0 and outer_updates_since_reset >= cfg.inner_reset_every_outer
            if reset_due_to_inner or reset_due_to_outer:
                reason_bits = []
                if reset_due_to_inner:
                    reason_bits.append(f"inner_steps={inner_steps_since_reset}")
                if reset_due_to_outer:
                    reason_bits.append(f"outer_updates={outer_updates_since_reset}")
                theta, inner_opt = _reset_inner_state("; ".join(reason_bits))
                inner_steps_since_reset = 0
                outer_updates_since_reset = 0
                inner_step_count = 0

            if it % cfg.log_every == 0 and last_outer_loss is not None:
                logger.info("[it %06d] loss_outer=%.4f", it, last_outer_loss)
                _wandb_log(
                    wandb_run,
                    {
                        "loss/outer": last_outer_loss,
                        "iter": it,
                    },
                    step=it,
                )

            if cfg.fme_eval_every and cfg.fme_eval_every > 0 and it % cfg.fme_eval_every == 0:
                fme_cfg = FMEConfig(
                    num_negatives=cfg.fme_train_num_negatives,
                    ks=(10, 20),
                    epochs=cfg.fme_train_epochs,
                    lr=cfg.fme_lr,
                    weight_decay=cfg.fme_weight_decay,
                    patience=cfg.fme_patience,
                    max_train_batches=cfg.fme_max_train_batches,
                    max_val_batches=cfg.fme_max_val_batches,
                    batch_size=cfg.fme_batch_size,
                    train_stride=cfg.train_stride,
                    show_progress=cfg.fme_show_progress,
                )
                def _builder():
                    return IDRecModel(
                        item_table=item_table,
                        hidden_dim=cfg.hidden_dim,
                        encoder_layers=cfg.encoder_layers,
                        decoder_layers=cfg.decoder_layers,
                        num_heads=cfg.num_heads,
                        ffn_dim=cfg.ffn_dim,
                        dropout=cfg.dropout,
                        attn_dropout=cfg.attn_dropout,
                        mode=cfg.model_mode,
                )

                fme_metrics = run_eval_online_fme(
                    patch=None,
                    patch_fn=patch_fn,
                    model_builder=_builder,
                    item_table=item_table,
                    train_dataset=ds_short,
                    val_dataset=ds_short,
                    device=device,
                    fme_cfg=fme_cfg,
                )
                msg = " ".join(f"{k}={v:.4f}" for k, v in fme_metrics.items())
                logger.info("[FME it %06d] %s", it, msg)
                _wandb_log(wandb_run, {f"fme/{k}": v for k, v in fme_metrics.items()}, step=it)
                ndcg = fme_metrics.get("fme_NDCG@10", 0.0)
                if keep_best_k > 0 and ndcg > best_patch_metric + 1e-12:
                    best_patch_metric = ndcg
                    ckpt_name = f"patch_step{it}_ndcg{ndcg:.4f}.pt"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    _save_patch_checkpoint(ckpt_path, patch, cfg, it, fme_metrics)
                    saved_ckpts.append(ckpt_path)
                    while len(saved_ckpts) > keep_best_k:
                        old = saved_ckpts.popleft()
                        try:
                            os.remove(old)
                        except OSError:
                            pass
                    best_patch_ckpt = ckpt_path
                if early_stopper.update(ndcg):
                    logger.info("Early stop triggered at it %06d (NDCG@10=%.4f)", it, ndcg)
                    stop_training = True
                    break

        if stop_training or it > cfg.max_iters:
            break

    logger.info("Training finished.")
    if load_best_for_test and keep_best_k > 0 and best_patch_ckpt and os.path.exists(best_patch_ckpt):
        state = torch.load(best_patch_ckpt, map_location=device, weights_only=False)
        patch.load_state_dict(state["patch_state_dict"])
        best_step = state.get("step", "n/a")
        best_ndcg = state.get("metrics", {}).get("fme_NDCG@10", float("nan"))
        logger.info("Loaded best patch checkpoint from %s (step=%s, NDCG@10=%.4f)", best_patch_ckpt, best_step, best_ndcg)
    elif keep_best_k > 0:
        logger.info("Best patch checkpoint not found; using last patch parameters.")

    if run_final_test_eval and ds_test_short is not None:
        fme_cfg = FMEConfig(
            num_negatives=cfg.fme_test_num_negatives,
            ks=(10, 20),
            epochs=cfg.fme_test_epochs,
            lr=cfg.fme_lr,
            weight_decay=cfg.fme_weight_decay,
            patience=cfg.fme_patience,
            max_train_batches=cfg.fme_max_train_batches,
            max_val_batches=cfg.fme_max_val_batches,
            batch_size=cfg.fme_batch_size,
            train_stride=cfg.train_stride,
            show_progress=cfg.fme_show_progress,
        )

        def _builder():
            return IDRecModel(
                item_table=item_table,
                hidden_dim=cfg.hidden_dim,
                encoder_layers=cfg.encoder_layers,
                decoder_layers=cfg.decoder_layers,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                mode=cfg.model_mode,
        )

        test_metrics = run_eval_online_fme(
            patch=None,
            patch_fn=patch_fn,
            model_builder=_builder,
            item_table=item_table,
            train_dataset=ds_short,
            val_dataset=ds_test_short,
            device=device,
            fme_cfg=fme_cfg,
        )
        msg = " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        logger.info("[FME test] %s", msg)
        _wandb_log(wandb_run, {f"test/{k}": v for k, v in test_metrics.items()}, step=it)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    cfg = MixFlowIDConfig()
    main(cfg)
