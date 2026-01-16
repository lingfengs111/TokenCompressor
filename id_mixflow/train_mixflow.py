#!/usr/bin/env python3
"""
MixFlow training scaffold for ID-based recommender.

Design:
  - Inner loop: train encoder/decoder on short sequences + soft patch.
  - Outer loop: update patch using longer real sequences (no patch).
  - Meta-gradients: reuse core/mixflow.py (eta-aware gradients).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple
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
    load_item_embeddings_from_npz,
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


@dataclass
class MixFlowIDConfig:
    # Data
    data_format: str = "xlong_pair"  # only xlong_pair supported for SASRec-style flow
    xlong_train_path: str = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/train_corpus_total_dual.txt"
    xlong_test_path: str = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_corpus_total_dual.txt"
    item_embeddings_path: Optional[str] = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/item_embeddings_sas128_len500.npy"
    embeddings_have_pad: bool = False
    L_recent: int = 64
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
    L_soft: int = 128
    num_patches: int = 30
    patch_routing: str = "kmeans_per_user"  # single | random_per_user | kmeans_per_user
    patch_seed: int = 2024
    kmeans_max_iters: int = 25
    user_embedding_method: str = "mean"  # mean | exp_decay
    user_embedding_decay: float = 1.0

    # Training
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    inner_steps: int = 5
    inner_lr: float = 1e-4
    inner_momentum: float = 0.9
    inner_grad_clip: float = 0.0
    outer_max_lr: float = 1e-4
    outer_min_lr: float = 1e-5
    outer_scheduler_type: str = "cosine_with_warmup"  # "cosine" | "cosine_with_warmup"
    outer_warmup_steps: int = 30
    outer_warmup_start_lr: float = 1e-7
    outer_grad_clip: float = 0.1
    weight_decay: float = 0.0
    meta_truncate_steps: int = 4 # inner steps,0 means no inner steps 
    outer_update_every: int = 3
    outer_warmup_steps: int = 30
    max_iters: int = 1000*10
    log_every: int = 50
    debug_nan_checks: bool = True
    # Gradient weighting
    lambda_direct: float = 1.0  # scales direct outer loss gradients wrt patch
    lambda_meta: float = 1.0    # scales meta-gradients wrt patch

    # Logging
    wandb_enabled: bool = True
    wandb_project: str = "mixflow-id2"
    wandb_run_name: Optional[str] = "30patch_kmeans_1000neg_decoder_only_recent128_full500_recent64"

    # Evaluation (FME)
    fme_eval_every: int = 1000
    fme_epochs: int = 50
    fme_lr: float = 1e-4
    fme_weight_decay: float = 0.0
    fme_patience: int = 5
    early_stop_patience: int = 5
    fme_num_negatives: int = 1000
    fme_batch_size: int = 256
    fme_max_train_batches: Optional[int] = 200
    fme_max_val_batches: Optional[int] = 50
    final_test_eval: bool = True
    outer_use_patch: bool = True # 外循环是否用patch
    fme_show_progress: bool = True

    def log_config(self):
        logger.info("=== MixFlow ID Config ===")
        logger.info("data_format: %s", self.data_format)
        logger.info("xlong_train_path: %s", self.xlong_train_path)
        logger.info("xlong_test_path: %s", self.xlong_test_path)
        logger.info("item_embeddings_path: %s", self.item_embeddings_path)
        logger.info("embeddings_have_pad: %s", self.embeddings_have_pad)
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
        logger.info("inner_grad_clip: %s | outer_grad_clip: %s", self.inner_grad_clip, self.outer_grad_clip)
        logger.info("wandb_enabled: %s | project: %s | run_name: %s", self.wandb_enabled, self.wandb_project, self.wandb_run_name)
        logger.info(
            "fme_eval_every: %s | fme_epochs: %s | fme_lr: %s | fme_patience: %s",
            self.fme_eval_every,
            self.fme_epochs,
            self.fme_lr,
            self.fme_patience,
        )
        logger.info(
            "fme_batch_size: %s | fme_num_negatives: %s | fme_show_progress: %s",
            self.fme_batch_size,
            self.fme_num_negatives,
            self.fme_show_progress,
        )
        logger.info("final_test_eval: %s", self.final_test_eval)
        logger.info("outer_use_patch: %s", self.outer_use_patch)
        logger.info("early_stop_patience: %s", self.early_stop_patience)
        logger.info("=========================")


def load_xlong_datasets(cfg: MixFlowIDConfig) -> Tuple[SequenceDataset, SequenceDataset, Optional[SequenceDataset], Dict[str, int]]:
    train_samples = load_xlong_samples(cfg.xlong_train_path)
    item_to_id = _build_xlong_item_map(train_samples)
    train_sequences, train_user_ids, dropped_train = _map_xlong_sequences(train_samples, item_to_id)

    test_samples = load_xlong_samples(cfg.xlong_test_path) if cfg.xlong_test_path else []
    test_sequences, test_user_ids, dropped_test = _map_xlong_sequences(test_samples, item_to_id) if test_samples else ([], [], 0)

    ds_short = SequenceDataset(train_sequences, train_user_ids, cfg.L_recent, item_to_id)
    ds_long = SequenceDataset(train_sequences, train_user_ids, cfg.L_full, item_to_id)
    ds_test = SequenceDataset(test_sequences, test_user_ids, cfg.L_recent, item_to_id) if test_sequences else None
    meta = {
        "num_items": len(item_to_id),
        "dropped_train": dropped_train,
        "dropped_test": dropped_test,
    }
    return ds_short, ds_long, ds_test, meta


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
    if cfg.num_patches <= 1 or cfg.patch_routing == "single":
        logger.info("Using single patch (no routing).")
        return PatchAssigner(1, {})
    user_to_seqs = _build_user_sequence_map(dataset)
    if not user_to_seqs:
        logger.warning("No user sequences found; falling back to single patch.")
        return PatchAssigner(1, {})
    logger.info("Building patch routing for %s users (strategy=%s)...", len(user_to_seqs), cfg.patch_routing)
    if cfg.patch_routing == "random_per_user":
        mapping = _random_patch_assignments(list(user_to_seqs.keys()), cfg.num_patches, cfg.patch_seed)
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
            mapping = _random_patch_assignments(list(user_to_seqs.keys()), cfg.num_patches, cfg.patch_seed)
        else:
            mapping = _kmeans_assignments(user_embs, cfg.num_patches, cfg.kmeans_max_iters, cfg.patch_seed)
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
    has_pad = cfg.embeddings_have_pad
    if cfg.item_embeddings_path:
        if cfg.item_embeddings_path.endswith(".npz"):
            emb_arr = load_item_embeddings_from_indexed_npz(cfg.item_embeddings_path)
            if emb_arr is not None:
                embeddings = torch.from_numpy(emb_arr).float()
                has_pad = True
        else:
            embeddings = load_item_embeddings(cfg.item_embeddings_path)
    elif cfg.data_path.endswith(".npz"):
        arr = load_item_embeddings_from_npz(cfg.data_path)
        if arr is not None:
            embeddings = torch.from_numpy(arr).float()

    if embeddings is None:
        if num_items is None:
            raise ValueError("num_items is required when building fresh embeddings")
        emb_dim = cfg.hidden_dim
        return ItemEmbeddingTable(num_items=num_items, embedding_dim=emb_dim, trainable=True)
    if cfg.debug_nan_checks:
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
    device = torch.device(cfg.device)

    if cfg.model_mode == "encoder_decoder" and cfg.patch_location != "encoder":
        logger.warning("encoder_decoder uses encoder-side patch; overriding patch_location=encoder")
        cfg.patch_location = "encoder"
    if cfg.model_mode == "decoder_only" and cfg.patch_location != "decoder":
        logger.warning("decoder_only uses decoder-side patch; overriding patch_location=decoder")
        cfg.patch_location = "decoder"

    if cfg.data_format != "xlong_pair":
        raise ValueError("MixFlow now supports only data_format='xlong_pair' for SASRec-style flow")
    ds_short, ds_long, ds_test, xlong_meta = load_xlong_datasets(cfg)
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
    )
    sampler_long = SequentialSampler(
        ds_long,
        batch_size=cfg.outer_batch_size,
        max_seq_length=cfg.L_full,
        train_stride=cfg.train_stride,
    )
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
    steps_total = cfg.max_iters
    if cfg.outer_scheduler_type == "cosine":
        warmup_scheduler = None
        scheduler_eta = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_eta,
            T_max=steps_total,
            eta_min=cfg.outer_min_lr,
        )
    elif cfg.outer_scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt_eta,
            start_factor=cfg.outer_warmup_start_lr / cfg.outer_max_lr,
            total_iters=cfg.outer_warmup_steps,
        )
        cosine_steps = max(steps_total - cfg.outer_warmup_steps, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_eta,
            T_max=cosine_steps,
            eta_min=cfg.outer_min_lr,
        )
        scheduler_eta = torch.optim.lr_scheduler.SequentialLR(
            opt_eta,
            schedulers=[warmup, cosine],
            milestones=[cfg.outer_warmup_steps],
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
        if cfg.debug_nan_checks:
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
    if ds_short is not None and len(ds_short.users) > 0:
        fme_cfg = FMEConfig(
            num_negatives=cfg.fme_num_negatives,
            ks=(10, 20),
            epochs=cfg.fme_epochs,
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
            patch=patch.select(None),
            patch_fn=patch_fn,
            model_builder=_builder,
            item_table=item_table,
            train_dataset=ds_short,
            val_dataset=ds_short,
            device=device,
            fme_cfg=fme_cfg,
        )
        msg = " ".join(f"{k}={v:.4f}" for k, v in init_metrics.items())
        logger.info("[FME init] %s", msg)
        _wandb_log(wandb_run, {f"fme_init/{k}": v for k, v in init_metrics.items()}, step=0)

    truncate_steps = cfg.meta_truncate_steps if cfg.meta_truncate_steps > 0 else 1
    recent_steps = deque(maxlen=truncate_steps)
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

            patch_emb = patch_fn(batch, device)
            for _ in range(cfg.inner_steps):
                if truncate_steps > 0:
                    w_state, m_state = inner_opt.snapshot()
                    recent_steps.append(
                        (
                            w_state,
                            m_state,
                            {
                                "input_ids": batch["input_ids"],
                                "pos_ids": batch["pos_ids"],
                                "neg_ids": batch["neg_ids"],
                                "user_ids": batch.get("user_ids"),
                            },
                        )
                    )
                gflat = grad_fn(
                    theta,
                    patch_emb,
                    batch["input_ids"].to(device),
                    batch["pos_ids"].to(device),
                    batch["neg_ids"].to(device),
                    batch.get("user_ids").to(device) if batch.get("user_ids") is not None else None,
                )
                gflat = _clip_flat_grad(gflat, cfg.inner_grad_clip)
                if cfg.debug_nan_checks:
                    _check_finite("inner/gflat", gflat)
                inner_opt.step(gflat)
                inner_step_count += 1

            do_outer = cfg.outer_update_every > 0 and (inner_step_count % cfg.outer_update_every == 0)
            if do_outer and len(recent_steps) >= truncate_steps:
                try:
                    batch_long = next(iter_long)
                except StopIteration:
                    iter_long = iter(sampler_long)
                    batch_long = next(iter_long)

                latest_w, latest_m = inner_opt.snapshot()
                start_w, start_m, _ = recent_steps[0]
                inner_opt.restore(start_w, start_m)

                for _, _, step_batch in recent_steps:
                    patch_emb = patch_fn(step_batch, device)
                    gflat = grad_fn_meta(
                        theta,
                        patch_emb,
                        step_batch["input_ids"].to(device),
                        step_batch["pos_ids"].to(device),
                        step_batch["neg_ids"].to(device),
                        step_batch.get("user_ids").to(device) if step_batch.get("user_ids") is not None else None,
                    )
                    gflat = _clip_flat_grad(gflat, cfg.inner_grad_clip)
                    if cfg.debug_nan_checks:
                        _check_finite("outer/gflat", gflat)
                    inner_opt.step(gflat)

                opt_eta.zero_grad(set_to_none=True)
                patch_for_outer = patch_fn(batch_long, device) if cfg.outer_use_patch else None
                loss_outer = cfg.lambda_direct * inner_loss(
                    theta,
                    patch_emb=patch_for_outer,
                    input_ids=batch_long["input_ids"].to(device),
                    pos_ids=batch_long["pos_ids"].to(device),
                    neg_ids=batch_long["neg_ids"].to(device),
                    user_ids=batch_long.get("user_ids").to(device) if batch_long.get("user_ids") is not None else None,
                )
                if cfg.debug_nan_checks:
                    _check_finite("outer/loss", loss_outer)
                loss_outer.backward()
                if cfg.outer_grad_clip and cfg.outer_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(outer_params, cfg.outer_grad_clip)
                opt_eta.step()
                if scheduler_eta is not None:
                    scheduler_eta.step()

                last_outer_loss = loss_outer.item()
                inner_opt.restore(latest_w, latest_m)

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
                    num_negatives=cfg.fme_num_negatives,
                    ks=(10, 20),
                    epochs=cfg.fme_epochs,
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
                    patch=patch.select(None),
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
                if early_stopper.update(ndcg):
                    logger.info("Early stop triggered at it %06d (NDCG@10=%.4f)", it, ndcg)
                    stop_training = True
                    break

        if stop_training or it > cfg.max_iters:
            break

    logger.info("Training finished.")
    if cfg.final_test_eval and ds_test is not None:
        fme_cfg = FMEConfig(
            num_negatives=cfg.fme_num_negatives,
            ks=(10, 20),
            epochs=cfg.fme_epochs,
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
            patch=patch.select(None),
            patch_fn=patch_fn,
            model_builder=_builder,
            item_table=item_table,
            train_dataset=ds_short,
            val_dataset=ds_test,
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
