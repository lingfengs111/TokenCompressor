#!/usr/bin/env python3
"""Train ID-based recommender without MixFlow (baseline).

This file now mirrors the data loading, negative sampling, and loss logic used in
`codes/semantic-ids-llm-main/src/train_sasrec.py`, adapted for xlong data and
supporting a configurable training stride.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import argparse
import inspect
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb

from data import (
    _build_xlong_item_map,
    _map_xlong_sequences,
    load_item_embeddings_from_indexed_npz,
    load_xlong_samples,
)
from evaluate import EarlyStopper
from model import IDRecModel, ItemEmbeddingTable, load_item_embeddings

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from core.logger import setup_logger

logger = setup_logger("train-id", log_to_file=True)


@dataclass
class IDTrainConfig:
    # Data
    data_format: str = "xlong_pair"  # Only xlong is supported for SASRec-style training here
    xlong_train_path: Optional[str] = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/train_corpus_total_dual.txt"
    xlong_test_path: Optional[str] = "/home/lingfengs111/codes/soft_patch_training/data/pure_id-based/xlong2018/test_corpus_total_dual.txt"
    item_embeddings_path: Optional[str] = "/home/lingfengs111/codes/soft_patch_training/data/text-based/taobao-mm/2k_history_item_emb.npz"
    embeddings_have_pad: bool = False
    max_seq_len: int = 128
    train_stride: int = 5  # train every stride positions
    batch_size: int = 1024
    num_workers: int = 0  # unused in SASRec-style sampler
    item_embedding_dim: Optional[int] = None # if None, defaults to hidden_dim
    train_item_embeddings: bool = True

    # Model
    hidden_dim: int = 128
    encoder_layers: int = 3
    decoder_layers: int = 3
    num_heads: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    attn_dropout: float = 0.1
    model_mode: str = "decoder_only"  # "encoder-only", "decoder-only", "encoder-decoder"

    # Training
    device: str = "cuda:2" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 200
    max_steps: Optional[int] = None  # cap total optimizer steps (for budget-matched runs)
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    scheduler_type: str = "cosine_with_warmup"  # "cosine" or "cosine_with_warmup"
    warmup_steps: int = 50
    warmup_start_lr: float = 1e-8
    weight_decay: float = 0.0
    log_every: int = 100

    # Eval
    eval_every: int = 400
    eval_negatives: int = 1000  # number of negatives per user for val/test ranking
    eval_batch_size: int = 256
    early_stop_patience: int = 5

    # Logging
    wandb_enabled: bool = True
    wandb_project: str = "id-rec2"
    wandb_run_name: Optional[str] = "seq128_neg1000_decoder_only"
    embedding_save_path: Optional[str] = None

    def log_config(self):
        emb_path = _resolve_embedding_path(self)
        logger.info("=== ID Train Config ===")
        logger.info("data_format: %s", self.data_format)
        logger.info("xlong_train_path: %s", self.xlong_train_path)
        logger.info("xlong_test_path: %s", self.xlong_test_path)
        logger.info("item_embeddings_path: %s", self.item_embeddings_path)
        logger.info("item_embedding_dim: %s | train_item_embeddings: %s", self.item_embedding_dim, self.train_item_embeddings)
        logger.info("embeddings_have_pad: %s", self.embeddings_have_pad)
        logger.info("max_seq_len: %s", self.max_seq_len)
        logger.info("train_stride: %s", self.train_stride)
        logger.info("batch_size: %s | num_workers: %s", self.batch_size, self.num_workers)
        logger.info("hidden_dim: %s | heads: %s", self.hidden_dim, self.num_heads)
        logger.info("encoder_layers: %s | decoder_layers: %s", self.encoder_layers, self.decoder_layers)
        logger.info("ffn_dim: %s | dropout: %s | attn_dropout: %s", self.ffn_dim, self.dropout, self.attn_dropout)
        logger.info("model_mode: %s", self.model_mode)
        logger.info(
            "device: %s | num_epochs: %s | lr: %s | wd: %s | max_steps: %s",
            self.device,
            self.num_epochs,
            self.max_learning_rate,
            self.weight_decay,
            self.max_steps,
        )
        logger.info(
            "lr schedule: %s | max_lr: %s | min_lr: %s | warmup_steps: %s | warmup_start_lr: %s",
            self.scheduler_type,
            self.max_learning_rate,
            self.min_learning_rate,
            self.warmup_steps,
            self.warmup_start_lr,
        )
        logger.info("eval_every: %s | eval_negatives: %s | eval_batch_size: %s", self.eval_every, self.eval_negatives, self.eval_batch_size)
        logger.info("early_stop_patience: %s", self.early_stop_patience)
        logger.info("embedding_save_path: %s", emb_path)
        logger.info("========================")


class SequenceDataset:
    """Sequence dataset following the SASRec-style setup, using xlong data."""

    def __init__(
        self,
        sequences: List[List[int]],
        user_ids: List[int],
        max_seq_length: int,
        item_to_id: Dict[int, int],
    ):
        self.max_seq_length = max_seq_length
        self.item_to_id = item_to_id
        self.id_to_item = {v: k for k, v in item_to_id.items()}

        # Filter out users with too-short histories and build mappings
        self.users: List[int] = []
        self.user_seq: Dict[int, List[int]] = {}
        item_set = set()
        for user, seq in zip(user_ids, sequences):
            if len(seq) < 3:
                continue
            self.users.append(user)
            self.user_seq[user] = seq
            item_set.update(seq)

        self.max_item = len(item_to_id)
        self.num_users = len(self.users)
        self.num_items = len(item_set)
        self.all_items = list(range(1, self.max_item + 1))

        if self.num_users == 0:
            raise ValueError("No valid users found after filtering sequences with length >= 3")

        logger.info(
            "Loaded %d users (%d items) for SASRec-style training | avg seq len=%.2f",
            self.num_users,
            self.num_items,
            np.mean([len(self.user_seq[u]) for u in self.users]),
        )


class SequentialSampler:
    """
    Sampler that mirrors the SASRec training procedure with optional stride.
    Generates input/positive/negative tensors for each user sequence.
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int,
        max_seq_length: int,
        train_stride: int = 1,
        vectorized_neg_sampling: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_item = dataset.max_item
        self.train_stride = max(train_stride, 1)
        # When True, use a batched negative sampler that resamples collisions; otherwise keep the original per-position sampler.
        self.vectorized_neg_sampling = vectorized_neg_sampling
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.valid_user_seqs: List[Tuple[int, List[int]]] = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            if len(seq) > 1:
                # Exclude only the last item for validation/test target
                self.valid_user_seqs.append((user, seq[:-1]))

    @staticmethod
    def sample_negative_item(min_id: int, max_id_exclusive: int, seen_items: set) -> int:
        item_id = np.random.randint(min_id, max_id_exclusive)
        while item_id in seen_items:
            item_id = np.random.randint(min_id, max_id_exclusive)
        return item_id

    def __iter__(self):
        indices = np.random.permutation(len(self.valid_user_seqs))

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data = [self.valid_user_seqs[idx] for idx in batch_indices]

            actual_batch_size = len(batch_data)
            seq_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long, device=self.device)
            pos_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long, device=self.device)
            neg_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long, device=self.device)
            keep_mask = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.bool, device=self.device)

            seen_lists = []  # used only when vectorized_neg_sampling is True

            for idx, (user, seq) in enumerate(batch_data):
                seq_len = min(len(seq), self.max_seq_length)
                if seq_len < 1:
                    continue

                if len(seq) > self.max_seq_length:
                    seq = seq[-self.max_seq_length :]
                    seq_len = self.max_seq_length

                seq_tensors[idx, -seq_len:] = torch.tensor(seq[:seq_len], device=self.device)

                # Determine which positions to train on using stride (counting from the end).
                keep_positions = set(range(seq_len - 1, -1, -self.train_stride))
                keep_positions.add(seq_len - 1)  # Always keep the last position

                for pos in range(seq_len):
                    if pos not in keep_positions:
                        continue
                    keep_mask[idx, -seq_len + pos] = True
                    if pos < seq_len - 1:
                        pos_tensors[idx, -seq_len + pos] = seq[pos + 1]
                    else:
                        full_seq = self.dataset.user_seq[user]
                        next_idx = len(seq)
                        if next_idx < len(full_seq):
                            pos_tensors[idx, -1] = full_seq[next_idx]

                if self.vectorized_neg_sampling:
                    seen_lists.append(list(set(self.dataset.user_seq[user])))
                else:
                    seen_set = set(self.dataset.user_seq[user])
                    for pos in range(seq_len):
                        if pos not in keep_positions:
                            continue
                        neg_item = self.sample_negative_item(1, self.max_item + 1, seen_set)
                        neg_tensors[idx, -seq_len + pos] = neg_item

            if self.vectorized_neg_sampling and actual_batch_size > 0:
                max_seen = max((len(s) for s in seen_lists), default=0)
                if max_seen > 0:
                    seen_pad = torch.zeros((actual_batch_size, max_seen), dtype=torch.long, device=self.device)
                    for j, seen in enumerate(seen_lists):
                        if seen:
                            seen_pad[j, : len(seen)] = torch.tensor(seen[:max_seen], device=self.device)

                    neg = torch.randint(1, self.max_item + 1, (actual_batch_size, self.max_seq_length), dtype=torch.long, device=self.device)
                    neg = neg * keep_mask  # keep non-training positions at 0
                    collision = keep_mask & (neg.unsqueeze(-1) == seen_pad.unsqueeze(1)).any(-1)
                    while collision.any():
                        num_collision = int(collision.sum().item())
                        n_new = torch.randint(1, self.max_item + 1, (num_collision,), dtype=torch.long, device=self.device)
                        neg[collision] = n_new
                        collision = keep_mask & (neg.unsqueeze(-1) == seen_pad.unsqueeze(1)).any(-1)
                    neg_tensors = neg
                else:
                    neg_tensors = torch.randint(
                        1, self.max_item + 1, (actual_batch_size, self.max_seq_length), dtype=torch.long, device=self.device
                    ) * keep_mask

            user_ids = torch.tensor([u for u, _ in batch_data], dtype=torch.long, device=self.device)
            yield {"input_ids": seq_tensors, "pos_ids": pos_tensors, "neg_ids": neg_tensors, "user_ids": user_ids}

    def __len__(self):
        return (len(self.valid_user_seqs) + self.batch_size - 1) // self.batch_size


def predict_scores(
    model: IDRecModel,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    attention_mask = input_ids != 0
    hidden, attn = model(input_ids, attention_mask=attention_mask, patch_emb=None)
    final_hidden = hidden[:, -1, :]
    candidate_embs = model.item_table(candidate_ids)
    scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)
    return scores


def evaluate(
    model: IDRecModel,
    dataset: SequenceDataset,
    mode: str = "val",
    batch_size: int = 256,
    num_negatives: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    model.eval()
    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_users = 0

    users = dataset.users
    for batch_start in range(0, len(users), batch_size):
        batch_users = users[batch_start : batch_start + batch_size]
        batch_seqs: List[List[int]] = []
        batch_targets: List[int] = []
        batch_valid_mask: List[bool] = []

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
            else:
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

        candidates_list: List[torch.Tensor] = []
        for i, (user, target) in enumerate(zip(batch_users, batch_targets)):
            if not batch_valid_mask[i]:
                candidates_list.append(torch.zeros(num_negatives + 1, dtype=torch.long))
                continue

            candidates = [target]
            seen_items = set(dataset.user_seq[user])
            while len(candidates) < num_negatives + 1:
                neg_item = np.random.randint(1, dataset.max_item + 1)
                if neg_item not in seen_items:
                    candidates.append(neg_item)

            candidates_list.append(torch.tensor(candidates))

        candidates_tensor = torch.stack(candidates_list).to(device)

        with torch.no_grad():
            valid_indices = [i for i, valid in enumerate(batch_valid_mask) if valid]
            if not valid_indices:
                continue
            valid_input = input_tensor[valid_indices]
            valid_candidates = candidates_tensor[valid_indices]
            scores = predict_scores(model, valid_input, valid_candidates)

        _, indices = torch.sort(scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1].cpu().numpy() + 1

        for rank in ranks:
            valid_users += 1
            if rank <= 10:
                hr_sum += 1
                ndcg_sum += 1 / np.log2(rank + 1)

    ndcg_10 = ndcg_sum / valid_users if valid_users > 0 else 0.0
    hr_10 = hr_sum / valid_users if valid_users > 0 else 0.0
    logger.info(f"Evaluated on {valid_users} users | NDCG@10={ndcg_10:.4f} HR@10={hr_10:.4f}")
    return {"ndcg@10": ndcg_10, "hr@10": hr_10}


def _init_wandb(cfg: IDTrainConfig):
    if not cfg.wandb_enabled:
        return None
    run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
    if run is not None:
        wandb.define_metric("step")
        wandb.define_metric("epoch")
        wandb.define_metric("loss/*", step_metric="step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")
        wandb.define_metric("test/*", step_metric="step")
    return run


_wandb_last_step = -1


def _wandb_log(run, metrics: Dict[str, float], step: int, commit: bool = True):
    if run is not None:
        global _wandb_last_step
        if step <= _wandb_last_step:
            step = _wandb_last_step + 1  # keep wandb step monotonically increasing
        _wandb_last_step = step
        run.log(metrics, step=step, commit=commit)

def _resolve_embedding_path(cfg: IDTrainConfig) -> str:
    fname = f"id_item_embeddings_seq{cfg.max_seq_len}_dec{cfg.decoder_layers}_{cfg.model_mode}.npy"
    return cfg.embedding_save_path or os.path.join(ROOT_DIR, "artifacts", fname)


def build_item_table(cfg: IDTrainConfig, num_items: Optional[int] = None) -> ItemEmbeddingTable:
    use_pretrained_path = cfg.item_embeddings_path if cfg.data_format != "xlong_pair" else None
    trainable = cfg.train_item_embeddings or cfg.data_format == "xlong_pair" or use_pretrained_path is None
    embeddings = None
    has_pad = cfg.embeddings_have_pad
    if use_pretrained_path:
        if use_pretrained_path.endswith(".npz"):
            emb_arr = load_item_embeddings_from_indexed_npz(use_pretrained_path)
            if emb_arr is not None:
                embeddings = torch.from_numpy(emb_arr).float()
                has_pad = True
        else:
            embeddings = load_item_embeddings(use_pretrained_path)
    if embeddings is None:
        if num_items is None:
            raise ValueError("num_items is required when building a fresh item table without embeddings")
        emb_dim = cfg.item_embedding_dim or cfg.hidden_dim
        return ItemEmbeddingTable(num_items=num_items, embedding_dim=emb_dim, trainable=True)

    if has_pad:
        table = ItemEmbeddingTable(
            num_items=embeddings.size(0) - 1,
            embedding_dim=embeddings.size(1),
            trainable=trainable,
        )
        with torch.no_grad():
            table.embedding.weight.copy_(embeddings)
    else:
        table = ItemEmbeddingTable.from_pretrained(embeddings, trainable=trainable, pad_zero=True)
    return table


def load_xlong_datasets(cfg: IDTrainConfig) -> Tuple[SequenceDataset, Optional[SequenceDataset], Dict[str, int]]:
    train_samples = load_xlong_samples(cfg.xlong_train_path)
    item_to_id = _build_xlong_item_map(train_samples)
    train_sequences, train_user_ids, dropped_train = _map_xlong_sequences(train_samples, item_to_id)

    test_samples = load_xlong_samples(cfg.xlong_test_path) if cfg.xlong_test_path else []
    if test_samples:
        test_sequences, test_user_ids, dropped_test = _map_xlong_sequences(test_samples, item_to_id)
    else:
        test_sequences, test_user_ids, dropped_test = [], [], 0

    train_dataset = SequenceDataset(train_sequences, train_user_ids, cfg.max_seq_len, item_to_id)
    test_dataset = SequenceDataset(test_sequences, test_user_ids, cfg.max_seq_len, item_to_id) if test_sequences else None
    meta = {
        "num_items": len(item_to_id),
        "dropped_train": dropped_train,
        "dropped_test": dropped_test,
    }
    return train_dataset, test_dataset, meta


def main(cfg: IDTrainConfig) -> None:
    cfg.log_config()
    device = torch.device(cfg.device)

    if cfg.data_format != "xlong_pair":
        raise ValueError("SASRec-style training currently supports only data_format='xlong_pair'")
    if not cfg.xlong_train_path:
        raise ValueError("xlong_train_path is required for xlong_pair data_format")

    train_dataset, test_dataset, meta = load_xlong_datasets(cfg)
    vocab_size = meta["num_items"]
    if cfg.item_embeddings_path:
        logger.info("data_format='xlong_pair': ignoring item_embeddings_path and training embeddings from scratch")
    logger.info(
        "Loaded xlong data | num_items=%d dropped_train=%d dropped_test=%d",
        meta["num_items"],
        meta["dropped_train"],
        meta["dropped_test"],
    )

    item_table = build_item_table(cfg, num_items=vocab_size).to(device)

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

    sampler = SequentialSampler(
        train_dataset,
        batch_size=cfg.batch_size,
        max_seq_length=cfg.max_seq_len,
        train_stride=cfg.train_stride,
    )
    steps_per_epoch = len(sampler)
    planned_steps = cfg.num_epochs * steps_per_epoch
    total_steps = cfg.max_steps if cfg.max_steps is not None else planned_steps
    logger.info(
        "Training for %d epochs (%d steps/epoch); total steps=%s (planned=%s, max_steps override=%s)",
        cfg.num_epochs,
        steps_per_epoch,
        f"{total_steps:,}",
        f"{planned_steps:,}",
        cfg.max_steps,
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.max_learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        fused=use_fused,
    )

    if cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=cfg.min_learning_rate,
        )
    elif cfg.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.warmup_start_lr / cfg.max_learning_rate,
            total_iters=cfg.warmup_steps,
        )
        cosine_steps = max(total_steps - cfg.warmup_steps, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=cfg.min_learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.warmup_steps],
        )
    else:
        scheduler = None

    bce_loss = nn.BCEWithLogitsLoss()

    wandb_run = _init_wandb(cfg)

    step = 0
    early_stopper = EarlyStopper(patience=cfg.early_stop_patience)
    stop_training = False
    pbar = tqdm(total=total_steps)
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        for batch in sampler:
            t0 = time.time()
            step += 1
            if step > total_steps:
                stop_training = True
                break
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)

            attention_mask = input_ids != 0
            hidden, attn = model(input_ids, attention_mask=attention_mask, patch_emb=None)

            pos_embs = model.item_table(pos_ids)
            neg_embs = model.item_table(neg_ids)

            pos_logits = (hidden * pos_embs).sum(dim=-1)
            neg_logits = (hidden * neg_embs).sum(dim=-1)

            valid_mask = pos_ids != 0
            loss_value = 0.0
            pos_loss_value = 0.0
            neg_loss_value = 0.0

            if valid_mask.any():
                pos_loss = bce_loss(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask]))
                neg_loss = bce_loss(neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask]))
                loss = pos_loss + neg_loss
                loss_value = loss.item()
                pos_loss_value = pos_loss.item()
                neg_loss_value = neg_loss.item()
                loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            pbar.update(1)
            current_lr = optimizer.param_groups[0]["lr"]

            if step == 1 or step % cfg.log_every == 0:
                t1 = time.time()
                batch_time_ms = (t1 - t0) * 1000
                samples_per_second = cfg.batch_size / max(t1 - t0, 1e-8)
                logger.info(
                    "[step %06d] loss=%.4f (pos=%.4f neg=%.4f) lr=%.2e time=%.0fms samples/s=%.0f",
                    step,
                    loss_value,
                    pos_loss_value,
                    neg_loss_value,
                    current_lr,
                    batch_time_ms,
                    samples_per_second,
                )
                _wandb_log(
                    wandb_run,
                    {
                        "loss/total": loss_value,
                        "loss/pos": pos_loss_value,
                        "loss/neg": neg_loss_value,
                        "train/learning_rate": current_lr,
                        "train/batch_time_ms": batch_time_ms,
                        "train/samples_per_second": samples_per_second,
                        "step": step,
                        "epoch": epoch,
                    },
                    step=step,
                )

            if cfg.eval_every and step % cfg.eval_every == 0:
                logger.info("Running validation at step %d", step)
                val_metrics = evaluate(
                    model,
                    train_dataset,
                    mode="val",
                    batch_size=cfg.eval_batch_size,
                    num_negatives=cfg.eval_negatives,
                    device=device,
                )
                _wandb_log(wandb_run, {f"val/{k}": v for k, v in val_metrics.items()}, step=step)
                ndcg = val_metrics.get("ndcg@10", 0.0)
                if early_stopper.update(ndcg):
                    logger.info("Early stop triggered at step %06d (NDCG@10=%.4f)", step, ndcg)
                    stop_training = True
                    break
        if stop_training:
            break
    pbar.close()

    if stop_training:
        logger.info("Training stopped early; proceeding to test evaluation.")

    target_test_dataset = test_dataset or train_dataset
    if target_test_dataset is not None:
        logger.info("Running test evaluation...")
        test_metrics = evaluate(
            model,
            target_test_dataset,
            mode="test",
            batch_size=cfg.eval_batch_size,
            num_negatives=cfg.eval_negatives,
            device=device,
        )
        _wandb_log(wandb_run, {f"test/{k}": v for k, v in test_metrics.items()}, step=step)
        logger.info("[test] %s", " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    if wandb_run is not None:
        wandb_run.finish()

    embedding_path = _resolve_embedding_path(cfg)
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    np.save(embedding_path, item_table.weight.detach().cpu().numpy())
    logger.info("Saved item embeddings to %s (shape=%s)", embedding_path, tuple(item_table.weight.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ID baseline (no patch).")
    parser.add_argument("--max-seq-len", type=int, help="Maximum sequence length (e.g., 50 for short, 500 for long).")
    parser.add_argument("--train-stride", type=int, help="Stride for selecting training positions.")
    parser.add_argument("--max-steps", type=int, help="Optional cap on total optimizer steps (budget matching).")
    parser.add_argument("--device", type=str, help="Device string, e.g., cuda:0 or cpu.")
    args = parser.parse_args()

    cfg = IDTrainConfig()
    if args.max_seq_len:
        cfg.max_seq_len = args.max_seq_len
    if args.train_stride:
        cfg.train_stride = args.train_stride
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.device:
        cfg.device = args.device

    main(cfg)
