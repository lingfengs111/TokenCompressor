"""Evaluation utilities for ID-based MixFlow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import IDRecModel, ItemEmbeddingTable


def compute_topk_metrics(scores: torch.Tensor, ks: Iterable[int]) -> Dict[str, float]:
    if scores.numel() == 0:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}

    maxk = max(ks)
    _, topk_idx = torch.topk(scores, k=maxk, dim=1)

    out: Dict[str, float] = {}
    for k in ks:
        hits = (topk_idx[:, :k] == 0)
        h = hits.any(dim=1).float()
        hr = h.mean().item()
        pos = torch.argmax(hits.float(), dim=1)
        ndcg = (h * (1.0 / torch.log2(pos.float() + 2.0))).mean().item()
        out[f"HR@{k}"] = hr
        out[f"NDCG@{k}"] = ndcg
    return out


def sample_negatives(
    targets: torch.Tensor,
    history_ids: torch.Tensor,
    num_items: int,
    num_negatives: int,
) -> torch.Tensor:
    device = targets.device
    negs = torch.randint(1, num_items + 1, (targets.size(0), num_negatives), device=device)
    invalid = (negs == targets.unsqueeze(1)) | (negs == 0)
    if history_ids is not None:
        hist = history_ids.to(device)
        invalid |= torch.isin(negs, hist)
    while invalid.any():
        resample = torch.randint(1, num_items + 1, (invalid.sum().item(),), device=device)
        negs[invalid] = resample
        invalid = (negs == targets.unsqueeze(1)) | (negs == 0)
        if history_ids is not None:
            invalid |= torch.isin(negs, hist)
    return negs


def sampled_scores(
    user_vec: torch.Tensor,
    item_table: ItemEmbeddingTable,
    targets: torch.Tensor,
    history_ids: torch.Tensor,
    num_negatives: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_items = item_table.weight.size(0) - 1
    negs = sample_negatives(targets, history_ids, num_items, num_negatives)
    cand_ids = torch.cat([targets.unsqueeze(1), negs], dim=1)
    cand_emb = item_table.weight[cand_ids]
    scores = torch.einsum("bd,bkd->bk", user_vec, cand_emb)
    return scores, cand_ids


def pointwise_scores(
    user_vec: torch.Tensor,
    item_table: ItemEmbeddingTable,
    targets: torch.Tensor,
) -> torch.Tensor:
    target_emb = item_table.weight[targets]
    return (user_vec * target_emb).sum(dim=1)


def get_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_pos = attention_mask.long().sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos, :]


def _sasrec_sampler(
    dataset,
    batch_size: int,
    max_seq_length: int,
    train_stride: int,
):
    """Yield SASRec-style batches with pos/neg ids and user ids."""
    valid_user_seqs = []
    for user in dataset.users:
        seq = dataset.user_seq[user]
        if len(seq) > 2:
            valid_user_seqs.append((user, seq[:-2]))

    indices = np.random.permutation(len(valid_user_seqs))
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_data = [valid_user_seqs[idx] for idx in batch_indices]

        actual_batch_size = len(batch_data)
        seq_tensors = torch.zeros((actual_batch_size, max_seq_length), dtype=torch.long)
        pos_tensors = torch.zeros((actual_batch_size, max_seq_length), dtype=torch.long)
        neg_tensors = torch.zeros((actual_batch_size, max_seq_length), dtype=torch.long)
        user_ids = torch.zeros(actual_batch_size, dtype=torch.long)

        for idx, (user, seq) in enumerate(batch_data):
            user_ids[idx] = int(user)
            seq_len = min(len(seq), max_seq_length)
            if seq_len < 1:
                continue
            if len(seq) > max_seq_length:
                seq = seq[-max_seq_length:]
                seq_len = max_seq_length
            seq_tensors[idx, -seq_len:] = torch.tensor(seq[:seq_len])

            keep_positions = set(range(seq_len - 1, -1, -train_stride))
            keep_positions.add(seq_len - 1)
            for pos in range(seq_len):
                if pos not in keep_positions:
                    continue
                if pos < seq_len - 1:
                    pos_tensors[idx, -seq_len + pos] = seq[pos + 1]
                else:
                    full_seq = dataset.user_seq[user]
                    next_idx = len(seq)
                    if next_idx < len(full_seq):
                        pos_tensors[idx, -1] = full_seq[next_idx]

            seen_set = set(dataset.user_seq[user])
            for pos in range(seq_len):
                if pos not in keep_positions:
                    continue
                neg_item = np.random.randint(1, dataset.max_item + 1)
                while neg_item in seen_set:
                    neg_item = np.random.randint(1, dataset.max_item + 1)
                neg_tensors[idx, -seq_len + pos] = neg_item

        yield {"input_ids": seq_tensors, "pos_ids": pos_tensors, "neg_ids": neg_tensors, "user_ids": user_ids}


def predict_scores(model: IDRecModel, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    attention_mask = input_ids != 0
    hidden, attn = model(input_ids, attention_mask=attention_mask, patch_emb=None)
    final_hidden = hidden[:, -1, :]
    candidate_embs = model.item_table(candidate_ids)
    scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)
    return scores


def _align_hidden(hidden: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """When patch is prepended, hidden length can exceed target length; align to last positions."""
    if hidden.size(1) == target_ids.size(1):
        return hidden
    return hidden[:, -target_ids.size(1) :, :]


def evaluate_sasrec(
    model: IDRecModel,
    dataset,
    mode: str,
    batch_size: int,
    num_negatives: int,
    device: torch.device,
):
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

        candidates_list = []
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
    return {"NDCG@10": ndcg_10, "HR@10": hr_10}


class EarlyStopper:
    def __init__(self, patience: int = 2, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.count = 0

    def update(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


@dataclass
class FMEConfig:
    num_negatives: int = 1000
    ks: Sequence[int] = (10, 20)
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 2
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    batch_size: int = 256
    train_stride: int = 1
    show_progress: bool = False


def run_eval_online_fme(
    patch: torch.Tensor,
    model_builder,
    item_table: ItemEmbeddingTable,
    train_dataset,
    val_dataset,
    device: torch.device,
    fme_cfg: FMEConfig,
    patch_fn: Optional[Callable[[dict, torch.device, Optional[torch.Tensor]], torch.Tensor]] = None,
) -> Dict[str, float]:
    model = model_builder().to(device)
    model.item_table = item_table

    for p in item_table.parameters():
        p.requires_grad = False

    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=fme_cfg.lr, weight_decay=fme_cfg.weight_decay)
    stopper = EarlyStopper(patience=fme_cfg.patience)

    patch = patch.detach()

    def _maybe_select_patch(batch: dict, dev: torch.device, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if patch_fn is None:
            return patch
        return patch_fn(batch, dev, mask).detach()

    def _sampler_for(dataset):
        return _sasrec_sampler(dataset, batch_size=fme_cfg.batch_size, max_seq_length=dataset.max_seq_length, train_stride=fme_cfg.train_stride)

    train_steps = math.ceil(train_dataset.num_users / fme_cfg.batch_size)

    for epoch_idx in range(fme_cfg.epochs):
        model.train()
        train_iter = _sampler_for(train_dataset)
        pbar = tqdm(
            train_iter,
            total=train_steps,
            desc=f"FME train e{epoch_idx + 1}/{fme_cfg.epochs}",
            leave=False,
            disable=not fme_cfg.show_progress,
        )

        for bidx, batch in enumerate(pbar):
            if fme_cfg.max_train_batches is not None and bidx >= fme_cfg.max_train_batches:
                break
            input_ids = batch["input_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            patch_for_batch = _maybe_select_patch(batch, device, batch.get("mask", None))
            attention_mask = input_ids != 0
            hidden, attn = model(input_ids, attention_mask=attention_mask, patch_emb=patch_for_batch)
            hidden = _align_hidden(hidden, pos_ids)
            pos_embs = item_table(pos_ids)
            neg_embs = item_table(neg_ids)
            pos_logits = (hidden * pos_embs).sum(dim=-1)
            neg_logits = (hidden * neg_embs).sum(dim=-1)
            valid_mask = pos_ids != 0
            if not valid_mask.any():
                continue
            bce = nn.BCEWithLogitsLoss()
            loss = bce(pos_logits[valid_mask], torch.ones_like(pos_logits[valid_mask])) + bce(
                neg_logits[valid_mask], torch.zeros_like(neg_logits[valid_mask])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if fme_cfg.show_progress:
                pbar.set_postfix(loss=float(loss))

        if fme_cfg.show_progress:
            pbar.close()

        metrics = evaluate_sasrec(
            model,
            val_dataset,
            mode="val",
            batch_size=fme_cfg.batch_size,
            num_negatives=fme_cfg.num_negatives,
            device=device,
        )
        metric_value = metrics.get("NDCG@10", 0.0) or metrics.get("ndcg@10", 0.0)
        if stopper.update(metric_value):
            break

    final_metrics = evaluate_sasrec(
        model,
        val_dataset,
        mode="val",
        batch_size=fme_cfg.batch_size,
        num_negatives=fme_cfg.num_negatives,
        device=device,
    )
    return {f"fme_{k}": v for k, v in final_metrics.items()}
