"""Evaluation utilities for ID-only training."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from data import make_dataloaders_from_txt
from model import GlobalSoftPatch, ItemTable, build_student, logits_from_ids
from utils import setup_logger

logger = setup_logger("id-only-eval")


@torch.no_grad()
def compute_topk_metrics(scores: torch.Tensor, targets: torch.Tensor, ks: Iterable[int], one_based_labels: bool = True):
    if scores.numel() == 0:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}

    scores = scores.float()
    device = scores.device
    tgt = targets.to(device)
    if one_based_labels:
        tgt = tgt - 1

    maxk = max(ks)
    _, topk_idx = torch.topk(scores, k=maxk, dim=1)

    out = {}
    for k in ks:
        hits = (topk_idx[:, :k] == tgt.unsqueeze(1))
        h = hits.any(dim=1).float()
        hr = h.mean().item()
        pos = torch.argmax(hits.float(), dim=1)
        ndcg = (h * (1.0 / torch.log2(pos.float() + 2.0))).mean().item()
        out[f"HR@{k}"] = hr
        out[f"NDCG@{k}"] = ndcg
    return out


def clip_right_pad(batch_ids: torch.Tensor, batch_mask: torch.Tensor, L_real_eval: int):
    B, _ = batch_ids.size()
    kept_list = []
    for i in range(B):
        ids = batch_ids[i]
        msk = batch_mask[i]
        length = int(msk.sum().item())
        keep = min(length, L_real_eval)
        kept = ids[:length][-keep:] if keep > 0 else ids[:0]
        kept_list.append(kept)

    max_len = max((x.size(0) for x in kept_list), default=0)
    device = batch_ids.device
    out_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    out_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, kept in enumerate(kept_list):
        L = kept.size(0)
        if L > 0:
            out_ids[i, :L] = kept
            out_mask[i, :L] = 1
    return out_ids, out_mask


def _parse_amp(amp_setting):
    if isinstance(amp_setting, str):
        s = amp_setting.lower()
        if "bf16" in s or "bfloat16" in s:
            return torch.bfloat16, False
        if "fp16" in s or "float16" in s or "half" in s:
            return torch.float16, True
        return None, False
    if isinstance(amp_setting, bool) and amp_setting:
        return torch.float16, True
    return None, False


def _amp_ctx(amp_setting):
    dtype, _ = _parse_amp(amp_setting)
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def _make_scaler(amp_setting):
    _, use_scaler = _parse_amp(amp_setting)
    return torch.amp.GradScaler("cuda", enabled=use_scaler)


@torch.no_grad()
def collect_scores_targets(
    student,
    item_table,
    eta_tensor: torch.Tensor,
    loader,
    L_soft_eval: int,
    L_real_eval: int,
    pool: str,
    amp_setting,
    clip_last: bool = True,
    max_batches: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    student.eval()
    device = next(student.parameters()).device
    all_scores, all_targets = [], []

    for bidx, (recent_ids, targets, mask_recent) in enumerate(loader):
        if (max_batches is not None) and (bidx >= max_batches):
            break
        recent_ids = recent_ids.to(device)
        mask_recent = mask_recent.to(device)
        targets = targets.to(device)

        if clip_last:
            ids_eval, mask_eval = clip_right_pad(recent_ids, mask_recent, L_real_eval)
        else:
            ids_eval = recent_ids[:, -L_real_eval:]
            mask_eval = mask_recent[:, -L_real_eval:]

        if hasattr(eta_tensor, "forward"):
            eta_b = eta_tensor(item_table, recent_ids, mask_recent)
        else:
            eta_b = eta_tensor

        with _amp_ctx(amp_setting):
            scores, _ = logits_from_ids(
                student=student,
                item_table=item_table,
                eta_tensor=eta_b,
                recent_ids=ids_eval,
                mask_recent=mask_eval,
                L_soft=L_soft_eval,
                pool=pool,
            )
            scores = scores.float()

        all_scores.append(scores.cpu())
        all_targets.append(targets.cpu())

    scores = torch.cat(all_scores, dim=0) if all_scores else torch.empty(0, item_table.table.size(0) - 1)
    targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, dtype=torch.long)
    student.train()
    return scores, targets


def cre_train_one_mode(
    student,
    item_table,
    eta_tensor,
    dl_tr,
    dl_va,
    L_soft,
    L_real,
    pool,
    ks,
    amp_setting,
    max_epochs,
    lr,
    weight_decay,
    patience,
    metric_key,
    grad_clip=0.0,
    max_train_batches=None,
    max_val_batches=None,
    metrics_fn=compute_topk_metrics,
):
    for p in student.parameters():
        p.requires_grad = True
    device = next(student.parameters()).device
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _make_scaler(amp_setting)

    best_metric = -1.0
    best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    wait = 0

    for _ in range(1, max_epochs + 1):
        student.train()

        for bidx, (recent_ids, targets, mask_recent) in enumerate(dl_tr):
            if (max_train_batches is not None) and (bidx >= max_train_batches):
                break
            recent_ids = recent_ids.to(device)
            mask_recent = mask_recent.to(device)
            targets = targets.to(device)

            ids_eval, mask_eval = clip_right_pad(recent_ids, mask_recent, L_real)

            eta_b = eta_tensor(item_table, recent_ids, mask_recent) if hasattr(eta_tensor, "forward") else eta_tensor
            with _amp_ctx(amp_setting):
                logits, _ = logits_from_ids(
                    student=student,
                    item_table=item_table,
                    eta_tensor=eta_b,
                    recent_ids=ids_eval,
                    mask_recent=mask_eval,
                    L_soft=L_soft,
                    pool=pool,
                )
                loss = F.cross_entropy(logits, targets - 1)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                opt.step()

        scores, targets = collect_scores_targets(
            student=student,
            item_table=item_table,
            eta_tensor=eta_tensor,
            loader=dl_va,
            L_soft_eval=L_soft,
            L_real_eval=L_real,
            pool=pool,
            amp_setting=amp_setting,
            clip_last=True,
            max_batches=max_val_batches,
        )
        metrics = metrics_fn(scores, targets, ks) if scores.numel() else {metric_key: 0.0}
        val_score = float(metrics.get(metric_key, 0.0))

        if val_score > best_metric:
            best_metric = val_score
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_state, best_metric


def _load_patch_state(ckpt: dict) -> dict:
    if "patch" in ckpt:
        return ckpt["patch"]
    if "phi" in ckpt:
        return ckpt["phi"]
    raise KeyError("Checkpoint missing patch state (expected 'patch' or 'phi').")


def _mode_cfg(mode: dict) -> tuple[str, int, int]:
    name = str(mode.get("name", "")).lower()
    return name, int(mode.get("L_soft", 0)), int(mode.get("L_real", 0))


def evaluate(
    *,
    data_name: str,
    L_real: int,
    batch_size: int,
    d_model: int,
    t5_name: str,
    grad_ckpt: bool,
    pool: str,
    compressor_type: str,
    L_soft: int,
    n_heads: int,
    dropout: float,
    patch_norm: str,
    k_list: Sequence[int],
    clip_last: bool,
    modes: Sequence[dict],
    batch_size_full: int,
    best_mode: str,
    device: str,
    amp: str,
    ckpt_num: int,
    cre_epochs: int,
    cre_lr: float,
    cre_wd: float,
    cre_early_stop: int,
    cre_metric: str,
    cre_grad_clip: float,
    cre_max_train_batches: Optional[int],
    cre_max_val_batches: Optional[int],
) -> dict:
    base_dir = Path(__file__).resolve().parent
    proc_dir = base_dir / "data" / data_name / "proc"

    base_dl_tr, base_dl_va, base_dl_te = make_dataloaders_from_txt(
        proc_dir,
        L_real,
        batch_size,
    )

    need_full = any(str(mode.get("name", "")).lower() == "full" for mode in modes)
    if need_full:
        L_full = max(int(mode.get("L_real", 0)) for mode in modes if str(mode.get("name", "")).lower() == "full")
        dl_tr_full, dl_va_full, dl_te_full = make_dataloaders_from_txt(
            proc_dir,
            L_full,
            int(batch_size_full),
        )
    else:
        dl_tr_full = dl_va_full = dl_te_full = None

    logger.info("Loading ckpt #%s ...", ckpt_num)
    ckpt_path = base_dir / "artifacts" / data_name / f"checkpoint_{ckpt_num}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    with (proc_dir / "item2idx.json").open("r") as f:
        num_items = len(json.load(f))

    E = ItemTable(num_items, d_model, trainable=False).to(device)
    E.load_state_dict(ckpt["item_table"])
    logger.info("ItemTable: num_items=%s, d_model=%s, trainable=False", num_items, d_model)

    E.use_cosine_default = True
    if "logit_scale" in ckpt:
        E.logit_scale = ckpt["logit_scale"].to(device)
    else:
        E.logit_scale = torch.tensor(0.0, device=device)
    E.normalize_patch_default = True
    E.patch_norm_kind = patch_norm

    patch_state = _load_patch_state(ckpt)

    results = {}
    for mode in modes:
        name, L_soft_eval, L_real_eval = _mode_cfg(mode)
        logger.info("Evaluating mode: %s", mode)

        if name == "full" and dl_tr_full is not None:
            dl_tr_mode, dl_va_mode, dl_te_mode = dl_tr_full, dl_va_full, dl_te_full
        else:
            dl_tr_mode, dl_va_mode, dl_te_mode = base_dl_tr, base_dl_va, base_dl_te

        if compressor_type != "soft" and L_soft_eval > 0:
            raise ValueError("Only soft patch is supported in id_only eval.")

        if L_soft_eval > 0:
            phi = GlobalSoftPatch(L_soft_eval, E.table.size(1), device=device).to(device)
            phi.load_state_dict(patch_state)
            eta_eval = phi.phi
        else:
            eta_eval = torch.empty(0, E.table.size(1), device=device)

        student = build_student(t5_name, device, grad_ckpt)

        best_state, best_val = cre_train_one_mode(
            student=student,
            item_table=E,
            eta_tensor=eta_eval,
            dl_tr=dl_tr_mode,
            dl_va=dl_va_mode,
            L_soft=L_soft_eval,
            L_real=L_real_eval,
            pool=pool,
            ks=k_list,
            amp_setting=amp,
            max_epochs=cre_epochs,
            lr=cre_lr,
            weight_decay=cre_wd,
            patience=cre_early_stop,
            metric_key=cre_metric,
            grad_clip=cre_grad_clip,
            max_train_batches=cre_max_train_batches,
            max_val_batches=cre_max_val_batches,
            metrics_fn=compute_topk_metrics,
        )

        student.load_state_dict(best_state)
        scores, targets = collect_scores_targets(
            student=student,
            item_table=E,
            eta_tensor=eta_eval,
            loader=dl_te_mode,
            L_soft_eval=L_soft_eval,
            L_real_eval=L_real_eval,
            pool=pool,
            amp_setting=amp,
            clip_last=True,
            max_batches=None,
        )
        metrics = compute_topk_metrics(scores, targets, k_list)
        results[name] = metrics
        logger.info(
            "[CRE|%s] val_best=%.4f %s",
            name,
            best_val,
            " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )

    return results


def run_eval_online_cre(
    E,
    phi,
    dl_tr,
    dl_va,
    *,
    data_name: str,
    L_real: int,
    batch_size: int,
    d_model: int,
    t5_name: str,
    grad_ckpt: bool,
    pool: str,
    compressor_type: str,
    L_soft: int,
    n_heads: int,
    dropout: float,
    patch_norm: str,
    k_list: Sequence[int],
    clip_last: bool,
    modes: Sequence[dict],
    batch_size_full: int,
    best_mode: str,
    device: str,
    amp: str,
    ckpt_num: int,
    cre_online_epochs: int,
    cre_online_lr: float,
    cre_online_wd: float,
    cre_online_early_stop: int,
    cre_online_metric: str,
    cre_online_grad_clip: float,
    cre_online_max_train_batches: Optional[int],
    cre_online_max_val_batches: Optional[int],
    max_eval_batches: Optional[int],
    **_ignored,
):
    ks = tuple(k_list)
    mode = next((m for m in modes if m.get("name") == best_mode), None)
    if mode is None:
        raise ValueError(f"best_mode='{best_mode}' not found in eval.modes")

    _, L_soft_mode, L_real_mode = _mode_cfg(mode)

    if compressor_type != "soft" and L_soft_mode > 0:
        raise ValueError("Only soft patch is supported in id_only eval.")

    eta = phi.phi.to(device) if L_soft_mode > 0 else torch.empty(0, E.table.size(1), device=device)

    max_epochs = int(cre_online_epochs)
    lr = float(cre_online_lr)
    weight_decay = float(cre_online_wd)
    patience = int(cre_online_early_stop)
    metric_key = cre_online_metric
    grad_clip = float(cre_online_grad_clip)
    max_train_batches = cre_online_max_train_batches
    max_val_batches = cre_online_max_val_batches or max_eval_batches

    student = build_student(t5_name, device, grad_ckpt)

    best_state, _ = cre_train_one_mode(
        student=student,
        item_table=E,
        eta_tensor=eta,
        dl_tr=dl_tr,
        dl_va=dl_va,
        L_soft=L_soft_mode,
        L_real=L_real_mode,
        pool=pool,
        ks=ks,
        amp_setting=amp,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        metric_key=metric_key,
        grad_clip=grad_clip,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        metrics_fn=compute_topk_metrics,
    )

    student.load_state_dict(best_state)
    scores, targets = collect_scores_targets(
        student=student,
        item_table=E,
        eta_tensor=eta,
        loader=dl_va,
        L_soft_eval=L_soft_mode,
        L_real_eval=L_real_mode,
        pool=pool,
        amp_setting=amp,
        clip_last=clip_last,
        max_batches=max_val_batches,
    )
    return compute_topk_metrics(scores, targets, ks)


if __name__ == "__main__":
    raise SystemExit(
        "config.yaml has been removed. Call evaluate(...) with explicit arguments from train script."
    )
