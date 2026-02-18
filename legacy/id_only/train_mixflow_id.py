#!/usr/bin/env python3
"""
MixFlow training for ID-only encoder with soft patch.

Design:
  - Inner loop: train encoder on short sequences + soft patch.
  - Outer loop: update patch using longer real sequences (no patch).
  - Meta-gradients: core/mixflow.py (eta-aware gradients).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import os
import sys

import torch
import torch.nn.functional as F
from torch.func import functional_call

from data import make_dataloaders_from_txt
from eval import run_eval_online_cre
from model import GlobalSoftPatch, ItemTable, build_student
from utils import AmpAutocast, cuda_mem_gb, pack_theta_state, set_seed, setup_logger

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from core.mixflow import get_fwdrev_grad_fn_eta, MomentumInner  # type: ignore

logger = setup_logger("train-mixflow-id", log_to_file=True)


@dataclass
class MixFlowIDConfig:
    # Data
    data_name: str = "movielens1m"
    L_recent: int = 128
    L_full: int = 256
    batch_size: int = 64
    num_workers: int = 0

    # Model
    t5_name: str = "t5-small"
    grad_ckpt: bool = False
    d_model: int = 512
    num_items: Optional[int] = None
    items_trainable: bool = False
    init_from_text: bool = True
    init_path: str = "emb/e_text_init.pt"
    pool: str = "last"

    # Patch
    L_soft: int = 96
    patch_norm: str = "l2"

    # Training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: str = "bf16"
    seed: int = 1234
    inner_steps: int = 3
    inner_lr: float = 3.0e-2
    inner_momentum: float = 0.9
    outer_lr: float = 5.0e-3
    max_iters: int = 1000
    log_every: int = 20

    # Online CRE eval
    eval_every: int = 200
    eval_k: Sequence[int] = (10, 20, 50)
    eval_clip_last: bool = True
    eval_best_mode: str = "patch"
    eval_modes: Optional[Sequence[dict]] = None
    max_eval_batches: Optional[int] = 50

    # CRE (online)
    cre_epochs: int = 10
    cre_lr: float = 1.0e-4
    cre_wd: float = 0.0
    cre_early_stop: int = 2
    cre_metric: str = "NDCG@20"
    cre_grad_clip: float = 0.0
    cre_max_train_batches: Optional[int] = 500
    cre_max_val_batches: Optional[int] = 200

    # Checkpoints
    ckpt_num: int = 0
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.eval_modes is None:
            self.eval_modes = (
                {"name": "patch", "L_soft": self.L_soft, "L_real": self.L_recent},
            )

    def log_config(self) -> None:
        logger.info("=== MixFlow ID Config ===")
        logger.info("data_name: %s", self.data_name)
        logger.info("L_recent: %s | L_full: %s", self.L_recent, self.L_full)
        logger.info("batch_size: %s | num_workers: %s", self.batch_size, self.num_workers)
        logger.info("t5_name: %s | grad_ckpt: %s", self.t5_name, self.grad_ckpt)
        logger.info("d_model: %s | num_items: %s | items_trainable: %s", self.d_model, self.num_items, self.items_trainable)
        logger.info("init_from_text: %s | init_path: %s", self.init_from_text, self.init_path)
        logger.info("pool: %s", self.pool)
        logger.info("L_soft: %s | patch_norm: %s", self.L_soft, self.patch_norm)
        logger.info(
            "device: %s | amp: %s | inner_steps: %s | inner_lr: %s | outer_lr: %s | max_iters: %s",
            self.device,
            self.amp,
            self.inner_steps,
            self.inner_lr,
            self.outer_lr,
            self.max_iters,
        )
        logger.info(
            "eval_every: %s | eval_k: %s | eval_best_mode: %s | max_eval_batches: %s",
            self.eval_every,
            list(self.eval_k),
            self.eval_best_mode,
            self.max_eval_batches,
        )
        logger.info("eval_modes: %s", list(self.eval_modes or []))
        logger.info(
            "cre_epochs: %s | cre_lr: %s | cre_early_stop: %s | cre_metric: %s",
            self.cre_epochs,
            self.cre_lr,
            self.cre_early_stop,
            self.cre_metric,
        )
        logger.info("=========================")

    def eval_kwargs(self) -> dict:
        return {
            "data_name": self.data_name,
            "L_real": self.L_recent,
            "batch_size": self.batch_size,
            "d_model": self.d_model,
            "t5_name": self.t5_name,
            "grad_ckpt": self.grad_ckpt,
            "pool": self.pool,
            "compressor_type": "soft",
            "L_soft": self.L_soft,
            "n_heads": 0,
            "dropout": 0.0,
            "patch_norm": self.patch_norm,
            "k_list": self.eval_k,
            "clip_last": self.eval_clip_last,
            "modes": self.eval_modes,
            "batch_size_full": self.batch_size,
            "best_mode": self.eval_best_mode,
            "device": self.device,
            "amp": self.amp,
            "ckpt_num": self.ckpt_num,
            "cre_online_epochs": self.cre_epochs,
            "cre_online_lr": self.cre_lr,
            "cre_online_wd": self.cre_wd,
            "cre_online_early_stop": self.cre_early_stop,
            "cre_online_metric": self.cre_metric,
            "cre_online_grad_clip": self.cre_grad_clip,
            "cre_online_max_train_batches": self.cre_max_train_batches,
            "cre_online_max_val_batches": self.cre_max_val_batches,
            "max_eval_batches": self.max_eval_batches,
        }


def ce_full_softmax(user_vec: torch.Tensor, item_table: torch.Tensor, targets: torch.Tensor):
    logits = user_vec @ item_table[1:].T
    loss = F.cross_entropy(logits, targets - 1)
    return loss, logits


def train(cfg: MixFlowIDConfig) -> None:
    base_dir = Path(__file__).resolve().parent
    set_seed(cfg.seed)
    device = cfg.device

    proc_dir = base_dir / "data" / cfg.data_name / "proc"
    logger.info("Loading %s dataset from %s", cfg.data_name, proc_dir)

    with (proc_dir / "item2idx.json").open("r") as f:
        num_items = len(json.load(f))
    if cfg.num_items is None:
        cfg.num_items = num_items
    logger.info("num_items=%s, d_model=%s", f"{num_items:,}", cfg.d_model)

    logger.info("Creating train/val dataloaders (L_recent=%s)...", cfg.L_recent)
    dl_tr, dl_va, _ = make_dataloaders_from_txt(proc_dir, cfg.L_recent, cfg.batch_size, cfg.num_workers)

    logger.info("Creating full dataloader for outer loop (L_full=%s)...", cfg.L_full)
    dl_tr_full, _, _ = make_dataloaders_from_txt(proc_dir, cfg.L_full, cfg.batch_size, cfg.num_workers)
    iter_full = iter(dl_tr_full)

    E = ItemTable(cfg.num_items, cfg.d_model, trainable=cfg.items_trainable).to(device)
    if cfg.init_from_text:
        txt_path = base_dir / "data" / cfg.data_name / cfg.init_path
        E.table.data.copy_(torch.load(txt_path, map_location="cpu"))
        logger.info("Loaded item table from %s", txt_path)
    for p in E.parameters():
        p.requires_grad = cfg.items_trainable

    student = build_student(cfg.t5_name, device, cfg.grad_ckpt)
    student.config.use_cache = False

    try:
        student.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        logger.warning("Transformers is older; falling back to reentrant checkpointing")
        student.encoder.gradient_checkpointing_enable()

    for p in student.parameters():
        p.requires_grad = False

    theta: List[torch.Tensor] = []
    for _, param in student.encoder.named_parameters():
        param.requires_grad = True
        theta.append(param)
    logger.info("Full-FT encoder params: %s", f"{sum(p.numel() for p in theta):,}")

    logit_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))
    patch = GlobalSoftPatch(cfg.L_soft, student.config.d_model, device=device).to(device)
    outer_params = list(patch.parameters()) + [logit_scale]
    opt_eta = torch.optim.AdamW(outer_params, lr=float(cfg.outer_lr))

    E.use_cosine_default = True
    E.logit_scale = logit_scale.data
    E.normalize_patch_default = True
    E.patch_norm_kind = cfg.patch_norm

    inner_opt = MomentumInner(theta, lr=cfg.inner_lr, momentum=cfg.inner_momentum)

    BASE_PARAMS_ENC, BASE_BUFFERS_ENC = {}, {}
    for name, param in student.named_parameters():
        if name.startswith("encoder."):
            BASE_PARAMS_ENC[name[len("encoder.") :]] = param
    for name, buffer in student.named_buffers():
        if name.startswith("encoder."):
            BASE_BUFFERS_ENC[name[len("encoder.") :]] = buffer

    theta_names_enc = []
    for name, param in student.named_parameters():
        if name.startswith("encoder.") and param.requires_grad:
            theta_names_enc.append(name[len("encoder.") :])

    def get_patch_emb() -> torch.Tensor:
        patch_emb = patch.phi
        if cfg.patch_norm == "l2":
            patch_emb = F.normalize(patch_emb, dim=-1, eps=1e-6)
        elif cfg.patch_norm == "ln":
            patch_emb = (patch_emb - patch_emb.mean(dim=-1, keepdim=True)) / (
                patch_emb.std(dim=-1, keepdim=True) + 1e-6
            )
        return patch_emb

    def inner_loss(theta_list, patch_emb, recent_ids, targets, mask_recent):
        override = {n: t for n, t in zip(theta_names_enc, theta_list)}
        param_and_buffers = {**BASE_PARAMS_ENC, **override, **BASE_BUFFERS_ENC}

        B = recent_ids.size(0)
        emb_recent = E(recent_ids.to(device))

        if patch_emb is None or patch_emb.numel() == 0:
            patch_tensor = emb_recent.new_zeros((B, 0, emb_recent.size(-1)))
        else:
            patch_tensor = patch_emb.unsqueeze(0).expand(B, -1, -1)

        L_soft = patch_tensor.size(1)
        inputs = torch.cat([patch_tensor, emb_recent], dim=1)
        attn = torch.cat(
            [
                torch.ones((B, L_soft), dtype=torch.long, device=device),
                mask_recent.to(device).long(),
            ],
            dim=1,
        )

        was_gc = getattr(student.encoder, "gradient_checkpointing", False)
        if was_gc:
            student.encoder.gradient_checkpointing_disable()

        try:
            with AmpAutocast(cfg.amp):
                enc_out = functional_call(
                    student.encoder,
                    param_and_buffers,
                    args=(),
                    kwargs=dict(inputs_embeds=inputs, attention_mask=attn, return_dict=True),
                )
                H = enc_out.last_hidden_state

                if cfg.pool == "last":
                    lengths = mask_recent.sum(dim=1)
                    idx_last = (lengths - 1).clamp_min(0)
                    pos = L_soft + idx_last
                    u = H[torch.arange(B, device=device), pos, :]
                elif cfg.pool == "mean":
                    recent = H[:, L_soft:, :]
                    denom = mask_recent.sum(dim=1).clamp_min(1).unsqueeze(1).to(device).float()
                    u = (recent * mask_recent.unsqueeze(-1).float().to(device)).sum(dim=1) / denom
                else:
                    raise ValueError("pool must be 'last' or 'mean'")

                u = F.normalize(u, dim=-1)
                tab = F.normalize(E.table, dim=-1)
                logits = (u @ tab[1:].T).float() * torch.exp(logit_scale)
                loss = F.cross_entropy(logits, (targets.to(device) - 1))
            return loss
        finally:
            try:
                student.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                student.encoder.gradient_checkpointing_enable()

    grad_fn = get_fwdrev_grad_fn_eta(inner_loss)

    theta_param_names = [n for n, p in student.named_parameters() if p.requires_grad]
    best_ndcg = -1.0
    it = 0

    while it < cfg.max_iters:
        for recent_ids, targets, mask_recent in dl_tr:
            it += 1

            w_state, m_state = inner_opt.snapshot()

            patch_emb = get_patch_emb()
            for _ in range(cfg.inner_steps):
                gflat = grad_fn(theta, patch_emb, recent_ids, targets, mask_recent)
                inner_opt.step(gflat)

            try:
                full_ids, full_tgts, full_mask = next(iter_full)
            except StopIteration:
                iter_full = iter(dl_tr_full)
                full_ids, full_tgts, full_mask = next(iter_full)

            opt_eta.zero_grad(set_to_none=True)
            loss_outer = inner_loss(theta, None, full_ids, full_tgts, full_mask)
            loss_outer.backward()
            torch.nn.utils.clip_grad_norm_(outer_params, 1.0)
            opt_eta.step()

            inner_opt.restore(w_state, m_state)

            if it % cfg.log_every == 0:
                peak = cuda_mem_gb(model=student, device_str=cfg.device, kind="alloc")
                logger.info(
                    "[it %06d] loss_outer=%.4f | temp=%.2f | max CUDA(GB)=%.3f",
                    it,
                    loss_outer.item(),
                    torch.exp(logit_scale).item(),
                    peak,
                )

            if cfg.eval_every and (it % cfg.eval_every == 0):
                metrics = run_eval_online_cre(E, patch, dl_tr, dl_va, **cfg.eval_kwargs())
                ndcg20 = metrics.get("NDCG@20", 0.0)
                logger.info("[eval it %s] %s", it, " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

                if cfg.save_best and ndcg20 > best_ndcg:
                    best_ndcg = ndcg20
                    Path("artifacts").mkdir(parents=True, exist_ok=True)
                    state = {
                        "theta": pack_theta_state(student, theta_param_names),
                        "item_table": E.state_dict(),
                        "cfg": cfg.eval_kwargs(),
                        "it": it,
                        "metrics": metrics,
                        "logit_scale": logit_scale.detach().cpu(),
                        "patch": patch.state_dict(),
                    }
                    torch.save(state, Path("artifacts") / "best.pt")
                    logger.info("Best checkpoint at it=%s (NDCG@20=%.4f)", it, ndcg20)

            if it >= cfg.max_iters:
                break

    artifacts_dir = Path("artifacts") / cfg.data_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    final_state = {
        "item_table": E.state_dict(),
        "cfg": cfg.eval_kwargs(),
        "logit_scale": logit_scale.detach().cpu(),
        "patch": patch.state_dict(),
    }
    ckpt = artifacts_dir / f"checkpoint_{cfg.ckpt_num}.pt"
    torch.save(final_state, ckpt)
    logger.info("Saved %s", ckpt)


if __name__ == "__main__":
    config = MixFlowIDConfig()
    config.log_config()
    train(config)
