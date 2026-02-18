#!/usr/bin/env python3
"""
MixFlow training scaffold for GR decoder (semantic codes).

Design:
  - Inner loop: train decoder/encoder (excluding frozen token embeddings) on short sequences + soft patch.
  - Outer loop: update patch by supervising on longer real sequences (no patch).
  - Meta-gradients: reuse core/mixflow.py (fwdrev with eta-aware gradients).

This script mirrors the logging/config style used in RQ_VAE_tokenizer/refer.py.
Evaluation (FME - Fresh-model Eval) is intentionally deferred to keep the loop minimal.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from model import SemanticCodeDecoder
from data import create_dataloaders_from_txt
from evaluate import run_eval_online_fme

# Reuse MixFlow primitives from core
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from core.mixflow import get_fwdrev_grad_fn_eta, MomentumInner  # type: ignore

from core.logger import setup_logger

logger = setup_logger("train-mixflow-gr", log_to_file=True)


@dataclass
class MixFlowConfig:
    # Data
    data_txt_path: str = "data/movielens1m/proc/data.txt"
    semantic_codes_path: str = "data/movielens1m/codes/item_semantic_codes.pt"
    num_levels: int = 3
    L_recent: int = 20  # short length for inner loop (patch)
    L_full: int = 100  # long length for outer loop (no patch)
    batch_size: int = 32
    num_workers: int = 0

    # Model
    pretrained_ckpt: str = "/home/lingfengs111/codes/soft_patch_training/checkpoints/decoder/checkpoint_best.pt" # load code/bos embeddings from a decoder checkpoint
    codebook_size: int = 256
    hidden_dim: int = 512
    encoder_layers: int = 3
    decoder_layers: int = 3
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    attn_dropout: float = 0.1
    carry_decoder_state: bool = False

    # Patch
    L_soft: int = 16

    # Training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    inner_steps: int = 3
    inner_lr: float = 5e-4
    inner_momentum: float = 0.9
    outer_lr: float = 5e-5
    weight_decay: float = 0.0
    max_iters: int = 1000
    log_every: int = 50
    wandb_enabled: bool = True
    wandb_project: str = "mixflow-gr"
    wandb_run_name: Optional[str] = None
    # FME eval (fresh-model eval)
    fme_eval_every: int = 30  # 0 disables online FME eval
    fme_epochs: int = 3
    fme_lr: float = 1e-4
    fme_weight_decay: float = 0.0
    fme_patience: int = 2
    fme_max_train_batches: Optional[int] = 200
    fme_max_val_batches: Optional[int] = 50

    def log_config(self):
        logger.info("=== MixFlow Config ===")
        logger.info(f"data_txt_path: {self.data_txt_path}")
        logger.info(f"semantic_codes_path: {self.semantic_codes_path}")
        logger.info(f"num_levels: {self.num_levels}")
        logger.info(f"L_recent: {self.L_recent} | L_full: {self.L_full}")
        logger.info(f"batch_size: {self.batch_size} | num_workers: {self.num_workers}")
        logger.info(f"hidden_dim: {self.hidden_dim} | heads: {self.num_heads}")
        logger.info(f"encoder_layers: {self.encoder_layers} | decoder_layers: {self.decoder_layers}")
        logger.info(f"ffn_dim: {self.ffn_dim} | dropout: {self.dropout} | attn_dropout: {self.attn_dropout}")
        logger.info(f"L_soft: {self.L_soft}")
        logger.info(
            f"device: {self.device} | inner_steps: {self.inner_steps} | inner_lr: {self.inner_lr} "
            f"| outer_lr: {self.outer_lr} | wd: {self.weight_decay} | max_iters: {self.max_iters}"
        )
        logger.info(f"pretrained_ckpt: {self.pretrained_ckpt}")
        logger.info(f"wandb_enabled: {self.wandb_enabled} | project: {self.wandb_project} | run_name: {self.wandb_run_name}")
        logger.info(
            f"fme_eval_every: {self.fme_eval_every} | fme_epochs: {self.fme_epochs} | fme_lr: {self.fme_lr} "
            f"| fme_patience: {self.fme_patience} | fme_max_train_batches: {self.fme_max_train_batches} "
            f"| fme_max_val_batches: {self.fme_max_val_batches}"
        )
        logger.info("======================")


class SoftPatch(nn.Module):
    """Learnable soft patch applied after token embeddings."""

    def __init__(self, L_soft: int, hidden_dim: int):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(L_soft, hidden_dim) * 0.02)


def unwrap_core(model: SemanticCodeDecoder):
    """SemanticCodeDecoder is a wrapper; unwrap to the underlying SemanticCodePredictor."""
    return model.model if hasattr(model, "model") else model


def freeze_code_embeddings(model: SemanticCodeDecoder):
    core = unwrap_core(model)
    if not hasattr(core, "code_embeddings"):
        raise AttributeError("Underlying model lacks code_embeddings")
    for emb in core.code_embeddings:
        for p in emb.parameters():
            p.requires_grad = False
    logger.info("[Init] code_embeddings frozen.")


def load_pretrained_embeddings(
    core_model: nn.Module,
    ckpt_path: Optional[str],
    device: torch.device,
) -> None:
    """
    Load code_embeddings.*.weight and bos_embedding from a decoder checkpoint.
    RoPE uses buffers only (inv_freq), so nothing to load there.
    """
    if not ckpt_path:
        logger.info("[Init] No pretrained_ckpt provided; skip embedding load.")
        return
    path = Path(ckpt_path)
    if not path.exists():
        logger.warning(f"[Init] pretrained_ckpt not found: {ckpt_path}, skip load.")
        return
    logger.info(f"[Init] Loading pretrained embeddings from {ckpt_path}")
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)

    def maybe_load(key_in_state: str, target_param: nn.Parameter):
        if key_in_state in sd:
            src = sd[key_in_state]
            if src.shape != target_param.shape:
                logger.warning(
                    f"[Init] Skip {key_in_state} due to shape mismatch: ckpt {tuple(src.shape)} vs model {tuple(target_param.shape)}"
                )
                return False
            with torch.no_grad():
                target_param.copy_(src.to(device))
            return True
        return False

    # code embeddings
    loaded = 0
    for i, emb in enumerate(core_model.code_embeddings):
        # keys might be with or without "model." prefix
        found = False
        for prefix in ["", "model.", "model.model."]:
            if maybe_load(f"{prefix}code_embeddings.{i}.weight", emb.weight):
                loaded = loaded + 1
                found = True
                break
        # fallback: scan keys containing suffix
        if not found:
            suffix = f"code_embeddings.{i}.weight"
            match = next((k for k in sd.keys() if k.endswith(suffix)), None)
            if match is not None:
                src = sd[match]
                if src.shape != emb.weight.shape:
                    logger.warning(
                        f"[Init] Skip {match} due to shape mismatch: ckpt {tuple(src.shape)} vs model {tuple(emb.weight.shape)}"
                    )
                else:
                    with torch.no_grad():
                        emb.weight.copy_(src.to(device))
                    loaded = loaded + 1
    if loaded == 0:
        logger.warning("[Init] No code_embeddings.* loaded from checkpoint.")
    else:
        logger.info(f"[Init] Loaded {loaded} code_embedding tables.")

    # bos_embedding
    if hasattr(core_model, "bos_embedding"):
        loaded_bos = False
        for k in ["bos_embedding", "model.bos_embedding", "model.model.bos_embedding"]:
            if maybe_load(k, core_model.bos_embedding):
                loaded_bos = True
                break
        if not loaded_bos:
            # fallback scan
            match = next((k for k in sd.keys() if k.endswith("bos_embedding")), None)
            if match is not None:
                src = sd[match]
                if src.shape != core_model.bos_embedding.shape:
                    logger.warning(
                        f"[Init] Skip {match} due to shape mismatch: ckpt {tuple(src.shape)} vs model {tuple(core_model.bos_embedding.shape)}"
                    )
                else:
                    with torch.no_grad():
                        core_model.bos_embedding.copy_(src.to(device))
                    loaded_bos = True
        if loaded_bos:
            logger.info("[Init] Loaded bos_embedding.")
        else:
            logger.warning("[Init] bos_embedding not found in checkpoint; keep random init.")


def collect_base_params_and_buffers(module: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params = {n: p for n, p in module.named_parameters()}
    buffers = {n: b for n, b in module.named_buffers()}
    return params, buffers


def build_override(names: List[str], tensors: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {n: t for n, t in zip(names, tensors)}


def run_encoder_with_patch(
    encoder: nn.Module,
    base_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    override_params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_emb: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encoder forward with patch injection after token embeddings.
    Uses functional_call to apply overridden parameters.
    """
    params = {**base_params, **override_params, **base_buffers}
    num_levels = input_ids.shape[2]

    embeddings = []
    for level in range(num_levels):
        weight = params.get(f"code_embeddings.{level}.weight", None)
        if weight is None:
            raise KeyError(f"Missing code_embeddings.{level}.weight")
        codes_at_level = input_ids[:, :, level]
        emb = F.embedding(codes_at_level, weight, padding_idx=0)
        embeddings.append(emb)

    x = torch.cat(embeddings, dim=-1)  # [B, S, num_levels*hidden_dim]
    x = F.linear(x, params["embed_proj.weight"], params["embed_proj.bias"])
    if hasattr(encoder, "dropout"):
        x = encoder.dropout(x)

    B = x.size(0)
    if patch_emb is None or patch_emb.numel() == 0:
        patch = x.new_zeros((B, 0, x.size(-1)))
    elif patch_emb.dim() == 2:
        patch = patch_emb.unsqueeze(0).expand(B, -1, -1)
    elif patch_emb.dim() == 3:
        patch = patch_emb
    else:
        raise ValueError("patch_emb must be [L_soft, H] or [B, L_soft, H]")

    x = torch.cat([patch, x], dim=1)
    patch_mask = torch.ones((B, patch.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
    attn = torch.cat([patch_mask, attention_mask], dim=1)

    for i, block in enumerate(encoder.transformer_blocks):
        prefix = f"transformer_blocks.{i}."
        block_params = {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}
        x = functional_call(
            block,
            block_params,
            args=(x,),
            kwargs={"attention_mask": attn, "is_causal": False},
        )

    return x, attn


def run_decoder_stateless(
    decoder: nn.Module,
    base_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    override_params: Dict[str, torch.Tensor],
    context: torch.Tensor,
    encoder_output: torch.Tensor,
    attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    params = {**base_params, **override_params, **base_buffers}
    out = functional_call(
        decoder,
        params,
        args=(),
        kwargs={
            "context": context,
            "encoder_output": encoder_output,
            "attention_mask": attention_mask,
            "decoder_input_ids": target_ids,
        },
    )
    return out.logits


def prepare_dataloaders(cfg: MixFlowConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dl_short_tr, dl_short_val, _ = create_dataloaders_from_txt(
        data_txt_path=cfg.data_txt_path,
        semantic_codes_path=cfg.semantic_codes_path,
        batch_size=cfg.batch_size,
        num_levels=cfg.num_levels,
        flatten_codes=False,
        max_seq_len=cfg.L_recent,
    )
    dl_long_tr, _, _ = create_dataloaders_from_txt(
        data_txt_path=cfg.data_txt_path,
        semantic_codes_path=cfg.semantic_codes_path,
        batch_size=cfg.batch_size,
        num_levels=cfg.num_levels,
        flatten_codes=False,
        max_seq_len=cfg.L_full,
    )
    logger.info(
        f"[Data] short batches train/val={len(dl_short_tr)}/{len(dl_short_val)}, "
        f"long train batches={len(dl_long_tr)}"
    )
    return dl_short_tr, dl_short_val, dl_long_tr


def main(cfg: MixFlowConfig):
    device = torch.device(cfg.device)
    logger.info(f"[Init] device={device}")
    cfg.log_config()

    wandb_run = None
    if cfg.wandb_enabled:
        run_name = cfg.wandb_run_name or f"mixflow-gr-Lsoft{cfg.L_soft}-Lr{cfg.L_recent}-Lf{cfg.L_full}"
        wandb_run = wandb.init(project=cfg.wandb_project, name=run_name, config=cfg.__dict__)
        logger.info(f"[Init] wandb run: {run_name}")

    dl_short_tr, dl_short_val, dl_long = prepare_dataloaders(cfg)
    iter_long = iter(dl_long)

    model = SemanticCodeDecoder(
        codebook_size=cfg.codebook_size,
        num_levels=cfg.num_levels,
        hidden_dim=cfg.hidden_dim,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        carry_decoder_state=cfg.carry_decoder_state,
    ).to(device)

    core = unwrap_core(model)
    load_pretrained_embeddings(core, cfg.pretrained_ckpt, device)
    freeze_code_embeddings(model)

    # Collect trainable names
    enc_trainable, dec_trainable = [], []
    for n, p in core.encoder.named_parameters():
        if n.startswith("code_embeddings"):
            p.requires_grad = False
        else:
            p.requires_grad = True
            enc_trainable.append(n)
    for n, p in core.decoder.named_parameters():
        if n.startswith("code_embeddings"):
            p.requires_grad = False
        else:
            p.requires_grad = True
            dec_trainable.append(n)
    core.bos_embedding.requires_grad = True

    theta = [dict(core.encoder.named_parameters())[n] for n in enc_trainable]
    theta += [dict(core.decoder.named_parameters())[n] for n in dec_trainable]
    theta.append(core.bos_embedding)

    theta_splits = {
        "enc": len(enc_trainable),
        "dec": len(enc_trainable) + len(dec_trainable),
    }

    base_params_enc, base_buffers_enc = collect_base_params_and_buffers(core.encoder)
    base_params_dec, base_buffers_dec = collect_base_params_and_buffers(core.decoder)

    patch = SoftPatch(cfg.L_soft, cfg.hidden_dim).to(device)
    outer_params = list(patch.parameters())
    opt_eta = torch.optim.AdamW(outer_params, lr=cfg.outer_lr, weight_decay=cfg.weight_decay)
    inner_opt = MomentumInner(theta, lr=cfg.inner_lr, momentum=cfg.inner_momentum)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def inner_loss(
        theta_list: List[torch.Tensor],
        patch_emb: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        enc_theta = theta_list[: theta_splits["enc"]]
        dec_theta = theta_list[theta_splits["enc"] : theta_splits["dec"]]
        bos_param = theta_list[-1]

        override_enc = build_override(enc_trainable, enc_theta)
        override_dec = build_override(dec_trainable, dec_theta)

        enc_out, attn = run_encoder_with_patch(
            encoder=core.encoder,
            base_params=base_params_enc,
            base_buffers=base_buffers_enc,
            override_params=override_enc,
            input_ids=input_ids,
            attention_mask=attention_mask,
            patch_emb=patch_emb,
        )

        last_pos = attn.long().sum(dim=1) - 1
        context = enc_out[torch.arange(enc_out.size(0), device=device), last_pos, :] + bos_param

        logits = run_decoder_stateless(
            decoder=core.decoder,
            base_params=base_params_dec,
            base_buffers=base_buffers_dec,
            override_params=override_dec,
            context=context,
            encoder_output=enc_out,
            attention_mask=attn,
            target_ids=target_ids,
        )

        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = target_ids.reshape(-1)
        return loss_fn(logits_flat, target_flat)

    grad_fn = get_fwdrev_grad_fn_eta(inner_loss)

    it = 0
    for epoch in range(10**6):  # loop controlled by max_iters
        for batch in tqdm(dl_short_tr, desc=f"Epoch {epoch}", leave=False):
            it += 1
            if it > cfg.max_iters:
                break

            # Inner loop on short sequences with patch
            w_state, m_state = inner_opt.snapshot()
            patch_emb = patch.phi
            for _ in range(cfg.inner_steps):
                gflat = grad_fn(
                    theta,
                    patch_emb,
                    batch["input_ids"].to(device),
                    batch["target_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                inner_opt.step(gflat)

            # Outer loop on long sequences without patch
            try:
                batch_long = next(iter_long)
            except StopIteration:
                iter_long = iter(dl_long)
                batch_long = next(iter_long)

            opt_eta.zero_grad(set_to_none=True)
            loss_outer = inner_loss(
                theta,
                patch_emb=None,
                input_ids=batch_long["input_ids"].to(device),
                target_ids=batch_long["target_ids"].to(device),
                attention_mask=batch_long["attention_mask"].to(device),
            )
            loss_outer.backward()
            torch.nn.utils.clip_grad_norm_(outer_params, 1.0)
            opt_eta.step()

            inner_opt.restore(w_state, m_state)

            if it % cfg.log_every == 0:
                logger.info(f"[it {it:06d}] loss_outer={loss_outer.item():.4f}")
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss_outer": loss_outer.item(),
                            "iter": it,
                            "inner_steps": cfg.inner_steps,
                        },
                        step=it,
                    )

            # Online FME (fresh-model eval) on current patch
            if cfg.fme_eval_every and cfg.fme_eval_every > 0 and it % cfg.fme_eval_every == 0:
                fme_metrics = run_eval_online_fme(
                    patch=patch.phi,
                    cfg=cfg,
                    dl_train=dl_short_tr,
                    dl_val=dl_short_val,
                    device=device,
                )
                msg = " ".join(f"{k}={v:.4f}" for k, v in fme_metrics.items())
                logger.info(f"[FME it {it:06d}] {msg}")
                if wandb_run is not None:
                    wandb.log(fme_metrics, step=it)

        if it > cfg.max_iters:
            break

    logger.info("Training finished.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    cfg = MixFlowConfig()
    main(cfg)
