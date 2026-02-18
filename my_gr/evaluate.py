"""
Evaluation utilities for semantic code decoder.
"""

from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SemanticCodeDecoder

# ---- Helpers for FME (soft-patch injection, stateless forward) ----
def unwrap_core(model: SemanticCodeDecoder):
    return model.model if hasattr(model, "model") else model


def collect_base_params_and_buffers(module: nn.Module):
    params = {n: p for n, p in module.named_parameters()}
    buffers = {n: b for n, b in module.named_buffers()}
    return params, buffers


def build_override(names, tensors):
    return {n: t for n, t in zip(names, tensors)}


def run_encoder_with_patch(
    encoder: nn.Module,
    base_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    override_params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_emb: torch.Tensor,
):
    params = {**base_params, **override_params, **base_buffers}
    num_levels = input_ids.shape[2]
    embeddings = []
    for level in range(num_levels):
        weight = params.get(f"code_embeddings.{level}.weight", None)
        if weight is None:
            raise KeyError(f"Missing code_embeddings.{level}.weight")
        codes_at_level = input_ids[:, :, level]
        emb = nn.functional.embedding(codes_at_level, weight, padding_idx=0)
        embeddings.append(emb)
    x = torch.cat(embeddings, dim=-1)
    x = nn.functional.linear(x, params["embed_proj.weight"], params["embed_proj.bias"])
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
):
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


def load_pretrained_embeddings(core_model: nn.Module, ckpt_path: Optional[str], device: torch.device) -> None:
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if not path.exists():
        return
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    for i, emb in enumerate(core_model.code_embeddings):
        loaded = False
        for prefix in ["", "model.", "model.model."]:
            key = f"{prefix}code_embeddings.{i}.weight"
            if key in sd:
                src = sd[key]
                if src.shape != emb.weight.shape:
                    continue
                with torch.no_grad():
                    emb.weight.copy_(src.to(device))
                loaded = True
                break
        if not loaded:
            suffix = f"code_embeddings.{i}.weight"
            match = next((k for k in sd.keys() if k.endswith(suffix)), None)
            if match is not None:
                src = sd[match]
                if src.shape == emb.weight.shape:
                    with torch.no_grad():
                        emb.weight.copy_(src.to(device))
    if hasattr(core_model, "bos_embedding"):
        loaded_bos = False
        for key in ["bos_embedding", "model.bos_embedding", "model.model.bos_embedding"]:
            if key in sd:
                src = sd[key]
                if src.shape != core_model.bos_embedding.shape:
                    continue
                with torch.no_grad():
                    core_model.bos_embedding.copy_(src.to(device))
                loaded_bos = True
                break
        if not loaded_bos:
            match = next((k for k in sd.keys() if k.endswith("bos_embedding")), None)
            if match is not None:
                src = sd[match]
                if src.shape == core_model.bos_embedding.shape:
                    with torch.no_grad():
                        core_model.bos_embedding.copy_(src.to(device))

def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    scores = np.asarray(scores, dtype=np.float64)[:k]
    if len(scores) == 0:
        return 0.0
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))


def ndcg_at_k(scores: np.ndarray, k: int) -> float:
    """Compute NDCG at rank k (binary relevance)."""
    scores = np.asarray(scores, dtype=np.float64)[:k]
    dcg_max = dcg_at_k(sorted(scores, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(scores, k) / dcg_max


def hit_rate_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Average hit rate across levels. A hit occurs when the target code is in top-k.
    
    Args:
        predictions: [batch, num_levels, vocab_size]
        targets: [batch, num_levels]
    """
    batch_size = targets.shape[0]
    num_levels = targets.shape[1]
    if batch_size == 0:
        return 0.0

    top_k_preds = torch.topk(predictions, k, dim=2)[1]  # [batch, num_levels, k]
    total_hits = 0
    total_count = 0

    for level in range(num_levels):
        for i in range(batch_size):
            target_code = targets[i, level].item()
            top_k_for_level = top_k_preds[i, level].cpu().numpy()
            if target_code in top_k_for_level:
                total_hits += 1
            total_count += 1

    return total_hits / total_count if total_count > 0 else 0.0


def ndcg_batch(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """Average NDCG@k across batch and levels."""
    batch_size = targets.shape[0]
    num_levels = targets.shape[1]
    if batch_size == 0:
        return 0.0

    top_k_preds = torch.topk(predictions, k, dim=2)[1]  # [batch, num_levels, k]
    ndcg_scores = []

    for i in range(batch_size):
        for level in range(num_levels):
            target_code = targets[i, level].item()
            top_k = top_k_preds[i, level].cpu().numpy()

            relevance = np.zeros(k)
            for pos, pred_code in enumerate(top_k):
                if pred_code == target_code:
                    relevance[pos] = 1
                    break

            ndcg_scores.append(ndcg_at_k(relevance, k))

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def evaluate(
    model: SemanticCodeDecoder,
    val_dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    config,
    patch_emb: Optional[torch.Tensor] = None,
) -> dict:
    """Run evaluation on validation set.

    Two modes:
      - default (logit) mode: uses teacher-forcing logits for code-level metrics.
      - beam mode (config.eval_use_beam=True): uses model.generate with beam search and
        evaluates sequence-level hits (any generated sequence matches target sequence).
      - item ranking (config.eval_item_ranking=True): uses logits to score all items via their codes
        and computes item-level HR/NDCG (assumes codes uniquely map to items).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_hit_rates = {5: [], 10: []}
    all_ndcgs = {5: [], 10: []}
    use_beam = getattr(config, "eval_use_beam", False)
    num_beams = getattr(config, "eval_num_beams", 5)
    num_return_sequences = getattr(config, "eval_num_return_sequences", 10)
    use_item_ranking = getattr(config, "eval_item_ranking", False)

    # Load item codes if item-level ranking is enabled
    item_codes = None
    if use_item_ranking:
        semantic_codes_path = getattr(config, "semantic_codes_path", None)
        if semantic_codes_path is None:
            raise ValueError("eval_item_ranking=True requires config.semantic_codes_path")
        item_codes = torch.load(semantic_codes_path).long() + 1  # shift PAD
        item_codes = item_codes.to(device)  # [num_items, num_levels]

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch.get("target_ids")
            attention_mask = batch["attention_mask"].to(device)

            if target_ids is None:
                continue

            target_ids = target_ids.to(device)

            # Always compute CE loss (teacher forcing) for stability/early stopping
            logits, _ = model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=target_ids,
                patch_emb=patch_emb,
            )

            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_ids_flat = target_ids.reshape(-1)
            loss = loss_fn(logits_flat, target_ids_flat)

            total_loss += loss.item()
            num_batches += 1

            if use_item_ranking and not use_beam:
                # Item-level ranking using logits
                log_probs = torch.log_softmax(logits, dim=-1)  # [B, L, V]
                B, L, V = log_probs.shape
                num_items = item_codes.shape[0]

                # Gather log prob for each item code per level and sum
                scores_per_level = []
                for level in range(L):
                    lvl_probs = log_probs[:, level, :]  # [B, V]
                    code_ids = item_codes[:, level].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
                    lvl_probs_expand = lvl_probs.unsqueeze(1).expand(-1, num_items, -1)  # [B, N, V]
                    gathered = torch.gather(lvl_probs_expand, 2, code_ids.expand(B, -1, -1)).squeeze(-1)  # [B, N]
                    scores_per_level.append(gathered)
                item_scores = torch.stack(scores_per_level, dim=0).sum(dim=0)  # [B, N]

                # Find target item indices by matching codes
                # Assumes codes uniquely map to items
                matches = (item_codes.unsqueeze(0) == target_ids.unsqueeze(1)).all(dim=2)  # [B, N]
                target_idx = matches.float().argmax(dim=1)  # [B]

                sorted_idx = torch.argsort(item_scores, dim=1, descending=True)
                # Rank position (1-based)
                eq = sorted_idx == target_idx.unsqueeze(1)
                positions = eq.float().argmax(dim=1) + 1  # [B]

                for k in [5, 10]:
                    hr = (positions <= k).float().mean().item()
                    ndcg_vals = torch.where(
                        positions <= k,
                        1.0 / torch.log2(positions.float() + 1),
                        torch.zeros_like(positions, dtype=torch.float),
                    )
                    ndcg = ndcg_vals.mean().item()
                    all_hit_rates[k].append(hr)
                    all_ndcgs[k].append(ndcg)
                continue

            if use_beam:
                # Sequence-level evaluation via beam search generation
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    patch_emb=patch_emb,
                )  # [batch, num_return_sequences, num_levels]

                batch_size, num_ret, num_levels = generated.shape
                targets_exp = target_ids.unsqueeze(1).expand(-1, num_ret, -1)
                matches = (generated == targets_exp).all(dim=2)  # [batch, num_ret]

                for k_orig in [5, 10]:
                    k = min(k_orig, num_ret)
                    topk_matches = matches[:, :k].float()
                    hr = (topk_matches.sum(dim=1) > 0).float().mean().item()
                    all_hit_rates[k_orig].append(hr)

                    ndcg_vals = []
                    for row in topk_matches:
                        ndcg_vals.append(ndcg_at_k(row.cpu().numpy(), k))
                    all_ndcgs[k_orig].append(np.mean(ndcg_vals) if ndcg_vals else 0.0)
            else:
                for k in [5, 10]:
                    hr = hit_rate_at_k(logits, target_ids, k=k)
                    all_hit_rates[k].append(hr)

                    ndcg = ndcg_batch(logits, target_ids, k=k)
                    all_ndcgs[k].append(ndcg)

    val_loss = total_loss / max(num_batches, 1)
    metrics = {"loss": val_loss}
    # In beam mode loss may be zero (not computed); still report HR/NDCG
    for k in [5, 10]:
        if all_hit_rates[k]:
            metrics[f"hit_rate@{k}"] = np.mean(all_hit_rates[k])
            metrics[f"ndcg@{k}"] = np.mean(all_ndcgs[k])
        else:
            metrics[f"hit_rate@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0

    return metrics


# =========================
# Online FME evaluation (fresh-model eval)
# =========================
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


def _fme_forward(
    core,
    base_params_enc: Dict[str, torch.Tensor],
    base_buffers_enc: Dict[str, torch.Tensor],
    override_enc: Dict[str, torch.Tensor],
    base_params_dec: Dict[str, torch.Tensor],
    base_buffers_dec: Dict[str, torch.Tensor],
    override_dec: Dict[str, torch.Tensor],
    bos_param: torch.Tensor,
    patch_emb: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    target_ids = batch["target_ids"]

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
    context = enc_out[torch.arange(enc_out.size(0), device=input_ids.device), last_pos, :] + bos_param

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


def run_eval_online_fme(
    patch: torch.Tensor,
    cfg,
    dl_train: DataLoader,
    dl_val: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train a fresh GR decoder (with frozen code embeddings) on dl_train using the given patch,
    early-stop on dl_val, then report HR/NDCG on dl_val.
    """
    # Build fresh model
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
    load_pretrained_embeddings(core, getattr(cfg, "pretrained_ckpt", None), device)
    # Freeze code embeddings
    for emb in core.code_embeddings:
        for p in emb.parameters():
            p.requires_grad = False

    # Trainable params
    enc_trainable, dec_trainable = [], []
    for n, p in core.encoder.named_parameters():
        if p.requires_grad:
            enc_trainable.append(n)
    for n, p in core.decoder.named_parameters():
        if p.requires_grad:
            dec_trainable.append(n)
    core.bos_embedding.requires_grad = True

    theta = [dict(core.encoder.named_parameters())[n] for n in enc_trainable]
    theta += [dict(core.decoder.named_parameters())[n] for n in dec_trainable]
    theta.append(core.bos_embedding)
    theta_splits = {"enc": len(enc_trainable), "dec": len(enc_trainable) + len(dec_trainable)}

    base_params_enc, base_buffers_enc = collect_base_params_and_buffers(core.encoder)
    base_params_dec, base_buffers_dec = collect_base_params_and_buffers(core.decoder)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=getattr(cfg, "fme_lr", 1e-4),
        weight_decay=getattr(cfg, "fme_weight_decay", 0.0),
    )
    max_epochs = int(getattr(cfg, "fme_epochs", 3))
    patience = int(getattr(cfg, "fme_patience", 2))
    max_train_batches = getattr(cfg, "fme_max_train_batches", None)
    max_val_batches = getattr(cfg, "fme_max_val_batches", None)
    stopper = EarlyStopper(patience=patience)

    def _override_from_theta(theta_list: List[torch.Tensor]):
        enc_theta = theta_list[: theta_splits["enc"]]
        dec_theta = theta_list[theta_splits["enc"] : theta_splits["dec"]]
        bos_param = theta_list[-1]
        override_enc = build_override(enc_trainable, enc_theta)
        override_dec = build_override(dec_trainable, dec_theta)
        return override_enc, override_dec, bos_param

    patch = patch.to(device).detach()

    for epoch in range(max_epochs):
        model.train()
        for b_idx, batch in enumerate(dl_train):
            if max_train_batches is not None and b_idx >= max_train_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            override_enc, override_dec, bos_param = _override_from_theta(theta)
            loss = _fme_forward(
                core,
                base_params_enc,
                base_buffers_enc,
                override_enc,
                base_params_dec,
                base_buffers_dec,
                override_dec,
                bos_param,
                patch,
                batch,
                loss_fn,
            )
            loss.backward()
            optimizer.step()

        # Simple early stop based on HR@10 computed via evaluate-style beam metrics
        model.eval()
        # Build a lightweight eval config mimicking train.py defaults
        class _EvalCfg:
            eval_use_beam = True
            eval_num_beams = getattr(cfg, "eval_num_beams", 30)
            eval_num_return_sequences = getattr(cfg, "eval_num_return_sequences", 30)
            eval_item_ranking = False
            semantic_codes_path = getattr(cfg, "semantic_codes_path", None)
        eval_cfg = _EvalCfg()
        metrics_val = evaluate(model, dl_val, device, loss_fn, eval_cfg, patch_emb=patch)
        mean_hr10 = metrics_val.get("hit_rate@10", 0.0)
        if stopper.update(mean_hr10):
            break

    # Final evaluation on val loader using the same evaluate() path
    final_metrics = evaluate(model, dl_val, device, loss_fn, eval_cfg, patch_emb=patch)
    # Prefix keys to distinguish FME
    prefixed = {f"fme_{k}": v for k, v in final_metrics.items()}
    return prefixed
