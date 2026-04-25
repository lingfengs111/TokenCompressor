#!/usr/bin/env python3
"""Train SASRec on LOO datasets (leave-two-out)."""

from __future__ import annotations

import os
import sys
import random
import argparse
import csv
import json
import platform
import re
import secrets
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    KMeans = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.func import functional_call
from tqdm import tqdm

import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.device_manager import DeviceManager
from core.mixflow import MomentumInner, get_fwdrev_grad_fn_eta
from core.logger import setup_logger
from core.loo_dataset import LooSequenceDataset, resolve_loo_dataset, infer_loo_min_len
from core.streaming_eval import (
    finalize_eval_metrics,
    flatten_streaming_eval_metrics,
    normalize_eval_protocol,
    resolve_eval_target_positions,
    resolve_train_cutoff,
    update_rank_metrics,
)
from backbones.SASRec import SASRec
from backbones.FMLP import FMLP
from backbones.LinRec import LinRec
from backbones.LRU import LRU
from backbones.Mamba4Rec import Mamba4Rec
from backbones.Bert4rec import Bert4Rec
from backbones.GRU4Rec import GRU4Rec
from backbones.HSTU import HSTU
from backbones.HSTUOfficialish import HSTUOfficialish
from backbones.HSTUResearchAligned import HSTUResearchAligned
from backbones.LONGER import LONGER
logger = setup_logger("train-sasrec-meta-patch", log_to_file=True)


class LocalMetricsLogger:
    """Append scalar metrics to a local JSONL file and export to CSV on demand."""

    def __init__(self, log_dir: str = "logs", run_name: Optional[str] = None, enable: bool = True) -> None:
        self.enable = enable
        if not self.enable:
            self.jsonl_path = None
            self.csv_path = None
            self._jsonl_file = None
            return
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = ""
        if run_name:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")
            safe_name = safe_name[:80]
        stem = f"metrics-{timestamp}" + (f"-{safe_name}" if safe_name else "")
        self.jsonl_path = Path(log_dir) / f"{stem}.jsonl"
        self.csv_path = Path(log_dir) / f"{stem}.csv"
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")

    @staticmethod
    def _to_scalar(val: Any) -> Optional[float | int | str | bool]:
        if val is None:
            return None
        if isinstance(val, (int, float, bool, str)):
            return val
        if isinstance(val, np.generic):
            return val.item()
        if torch.is_tensor(val):
            if val.numel() == 1:
                return val.item()
            return None
        return None

    def _sanitize(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for k, v in metrics.items():
            scalar = self._to_scalar(v)
            if scalar is not None:
                sanitized[k] = scalar
        return sanitized

    def log(self, metrics: Dict[str, Any]) -> None:
        if not self.enable or self._jsonl_file is None:
            return
        record = {"_timestamp": time.time()}
        record.update(self._sanitize(metrics))
        json.dump(record, self._jsonl_file, ensure_ascii=True)
        self._jsonl_file.write("\n")
        self._jsonl_file.flush()

    def export_csv(self) -> Optional[Path]:
        if not self.enable or self.jsonl_path is None or not self.jsonl_path.exists():
            return None
        # Two-pass export to capture all fields without holding everything in memory.
        fieldnames: set[str] = set()
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fieldnames.update(record.keys())
        if not fieldnames:
            return None
        ordered: List[str] = []
        for key in ("_timestamp", "progress/step", "progress/epoch"):
            if key in fieldnames:
                ordered.append(key)
                fieldnames.remove(key)
        ordered.extend(sorted(fieldnames))
        if self.csv_path is None:
            return None
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            with self.jsonl_path.open("r", encoding="utf-8") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    writer.writerow(record)
        return self.csv_path

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()


LOCAL_METRICS_LOGGER: Optional[LocalMetricsLogger] = None


def log_metrics(metrics: Dict[str, Any]) -> None:
    if not metrics:
        return
    wandb.log(metrics)
    if LOCAL_METRICS_LOGGER is not None:
        LOCAL_METRICS_LOGGER.log(metrics)


def _uses_similarity_scoring(config: Optional[Any]) -> bool:
    if config is None:
        return False
    norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
    item_l2 = bool(getattr(config, "item_l2_norm", False))
    temperature = float(getattr(config, "temperature", 1.0) or 1.0)
    return norm != "none" or item_l2 or abs(temperature - 1.0) > 1e-12


def _resolve_item_embedding(model: nn.Module) -> nn.Module:
    if hasattr(model, "item_emb"):
        return model.item_emb
    if hasattr(model, "embedding") and hasattr(model.embedding, "token"):
        return model.embedding.token
    raise AttributeError("Unable to locate item embedding on model (expected item_emb or embedding.token).")


def _extract_prefixed_state(
    state: Optional[Dict[str, torch.Tensor]],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not state:
        return {}
    plen = len(prefix)
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def _lookup_item_embeddings(
    model: nn.Module,
    item_ids: torch.Tensor,
    *,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    item_emb = _resolve_item_embedding(model)
    if state is not None:
        item_state = _extract_prefixed_state(state, "item_emb.")
        if item_state:
            try:
                return functional_call(item_emb, item_state, args=(item_ids,), kwargs={})
            except Exception:
                weight = item_state.get("weight")
                if weight is not None:
                    return F.embedding(item_ids, weight)
    return item_emb(item_ids)


def _all_item_embeddings(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "all_item_embeddings"):
        return model.all_item_embeddings()
    item_emb = _resolve_item_embedding(model)
    if hasattr(item_emb, "all_embeddings"):
        return item_emb.all_embeddings()
    weight = getattr(item_emb, "weight", None)
    if weight is None:
        raise AttributeError("Item embedding module does not expose weight/all_embeddings.")
    return weight


def _sequence_lengths_from_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return (input_ids != 0).sum(dim=1)


def _normalize_user_embeddings(
    projected: torch.Tensor,
    config: Any,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    if model is not None and hasattr(model, "postprocess_query_embeddings"):
        return model.postprocess_query_embeddings(projected)
    norm = str(getattr(config, "user_embedding_norm", "none") or "none").lower()
    eps = float(getattr(config, "l2_norm_eps", 1e-6) or 1e-6)
    if norm == "none":
        return projected
    if norm == "l2_norm":
        return F.normalize(projected, p=2, dim=-1, eps=eps)
    if norm == "layer_norm":
        return F.layer_norm(projected, normalized_shape=(projected.size(-1),), eps=eps)
    raise ValueError(f"Unsupported user_embedding_norm: {norm}")


def _normalize_item_embeddings(
    item_embeddings: torch.Tensor,
    config: Any,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    if model is not None and hasattr(model, "postprocess_item_embeddings"):
        return model.postprocess_item_embeddings(item_embeddings)
    if not bool(getattr(config, "item_l2_norm", False)):
        return item_embeddings
    eps = float(getattr(config, "l2_norm_eps", 1e-6) or 1e-6)
    return F.normalize(item_embeddings, p=2, dim=-1, eps=eps)


def _apply_similarity_logits(
    projected: torch.Tensor,
    item_embeddings: torch.Tensor,
    config: Any,
    model: Optional[nn.Module] = None,
    *,
    item_ids: Optional[torch.Tensor] = None,
    seq_lengths: Optional[torch.Tensor] = None,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    projected = _normalize_user_embeddings(projected, config, model=model)
    item_embeddings = _normalize_item_embeddings(item_embeddings.to(projected.dtype), config, model=model)
    if item_embeddings.dim() == projected.dim():
        logits = (projected * item_embeddings).sum(dim=-1)
    elif item_embeddings.dim() == projected.dim() + 1:
        logits = (projected.unsqueeze(-2) * item_embeddings).sum(dim=-1)
    elif projected.dim() == 2 and item_embeddings.dim() == 3:
        logits = torch.einsum("bd,bnd->bn", projected, item_embeddings)
    else:
        raise ValueError(
            f"Unsupported shapes for similarity logits: projected={tuple(projected.shape)}, "
            f"item_embeddings={tuple(item_embeddings.shape)}"
        )
    temperature = float(getattr(config, "temperature", 1.0) or 1.0)
    if temperature != 1.0:
        logits = logits / temperature
    if (
        model is not None
        and hasattr(model, "score_calibration")
        and item_ids is not None
        and seq_lengths is not None
    ):
        score_head = getattr(model, "score_calibration", None)
        if state is not None and score_head is not None:
            score_state = _extract_prefixed_state(state, "score_calibration.")
            if score_state:
                return functional_call(
                    score_head,
                    score_state,
                    args=(logits, item_ids, seq_lengths),
                    kwargs={},
                )
        if score_head is not None:
            logits = score_head(logits, item_ids, seq_lengths)
    return logits


def _strip_hidden_states_for_scoring(
    model: nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    use_patch: bool,
) -> torch.Tensor:
    if use_patch and int(getattr(model, "patch_len", 0) or 0) > 0:
        if hasattr(model, "strip_patch_tokens"):
            try:
                hidden_states = model.strip_patch_tokens(hidden_states)
            except TypeError:
                hidden_states = model.strip_patch_tokens(hidden_states, input_ids.size(1))
        elif hasattr(model, "_strip_patch_tokens"):
            try:
                hidden_states = model._strip_patch_tokens(hidden_states)
            except TypeError:
                hidden_states = model._strip_patch_tokens(hidden_states, input_ids.size(1))
    if hasattr(model, "_strip_shared_tokens"):
        hidden_states = model._strip_shared_tokens(hidden_states)
    if bool(getattr(model, "persrec_enable", False)) and int(getattr(model, "persrec_num_tokens", 0) or 0) > 0:
        pre, _ = model._resolve_persrec_lengths(input_ids.size(1))
        k = int(model.persrec_num_tokens)
        hidden_states = torch.cat([hidden_states[:, :pre, :], hidden_states[:, pre + k :, :]], dim=1)
    return hidden_states


def _call_forward_features_with_optional_user_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    user_ids: Optional[torch.Tensor] = None,
    patch_params: Optional[Any] = None,
    return_gating: bool = False,
    use_patch: bool = True,
) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
    kwargs = {
        "patch_params": patch_params,
        "use_patch": use_patch,
    }
    if return_gating:
        kwargs["return_gating"] = True
    try:
        if user_ids is not None:
            out = model.forward_features(input_ids, user_ids=user_ids, **kwargs)
        else:
            out = model.forward_features(input_ids, **kwargs)
    except TypeError:
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("return_gating", None)
        if user_ids is not None:
            out = model.forward_features(input_ids, **retry_kwargs)
        else:
            out = model.forward_features(input_ids, **retry_kwargs)
    if return_gating:
        if isinstance(out, tuple):
            return out
        return out, None
    if isinstance(out, tuple):
        return out[0]
    return out


def _compute_projected_hidden_with_optional_user_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    use_head: bool = True,
    patch_params: Optional[Any] = None,
    return_gating: bool = False,
    head_params: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = _call_forward_features_with_optional_user_ids(
        model,
        input_ids,
        user_ids=user_ids,
        patch_params=patch_params,
        return_gating=return_gating,
        use_patch=use_patch,
    )
    if return_gating:
        hidden_states, gating_weights = out
    else:
        hidden_states = out
        gating_weights = None
    hidden_states = _strip_hidden_states_for_scoring(
        model,
        hidden_states,
        input_ids,
        use_patch=use_patch,
    )
    if use_head:
        hidden_states = model.apply_head(hidden_states, head_params=head_params)
    if return_gating:
        return hidden_states, gating_weights
    return hidden_states


def _score_candidates_with_optional_user_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    item_indices: torch.Tensor,
    *,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    use_head: bool = True,
) -> torch.Tensor:
    projected = _compute_projected_hidden_with_optional_user_ids(
        model,
        input_ids,
        user_ids=user_ids,
        use_patch=use_patch,
        use_head=use_head,
        return_gating=False,
    )
    final_hidden = projected[:, -1, :]
    candidate_embs = _resolve_item_embedding(model)(item_indices).to(final_hidden.dtype)
    seq_lengths = _sequence_lengths_from_input_ids(input_ids)
    return _apply_similarity_logits(
        final_hidden,
        candidate_embs,
        getattr(model, "config", None),
        model=model,
        item_ids=item_indices,
        seq_lengths=seq_lengths,
    )


def _compute_sequence_logits_with_optional_user_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    *,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    use_head: bool = True,
    patch_params: Optional[Any] = None,
    return_gating: bool = False,
    head_params: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    projected_out = _compute_projected_hidden_with_optional_user_ids(
        model,
        input_ids,
        user_ids=user_ids,
        use_patch=use_patch,
        use_head=use_head,
        patch_params=patch_params,
        return_gating=return_gating,
        head_params=head_params,
    )
    if return_gating:
        projected, gating_weights = projected_out
    else:
        projected = projected_out
        gating_weights = None
    item_emb = _resolve_item_embedding(model)
    pos_embs = item_emb(pos_ids).to(projected.dtype)
    neg_embs = item_emb(neg_ids).to(projected.dtype)
    seq_lengths = _sequence_lengths_from_input_ids(input_ids)
    pos_logits = _apply_similarity_logits(
        projected,
        pos_embs,
        getattr(model, "config", None),
        model=model,
        item_ids=pos_ids,
        seq_lengths=seq_lengths,
    )
    neg_logits = _apply_similarity_logits(
        projected,
        neg_embs,
        getattr(model, "config", None),
        model=model,
        item_ids=neg_ids,
        seq_lengths=seq_lengths,
    )
    return pos_logits, neg_logits, gating_weights


def _call_training_step_with_optional_user_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    *,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    patch_params: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _uses_similarity_scoring(getattr(model, "config", None)) and hasattr(model, "forward_features"):
        pos_logits, neg_logits, _ = _compute_sequence_logits_with_optional_user_ids(
            model,
            input_ids,
            pos_ids,
            neg_ids,
            user_ids=user_ids,
            use_patch=use_patch,
            use_head=True,
            patch_params=patch_params,
            return_gating=False,
        )
        return pos_logits, neg_logits
    kwargs = {
        "patch_params": patch_params,
        "use_patch": use_patch,
    }
    if user_ids is not None:
        try:
            return model.training_step(input_ids, pos_ids, neg_ids, user_ids=user_ids, **kwargs)
        except TypeError:
            return model.training_step(input_ids, pos_ids, neg_ids, **kwargs)
    return model.training_step(input_ids, pos_ids, neg_ids, **kwargs)


def _call_predict_with_optional_user_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    item_indices: torch.Tensor,
    *,
    user_ids: Optional[torch.Tensor] = None,
    use_patch: bool = True,
    use_head: bool = True,
) -> torch.Tensor:
    if _uses_similarity_scoring(getattr(model, "config", None)) and hasattr(model, "forward_features"):
        return _score_candidates_with_optional_user_ids(
            model,
            input_ids,
            item_indices,
            user_ids=user_ids,
            use_patch=use_patch,
            use_head=use_head,
        )
    kwargs = {
        "use_patch": use_patch,
        "use_head": use_head,
    }
    if user_ids is not None:
        try:
            return model.predict(input_ids, item_indices, user_ids=user_ids, **kwargs)
        except TypeError:
            return model.predict(input_ids, item_indices, **kwargs)
    return model.predict(input_ids, item_indices, **kwargs)


def _str2bool(value: str | bool | None) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SASRec meta-patch with long/short adaptation.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--pretrained_ckpt_path", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--inner_seq_length", type=int, default=None)
    parser.add_argument("--eval_seq_length", type=int, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deterministic", type=_str2bool, default=None)
    parser.add_argument("--use_gradient_checkpointing", type=_str2bool, default=None)
    parser.add_argument("--use_flash_attention", type=_str2bool, default=None)
    parser.add_argument("--sasrec_attention_norm", type=str, choices=["softmax", "softmax_custom", "softmax1"], default=None)
    parser.add_argument("--sasrec_enable_relative_attention_bias", type=_str2bool, default=None)
    parser.add_argument("--sasrec_use_rope", type=_str2bool, default=None)
    parser.add_argument("--sasrec_rope_base", type=float, default=None)
    parser.add_argument("--hstu_linear_dim", type=int, default=None)
    parser.add_argument("--hstu_attention_dim", type=int, default=None)
    parser.add_argument("--hstu_linear_activation", type=str, default=None)
    parser.add_argument("--hstu_attn_dropout", type=float, default=None)
    parser.add_argument("--hstu_enable_relative_attention_bias", type=_str2bool, default=None)
    parser.add_argument("--hstu_normalization", type=str, default=None)
    parser.add_argument("--hstu_concat_ua", type=_str2bool, default=None)
    parser.add_argument("--hstu_epsilon", type=float, default=None)
    parser.add_argument("--hstu_parametric_block_norm", type=_str2bool, default=None)
    parser.add_argument("--lru_num_blocks", type=int, default=None)
    parser.add_argument("--lru_dropout", type=float, default=None)
    parser.add_argument("--lru_attn_dropout", type=float, default=None)
    parser.add_argument("--mamba_num_layers", type=int, default=None)
    parser.add_argument("--mamba_d_state", type=int, default=None)
    parser.add_argument("--mamba_d_conv", type=int, default=None)
    parser.add_argument("--mamba_expand", type=int, default=None)
    parser.add_argument("--mamba_dropout", type=float, default=None)
    parser.add_argument("--hllm_text_user_pretrain_dir", type=str, default=None)
    parser.add_argument("--hllm_text_item_feature_path", type=str, default=None)
    parser.add_argument("--hllm_text_dtype", type=str, default=None)
    parser.add_argument("--hllm_text_user_model_class", type=str, default=None)
    parser.add_argument("--hllm_text_freeze_user_llm", type=_str2bool, default=None)
    parser.add_argument("--hllm_text_normalize_logits", type=_str2bool, default=None)
    parser.add_argument("--hllm_text_logit_scale", type=float, default=None)
    parser.add_argument("--hllm_text_trust_remote_code", type=_str2bool, default=None)
    parser.add_argument("--hllm_text_lightweight_checkpoint", type=_str2bool, default=None)
    parser.add_argument("--hllm_item_pretrain_dir", type=str, default=None)
    parser.add_argument("--hllm_user_pretrain_dir", type=str, default=None)
    parser.add_argument("--hllm_text_path", type=str, default=None)
    parser.add_argument("--hllm_text_column", type=str, default=None)
    parser.add_argument("--hllm_item_id_column", type=str, default=None)
    parser.add_argument("--hllm_dtype", type=str, default=None)
    parser.add_argument("--hllm_item_model_class", type=str, default=None)
    parser.add_argument("--hllm_user_model_class", type=str, default=None)
    parser.add_argument("--hllm_item_llm_init", type=_str2bool, default=None)
    parser.add_argument("--hllm_user_llm_init", type=_str2bool, default=None)
    parser.add_argument("--hllm_freeze_item_llm", type=_str2bool, default=None)
    parser.add_argument("--hllm_freeze_user_llm", type=_str2bool, default=None)
    parser.add_argument("--hllm_normalize_logits", type=_str2bool, default=None)
    parser.add_argument("--hllm_logit_scale", type=float, default=None)
    parser.add_argument("--hllm_trust_remote_code", type=_str2bool, default=None)
    parser.add_argument("--hllm_max_text_length", type=int, default=None)
    parser.add_argument("--hllm_item_emb_token_n", type=int, default=None)
    parser.add_argument("--hllm_item_encode_batch_size", type=int, default=None)
    parser.add_argument("--hllm_eval_cache_full_items", type=_str2bool, default=None)
    parser.add_argument("--hllm_item_emb_init_std", type=float, default=None)
    parser.add_argument("--user_embedding_norm", type=str, default=None)
    parser.add_argument("--item_l2_norm", type=_str2bool, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--l2_norm_eps", type=float, default=None)
    parser.add_argument("--enable_score_item_bias", type=_str2bool, default=None)
    parser.add_argument("--enable_score_length_bias", type=_str2bool, default=None)
    parser.add_argument("--enable_score_length_scale", type=_str2bool, default=None)
    parser.add_argument("--score_length_bucket_size", type=int, default=None)

    # Optimization
    parser.add_argument("--inner_lr", type=float, default=None)
    parser.add_argument("--outer_lr", type=float, default=None)
    parser.add_argument("--outer_update_every", type=int, default=None)
    parser.add_argument("--inner_steps", type=int, default=None)
    parser.add_argument("--inner_momentum", type=float, default=None)
    parser.add_argument("--inner_grad_clip", type=float, default=None)
    parser.add_argument("--outer_grad_clip", type=float, default=None)
    parser.add_argument("--outer_weight_decay", type=float, default=None)
    parser.add_argument("--full_finetune", type=_str2bool, default=None)
    parser.add_argument("--meta_truncate_steps", type=int, default=None)
    parser.add_argument("--lambda_meta", type=float, default=None)
    parser.add_argument("--outer_loss_mode", type=str, default=None)
    parser.add_argument("--outer_loss_decay", type=float, default=None)
    parser.add_argument("--outer_distill", type=str, default=None)
    parser.add_argument("--outer_distill_temperature", type=float, default=None)
    parser.add_argument("--outer_neg_samples", type=int, default=None)
    parser.add_argument("--outer_tail_weight", type=float, default=None)
    parser.add_argument("--outer_mid_weight", type=float, default=None)
    parser.add_argument("--outer_mid_samples", type=int, default=None)
    parser.add_argument("--outer_gt_weight", type=float, default=None)
    parser.add_argument("--outer_mid_rel_weight", type=float, default=None)
    parser.add_argument("--outer_patch_future_weight", type=float, default=None)
    parser.add_argument("--outer_patch_future_steps", type=int, default=None)
    parser.add_argument("--outer_patch_boundary_weight", type=float, default=None)
    parser.add_argument("--outer_patch_boundary_steps", type=int, default=None)
    parser.add_argument("--inner_loss_mode", type=str, default=None)
    parser.add_argument("--prefix_len", type=int, default=None)
    parser.add_argument("--shared_prefix_len", type=int, default=None)
    parser.add_argument("--shared_token_len", dest="shared_prefix_len", type=int, default=None)
    parser.add_argument("--shared_prefix_init_std", type=float, default=None)
    parser.add_argument("--shared_token_init_std", dest="shared_prefix_init_std", type=float, default=None)
    parser.add_argument("--prefix_source", type=str, default=None)
    parser.add_argument("--prefix_tail_positions", type=_str2bool, default=None)
    parser.add_argument("--patch_after_prefix", type=_str2bool, default=None)
    parser.add_argument("--patch_use_position_embeddings", type=_str2bool, default=None)
    parser.add_argument("--inner_drop_prefix", type=_str2bool, default=None)
    parser.add_argument("--inner_reset_every", type=int, default=None)
    parser.add_argument("--supervise_prefix_targets", type=_str2bool, default=None)
    parser.add_argument("--drop_unseen_items", type=_str2bool, default=None)

    # Patch/Gating
    parser.add_argument("--num_patches", type=int, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--use_gating", type=_str2bool, default=None)
    parser.add_argument("--gating_hidden_dim", type=int, default=None)
    parser.add_argument("--gating_temperature", type=float, default=None)
    parser.add_argument("--gating_noise_std", type=float, default=None)
    parser.add_argument("--gating_balance_weight", type=float, default=None)
    parser.add_argument("--patch_orth_weight", type=float, default=None)
    parser.add_argument("--patch_inner_orth_weight", type=float, default=None)
    parser.add_argument("--patch_routing", type=str, default=None)
    parser.add_argument("--kmeans_max_samples", type=int, default=None)
    parser.add_argument("--kmeans_max_iters", type=int, default=None)
    parser.add_argument("--input_emb_lora_rank", type=int, default=None)
    parser.add_argument("--input_emb_lora_alpha", type=float, default=None)
    parser.add_argument("--train_input_emb_lora", type=_str2bool, default=None)
    parser.add_argument("--attn_lora_rank", type=int, default=None)
    parser.add_argument("--attn_lora_alpha", type=float, default=None)
    parser.add_argument("--attn_lora_blocks", type=str, default=None)
    parser.add_argument("--train_attn_lora", type=_str2bool, default=None)

    # Eval/logging
    parser.add_argument("--val_eval_every_epochs", type=int, default=None)
    parser.add_argument("--eval_sample_size", type=int, default=None)
    parser.add_argument("--steps_per_train_log", type=int, default=None)
    parser.add_argument("--save_best_model", type=_str2bool, default=None)
    parser.add_argument("--eval_before_train", type=_str2bool, default=None)
    parser.add_argument("--eval_after_train", type=_str2bool, default=None)
    parser.add_argument("--eval_protocol", type=str, default=None)
    parser.add_argument("--last_k_eval_test", type=int, default=None)
    parser.add_argument("--streaming_eval_last_k", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_epochs", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--checkpoint_mode", type=str, default=None)
    parser.add_argument("--strict_load_pretrained", type=_str2bool, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--persistent_workers", type=_str2bool, default=None)
    parser.add_argument("--pin_memory", type=_str2bool, default=None)
    parser.add_argument("--enable_timing", type=_str2bool, default=None)
    parser.add_argument("--timing_window", type=int, default=None)
    parser.add_argument("--train_bias", dest="inner_train_bias", type=_str2bool, default=None)
    parser.add_argument("--train_layernorm", dest="inner_train_layernorm", type=_str2bool, default=None)
    parser.add_argument("--train_score_head", dest="inner_train_score_head", type=_str2bool, default=None)

    return parser


def _maybe_path(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _maybe_none_string(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def apply_overrides_from_args(config: SASRecConfig, args: argparse.Namespace) -> None:
    for key, val in vars(args).items():
        if val is None:
            continue
        if not hasattr(config, key):
            continue
        val = _maybe_none_string(val)
        if val is None:
            setattr(config, key, None)
            continue
        if key in {
            "data_dir",
            "data_txt_path",
            "checkpoint_dir",
            "hllm_text_item_feature_path",
            "hllm_text_path",
        }:
            setattr(config, key, _maybe_path(val))
        else:
            setattr(config, key, val)


def apply_overrides_from_dict(config: SASRecConfig, overrides: Dict[str, Any]) -> None:
    for key, val in overrides.items():
        if key.startswith("_"):
            continue
        if not hasattr(config, key):
            continue
        val = _maybe_none_string(val)
        if key in {
            "data_dir",
            "data_txt_path",
            "checkpoint_dir",
            "hllm_text_item_feature_path",
            "hllm_text_path",
        }:
            setattr(config, key, _maybe_path(val))
        else:
            setattr(config, key, val)


def resolve_inner_bitfit_flags(config: Any) -> Tuple[bool, bool]:
    legacy = bool(getattr(config, "inner_train_bias_ln", True))
    enable_bias = getattr(config, "inner_train_bias", None)
    enable_layernorm = getattr(config, "inner_train_layernorm", None)
    if enable_bias is None:
        enable_bias = legacy
    if enable_layernorm is None:
        enable_layernorm = legacy
    return bool(enable_bias), bool(enable_layernorm)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_eval_seed(config: "SASRecConfig", mode: str, streaming_last_k: int = 0) -> int:
    base_seed = getattr(config, "eval_seed", None)
    if base_seed is None:
        base_seed = getattr(config, "seed", 2026)
    offset = 0 if str(mode).lower() == "val" else 1_000_003
    offset += int(streaming_last_k or 0) * 7_919
    return int(base_seed) + offset


def _build_eval_rng(config: "SASRecConfig", mode: str, streaming_last_k: int = 0) -> np.random.RandomState:
    return np.random.RandomState(_resolve_eval_seed(config, mode, streaming_last_k))


def _build_eval_torch_generator(
    config: "SASRecConfig",
    mode: str,
    device: torch.device,
    streaming_last_k: int = 0,
) -> torch.Generator:
    _ = device
    generator = torch.Generator()
    generator.manual_seed(_resolve_eval_seed(config, mode, streaming_last_k))
    return generator


def _get_eval_seen_items_cache(
    dataset,
    users: List[int],
) -> Dict[int, set[int]]:
    cache = getattr(dataset, "_eval_seen_items_cache", None)
    if cache is None:
        cache = {}
        setattr(dataset, "_eval_seen_items_cache", cache)
    for user in users:
        if user not in cache:
            cache[user] = {int(x) for x in dataset.user_seq[user] if x > 1}
    return cache


def _get_eval_target_positions_cache(
    dataset,
    users: List[int],
    *,
    mode: str,
    streaming_last_k: int,
    config: "SASRecConfig",
) -> Dict[int, List[int]]:
    protocol = normalize_eval_protocol(getattr(config, "eval_protocol", "legacy_loo"))
    holdout_last_k = int(getattr(config, "last_k_eval_test", 0) or 0)
    cache_key = (str(mode), int(streaming_last_k or 0), protocol, holdout_last_k)
    cache_store = getattr(dataset, "_eval_target_positions_cache", None)
    if cache_store is None:
        cache_store = {}
        setattr(dataset, "_eval_target_positions_cache", cache_store)
    per_user_cache = cache_store.setdefault(cache_key, {})
    for user in users:
        if user not in per_user_cache:
            per_user_cache[user] = resolve_eval_target_positions(
                len(dataset.user_seq[user]),
                mode=mode,
                streaming_last_k=streaming_last_k,
                eval_protocol=protocol,
                last_k_eval_test=holdout_last_k,
            )
    return per_user_cache


def _get_eval_fixed_negative_cache(
    dataset,
    users: List[int],
    seen_items_cache: Dict[int, set[int]],
) -> Dict[int, int]:
    if not hasattr(dataset, "neg_item_by_user"):
        return {}
    cache = getattr(dataset, "_eval_fixed_negative_cache", None)
    if cache is None:
        cache = {}
        setattr(dataset, "_eval_fixed_negative_cache", cache)
    for user in users:
        if user in cache:
            continue
        fixed_neg = dataset.neg_item_by_user.get(user)
        if fixed_neg and fixed_neg > 1 and fixed_neg not in seen_items_cache[user]:
            cache[user] = int(fixed_neg)
    return cache


@dataclass
class SASRecConfig:
    """Configuration for SASRec training (LOO datasets)."""

    dataset: str = "taobao_loo202"
    data_dir: Optional[Path] = None
    data_txt_path: Optional[Path] = None
    checkpoint_dir: Path = field(default_factory=lambda: Path(ROOT_DIR) / "checkpoints")

    # Model parameters
    backbone: str = "sasrec"  # sasrec | hstu | hllm | hllm_text | longer | fmlp | linrec | lru | mamba4rec | bert4rec | gru4rec
    max_seq_length: Optional[int] = None  # Maximum sequence length
    hidden_units: int = 128  # Hidden dimension size
    num_blocks: int = 2  # Number of transformer blocks
    num_heads: int = 1  # Number of attention heads
    dropout_rate: float = 0.2  # Dropout rate
    right_align_positions: bool = True  # Right-align positional ids for short inputs
    sasrec_attention_norm: str = "softmax"
    sasrec_enable_relative_attention_bias: bool = False
    sasrec_use_rope: bool = False
    sasrec_rope_base: float = 10000.0
    hstu_linear_dim: Optional[int] = None
    hstu_attention_dim: Optional[int] = None
    hstu_linear_activation: str = "silu"
    hstu_attn_dropout: Optional[float] = None
    hstu_enable_relative_attention_bias: bool = False
    hstu_normalization: str = "rel_bias"
    hstu_concat_ua: bool = False
    hstu_epsilon: float = 1e-6
    hstu_parametric_block_norm: bool = False

    # HLLMText-specific
    hllm_text_user_pretrain_dir: Optional[str] = None
    hllm_text_item_feature_path: Optional[Path] = None
    hllm_text_dtype: str = "float32"  # float32 | bfloat16 | float16 | auto
    hllm_text_user_model_class: str = "auto_model"  # auto_model | causal_lm
    hllm_text_freeze_user_llm: bool = True
    hllm_text_normalize_logits: bool = True
    hllm_text_logit_scale: float = 1.0 / 0.07
    hllm_text_trust_remote_code: bool = True
    hllm_text_lightweight_checkpoint: bool = True
    hllm_item_pretrain_dir: Optional[str] = None
    hllm_user_pretrain_dir: Optional[str] = None
    hllm_text_path: Optional[Path] = None
    hllm_text_column: str = "text"
    hllm_item_id_column: str = "model_item_id"
    hllm_dtype: str = "float32"  # float32 | bfloat16 | float16 | auto
    hllm_item_model_class: str = "causal_lm"  # auto_model | causal_lm
    hllm_user_model_class: str = "causal_lm"  # auto_model | causal_lm
    hllm_item_llm_init: bool = True
    hllm_user_llm_init: bool = True
    hllm_freeze_item_llm: bool = False
    hllm_freeze_user_llm: bool = False
    hllm_normalize_logits: bool = True
    hllm_logit_scale: float = 1.0 / 0.07
    hllm_trust_remote_code: bool = True
    hllm_max_text_length: int = 128
    hllm_item_emb_token_n: int = 1
    hllm_item_encode_batch_size: int = 64
    hllm_eval_cache_full_items: bool = True
    hllm_item_emb_init_std: float = 0.02
    user_embedding_norm: str = "none"  # none | l2_norm | layer_norm
    item_l2_norm: bool = False
    temperature: float = 1.0
    l2_norm_eps: float = 1e-6
    enable_score_item_bias: bool = False
    enable_score_length_bias: bool = False
    enable_score_length_scale: bool = False
    score_length_bucket_size: int = 20
    num_negatives: int = 128  # Number of sampled negatives per position
    sampled_softmax_chunk_size: int = 4096  # Chunk size for sampled-softmax style logits

    # LONGER-specific
    longer_global_tokens: int = 4
    longer_merge_size: int = 4
    longer_merge_pool: str = "last"  # last | mean
    longer_inner_num_layers: int = 1

    # LRU-specific
    lru_num_blocks: Optional[int] = 2
    lru_dropout: Optional[float] = None
    lru_attn_dropout: Optional[float] = None

    # Mamba4Rec-specific
    mamba_num_layers: Optional[int] = None
    mamba_d_state: int = 32
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dropout: Optional[float] = None

    # Short/long split for meta-training
    inner_seq_length: int = 50  # Short-view length (inner loop)
    eval_seq_length: int = 50  # Eval length (short-view)

    # Meta-patch parameters
    num_patches: int = 4  # Number of patches in the bank
    patch_len: int = 10  # Length of each patch (soft prompt tokens)
    use_gating: bool = True  # Use gating network to mix patches
    gating_hidden_dim: int = 64  # Hidden size for gating MLP
    patch_init_std: float = 0.0  # Init std for patch/gating params

    # head
    head_residual: bool = True  # Use residual head: output = x + head(x)
    head_zero_init: bool = True  # Zero-init head for identity start when residual
    enable_projection_head: bool = True  # If False, skip projection head entirely
    head_use_gelu: bool = False  # Keep head lightweight by default
    head_use_ln: bool = False  # Keep head lightweight by default
    # gating
    gating_pool: str = "mean"  # mean | last
    gating_init_std: float = 0.2  # Init std for gating params
    gating_temperature: float = 0.5  # Softmax temperature (<1 sharpens)
    gating_noise_std: float = 0.01  # Logit noise for symmetry breaking
    gating_balance_weight: float = 0.0  # encourage batch-average gating usage to stay spread
    patch_orth_weight: float = 0.0  # encourage patch bank entries to stay decorrelated
    patch_inner_orth_weight: float = 0.0  # encourage tokens inside each patch entry to stay decorrelated
    
    patch_routing: str = "learned"  # learned | kmeans | user_table | random | single
    kmeans_max_iters: int = 25
    kmeans_seed: int = 2026
    kmeans_max_samples: int = 20000
    input_emb_lora_rank: int = 8  # 0 disables input-side embedding delta
    input_emb_lora_alpha: float = 8.0  # scale for input embedding delta
    train_input_emb_lora: bool = True  # train input-side embedding delta in outer loop
    attn_lora_rank: int = 4  # 0 disables attention LoRA
    attn_lora_alpha: float = 8.0  # scale for attention LoRA
    attn_lora_blocks: str = "all"  # all | first | last | comma-separated block ids
    train_attn_lora: bool = True  # train attention LoRA in outer loop

    # Training parameters
    batch_size: int = 1024  # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    seed: int = 2026  # Global RNG seed
    eval_seed: Optional[int] = None  # Fixed RNG for sampled val/test negatives
    deterministic: bool = False  # Enable deterministic ops (slower)

    # Meta-learning (MixFlow) parameters
    inner_steps: int = 1  # Inner updates per outer update block
    inner_lr: float = 5e-5  # Inner (head) learning rate
    inner_momentum: float = 0.0  # Inner momentum (0 disables)
    inner_grad_clip: float = 0.0  # Clip inner gradients (0 disables)
    outer_update_every: int = 20  # Perform outer update every N inner steps
    outer_lr: float = 1e-4  # Outer (patch) learning rate
    outer_weight_decay: float = 0.0  # Outer weight decay
    outer_grad_clip: float = 1.0  # Clip patch gradients (0 disables)
    full_finetune: bool = False  # Train the whole student model instead of PEFT-only
    meta_truncate_steps: int = 1  # Truncated unroll steps for meta-gradients
    lambda_meta: float = 1.0  # Scaling for meta-gradients
    val_batch_size: int = 1024  # Batch size for outer loop (clean val)
    val_eval_every_epochs: int = 4  # Run meta-patch eval on val every N epochs
    outer_loss_mode: str = "decay"  # all | last | decay
    outer_loss_decay: float = 0.9  # decay factor for outer loss when mode=decay
    outer_distill: str = "kl"  # kl | soft_bce | mse
    outer_distill_temperature: float = 1.0  # temperature for distillation
    outer_neg_samples: int = 1  # number of negative samples for outer distillation
    outer_tail_weight: float = 1.0  # weight for tail distillation (0 disables)
    outer_mid_weight: float = 1.0  # weight for middle-segment distillation (0 disables)
    outer_mid_samples: int = 0  # number of middle positions to distill (0 => patch_len)
    outer_gt_weight: float = 1.0  # weight for direct GT loss on short+patch (0 disables)
    outer_gt_loss_type: str = "bce"  # bce | sampled_softmax
    outer_gt_num_negatives: int = 0  # <=0 => follow num_negatives
    outer_gt_chunk_size: int = 0  # <=0 => follow sampled_softmax_chunk_size
    outer_mid_rel_weight: float = 0.0  # weight for relational mid-segment alignment (0 disables)
    outer_patch_future_weight: float = 0.0  # weight for patch-token -> real future-item BCE (0 disables)
    outer_patch_future_steps: int = 0  # number of future items supervised by patch tokens (0 => patch_len)
    outer_patch_boundary_weight: float = 0.0  # weight for patch-token -> dropped boundary-item BCE (0 disables)
    outer_patch_boundary_steps: int = 0  # number of dropped boundary items supervised by patch tokens (0 => patch_len)
    inner_loss_mode: str = "match_outer"  # match_outer | all | last | decay
    prefix_len: int = 5  # number of real prefix items to retain in short-view inputs
    shared_prefix_len: int = 0  # number of shared learned tokens prepended before items when >0
    shared_prefix_init_std: float = 0.02  # init std for shared token embeddings
    prefix_source: str = "head"  # head | middle | uniform | random
    prefix_tail_positions: bool = True  # align prefix to start and tail to end positions
    patch_after_prefix: bool = True  # insert patch between prefix and tail when prefix_len>0
    patch_use_position_embeddings: bool = False  # add absolute position embeddings to patch tokens
    inner_drop_prefix: bool = False  # keep prefix in inner loop by default
    inner_reset_every: int = 0  # reset inner parameters every N steps (0 disables)
    supervise_prefix_targets: bool = False  # if True, keep GT labels on prefix positions instead of context-only
    

    # Unseen item handling
    drop_unseen_items: bool = True  # If True, drop sequences with unseen items (no UNK mapping)
    inner_unk_mask_prob: float = 0.0  # Simulate UNK in inner loop by masking input ids to 1

    # Inner-loop trainable parameters
    inner_train_bias_ln: bool = True  # Legacy BitFit switch: biases + LayerNorm
    inner_train_bias: Optional[bool] = None  # None => follow inner_train_bias_ln
    inner_train_layernorm: Optional[bool] = None  # None => follow inner_train_bias_ln
    inner_train_head: bool = True  # Projection head parameters
    inner_train_score_head: bool = True  # Explicit scorer calibration params

    # Meta-test adaptation (per-user TTA)
    meta_test_adapt_steps: int = 0 # 0 => no adaptation (copy head/bias/LN)
    meta_test_adapt_lr: float = 5e-5  # Adaptation LR for BitFit params
    meta_test_init_from_trained: bool = False  # Used when adapt_steps>0: True=start from trained theta; False=from initial defaults
    meta_test_unk_mask_prob: Optional[float] = None  # None -> follow inner_unk_mask_prob (if not dropping)

    # Training settings
    steps_per_train_log: int = 100  # Log training progress every N steps
    # ⚠️之前eval用的是100导致结果看起来很高
    eval_sample_size: int = 1000  # Total candidates per user when eval_mode="sampled" (includes target)
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing (memory fallback)
    use_flash_attention: bool = False  # Disable flash attention (use math kernel)
    eval_before_train: bool = False  # Run baseline/meta eval before training
    eval_after_train: bool = True  # Run val eval after training
    eval_protocol: str = "legacy_loo"  # legacy_loo | holdout_anchor
    last_k_eval_test: int = 10  # Used when eval_protocol=holdout_anchor
    streaming_eval_last_k: int = 0  # If >1, run extra rolling final-test eval on the last K targets
    early_stop_patience: int = 0  # Number of val evals without improvement before stopping (0 disables)
    early_stop_min_epochs: int = 0  # Ignore early stop before this epoch count
    early_stop_min_delta: float = 0.0  # Minimum NDCG@10 gain to count as improvement
    # DataLoader settings (vectorized batch building)
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    enable_timing: bool = False  # Log timing breakdowns for profiling
    timing_window: int = 50  # Number of steps per timing average

    # Checkpoint loading
    strict_load_pretrained: bool = False  # If True, load full model with strict=True
    ckpt_prefix_to_strip: Optional[str] = None  # Optional prefix to strip from checkpoint keys

    # Output settings
    save_item_embeddings: bool = False  # Save item embeddings after training
    save_best_model: bool = True  # Save best val model for test evaluation
    checkpoint_mode: str = "full"  # full | delta
    run_tag: Optional[str] = None  # Unique run folder suffix (timestamp + run id)

    # Device settings
    device: str = "cuda:1"  # e.g., "cuda:1", "cpu", "mps"

    # Pretrained checkpoint
    pretrained_ckpt_path: str = (
        "/home/lingfengs111/codes/soft_patch_training/checkpoints/sasrec_loo_standard/"
        "sasrec_taobao_loo202_seq202_dim128_L2_H1_best.pt"
    )

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== SASRec Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  data_txt_path: {self.data_txt_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  hidden_units: {self.hidden_units}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")
        logger.info(f"  right_align_positions: {self.right_align_positions}")
        logger.info(f"  sasrec_attention_norm: {self.sasrec_attention_norm}")
        logger.info(f"  sasrec_enable_relative_attention_bias: {self.sasrec_enable_relative_attention_bias}")
        logger.info("HLLMText Parameters:")
        logger.info(f"  hllm_text_user_pretrain_dir: {self.hllm_text_user_pretrain_dir}")
        logger.info(f"  hllm_text_item_feature_path: {self.hllm_text_item_feature_path}")
        logger.info(f"  hllm_text_dtype: {self.hllm_text_dtype}")
        logger.info(f"  hllm_text_user_model_class: {self.hllm_text_user_model_class}")
        logger.info(f"  hllm_text_freeze_user_llm: {self.hllm_text_freeze_user_llm}")
        logger.info(f"  hllm_text_normalize_logits: {self.hllm_text_normalize_logits}")
        logger.info(f"  hllm_text_logit_scale: {self.hllm_text_logit_scale}")
        logger.info(f"  hllm_text_lightweight_checkpoint: {self.hllm_text_lightweight_checkpoint}")
        logger.info("HLLM Parameters:")
        logger.info(f"  hllm_item_pretrain_dir: {self.hllm_item_pretrain_dir}")
        logger.info(f"  hllm_user_pretrain_dir: {self.hllm_user_pretrain_dir}")
        logger.info(f"  hllm_text_path: {self.hllm_text_path}")
        logger.info(f"  hllm_text_column: {self.hllm_text_column}")
        logger.info(f"  hllm_item_id_column: {self.hllm_item_id_column}")
        logger.info(f"  hllm_dtype: {self.hllm_dtype}")
        logger.info(f"  hllm_item_model_class: {self.hllm_item_model_class}")
        logger.info(f"  hllm_user_model_class: {self.hllm_user_model_class}")
        logger.info(f"  hllm_item_llm_init: {self.hllm_item_llm_init}")
        logger.info(f"  hllm_user_llm_init: {self.hllm_user_llm_init}")
        logger.info(f"  hllm_freeze_item_llm: {self.hllm_freeze_item_llm}")
        logger.info(f"  hllm_freeze_user_llm: {self.hllm_freeze_user_llm}")
        logger.info(f"  hllm_normalize_logits: {self.hllm_normalize_logits}")
        logger.info(f"  hllm_logit_scale: {self.hllm_logit_scale}")
        logger.info(f"  hllm_trust_remote_code: {self.hllm_trust_remote_code}")
        logger.info(f"  hllm_max_text_length: {self.hllm_max_text_length}")
        logger.info(f"  hllm_item_emb_token_n: {self.hllm_item_emb_token_n}")
        logger.info(f"  hllm_item_encode_batch_size: {self.hllm_item_encode_batch_size}")
        logger.info(f"  hllm_eval_cache_full_items: {self.hllm_eval_cache_full_items}")
        logger.info(f"  hllm_item_emb_init_std: {self.hllm_item_emb_init_std}")
        logger.info(f"  user_embedding_norm: {self.user_embedding_norm}")
        logger.info(f"  item_l2_norm: {self.item_l2_norm}")
        logger.info(f"  temperature: {self.temperature}")
        logger.info(f"  l2_norm_eps: {self.l2_norm_eps}")
        logger.info(f"  enable_score_item_bias: {self.enable_score_item_bias}")
        logger.info(f"  enable_score_length_bias: {self.enable_score_length_bias}")
        logger.info(f"  enable_score_length_scale: {self.enable_score_length_scale}")
        logger.info(f"  score_length_bucket_size: {self.score_length_bucket_size}")
        logger.info(f"  longer_global_tokens: {self.longer_global_tokens}")
        logger.info(f"  longer_merge_size: {self.longer_merge_size}")
        logger.info(f"  longer_merge_pool: {self.longer_merge_pool}")
        logger.info(f"  longer_inner_num_layers: {self.longer_inner_num_layers}")
        logger.info("Short/Long Split:")
        logger.info(f"  inner_seq_length: {self.inner_seq_length}")
        logger.info(f"  eval_seq_length: {self.eval_seq_length}")
        logger.info("Patch Parameters:")
        logger.info(f"  num_patches: {self.num_patches}")
        logger.info(f"  patch_len: {self.patch_len}")
        logger.info(f"  use_gating: {self.use_gating}")
        logger.info(f"  gating_hidden_dim: {self.gating_hidden_dim}")
        logger.info(f"  patch_init_std: {self.patch_init_std}")
        logger.info(f"  head_residual: {self.head_residual}")
        logger.info(f"  head_zero_init: {self.head_zero_init}")
        logger.info(f"  enable_projection_head: {self.enable_projection_head}")
        logger.info(f"  head_use_gelu: {self.head_use_gelu}")
        logger.info(f"  head_use_ln: {self.head_use_ln}")
        logger.info(f"  gating_pool: {self.gating_pool}")
        logger.info(f"  gating_init_std: {self.gating_init_std}")
        logger.info(f"  gating_temperature: {self.gating_temperature}")
        logger.info(f"  gating_noise_std: {self.gating_noise_std}")
        logger.info(f"  gating_balance_weight: {self.gating_balance_weight}")
        logger.info(f"  patch_orth_weight: {self.patch_orth_weight}")
        logger.info(f"  patch_inner_orth_weight: {self.patch_inner_orth_weight}")
        logger.info(f"  patch_routing: {self.patch_routing}")
        logger.info(f"  kmeans_max_iters: {self.kmeans_max_iters}")
        logger.info(f"  kmeans_seed: {self.kmeans_seed}")
        logger.info(f"  kmeans_max_samples: {self.kmeans_max_samples}")
        logger.info(f"  input_emb_lora_rank: {self.input_emb_lora_rank}")
        logger.info(f"  input_emb_lora_alpha: {self.input_emb_lora_alpha}")
        logger.info(f"  train_input_emb_lora: {self.train_input_emb_lora}")
        logger.info(f"  attn_lora_rank: {self.attn_lora_rank}")
        logger.info(f"  attn_lora_alpha: {self.attn_lora_alpha}")
        logger.info(f"  attn_lora_blocks: {self.attn_lora_blocks}")
        logger.info(f"  train_attn_lora: {self.train_attn_lora}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  seed: {self.seed}")
        logger.info(f"  eval_seed: {self.eval_seed if self.eval_seed is not None else self.seed}")
        logger.info(f"  deterministic: {self.deterministic}")
        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  eval_sample_size: {self.eval_sample_size}")
        logger.info("Meta-learning Settings:")
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
        logger.info(f"  full_finetune: {self.full_finetune}")
        logger.info(f"  meta_truncate_steps: {self.meta_truncate_steps} | lambda_meta: {self.lambda_meta}")
        logger.info(f"  val_batch_size: {self.val_batch_size}")
        logger.info(f"  val_eval_every_epochs: {self.val_eval_every_epochs}")
        logger.info(f"  outer_loss_mode: {self.outer_loss_mode}")
        logger.info(f"  outer_loss_decay: {self.outer_loss_decay}")
        logger.info(f"  outer_distill: {self.outer_distill}")
        logger.info(f"  outer_distill_temperature: {self.outer_distill_temperature}")
        logger.info(f"  outer_neg_samples: {self.outer_neg_samples}")
        logger.info(f"  outer_tail_weight: {self.outer_tail_weight}")
        logger.info(f"  outer_mid_weight: {self.outer_mid_weight}")
        logger.info(f"  outer_mid_samples: {self.outer_mid_samples}")
        logger.info(f"  outer_gt_weight: {self.outer_gt_weight}")
        logger.info(f"  outer_mid_rel_weight: {self.outer_mid_rel_weight}")
        logger.info(f"  outer_patch_future_weight: {self.outer_patch_future_weight}")
        logger.info(f"  outer_patch_future_steps: {self.outer_patch_future_steps}")
        logger.info(f"  outer_patch_boundary_weight: {self.outer_patch_boundary_weight}")
        logger.info(f"  outer_patch_boundary_steps: {self.outer_patch_boundary_steps}")
        logger.info(f"  inner_loss_mode: {self.inner_loss_mode}")
        logger.info(f"  prefix_len: {self.prefix_len}")
        logger.info(f"  shared_prefix_len: {self.shared_prefix_len}")
        logger.info(f"  shared_prefix_init_std: {self.shared_prefix_init_std}")
        logger.info(f"  prefix_source: {self.prefix_source}")
        logger.info(f"  prefix_tail_positions: {self.prefix_tail_positions}")
        logger.info(f"  patch_after_prefix: {self.patch_after_prefix}")
        logger.info(f"  patch_use_position_embeddings: {self.patch_use_position_embeddings}")
        logger.info(f"  inner_drop_prefix: {self.inner_drop_prefix}")
        logger.info(f"  inner_reset_every: {self.inner_reset_every}")
        logger.info(f"  supervise_prefix_targets: {self.supervise_prefix_targets}")
        logger.info(f"  inner_unk_mask_prob: {self.inner_unk_mask_prob}")
        logger.info(f"  drop_unseen_items: {self.drop_unseen_items}")
        train_bias, train_layernorm = resolve_inner_bitfit_flags(self)
        logger.info(f"  inner_train_bias_ln: {self.inner_train_bias_ln}")
        logger.info(f"  inner_train_bias: {train_bias}")
        logger.info(f"  inner_train_layernorm: {train_layernorm}")
        logger.info(f"  inner_train_head: {self.inner_train_head}")
        logger.info(f"  inner_train_score_head: {self.inner_train_score_head}")
        logger.info("Meta-Test Adaptation:")
        logger.info(f"  meta_test_adapt_steps: {self.meta_test_adapt_steps}")
        logger.info(f"  meta_test_adapt_lr: {self.meta_test_adapt_lr}")
        logger.info(f"  meta_test_init_from_trained: {self.meta_test_init_from_trained}")
        logger.info(f"  meta_test_unk_mask_prob: {self.meta_test_unk_mask_prob}")
        logger.info(f"  use_gradient_checkpointing: {self.use_gradient_checkpointing}")
        logger.info(f"  use_flash_attention: {self.use_flash_attention}")
        logger.info(f"  eval_before_train: {self.eval_before_train}")
        logger.info(f"  eval_after_train: {self.eval_after_train}")
        logger.info(f"  eval_protocol: {self.eval_protocol}")
        logger.info(f"  last_k_eval_test: {self.last_k_eval_test}")
        logger.info(f"  streaming_eval_last_k: {self.streaming_eval_last_k}")
        logger.info(f"  early_stop_patience: {self.early_stop_patience}")
        logger.info(f"  early_stop_min_epochs: {self.early_stop_min_epochs}")
        logger.info(f"  early_stop_min_delta: {self.early_stop_min_delta}")
        logger.info("DataLoader Settings:")
        logger.info(f"  num_workers: {self.num_workers}")
        logger.info(f"  prefetch_factor: {self.prefetch_factor}")
        logger.info(f"  persistent_workers: {self.persistent_workers}")
        logger.info(f"  pin_memory: {self.pin_memory}")
        logger.info(f"  enable_timing: {self.enable_timing}")
        logger.info(f"  timing_window: {self.timing_window}")
        logger.info(f"  strict_load_pretrained: {self.strict_load_pretrained}")
        logger.info(f"  ckpt_prefix_to_strip: {self.ckpt_prefix_to_strip}")

        logger.info("Output Settings:")
        logger.info(f"  save_item_embeddings: {self.save_item_embeddings}")
        logger.info(f"  save_best_model: {self.save_best_model}")
        logger.info(f"  checkpoint_mode: {self.checkpoint_mode}")
        logger.info(f"  run_tag: {self.run_tag}")

        logger.info("Device Settings:")
        logger.info(f"  device: {self.device}")
        logger.info(f"  pretrained_ckpt_path: {self.pretrained_ckpt_path}")
        logger.info("===========================")


def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict or a state_dict-like object.")
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return None
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _serialize_config(config: SASRecConfig) -> Dict[str, Any]:
    raw = dict(config.__dict__)
    return {k: _jsonify(v) for k, v in raw.items()}


def build_checkpoint_tag(config: SASRecConfig) -> str:
    backbone = str(config.backbone).lower()
    gate_tag = "nogate"
    if config.use_gating:
        gate_tag = f"gate{str(config.gating_pool).lower()}"
    route_tag = f"route{str(config.patch_routing).lower()}"
    train_tag = "_fullft" if getattr(config, "full_finetune", False) else ""
    extra = ""
    if float(getattr(config, "gating_balance_weight", 0.0) or 0.0) > 0:
        extra += f"_gb{_format_run_float(float(config.gating_balance_weight))}"
    if float(getattr(config, "patch_orth_weight", 0.0) or 0.0) > 0:
        extra += f"_po{_format_run_float(float(config.patch_orth_weight))}"
    if float(getattr(config, "patch_inner_orth_weight", 0.0) or 0.0) > 0:
        extra += f"_pio{_format_run_float(float(config.patch_inner_orth_weight))}"
    if int(getattr(config, "shared_prefix_len", 0) or 0) > 0:
        extra += f"_SP{int(config.shared_prefix_len)}"
    if backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        extra += "_SM1"
    elif backbone == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax_custom":
        extra += "_SMCUSTOM"
    if backbone == "sasrec" and bool(getattr(config, "sasrec_enable_relative_attention_bias", False)):
        extra += "_RB"
    return (
        f"{backbone}_{config.dataset}_short{config.inner_seq_length}_long{config.max_seq_length}"
        f"_dim{config.hidden_units}_L{config.num_blocks}_H{config.num_heads}"
        f"_P{config.patch_len}x{config.num_patches}_{gate_tag}_{route_tag}{train_tag}{extra}"
    )


def _extract_routing_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    routing_state: Dict[str, torch.Tensor] = {}
    meta_patch = getattr(model, "meta_patch", None)
    if meta_patch is None:
        return routing_state
    centers = getattr(meta_patch, "kmeans_centers", None)
    if torch.is_tensor(centers) and centers.numel() > 0:
        routing_state["kmeans_centers"] = centers.detach().cpu()
    user_table = getattr(meta_patch, "user_to_patch", None)
    if torch.is_tensor(user_table) and user_table.numel() > 0:
        routing_state["user_to_patch"] = user_table.detach().cpu()
    return routing_state


def _restore_routing_state(model: nn.Module, payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    routing_state = payload.get("routing_state")
    if not isinstance(routing_state, dict):
        return
    meta_patch = getattr(model, "meta_patch", None)
    if meta_patch is None:
        return
    centers = routing_state.get("kmeans_centers")
    if torch.is_tensor(centers) and centers.numel() > 0 and hasattr(meta_patch, "set_kmeans_centers"):
        meta_patch.set_kmeans_centers(centers)
    user_table = routing_state.get("user_to_patch")
    if torch.is_tensor(user_table) and user_table.numel() > 0 and hasattr(meta_patch, "set_user_table"):
        meta_patch.set_user_table(user_table)


def _collect_trainable_state_dict(model: nn.Module) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    delta_state = {k: v for k, v in state.items() if k in trainable}
    return delta_state, sorted(trainable)


def _build_run_tag(config: SASRecConfig, run: Optional[Any]) -> str:
    if config.run_tag:
        return str(config.run_tag)
    run_id = None
    if run is not None:
        run_id = getattr(run, "id", None) or getattr(wandb.run, "id", None)
    tag = time.strftime("%Y%m%d_%H%M%S")
    suffix = run_id or secrets.token_hex(4)
    return f"{tag}-{suffix}"


def _format_run_float(val: float) -> str:
    if val is None:
        return "na"
    if abs(val - int(val)) < 1e-6:
        return str(int(val))
    text = f"{val:.3g}"
    return text.replace(".", "p")


def _build_run_name(config: SASRecConfig) -> str:
    base = (
        f"{config.backbone}-meta-patch-{config.dataset}-L{config.num_blocks}-H{config.hidden_units}"
        f"-P{config.num_patches}x{config.patch_len}"
        f"-short{config.inner_seq_length}-long{config.max_seq_length}"
    )
    suffix = []
    sweep_id = os.getenv("WANDB_SWEEP_ID")
    if sweep_id:
        suffix.append(f"sw{sweep_id[:4]}")
    suffix.append(f"route{str(config.patch_routing).lower()}")
    suffix.append(f"pref{config.prefix_len}")
    prefix_source = str(getattr(config, "prefix_source", "head") or "head").lower()
    if prefix_source != "head":
        suffix.append(f"ps{prefix_source}")
    if int(getattr(config, "shared_prefix_len", 0) or 0) > 0:
        suffix.append(f"sp{int(config.shared_prefix_len)}")
    if str(config.backbone).lower() == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax1":
        suffix.append("sm1")
    elif str(config.backbone).lower() == "sasrec" and str(getattr(config, "sasrec_attention_norm", "softmax")).lower() == "softmax_custom":
        suffix.append("smcustom")
    if str(config.backbone).lower() == "sasrec" and bool(getattr(config, "sasrec_enable_relative_attention_bias", False)):
        suffix.append("rbias")
    if bool(getattr(config, "sasrec_use_rope", False)):
        suffix.append("rope")
    suffix.append("peT" if config.prefix_tail_positions else "peF")
    suffix.append("papT" if config.patch_after_prefix else "papF")
    if bool(getattr(config, "patch_use_position_embeddings", False)):
        suffix.append("ppos")
    if config.inner_drop_prefix:
        suffix.append("idp")
    if getattr(config, "inner_reset_every", 0):
        suffix.append(f"ir{config.inner_reset_every}")
    suffix.append(f"tw{_format_run_float(float(getattr(config, 'outer_tail_weight', 1.0)))}")
    suffix.append(f"mw{_format_run_float(float(getattr(config, 'outer_mid_weight', 0.0)))}")
    suffix.append(f"gw{_format_run_float(float(getattr(config, 'outer_gt_weight', 1.0)))}")
    if getattr(config, "full_finetune", False):
        suffix.append("fullft")
    if float(getattr(config, "gating_balance_weight", 0.0) or 0.0) > 0:
        suffix.append(f"gb{_format_run_float(float(config.gating_balance_weight))}")
    if float(getattr(config, "patch_orth_weight", 0.0) or 0.0) > 0:
        suffix.append(f"po{_format_run_float(float(config.patch_orth_weight))}")
    if float(getattr(config, "patch_inner_orth_weight", 0.0) or 0.0) > 0:
        suffix.append(f"pio{_format_run_float(float(config.patch_inner_orth_weight))}")
    if int(getattr(config, "input_emb_lora_rank", 0) or 0) > 0:
        suffix.append(f"iel{int(config.input_emb_lora_rank)}")
    if int(getattr(config, "attn_lora_rank", 0) or 0) > 0:
        suffix.append(f"alr{int(config.attn_lora_rank)}")
        suffix.append(f"alb{str(getattr(config, 'attn_lora_blocks', 'all')).replace(',', '_')}")
    if normalize_eval_protocol(getattr(config, "eval_protocol", "legacy_loo")) != "legacy_loo":
        suffix.append(f"anchork{int(getattr(config, 'last_k_eval_test', 0) or 0)}")
    run_label = os.getenv("PATCH_RUN_LABEL")
    if run_label:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label)).strip("_")
        if safe_label:
            suffix.append(f"lbl{safe_label}")
    return base + "-" + "-".join(suffix)


def save_run_config(
    config: SASRecConfig,
    run_name: str,
    argv: List[str],
) -> Optional[Path]:
    try:
        payload = _serialize_config(config)
        payload.update(
            {
                "run_name": run_name,
                "argv": list(argv),
                "cwd": os.getcwd(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        out_path = config.checkpoint_dir / "config.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return out_path
    except Exception as exc:
        logger.warning("Failed to write config.json: %s", exc)
        return None


def save_run_summary(
    config: SASRecConfig,
    run_name: str,
    best_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    meta_metrics: Dict[str, float],
    best_ckpt_path: Optional[Path],
    metrics_jsonl: Optional[Path] = None,
    metrics_csv: Optional[Path] = None,
) -> Optional[Path]:
    try:
        payload = {
            "run_name": run_name,
            "run_tag": config.run_tag,
            "checkpoint_dir": str(config.checkpoint_dir),
            "checkpoint_tag": build_checkpoint_tag(config),
            "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path else None,
            "best_val": _jsonify(best_metrics),
            "test": {
                "baseline": _jsonify(baseline_metrics),
                "meta_patch": _jsonify(meta_metrics),
            },
            "metrics_jsonl": str(metrics_jsonl) if metrics_jsonl else None,
            "metrics_csv": str(metrics_csv) if metrics_csv else None,
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_path = config.checkpoint_dir / "summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return out_path
    except Exception as exc:
        logger.warning("Failed to write summary.json: %s", exc)
        return None


def load_checkpoint(path: str, trust_pickle: bool = True) -> Dict:
    """Load a checkpoint with PyTorch 2.6+ weights_only safety handling."""
    if trust_pickle:
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            import numpy as np
            from torch.serialization import safe_globals

            with safe_globals([np.core.multiarray.scalar]):
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, val in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = val
        else:
            cleaned[key] = val
    return cleaned


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: Optional[str]) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    cleaned = {}
    for key, val in state_dict.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix) :]] = val
        else:
            cleaned[key] = val
    return cleaned


def _auto_strip_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Avoid stripping "encoder.": it is a real submodule prefix for several backbones
    # (for example LinRec/Bert-style models), not just a wrapper namespace.
    prefixes = ["_orig_mod.", "model.", "net.", "sasrec.", "student."]
    total = len(state_dict)
    if total == 0:
        return state_dict
    best_prefix = None
    best_count = 0
    for prefix in prefixes:
        count = sum(1 for k in state_dict.keys() if k.startswith(prefix))
        if count > best_count:
            best_count = count
            best_prefix = prefix
    if best_prefix is not None and best_count / total >= 0.5:
        logger.info("Auto-stripping checkpoint prefix: %s", best_prefix)
        return _strip_prefix(state_dict, best_prefix)
    return state_dict


def _maybe_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: Optional[str]) -> Dict[str, torch.Tensor]:
    if prefix:
        return _strip_prefix(state_dict, prefix)
    return _auto_strip_prefix(state_dict)


def infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor], config: SASRecConfig) -> SASRecConfig:
    if any(k.endswith(".uvqk") for k in state_dict.keys()):
        config.backbone = "hstu"
    if "item_emb.weight" in state_dict:
        config.hidden_units = int(state_dict["item_emb.weight"].shape[1])
    if "shared_prefix_tokens" in state_dict and state_dict["shared_prefix_tokens"].ndim == 3:
        config.shared_prefix_len = int(state_dict["shared_prefix_tokens"].shape[1])
    if "pos_emb.weight" in state_dict:
        if config.max_seq_length is None or int(config.max_seq_length) <= 0:
            config.max_seq_length = int(state_dict["pos_emb.weight"].shape[0]) - 1
    if "input_emb_lora.rows.weight" in state_dict:
        config.input_emb_lora_rank = int(state_dict["input_emb_lora.rows.weight"].shape[1])
    attn_lora_blocks: set[int] = set()
    for key, value in state_dict.items():
        if key.endswith(".attn.c_attn_lora.down.weight") and key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                attn_lora_blocks.add(int(parts[1]))
                config.attn_lora_rank = int(value.shape[0])
    if attn_lora_blocks:
        config.attn_lora_blocks = ",".join(str(i) for i in sorted(attn_lora_blocks))

    block_indices = []
    for key in state_dict.keys():
        if key.startswith("blocks.") or key.startswith("encoder.layer.") or key.startswith("item_encoder.layer."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
            elif len(parts) > 3 and parts[2].isdigit():
                block_indices.append(int(parts[2]))
    if block_indices:
        config.num_blocks = max(block_indices) + 1

    rel_bias_keys = [k for k in state_dict.keys() if k.endswith(".rel_pos_bias.bias")]
    if rel_bias_keys:
        if any(k.startswith("blocks.") and ".attn." in k for k in rel_bias_keys):
            config.sasrec_enable_relative_attention_bias = True
        else:
            config.hstu_enable_relative_attention_bias = True

    if config.hidden_units % config.num_heads != 0:
        logger.warning(
            "hidden_units (%s) not divisible by num_heads (%s). Forcing num_heads=1.",
            config.hidden_units,
            config.num_heads,
        )
        config.num_heads = 1
    return config


def apply_config_from_checkpoint_payload(config: SASRecConfig, ckpt_payload: Optional[Dict[str, Any]]) -> SASRecConfig:
    if not isinstance(ckpt_payload, dict):
        return config
    ckpt_config = ckpt_payload.get("config")
    if not isinstance(ckpt_config, dict):
        return config

    keys = [
        "backbone",
        "max_seq_length",
        "hidden_units",
        "num_blocks",
        "num_heads",
        "dropout_rate",
        "right_align_positions",
        "sasrec_attention_norm",
        "sasrec_enable_relative_attention_bias",
        "hstu_linear_dim",
        "hstu_attention_dim",
        "hstu_linear_activation",
        "hstu_attn_dropout",
        "hstu_enable_relative_attention_bias",
        "hstu_normalization",
        "hstu_concat_ua",
        "hstu_epsilon",
        "hstu_parametric_block_norm",
        "user_embedding_norm",
        "item_l2_norm",
        "temperature",
        "l2_norm_eps",
        "enable_score_item_bias",
        "enable_score_length_bias",
        "enable_score_length_scale",
        "score_length_bucket_size",
        "num_negatives",
        "sampled_softmax_chunk_size",
        "shared_prefix_len",
        "shared_prefix_init_std",
    ]
    applied = {}
    for key in keys:
        if key in ckpt_config and hasattr(config, key):
            setattr(config, key, ckpt_config[key])
            applied[key] = ckpt_config[key]
    if applied:
        logger.info("Applied architecture config from checkpoint: %s", applied)
    return config


def load_pretrained_backbone(
    model: nn.Module, ckpt_path: str, state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> None:
    ckpt_payload = None
    if state_dict is None:
        if not ckpt_path:
            logger.warning("No pretrained_ckpt_path provided; skipping backbone load.")
            return
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_payload = load_checkpoint(ckpt_path, trust_pickle=True)
        state_dict = _strip_module_prefix(_extract_state_dict(ckpt_payload))
    state_dict = _maybe_strip_prefix(state_dict, getattr(model.config, "ckpt_prefix_to_strip", None))
    filtered = {}
    pos_weight = None
    rel_bias_weights: Dict[str, torch.Tensor] = {}
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if k not in model_state and not k.startswith("item_emb.") and not k.startswith("pos_emb."):
            continue
        if k == "item_emb.weight":
            if hasattr(model, "item_emb") and v.shape == model.item_emb.weight.shape:
                filtered[k] = v
            elif hasattr(model, "item_emb") and v.shape[0] + 1 == model.item_emb.weight.shape[0] and v.shape[1] == model.item_emb.weight.shape[1]:
                # ckpt has PAD only; insert UNK row at index 1 and shift real items
                new_weight = model.item_emb.weight.detach().clone()
                new_weight.zero_()
                new_weight[0] = v[0]
                new_weight[1] = v[1:].mean(dim=0) if v.size(0) > 1 else torch.zeros_like(v[0])
                new_weight[2:] = v[1:]
                filtered[k] = new_weight
                logger.info("Expanded item_emb.weight with UNK row (shifted by +1).")
            else:
                logger.warning(
                    "Skipped item_emb.weight due to shape mismatch (ckpt=%s, model=%s).",
                    v.shape,
                    model.item_emb.weight.shape,
                )
        elif k == "pos_emb.weight":
            pos_weight = v
            if hasattr(model, "pos_emb") and v.shape == model.pos_emb.weight.shape:
                filtered[k] = v
        elif k.endswith(".rel_pos_bias.bias") and k in model_state:
            if v.shape == model_state[k].shape:
                filtered[k] = v
            elif v.ndim == 1 and model_state[k].ndim == 1:
                rel_bias_weights[k] = v
        elif k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("Missing keys when loading backbone: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading backbone: %s", unexpected)
    if pos_weight is not None and pos_weight.shape != model.pos_emb.weight.shape:
        logger.warning(
            "Skipped pos_emb.weight due to shape mismatch (ckpt=%s, model=%s).",
            pos_weight.shape,
            model.pos_emb.weight.shape,
        )
    for key, src_weight in rel_bias_weights.items():
        target = model_state.get(key)
        if target is None:
            continue
        src_len = int(src_weight.numel())
        tgt_len = int(target.numel())
        if src_len >= tgt_len or (tgt_len - src_len) % 2 != 0:
            logger.warning(
                "Skipped %s due to shape mismatch (ckpt=%s, model=%s).",
                key,
                tuple(src_weight.shape),
                tuple(target.shape),
            )
            continue
        offset = (tgt_len - src_len) // 2
        with torch.no_grad():
            param = model.get_parameter(key)
            param.data[offset : offset + src_len].copy_(
                src_weight.to(device=param.device, dtype=param.dtype)
            )
        logger.info(
            "Expanded %s by center-aligning relative bias (ckpt=%s -> model=%s, offset=%s).",
            key,
            tuple(src_weight.shape),
            tuple(target.shape),
            offset,
        )
    if ckpt_payload is not None:
        _restore_routing_state(model, ckpt_payload)


def initialize_head_as_identity(model: nn.Module) -> None:
    """Initialize projection head to preserve pretrained features at start."""
    if not hasattr(model, "proj_linear"):
        return
    residual = bool(getattr(model.config, "head_residual", False))
    zero_init = bool(getattr(model.config, "head_zero_init", False))
    use_ln = bool(getattr(model.config, "head_use_ln", True))
    if residual and zero_init:
        logger.info("Initializing projection head as residual-zero (output starts as identity).")
    else:
        logger.info("Initializing projection head as identity-like (linear + layernorm).")
    with torch.no_grad():
        linear = model.proj_linear
        if hasattr(linear, "weight") and linear.weight is not None:
            if residual and zero_init:
                linear.weight.zero_()
            else:
                linear.weight.zero_()
                dim = min(linear.weight.size(0), linear.weight.size(1))
                linear.weight[:dim, :dim].copy_(torch.eye(dim, device=linear.weight.device, dtype=linear.weight.dtype))
        if hasattr(linear, "bias") and linear.bias is not None:
            linear.bias.zero_()
        if use_ln and hasattr(model, "proj_ln"):
            if model.proj_ln.weight is not None:
                model.proj_ln.weight.fill_(1.0)
            if model.proj_ln.bias is not None:
                model.proj_ln.bias.zero_()


def collect_base_params_and_buffers(module: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params = {n: p for n, p in module.named_parameters()}
    buffers = {n: b for n, b in module.named_buffers()}
    return params, buffers


def build_override(names: List[str], tensors: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {n: t for n, t in zip(names, tensors)}


def _is_frozen_hllm_user_param(model: nn.Module, name: str) -> bool:
    return (
        bool(
            getattr(getattr(model, "config", None), "hllm_text_freeze_user_llm", False)
            or getattr(getattr(model, "config", None), "hllm_freeze_user_llm", False)
        )
        and hasattr(model, "user_llm")
        and name.startswith("user_llm.")
    )


def build_bitfit_param_names(
    model: nn.Module,
    enable_bias: bool = True,
    enable_layernorm: bool = True,
    enable_head: bool = True,
    enable_score_head: bool = True,
) -> List[str]:
    """Select parameters for BitFit-style inner loop (biases / LayerNorm / head)."""
    trainable = set()
    if enable_bias:
        for name, _ in model.named_parameters():
            if _is_frozen_hllm_user_param(model, name):
                continue
            if name.endswith(".bias"):
                trainable.add(name)
    if enable_layernorm:
        for module_name, module in model.named_modules():
            if _is_frozen_hllm_user_param(model, f"{module_name}."):
                continue
            if isinstance(module, nn.LayerNorm):
                for param_name, _ in module.named_parameters(recurse=False):
                    full = f"{module_name}.{param_name}" if module_name else param_name
                    trainable.add(full)
    if enable_head and hasattr(model, "proj_linear"):
        for name, _ in model.proj_linear.named_parameters(prefix="proj_linear"):
            trainable.add(name)
    if enable_head and hasattr(model, "proj_ln"):
        for name, _ in model.proj_ln.named_parameters(prefix="proj_ln"):
            trainable.add(name)
    if enable_score_head and hasattr(model, "score_calibration"):
        for name, _ in model.score_calibration.named_parameters(prefix="score_calibration"):
            trainable.add(name)
    ordered = [name for name, _ in model.named_parameters() if name in trainable]
    return ordered


def _has_active_patch_bank(model: nn.Module) -> bool:
    meta_patch = getattr(model, "meta_patch", None)
    if meta_patch is None or not hasattr(meta_patch, "eta"):
        return False
    patch_len = int(getattr(meta_patch, "patch_len", getattr(model, "patch_len", 0)) or 0)
    return patch_len > 0


def build_outer_param_names(model: nn.Module) -> List[str]:
    """Select slow outer-loop params: patch bank/gating plus optional LoRA-style adapters."""
    trainable = set()
    include_patch_bank = _has_active_patch_bank(model)
    for name, _ in model.named_parameters():
        if name == "meta_patch.eta" and not include_patch_bank:
            continue
        if name == "meta_patch.eta" or "_lora." in name or name == "shared_prefix_tokens":
            trainable.add(name)
    return [name for name, _ in model.named_parameters() if name in trainable]


def apply_bitfit_freeze(
    model: nn.Module,
    enable_bias: bool = True,
    enable_layernorm: bool = True,
    enable_head: bool = True,
    enable_score_head: bool = True,
    enable_input_emb_lora: bool = False,
    enable_attn_lora: bool = False,
) -> None:
    """Freeze backbone weights; keep fast theta plus selected slow PEFT params trainable."""
    for p in model.parameters():
        p.requires_grad = False
    if enable_bias:
        for name, p in model.named_parameters():
            if _is_frozen_hllm_user_param(model, name):
                continue
            if name.endswith(".bias"):
                p.requires_grad = True
    if enable_layernorm:
        for module_name, module in model.named_modules():
            if _is_frozen_hllm_user_param(model, f"{module_name}."):
                continue
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad = True
    if enable_head and hasattr(model, "proj_linear"):
        for p in model.proj_linear.parameters():
            p.requires_grad = True
    if enable_head and hasattr(model, "proj_ln"):
        for p in model.proj_ln.parameters():
            p.requires_grad = True
    if enable_score_head and hasattr(model, "score_calibration"):
        for p in model.score_calibration.parameters():
            p.requires_grad = True
    if enable_input_emb_lora and hasattr(model, "input_emb_lora") and model.input_emb_lora is not None:
        for p in model.input_emb_lora.parameters():
            p.requires_grad = True
    if enable_attn_lora and hasattr(model, "blocks"):
        for block in model.blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            for attr in ("c_attn_lora", "c_proj_lora"):
                adapter = getattr(attn, attr, None)
                if adapter is None:
                    continue
                for p in adapter.parameters():
                    p.requires_grad = True
    if hasattr(model, "shared_prefix_tokens") and model.shared_prefix_tokens is not None:
        model.shared_prefix_tokens.requires_grad_(True)
    if _has_active_patch_bank(model):
        model.meta_patch.eta.requires_grad_(True)
    if bool(
        getattr(getattr(model, "config", None), "hllm_text_freeze_user_llm", False)
        or getattr(getattr(model, "config", None), "hllm_freeze_user_llm", False)
    ) and hasattr(model, "user_llm"):
        for p in model.user_llm.parameters():
            p.requires_grad = False
        model.user_llm.eval()


def sasrec_training_step_stateless(
    model: SASRec,
    base_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    theta_names: List[str],
    theta_list: List[torch.Tensor],
    eta: torch.Tensor,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    user_ids: Optional[torch.Tensor] = None,
    return_gating: bool = False,
    use_patch: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    override = build_override(theta_names, theta_list)
    params = {**base_params, **override, **base_buffers}
    if _uses_similarity_scoring(getattr(model, "config", None)):
        kwargs = {
            "input_ids": input_ids,
            "patch_params": eta,
            "return_gating": True,
            "use_patch": use_patch,
        }
        if user_ids is not None:
            kwargs["user_ids"] = user_ids
        hidden_states, gating_weights = functional_call(
            model,
            params,
            args=(),
            kwargs=kwargs,
        )
        hidden_states = _strip_hidden_states_for_scoring(
            model,
            hidden_states,
            input_ids,
            use_patch=use_patch,
        )
        head_params = _collect_head_params_from_params(model, params, model.config)
        projected = model.apply_head(hidden_states, head_params=head_params)
        pos_embs = _lookup_item_embeddings(model, pos_ids, state=params).to(projected.dtype)
        neg_embs = _lookup_item_embeddings(model, neg_ids, state=params).to(projected.dtype)
        seq_lengths = _sequence_lengths_from_input_ids(input_ids)
        pos_logits = _apply_similarity_logits(
            projected,
            pos_embs,
            getattr(model, "config", None),
            model=model,
            item_ids=pos_ids,
            seq_lengths=seq_lengths,
            state=params,
        )
        neg_logits = _apply_similarity_logits(
            projected,
            neg_embs,
            getattr(model, "config", None),
            model=model,
            item_ids=neg_ids,
            seq_lengths=seq_lengths,
            state=params,
        )
        if return_gating:
            return pos_logits, neg_logits, gating_weights
        return pos_logits, neg_logits, None
    kwargs = {
        "input_ids": input_ids,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "patch_params": eta,
        "return_gating": True,
        "use_patch": use_patch,
    }
    if user_ids is not None:
        kwargs["user_ids"] = user_ids
    pos_logits, neg_logits, gating_weights = functional_call(
        model,
        params,
        args=(),
        kwargs=kwargs,
    )
    if return_gating:
        return pos_logits, neg_logits, gating_weights
    return pos_logits, neg_logits, None


def _last_position_loss(
    raw_loss: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if not valid_mask.any():
        return raw_loss.sum() * 0.0
    lengths = valid_mask.sum(dim=1)
    valid_rows = lengths > 0
    if not valid_rows.any():
        return raw_loss.sum() * 0.0
    last_idx = (lengths - 1).clamp_min(0)
    row_indices = torch.arange(raw_loss.size(0), device=raw_loss.device)
    last_loss = raw_loss[row_indices, last_idx]
    return last_loss[valid_rows].mean()


def _decay_position_loss(
    raw_loss: torch.Tensor, valid_mask: torch.Tensor, decay: float
) -> torch.Tensor:
    if not valid_mask.any():
        return raw_loss.sum() * 0.0
    lengths = valid_mask.sum(dim=1)
    max_len = raw_loss.size(1)
    pos_idx = torch.arange(max_len, device=raw_loss.device).view(1, -1)
    dist = (lengths.unsqueeze(1) - 1) - pos_idx
    dist = dist.clamp(min=0)
    gamma = float(decay)
    if gamma <= 0:
        gamma = 1e-6
    if gamma > 1:
        gamma = 1.0
    weights = (gamma ** dist) * valid_mask.float()
    denom = weights.sum()
    if denom <= 0:
        return raw_loss.sum() * 0.0
    return (raw_loss * weights).sum() / denom


def _reduce_loss(
    raw_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str,
    decay: float,
) -> torch.Tensor:
    if mode == "all":
        if not valid_mask.any():
            return raw_loss.sum() * 0.0
        return raw_loss[valid_mask].mean()
    if mode == "last":
        return _last_position_loss(raw_loss, valid_mask)
    if mode == "decay":
        return _decay_position_loss(raw_loss, valid_mask, decay)
    raise ValueError(f"Unknown loss mode: {mode}")


def _gating_balance_loss(gating_weights: torch.Tensor) -> torch.Tensor:
    if gating_weights is None or gating_weights.numel() == 0:
        ref = gating_weights if torch.is_tensor(gating_weights) else None
        device = ref.device if ref is not None else None
        return torch.zeros((), device=device)
    avg_weights = gating_weights.mean(dim=0).clamp_min(1e-8)
    uniform = torch.full_like(avg_weights, 1.0 / float(avg_weights.numel()))
    return torch.sum(avg_weights * (avg_weights.log() - uniform.log()))


def _patch_orthogonality_loss(model: nn.Module) -> torch.Tensor:
    meta_patch = getattr(model, "meta_patch", None)
    if meta_patch is None or not hasattr(meta_patch, "get_patch_bank"):
        param = next(model.parameters(), None)
        device = param.device if param is not None else None
        return torch.zeros((), device=device)
    patch_bank = meta_patch.get_patch_bank()
    if patch_bank.numel() == 0 or patch_bank.size(0) <= 1:
        return patch_bank.sum() * 0.0
    flat = patch_bank.reshape(patch_bank.size(0), -1)
    flat = F.normalize(flat, dim=-1, eps=1e-8)
    gram = flat @ flat.transpose(0, 1)
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    return F.mse_loss(gram, eye)


def _patch_inner_orthogonality_loss(model: nn.Module) -> torch.Tensor:
    meta_patch = getattr(model, "meta_patch", None)
    if meta_patch is None or not hasattr(meta_patch, "get_patch_bank"):
        param = next(model.parameters(), None)
        device = param.device if param is not None else None
        return torch.zeros((), device=device)
    patch_bank = meta_patch.get_patch_bank()
    if patch_bank.numel() == 0 or patch_bank.size(1) <= 1:
        return patch_bank.sum() * 0.0
    tokens = F.normalize(patch_bank, dim=-1, eps=1e-8)
    gram = tokens @ tokens.transpose(1, 2)
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return F.mse_loss(gram, eye.expand_as(gram))


def _resolve_inner_loss_mode(config: SASRecConfig) -> str:
    mode = getattr(config, "inner_loss_mode", "match_outer")
    if mode == "match_outer":
        mode = getattr(config, "outer_loss_mode", "all")
    return mode

def resolve_dataset_config(config: SASRecConfig) -> None:
    spec = resolve_loo_dataset(config.dataset, str(config.data_dir) if config.data_dir else None)
    config.dataset = spec.name
    config.data_dir = spec.root
    config.data_txt_path = spec.data_txt
    if config.max_seq_length is None or int(config.max_seq_length) <= 0:
        min_len = infer_loo_min_len(spec)
        if min_len is None or min_len <= 0:
            raise ValueError(f"Unable to infer max_seq_length from {spec.name}.")
        logger.info("Setting max_seq_length to %s based on %s.", min_len, spec.name)
        config.max_seq_length = int(min_len)


def resolve_eval_protocol_config(config: SASRecConfig) -> None:
    config.eval_protocol = normalize_eval_protocol(getattr(config, "eval_protocol", "legacy_loo"))
    config.last_k_eval_test = int(getattr(config, "last_k_eval_test", 0) or 0)
    if config.eval_protocol != "legacy_loo" and config.last_k_eval_test < 2:
        raise ValueError(
            f"holdout_anchor protocol requires last_k_eval_test >= 2, got {config.last_k_eval_test}."
        )


def resolve_shared_prefix_config(config: SASRecConfig) -> None:
    shared_prefix_len = int(getattr(config, "shared_prefix_len", 0) or 0)
    if shared_prefix_len <= 0:
        return
    if str(getattr(config, "backbone", "")).lower() not in {
        "sasrec",
        "hstu",
        "hstu_officialish",
        "hstu_official",
        "hstu_orig",
        "hstu_research_aligned",
        "hstu_research",
        "hstu_ra",
        "hllm",
        "hllm_text",
        "hllmtext",
    }:
        logger.warning(
            "shared_prefix_len=%s is currently only implemented for SASRec/HSTU/HLLM/HLLMText; backbone=%s will ignore it.",
            shared_prefix_len,
            getattr(config, "backbone", None),
        )
        return


def get_shared_token_len(config: SASRecConfig) -> int:
    return int(getattr(config, "shared_prefix_len", 0) or 0)


def get_patch_token_start(config: SASRecConfig, seq_length: int) -> int:
    # patch 的绝对起点由两部分决定：
    # 1. 前面有没有 shared token
    # 2. patch 是否被要求插在真实 prefix 后面
    start = get_shared_token_len(config)
    prefix_len = int(getattr(config, "prefix_len", 0) or 0)
    if bool(getattr(config, "patch_after_prefix", False)) and prefix_len > 0:
        start += min(prefix_len, max(int(seq_length), 0))
    return start


def _run_kmeans(data: torch.Tensor, num_clusters: int, max_iters: int, seed: int) -> torch.Tensor:
    if data.numel() == 0 or num_clusters <= 0:
        raise ValueError("KMeans requires non-empty data and num_clusters > 0.")
    if KMeans is not None:
        arr = data.cpu().numpy()
        try:
            km = KMeans(n_clusters=num_clusters, n_init="auto", max_iter=max_iters, random_state=seed)
        except TypeError:
            km = KMeans(n_clusters=num_clusters, n_init=10, max_iter=max_iters, random_state=seed)
        km.fit(arr)
        centers = torch.from_numpy(km.cluster_centers_).to(data.dtype)
        return centers

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.size(0), generator=g)
    centers = data[perm[:num_clusters]].clone()
    assignments = torch.zeros(data.size(0), dtype=torch.long)
    for _ in range(max_iters):
        dist = torch.cdist(data, centers)
        new_assign = dist.argmin(dim=1)
        if torch.equal(new_assign, assignments):
            break
        assignments = new_assign
        for ci in range(num_clusters):
            mask = assignments == ci
            if mask.any():
                centers[ci] = data[mask].mean(dim=0)
    return centers


def build_kmeans_centers(
    dataset: "LooSequenceDataset",
    model: "SASRec",
    config: SASRecConfig,
) -> torch.Tensor:
    users = list(dataset.users)
    total_users = len(users)
    if config.kmeans_max_samples > 0 and total_users > config.kmeans_max_samples:
        rng = np.random.RandomState(config.kmeans_seed)
        idx = rng.choice(total_users, size=config.kmeans_max_samples, replace=False)
        users = [users[i] for i in idx]
    logger.info(
        "Building kmeans routing with %s/%s sequences (num_patches=%s)...",
        len(users),
        total_users,
        config.num_patches,
    )

    weight = _all_item_embeddings(model).detach()
    emb_list = []
    with torch.no_grad():
        for user in users:
            seq = dataset.user_seq.get(user, [])
            if not seq:
                continue
            train_end = resolve_train_cutoff(
                len(seq),
                eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            seq = seq[:train_end]
            if not seq:
                continue
            if len(seq) > config.max_seq_length:
                seq = seq[-config.max_seq_length :]
            ids = torch.tensor([x for x in seq if x > 1], dtype=torch.long, device=weight.device)
            emb = weight.index_select(0, ids)
            if emb.numel() == 0:
                continue
            seq_emb = emb.mean(dim=0).float().cpu()
            emb_list.append(seq_emb)

    if not emb_list:
        raise RuntimeError("No sequence embeddings available for kmeans routing.")

    data = torch.stack(emb_list, dim=0)
    k = min(config.num_patches, data.size(0))
    centers = _run_kmeans(data, k, config.kmeans_max_iters, config.kmeans_seed)
    if centers.size(0) < config.num_patches:
        repeats = config.num_patches - centers.size(0)
        extra = centers[torch.arange(repeats) % centers.size(0)].clone()
        centers = torch.cat([centers, extra], dim=0)
        logger.warning("KMeans centers < num_patches; padding centers to %s.", config.num_patches)
    return centers


def build_user_patch_table(
    dataset: "LooSequenceDataset",
    model: "SASRec",
    centers: torch.Tensor,
    config: SASRecConfig,
) -> torch.Tensor:
    users = list(dataset.users)
    weight = _all_item_embeddings(model).detach().cpu()
    centers_cpu = centers.detach().cpu()
    table = torch.zeros((dataset.num_users,), dtype=torch.long)
    emb_list: List[torch.Tensor] = []
    for user in users:
        seq = dataset.user_seq.get(user, [])
        if not seq:
            emb_list.append(torch.zeros(config.hidden_units))
            continue
        train_end = resolve_train_cutoff(
            len(seq),
            eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
            last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
        )
        seq = seq[:train_end]
        if not seq:
            emb_list.append(torch.zeros(config.hidden_units))
            continue
        if len(seq) > config.max_seq_length:
            seq = seq[-config.max_seq_length :]
        ids = torch.tensor([x for x in seq if x > 1], dtype=torch.long)
        if ids.numel() == 0:
            emb_list.append(torch.zeros(config.hidden_units))
            continue
        emb = weight.index_select(0, ids).mean(dim=0).float()
        emb_list.append(emb)
    if not emb_list:
        return table
    data = torch.stack(emb_list, dim=0)
    dist = torch.cdist(data, centers_cpu)
    idx = dist.argmin(dim=1).to(torch.long)
    for i, user in enumerate(users):
        if user < table.numel():
            table[user] = idx[i]
    return table


class LooTrainDataset(Dataset):
    """Training dataset for protocol-aware holdout sequences."""

    def __init__(self, dataset: LooSequenceDataset, config: SASRecConfig):
        self.dataset = dataset
        self.max_seq_length = config.max_seq_length
        self.samples = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            train_end = resolve_train_cutoff(
                len(seq),
                eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            if train_end > 1:
                self.samples.append((user, seq[:train_end]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _resample_negatives(
    neg_row: torch.Tensor,
    seen_tensor: torch.Tensor,
    min_item_id: int,
    max_item: int,
    valid_mask: Optional[torch.Tensor] = None,
    max_tries: int = 10,
) -> None:
    if seen_tensor.numel() == 0:
        return
    mask = torch.isin(neg_row, seen_tensor)
    if valid_mask is not None:
        mask = mask & valid_mask
    tries = 0
    while mask.any() and tries < max_tries:
        neg_row[mask] = torch.randint(
            min_item_id,
            max_item + 1,
            size=(int(mask.sum().item()),),
            dtype=neg_row.dtype,
            device=neg_row.device,
        )
        mask = torch.isin(neg_row, seen_tensor)
        if valid_mask is not None:
            mask = mask & valid_mask
        tries += 1


def build_train_collate_fn(train_data: LooTrainDataset, config: SASRecConfig):
    base_dataset = train_data.dataset
    max_seq_length = config.max_seq_length
    min_item_id = base_dataset.min_item_id
    max_item = base_dataset.max_item
    sample_id_stride = max(1, base_dataset.max_train_seq_len + 1)

    def collate(batch):
        batch_size = len(batch)
        seq_tensors = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        pos_tensors = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        neg_tensors = torch.randint(
            min_item_id,
            max_item + 1,
            size=(batch_size, max_seq_length),
            dtype=torch.long,
        )
        sample_id_tensors = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
        user_id_tensors = torch.zeros((batch_size,), dtype=torch.long)
        internal_user_id_tensors = torch.zeros((batch_size,), dtype=torch.long)

        for i, (user, seq) in enumerate(batch):
            internal_user_id_tensors[i] = user
            user_id_tensors[i] = base_dataset.internal_to_user_id.get(user, user)
            seq_len = min(len(seq), max_seq_length)
            if seq_len < 1:
                continue
            if len(seq) > max_seq_length:
                start_idx = len(seq) - max_seq_length
                seq = seq[-max_seq_length:]
                seq_len = max_seq_length
            else:
                start_idx = 0
            seq_tensors[i, -seq_len:] = torch.as_tensor(seq[:seq_len], dtype=torch.long)
            if seq_len > 1:
                pos_tensors[i, -seq_len:-1] = torch.as_tensor(seq[1:seq_len], dtype=torch.long)
            base_offset = user * sample_id_stride + start_idx
            positions = torch.arange(seq_len, dtype=torch.long)
            sample_id_tensors[i, -seq_len:] = base_offset + positions

        valid_mask = pos_tensors != 0
        for i, (user, _) in enumerate(batch):
            seen = base_dataset.user_seq.get(user, [])
            if not seen:
                continue
            seen_tensor = torch.as_tensor([x for x in seen if x > 1], dtype=torch.long)
            _resample_negatives(neg_tensors[i], seen_tensor, min_item_id, max_item, valid_mask[i])

        return {
            "input_ids": seq_tensors,
            "pos_ids": pos_tensors,
            "neg_ids": neg_tensors,
            "sample_ids": sample_id_tensors,
            "user_ids": user_id_tensors,
            "internal_user_ids": internal_user_id_tensors,
        }

    return collate


class SequentialSampler:
    """
    Sampler for sequential data that generates training batches.
    Each batch contains user sequences with positive and negative samples.
    """

    def __init__(self, dataset: LooSequenceDataset, config: SASRecConfig):
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
            train_end = resolve_train_cutoff(
                len(seq),
                eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
                last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
            )
            if train_end > 1:
                self.valid_user_seqs.append((user, seq[:train_end]))

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
            internal_user_id_tensors = torch.zeros((actual_batch_size,), dtype=torch.long)

            for idx, (user, seq) in enumerate(batch_data):
                internal_user_id_tensors[idx] = user
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

                # Positive items: for each position i, predict item at position i+1.
                # The last position has no next item, so target stays 0.
                for pos in range(seq_len):
                    if pos < seq_len - 1:
                        pos_tensors[idx, -seq_len + pos] = seq[pos + 1]
                    # Sample_ID for (user, position) in the full training sequence
                    global_pos = start_idx + pos
                    sample_id_tensors[idx, -seq_len + pos] = user * self.sample_id_stride + global_pos

                # Sample negative items for each position
                # Use full user history to avoid sampling held-out positives as negatives.
                seen_set = {x for x in self.dataset.user_seq[user] if x > 1}
                for pos in range(seq_len):
                    neg_item = self.sample_negative_item(2, self.max_item + 1, seen_set)
                    neg_tensors[idx, -seq_len + pos] = neg_item

            yield {
                "input_ids": seq_tensors,
                "pos_ids": pos_tensors,
                "neg_ids": neg_tensors,
                "sample_ids": sample_id_tensors,
                "user_ids": user_id_tensors,
                "internal_user_ids": internal_user_id_tensors,
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
    use_patch: bool = True,
    use_head: bool = True,
    max_seq_length: Optional[int] = None,
    truncate_len: Optional[int] = None,
    theta_names: Optional[List[str]] = None,
    bitfit_init_state: Optional[Dict[str, torch.Tensor]] = None,
    streaming_last_k: int = 0,
) -> Dict[str, float]:
    """Evaluate on LOO split using sampled negatives."""
    model.eval()
    rng = _build_eval_rng(config, mode=mode, streaming_last_k=streaming_last_k)
    eval_generator = _build_eval_torch_generator(
        config,
        mode=mode,
        device=torch.device(device),
        streaming_last_k=streaming_last_k,
    )
    if mode == "meta-test" and max_seq_length is None:
        max_seq_length = config.eval_seq_length

    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_users = 0
    per_position: Dict[int, Dict[str, float]] = {}

    param_dict = dict(model.named_parameters())
    if mode == "meta-test":
        if theta_names is None:
            raise ValueError("theta_names is required for meta-test adaptation.")
        if bitfit_init_state is None:
            raise ValueError("bitfit_init_state is required for meta-test adaptation.")
        trained_state = _snapshot_params_by_name(model, theta_names)
        adapt_steps = max(0, int(config.meta_test_adapt_steps))
        if adapt_steps > 0 and not config.meta_test_init_from_trained:
            init_state = bitfit_init_state
        else:
            # Without adaptation, evaluate the currently trained theta instead
            # of reverting to the initial BitFit/head state.
            init_state = trained_state
        restore_state = trained_state
        adapt_lr = float(config.meta_test_adapt_lr)
        if config.meta_test_unk_mask_prob is None:
            meta_unk_prob = config.inner_unk_mask_prob if not config.drop_unseen_items else 0.0
        else:
            meta_unk_prob = float(config.meta_test_unk_mask_prob)
        eta_requires_grad = None
        if hasattr(model, "meta_patch") and hasattr(model.meta_patch, "eta"):
            eta_requires_grad = model.meta_patch.eta.requires_grad
            model.meta_patch.eta.requires_grad_(False)

    users = list(dataset.users)
    target_positions_cache = _get_eval_target_positions_cache(
        dataset,
        users,
        mode=mode,
        streaming_last_k=streaming_last_k,
        config=config,
    )
    seen_items_cache = _get_eval_seen_items_cache(dataset, users) if mode != "val" else {}
    fixed_negative_cache = _get_eval_fixed_negative_cache(dataset, users, seen_items_cache) if mode != "val" else {}
    user_seq_map = dataset.user_seq

    for batch_start in range(0, len(users), batch_size):
        batch_users = users[batch_start : batch_start + batch_size]
        batch_examples: List[Tuple[int, List[int], int, int]] = []

        for user in batch_users:
            seq = user_seq_map[user]
            target_positions = target_positions_cache[user]
            for target_idx in target_positions:
                target = seq[target_idx]
                if target == 1 and config.drop_unseen_items:
                    continue
                batch_examples.append((user, seq[:target_idx], target, len(seq) - target_idx))

        if not batch_examples:
            continue

        batch_seqs = [input_seq for _, input_seq, _, _ in batch_examples]
        max_len = min(max(len(s) for s in batch_seqs), dataset.max_seq_length)
        if max_seq_length is not None and max_seq_length > 0:
            max_len = min(max_len, max_seq_length)
        use_len = max_len
        if truncate_len is not None and truncate_len > 0:
            use_len = min(truncate_len, max_len)
        prefix_len = int(getattr(config, "prefix_len", 0) or 0)
        prefix_source = str(getattr(config, "prefix_source", "head") or "head")
        if prefix_len > 0:
            total_len = prefix_len + use_len
            input_tensor = torch.zeros((len(batch_examples), total_len), dtype=torch.long)
            for i, seq in enumerate(batch_seqs):
                full_len = min(len(seq), max_len)
                trimmed = seq[-full_len:]
                prefix_eff = min(prefix_len, full_len)
                tail_eff = min(use_len, max(0, full_len - prefix_eff))
                candidate_len = max(0, full_len - tail_eff)
                prefix_eff = min(prefix_eff, candidate_len)
                if prefix_eff > 0:
                    offsets = _select_prefix_offsets_from_list(
                        [int(x) for x in trimmed[:candidate_len]],
                        prefix_eff,
                        prefix_source,
                    )
                    prefix_tokens = torch.tensor([trimmed[pos] for pos in offsets], dtype=torch.long)
                    input_tensor[i, :prefix_eff] = prefix_tokens
                if tail_eff > 0:
                    tail_tokens = torch.tensor(trimmed[-tail_eff:])
                    tail_start = total_len - tail_eff
                    input_tensor[i, tail_start : tail_start + tail_eff] = tail_tokens
        else:
            input_tensor = torch.zeros((len(batch_examples), use_len), dtype=torch.long)
            for i, seq in enumerate(batch_seqs):
                seq_len = min(len(seq), use_len)
                input_tensor[i, -seq_len:] = torch.tensor(seq[-seq_len:])

        input_tensor = input_tensor.to(device)
        valid_input = input_tensor
        valid_targets = [target for _, _, target, _ in batch_examples]
        valid_user_ids = [user for user, _, _, _ in batch_examples]
        valid_rel_positions = [rel_from_end for _, _, _, rel_from_end in batch_examples]

        if mode == "meta-test" and adapt_steps > 0:
            _load_params_by_name(model, init_state, theta_names)
            adapt_input, adapt_pos, adapt_neg = _build_adapt_tensors(
                valid_input,
                dataset.max_item,
                prefix_len=int(getattr(config, "prefix_len", 0)),
                supervise_prefix_targets=bool(getattr(config, "supervise_prefix_targets", False)),
            )
            if meta_unk_prob > 0:
                adapt_input = _mask_inputs_with_unk(adapt_input, meta_unk_prob)
            adapt_input = adapt_input.to(device)
            adapt_pos = adapt_pos.to(device)
            adapt_neg = adapt_neg.to(device)
            loss_mode = _resolve_inner_loss_mode(config)
            for _ in range(adapt_steps):
                model.zero_grad(set_to_none=True)
                pos_logits, neg_logits = _call_training_step_with_optional_user_ids(
                    model,
                    adapt_input,
                    adapt_pos,
                    adapt_neg,
                    patch_params=None,
                    user_ids=torch.as_tensor(valid_user_ids, device=device),
                    use_patch=use_patch,
                )
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits), reduction="none"
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits), reduction="none"
                )
                raw_loss = pos_loss + neg_loss
                valid_mask = adapt_pos != 0
                loss = _reduce_loss(raw_loss, valid_mask, loss_mode, config.outer_loss_decay)
                loss.backward()
                with torch.no_grad():
                    for name in theta_names:
                        p = param_dict.get(name)
                        if p is None or p.grad is None:
                            continue
                        p.add_(p.grad, alpha=-adapt_lr)
                for name in theta_names:
                    p = param_dict.get(name)
                    if p is not None:
                        p.grad = None

        with torch.no_grad():
            sample_size = max(2, config.eval_sample_size)
            if mode == "val":
                candidates_tensor = _build_eval_candidates_fast(
                    dataset=dataset,
                    users=valid_user_ids,
                    targets=valid_targets,
                    sample_size=sample_size,
                    device=torch.device(device),
                    generator=eval_generator,
                )
            else:
                candidates_list: List[torch.Tensor] = []
                for idx, user in enumerate(valid_user_ids):
                    target = valid_targets[idx]
                    candidates = [target]
                    seen_items = seen_items_cache[user]
                    fixed_neg = fixed_negative_cache.get(user)
                    if fixed_neg is not None and fixed_neg != target:
                        candidates.append(fixed_neg)
                    while len(candidates) < sample_size:
                        neg_item = rng.randint(2, dataset.max_item + 1)
                        if neg_item not in seen_items and neg_item not in candidates:
                            candidates.append(neg_item)
                    candidates_list.append(torch.tensor(candidates, device=device))
                candidates_tensor = torch.stack(candidates_list, dim=0)

            scores = _call_predict_with_optional_user_ids(
                user_ids=torch.as_tensor(valid_user_ids, device=device),
                model=model,
                input_ids=valid_input,
                item_indices=candidates_tensor,
                use_patch=use_patch,
                use_head=use_head,
            )

        _, indices = torch.sort(scores, dim=1, descending=True)
        ranks = (indices == 0).nonzero(as_tuple=True)[1].cpu().numpy() + 1  # 1-indexed ranks

        for rel_from_end, rank in zip(valid_rel_positions, ranks):
            valid_users += 1
            update_rank_metrics(per_position, rel_from_end, int(rank))
            if rank <= 10:
                hr_sum += 1
                ndcg_sum += 1 / np.log2(rank + 1)

    eval_entity = "examples" if int(streaming_last_k or 0) > 1 else "users"
    logger.info("Evaluated on %s %s", f"{valid_users:,}", eval_entity)

    if mode == "meta-test":
        _load_params_by_name(model, restore_state, theta_names)
        if eta_requires_grad is not None and hasattr(model, "meta_patch") and hasattr(model.meta_patch, "eta"):
            model.meta_patch.eta.requires_grad_(eta_requires_grad)

    return finalize_eval_metrics(
        ndcg_sum=ndcg_sum,
        hr_sum=hr_sum,
        num_examples=valid_users,
        per_position=per_position,
        streaming_last_k=streaming_last_k,
    )


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _slice_batch_tail(batch: Dict[str, torch.Tensor], tail_len: int) -> Dict[str, torch.Tensor]:
    if tail_len <= 0:
        return batch
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and k in {"input_ids", "pos_ids", "neg_ids", "sample_ids"}:
            # 最朴素的 short view：直接只保留最后 tail_len 个位置。
            out[k] = v[:, -tail_len:]
        else:
            out[k] = v
    return out


def _normalize_prefix_source(prefix_source: str) -> str:
    source = str(prefix_source or "head").strip().lower()
    aliases = {
        "front": "head",
        "start": "head",
        "first": "head",
        "mid": "middle",
        "center": "middle",
        "centre": "middle",
        "hm": "head_middle",
        "headmid": "head_middle",
        "head_middle": "head_middle",
        "mixhm": "head_middle",
        "hybrid_hm": "head_middle",
        "even": "uniform",
        "evenly": "uniform",
        "spread": "uniform",
        "uniform_history": "uniform",
        "history_uniform": "uniform",
        "spread_history": "uniform",
        "sample": "random",
        "rand": "random",
    }
    source = aliases.get(source, source)
    if source not in {"head", "middle", "head_middle", "uniform", "random"}:
        raise ValueError(
            f"Unsupported prefix_source={prefix_source!r}; expected head, middle, head_middle, uniform, or random."
        )
    return source


def _dedup_and_fill_offsets(offsets: List[int], candidate_len: int, prefix_eff: int) -> List[int]:
    seen = set()
    out: List[int] = []
    for pos in offsets:
        pos = int(pos)
        if pos < 0 or pos >= candidate_len or pos in seen:
            continue
        out.append(pos)
        seen.add(pos)
        if len(out) >= prefix_eff:
            return sorted(out[:prefix_eff])
    for pos in range(candidate_len):
        if pos in seen:
            continue
        out.append(pos)
        if len(out) >= prefix_eff:
            break
    return sorted(out[:prefix_eff])


def _ensure_first_token_in_offsets(offsets: List[int], candidate_len: int, prefix_eff: int) -> List[int]:
    if candidate_len <= 0 or prefix_eff <= 0:
        return []
    cleaned = _dedup_and_fill_offsets(offsets, candidate_len, prefix_eff)
    if 0 in cleaned:
        return cleaned
    if not cleaned:
        return [0]
    cleaned = [0] + [pos for pos in cleaned if pos != 0]
    return sorted(cleaned[:prefix_eff])


def _stable_random_prefix_offsets(candidate_tokens: torch.Tensor, prefix_eff: int) -> torch.Tensor:
    if prefix_eff <= 0:
        return torch.empty(0, dtype=torch.long, device=candidate_tokens.device)
    positions = torch.arange(candidate_tokens.numel(), dtype=torch.long, device=candidate_tokens.device)
    scores = (candidate_tokens.long() * 1_103_515_245 + (positions + 1) * 12_345) % 2_147_483_647
    selected = torch.topk(scores, k=prefix_eff, largest=False).indices
    return torch.sort(selected).values


def _select_prefix_offsets(
    candidate_len: int,
    prefix_eff: int,
    prefix_source: str,
    *,
    device: torch.device,
    candidate_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    prefix_eff = min(max(int(prefix_eff), 0), max(int(candidate_len), 0))
    if prefix_eff <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    source = _normalize_prefix_source(prefix_source)
    if source == "head":
        return torch.arange(prefix_eff, dtype=torch.long, device=device)
    if source == "middle":
        start = max((int(candidate_len) - prefix_eff) // 2, 0)
        middle_offsets = list(range(start, start + prefix_eff))
        middle_offsets = _ensure_first_token_in_offsets(middle_offsets, int(candidate_len), prefix_eff)
        return torch.tensor(middle_offsets, dtype=torch.long, device=device)
    if source == "head_middle":
        head_take = prefix_eff // 2
        middle_take = prefix_eff - head_take
        head_offsets = list(range(head_take))
        middle_start = max((int(candidate_len) - middle_take) // 2, 0)
        middle_offsets = list(range(middle_start, middle_start + middle_take))
        merged = _ensure_first_token_in_offsets(head_offsets + middle_offsets, int(candidate_len), prefix_eff)
        return torch.tensor(merged, dtype=torch.long, device=device)
    if source == "uniform":
        uniform_offsets = (
            torch.linspace(0, int(candidate_len) - 1, steps=prefix_eff, device=device).round().long().tolist()
        )
        uniform_offsets = _ensure_first_token_in_offsets(uniform_offsets, int(candidate_len), prefix_eff)
        return torch.tensor(uniform_offsets, dtype=torch.long, device=device)
    if candidate_tokens is None:
        raise ValueError("candidate_tokens is required when prefix_source='random'.")
    random_offsets = _stable_random_prefix_offsets(candidate_tokens, prefix_eff).tolist()
    random_offsets = _ensure_first_token_in_offsets(random_offsets, int(candidate_len), prefix_eff)
    return torch.tensor(random_offsets, dtype=torch.long, device=device)


def _select_prefix_offsets_from_list(
    candidate_tokens: List[int],
    prefix_eff: int,
    prefix_source: str,
) -> List[int]:
    candidate_len = len(candidate_tokens)
    prefix_eff = min(max(int(prefix_eff), 0), candidate_len)
    if prefix_eff <= 0:
        return []

    source = _normalize_prefix_source(prefix_source)
    if source == "head":
        return list(range(prefix_eff))
    if source == "middle":
        start = max((candidate_len - prefix_eff) // 2, 0)
        return _ensure_first_token_in_offsets(list(range(start, start + prefix_eff)), candidate_len, prefix_eff)
    if source == "head_middle":
        head_take = prefix_eff // 2
        middle_take = prefix_eff - head_take
        head_offsets = list(range(head_take))
        middle_start = max((candidate_len - middle_take) // 2, 0)
        middle_offsets = list(range(middle_start, middle_start + middle_take))
        return _ensure_first_token_in_offsets(head_offsets + middle_offsets, candidate_len, prefix_eff)
    if source == "uniform":
        uniform_offsets = [int(round(x)) for x in np.linspace(0, candidate_len - 1, num=prefix_eff)]
        return _ensure_first_token_in_offsets(uniform_offsets, candidate_len, prefix_eff)
    scores = [
        ((int(token) * 1_103_515_245 + (pos + 1) * 12_345) % 2_147_483_647, pos)
        for pos, token in enumerate(candidate_tokens)
    ]
    random_offsets = sorted(pos for _, pos in sorted(scores)[:prefix_eff])
    return _ensure_first_token_in_offsets(random_offsets, candidate_len, prefix_eff)


def _drop_prefix_from_batch(
    batch: Dict[str, torch.Tensor],
    prefix_len: int,
    tail_len: int,
    dataset: LooSequenceDataset,
) -> Dict[str, torch.Tensor]:
    """Remove prefix positions, keep tail length, and zero prefix losses."""
    if prefix_len <= 0:
        return _slice_batch_tail(batch, tail_len)
    input_ids = batch["input_ids"]
    device = input_ids.device
    batch_size, full_len = input_ids.size()
    tail_len = max(1, int(tail_len))
    prefix_len = max(1, int(prefix_len))
    out_len = tail_len

    out_input = torch.zeros((batch_size, out_len), dtype=input_ids.dtype, device=device)
    out_pos = torch.zeros((batch_size, out_len), dtype=input_ids.dtype, device=device)
    seq_lens = (input_ids != 0).sum(dim=1)
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        if seq_len <= 0:
            continue
        start = full_len - seq_len
        prefix_eff = min(prefix_len, seq_len)
        tail_eff = min(tail_len, max(0, seq_len - prefix_eff))
        if tail_eff <= 0:
            continue
        # 这个分支会把 prefix 完全丢掉，只保留最末尾 recent。
        tail_tokens = input_ids[i, full_len - tail_eff : full_len]
        out_input[i, -tail_eff:] = tail_tokens
        if tail_eff > 1:
            out_pos[i, -tail_eff:-1] = tail_tokens[1:tail_eff]

    user_ids = batch.get("internal_user_ids")
    if user_ids is None:
        raise KeyError("internal_user_ids missing from batch; needed for tail negatives.")
    neg_ids = _build_outer_neg_ids(
        dataset=dataset,
        user_ids=user_ids,
        pos_ids=out_pos,
        num_neg=1,
        device=device,
    ).squeeze(-1)
    return {
        "input_ids": out_input,
        "pos_ids": out_pos,
        "neg_ids": neg_ids,
        "internal_user_ids": user_ids,
    }


def _sample_negative_item(min_id: int, max_id_exclusive: int, seen_items: set) -> int:
    item_id = np.random.randint(min_id, max_id_exclusive)
    while item_id in seen_items:
        item_id = np.random.randint(min_id, max_id_exclusive)
    return item_id


def _build_adapt_tensors(
    input_ids: torch.Tensor,
    max_item: int,
    prefix_len: int = 0,
    supervise_prefix_targets: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_ids = torch.zeros_like(input_ids)
    pos_ids[:, :-1] = input_ids[:, 1:]
    if (not supervise_prefix_targets) and prefix_len and prefix_len > 0 and input_ids.size(1) >= prefix_len:
        pos_ids[:, :prefix_len] = 0
    pos_ids[input_ids == 0] = 0
    pos_ids[pos_ids <= 1] = 0
    neg_ids = torch.zeros_like(input_ids)
    input_cpu = input_ids.detach().cpu().numpy()
    for i in range(input_ids.size(0)):
        seen = set(int(x) for x in input_cpu[i] if x > 1)
        if not seen:
            continue
        for j in range(input_ids.size(1)):
            if input_cpu[i][j] == 0:
                continue
            neg_ids[i, j] = _sample_negative_item(2, max_item + 1, seen)
    if (not supervise_prefix_targets) and prefix_len and prefix_len > 0 and input_ids.size(1) >= prefix_len:
        neg_ids[:, :prefix_len] = 0
    return input_ids, pos_ids, neg_ids


def _build_prefix_tail_batch(
    batch: Dict[str, torch.Tensor],
    prefix_len: int,
    tail_len: int,
    dataset: LooSequenceDataset,
    supervise_prefix_targets: bool = False,
    prefix_source: str = "head",
) -> Dict[str, torch.Tensor]:
    if prefix_len <= 0:
        return _slice_batch_tail(batch, tail_len)
    input_ids = batch["input_ids"]
    device = input_ids.device
    batch_size, full_len = input_ids.size()
    tail_len = max(1, int(tail_len))
    prefix_len = max(1, int(prefix_len))
    out_len = prefix_len + tail_len

    out_input = torch.zeros((batch_size, out_len), dtype=input_ids.dtype, device=device)
    out_pos = torch.zeros((batch_size, out_len), dtype=input_ids.dtype, device=device)

    seq_lens = (input_ids != 0).sum(dim=1)
    for i in range(batch_size):
        seq_len = int(seq_lens[i].item())
        if seq_len <= 0:
            continue
        start = full_len - seq_len
        prefix_eff = min(prefix_len, seq_len)
        tail_eff = min(tail_len, max(0, seq_len - prefix_eff))
        candidate_len = max(0, seq_len - tail_eff)
        prefix_eff = min(prefix_eff, candidate_len)

        if prefix_eff > 0:
            candidate_tokens = input_ids[i, start : start + candidate_len]
            offsets = _select_prefix_offsets(
                candidate_len,
                prefix_eff,
                prefix_source,
                device=device,
                candidate_tokens=candidate_tokens,
            )
            prefix_tokens = candidate_tokens[offsets]
            out_input[i, :prefix_eff] = prefix_tokens
            if prefix_eff > 1:
                # These labels are only meaningful for contiguous sources; default configs zero them below.
                out_pos[i, : prefix_eff - 1] = prefix_tokens[1:prefix_eff]

        if tail_eff > 0:
            tail_tokens = input_ids[i, full_len - tail_eff : full_len]
            tail_start = out_len - tail_eff
            out_input[i, tail_start : tail_start + tail_eff] = tail_tokens
            if tail_eff > 1:
                out_pos[i, tail_start : tail_start + tail_eff - 1] = tail_tokens[1:tail_eff]

    if (not supervise_prefix_targets) and prefix_len > 0:
        # 默认把 prefix 当作“给上下文看的 anchor”，而不是要求它本身也预测下一个 item。
        out_pos[:, :prefix_len] = 0

    user_ids = batch.get("internal_user_ids")
    if user_ids is None:
        raise KeyError("internal_user_ids missing from batch; needed for prefix+tail negatives.")
    neg_ids = _build_outer_neg_ids(
        dataset=dataset,
        user_ids=user_ids,
        pos_ids=out_pos,
        num_neg=1,
        device=device,
    ).squeeze(-1)
    if (not supervise_prefix_targets) and prefix_len > 0:
        neg_ids[:, :prefix_len] = 0

    return {
        "input_ids": out_input,
        "pos_ids": out_pos,
        "neg_ids": neg_ids,
        "internal_user_ids": user_ids,
    }


def _build_outer_neg_ids(
    dataset: LooSequenceDataset,
    user_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    num_neg: int,
    device: torch.device,
    max_tries: int = 5,
) -> torch.Tensor:
    if num_neg <= 0:
        raise ValueError("num_neg must be positive.")
    batch_size, seq_len = pos_ids.shape
    min_id = max(2, getattr(dataset, "min_item_id", 1))
    max_item = dataset.max_item
    neg = torch.randint(min_id, max_item + 1, size=(batch_size, seq_len, num_neg), device=device)
    if batch_size == 0:
        return neg
    valid_mask = pos_ids != 0
    user_ids_list = user_ids.detach().cpu().tolist()
    for i, uid in enumerate(user_ids_list):
        seen = dataset.user_seq.get(uid, [])
        if not seen:
            continue
        seen_tensor = torch.as_tensor([x for x in seen if x > 1], device=device, dtype=torch.long)
        row = neg[i].reshape(-1)
        mask = valid_mask[i].unsqueeze(-1).expand(seq_len, num_neg).reshape(-1)
        _resample_negatives(row, seen_tensor, min_id, max_item, mask, max_tries=max_tries)
    return neg


def _select_middle_positions(
    input_ids: torch.Tensor,
    prefix_len: int,
    tail_len: int,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select middle-segment indices (between prefix and tail) for each sequence."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    mid_idx = torch.zeros((batch_size, num_samples), dtype=torch.long, device=device)
    mid_mask = torch.zeros((batch_size, num_samples), dtype=torch.bool, device=device)
    if num_samples <= 0:
        return mid_idx, mid_mask
    seq_lens = (input_ids != 0).sum(dim=1)
    for i in range(batch_size):
        length = int(seq_lens[i].item())
        if length <= 0:
            continue
        prefix_eff = min(prefix_len, length)
        tail_eff = min(tail_len, max(0, length - prefix_eff))
        mid_len = length - prefix_eff - tail_eff
        if mid_len <= 0:
            continue
        k = min(num_samples, mid_len)
        start = seq_len - length
        if k == 1:
            offsets = torch.tensor([mid_len // 2], device=device)
        else:
            offsets = torch.linspace(0, mid_len - 1, steps=k, device=device).round().long()
        idx = start + prefix_eff + offsets
        mid_idx[i, :k] = idx
        mid_mask[i, :k] = True
    return mid_idx, mid_mask


def _collect_head_params_from_params(
    model: SASRec,
    params: Dict[str, torch.Tensor],
    config: SASRecConfig,
) -> Optional[List[torch.Tensor]]:
    if not getattr(config, "enable_projection_head", True):
        return None
    head_params: List[torch.Tensor] = []
    for name, _ in model.proj_linear.named_parameters(prefix="proj_linear"):
        head_params.append(params[name])
    if getattr(config, "head_use_ln", True) and hasattr(model, "proj_ln"):
        for name, _ in model.proj_ln.named_parameters(prefix="proj_ln"):
            head_params.append(params[name])
    return head_params


def _build_eval_candidates_fast(
    dataset: LooSequenceDataset,
    users: List[int],
    targets: List[int],
    sample_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    max_tries: int = 3,
) -> torch.Tensor:
    if sample_size < 2:
        sample_size = 2
    batch_size = len(users)
    if batch_size == 0:
        return torch.empty((0, sample_size), device=device, dtype=torch.long)
    min_id = max(2, getattr(dataset, "min_item_id", 1))
    max_item = dataset.max_item
    neg_size = sample_size - 1
    targets_t = torch.as_tensor(targets, dtype=torch.long)
    if neg_size <= 0:
        return targets_t.unsqueeze(1).to(device)

    neg = torch.randint(min_id, max_item + 1, size=(batch_size, neg_size), generator=generator, dtype=torch.long)

    use_fixed_neg = hasattr(dataset, "neg_item_by_user")
    seen_lists: List[List[int]] = []
    fixed_vals = torch.zeros((batch_size,), dtype=torch.long)
    fixed_mask = torch.zeros((batch_size,), dtype=torch.bool)
    for i, user in enumerate(users):
        seen = [x for x in dataset.user_seq[user] if x > 1]
        seen_lists.append(seen)
        if use_fixed_neg:
            fixed_neg = dataset.neg_item_by_user.get(user)
            if fixed_neg and fixed_neg > 1 and fixed_neg != targets[i] and fixed_neg not in seen:
                fixed_vals[i] = fixed_neg
                fixed_mask[i] = True

    max_seen = max((len(s) for s in seen_lists), default=0)
    if max_seen > 0:
        seen_padded = torch.full((batch_size, max_seen), -1, dtype=torch.long)
        for i, seen in enumerate(seen_lists):
            if seen:
                seen_padded[i, : len(seen)] = torch.as_tensor(seen, dtype=torch.long)
        for _ in range(max_tries):
            mask = torch.isin(neg, seen_padded) | (neg == targets_t.unsqueeze(1))
            if not mask.any():
                break
            neg[mask] = torch.randint(
                min_id,
                max_item + 1,
                size=(int(mask.sum().item()),),
                generator=generator,
                dtype=torch.long,
            )
    else:
        for _ in range(max_tries):
            mask = neg == targets_t.unsqueeze(1)
            if not mask.any():
                break
            neg[mask] = torch.randint(
                min_id,
                max_item + 1,
                size=(int(mask.sum().item()),),
                generator=generator,
                dtype=torch.long,
            )

    if use_fixed_neg and fixed_mask.any():
        neg[fixed_mask, 0] = fixed_vals[fixed_mask]

    return torch.cat([targets_t.unsqueeze(1), neg], dim=1).to(device)


def _snapshot_params_by_name(
    model: nn.Module, names: List[str]
) -> Dict[str, torch.Tensor]:
    param_dict = dict(model.named_parameters())
    return {name: param_dict[name].detach().clone() for name in names if name in param_dict}


def _load_params_by_name(
    model: nn.Module, state: Dict[str, torch.Tensor], names: List[str]
) -> None:
    param_dict = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            if name in param_dict and name in state:
                src = state[name].to(device=param_dict[name].device, dtype=param_dict[name].dtype)
                param_dict[name].copy_(src)


def _mask_inputs_with_unk(input_ids: torch.Tensor, mask_prob: float) -> torch.Tensor:

    """Randomly mask real item ids (>1) to UNK=1 with probability mask_prob."""
    if mask_prob <= 0:
        return input_ids
    mask = (torch.rand_like(input_ids.float()) < mask_prob) & (input_ids > 1)
    if not mask.any():
        return input_ids
    masked = input_ids.clone()
    masked[mask] = 1
    return masked


def train_sasrec_meta(
    model: SASRec,
    train_dataset,
    config: SASRecConfig,
    device: str = "cpu",
    val_dataset=None,
) -> Tuple[Dict[str, float], Optional[Path]]:
    """Bi-level optimization with meta-patches (inner: short-view, outer: long-view distillation)."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.train()

    if config.inner_seq_length > config.max_seq_length:
        logger.warning(
            "inner_seq_length (%s) > max_seq_length (%s); clamping to max_seq_length.",
            config.inner_seq_length,
            config.max_seq_length,
        )
        config.inner_seq_length = config.max_seq_length
    if config.eval_seq_length > config.max_seq_length:
        logger.warning(
            "eval_seq_length (%s) > max_seq_length (%s); clamping to max_seq_length.",
            config.eval_seq_length,
            config.max_seq_length,
        )
        config.eval_seq_length = config.max_seq_length

    base_params, base_buffers = collect_base_params_and_buffers(model)
    train_bias, train_layernorm = resolve_inner_bitfit_flags(config)
    theta_names = build_bitfit_param_names(
        model,
        enable_bias=train_bias,
        enable_layernorm=train_layernorm,
        enable_head=config.inner_train_head and config.enable_projection_head,
        enable_score_head=bool(getattr(config, "inner_train_score_head", True)),
    )
    if not theta_names:
        raise RuntimeError("No parameters selected for inner loop; check BitFit selection.")
    theta = [base_params[n].detach().clone().requires_grad_(True) for n in theta_names]
    inner_opt = MomentumInner(theta, lr=config.inner_lr, momentum=config.inner_momentum)
    theta_init = [t.detach().clone() for t in theta]
    logger.info(
        "BitFit inner params: %s tensors, %s parameters",
        len(theta_names),
        sum(t.numel() for t in theta),
    )
    eta = model.meta_patch.eta

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def _inner_loss(
        theta_list: List[torch.Tensor],
        eta_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_logits, neg_logits, _ = sasrec_training_step_stateless(
            model,
            base_params,
            base_buffers,
            theta_names,
            theta_list,
            eta_tensor,
            input_ids,
            pos_ids,
            neg_ids,
            user_ids,
            use_patch=True,
        )
        pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
        neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
        raw_loss = pos_loss + neg_loss
        valid_mask = pos_ids != 0
        mode = getattr(config, "inner_loss_mode", "match_outer")
        if mode == "match_outer":
            mode = getattr(config, "outer_loss_mode", "all")
        loss = _reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
        if not valid_mask.any():
            zero = eta_tensor.sum() * 0.0
            for t in theta_list:
                zero = zero + t.sum() * 0.0
            return zero
        return loss

    def _inner_loss_meta(
        theta_list: List[torch.Tensor],
        eta_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return config.lambda_meta * _inner_loss(theta_list, eta_tensor, input_ids, pos_ids, neg_ids, user_ids)

    def _outer_loss(
        theta_list: List[torch.Tensor],
        eta_tensor: torch.Tensor,
        short_input_ids: torch.Tensor,
        short_pos_ids: torch.Tensor,
        short_neg_ids: torch.Tensor,
        long_input_ids: torch.Tensor,
        long_pos_ids: torch.Tensor,
        long_neg_ids: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal last_outer_mid_loss
        nonlocal last_outer_tail_loss
        nonlocal last_outer_mid_rel_loss
        nonlocal last_outer_gt_loss
        num_neg = max(1, int(getattr(config, "outer_neg_samples", 1)))
        if num_neg > 1:
            neg_long_ids = _build_outer_neg_ids(
                dataset=train_dataset,
                user_ids=user_ids,
                pos_ids=long_pos_ids,
                num_neg=num_neg,
                device=long_pos_ids.device,
            )
            neg_short_ids = neg_long_ids[:, -short_pos_ids.size(1) :, :]
        else:
            neg_short_ids = short_neg_ids
            neg_long_ids = long_neg_ids

        pos_short, neg_short, _ = sasrec_training_step_stateless(
            model,
            base_params,
            base_buffers,
            theta_names,
            theta_list,
            eta_tensor,
            short_input_ids,
            short_pos_ids,
            neg_short_ids,
            user_ids,
            use_patch=True,
        )
        pos_long_full, neg_long_full, _ = sasrec_training_step_stateless(
            model,
            base_params,
            base_buffers,
            theta_names,
            theta_list,
            eta_tensor,
            long_input_ids,
            long_pos_ids,
            neg_long_ids,
            user_ids,
            use_patch=False,
        )

        pos_long_tail = pos_long_full
        neg_long_tail = neg_long_full
        if pos_long_tail.size(1) != pos_short.size(1):
            pos_long_tail = pos_long_tail[:, -pos_short.size(1) :]
            neg_long_tail = neg_long_tail[:, -pos_short.size(1) :]
        pos_long_tail = pos_long_tail.detach()
        neg_long_tail = neg_long_tail.detach()

        distill = getattr(config, "outer_distill", "kl")
        temp = float(getattr(config, "outer_distill_temperature", 1.0))
        if temp <= 0:
            temp = 1.0

        def _distill_logits(
            pos_student: torch.Tensor,
            neg_student: torch.Tensor,
            pos_teacher: torch.Tensor,
            neg_teacher: torch.Tensor,
        ) -> torch.Tensor:
            if distill == "mse":
                if neg_student.dim() == 3:
                    neg_term = (neg_student - neg_teacher).pow(2).mean(dim=-1)
                else:
                    neg_term = (neg_student - neg_teacher).pow(2)
                return (pos_student - pos_teacher).pow(2) + neg_term
            if distill == "soft_bce":
                pos_targets = torch.sigmoid(pos_teacher / temp)
                neg_targets = torch.sigmoid(neg_teacher / temp)
                pos_loss = F.binary_cross_entropy_with_logits(pos_student / temp, pos_targets, reduction="none")
                neg_loss = F.binary_cross_entropy_with_logits(neg_student / temp, neg_targets, reduction="none")
                if neg_loss.dim() == 3:
                    neg_loss = neg_loss.mean(dim=-1)
                return (pos_loss + neg_loss) * (temp ** 2)
            if distill == "kl":
                if neg_student.dim() == 3:
                    logits_student = torch.cat([pos_student.unsqueeze(-1), neg_student], dim=-1)
                    logits_teacher = torch.cat([pos_teacher.unsqueeze(-1), neg_teacher], dim=-1)
                else:
                    logits_student = torch.stack([pos_student, neg_student], dim=-1)
                    logits_teacher = torch.stack([pos_teacher, neg_teacher], dim=-1)
                logp_student = F.log_softmax(logits_student / temp, dim=-1)
                logp_teacher = F.log_softmax(logits_teacher / temp, dim=-1)
                p_teacher = logp_teacher.exp()
                return (p_teacher * (logp_teacher - logp_student)).sum(dim=-1) * (temp ** 2)
            raise ValueError(f"Unknown outer_distill: {distill}")

        raw_loss = _distill_logits(pos_short, neg_short, pos_long_tail, neg_long_tail)
        valid_mask = short_pos_ids != 0
        mode = getattr(config, "outer_loss_mode", "all")
        tail_loss = _reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
        last_outer_tail_loss = tail_loss.item() if isinstance(tail_loss, torch.Tensor) else float(tail_loss)
        tail_weight = float(getattr(config, "outer_tail_weight", 1.0))
        if tail_weight < 0:
            tail_weight = 0.0
        loss = tail_weight * tail_loss

        last_outer_mid_loss = None
        last_outer_mid_rel_loss = None
        last_outer_gt_loss = None
        short_seq_lengths = _sequence_lengths_from_input_ids(short_input_ids)
        mid_weight = float(getattr(config, "outer_mid_weight", 0.0))
        mid_rel_weight = float(getattr(config, "outer_mid_rel_weight", 0.0))
        if mid_weight > 0 and config.patch_len > 0:
            mid_samples = int(getattr(config, "outer_mid_samples", 0))
            if mid_samples <= 0:
                mid_samples = int(config.patch_len)
            mid_samples = min(mid_samples, int(config.patch_len))
            if mid_samples > 0:
                mid_idx, mid_mask = _select_middle_positions(
                    long_input_ids,
                    int(getattr(config, "prefix_len", 0) or 0),
                    int(getattr(config, "inner_seq_length", 0) or 0),
                    mid_samples,
                )
                if mid_mask.any():
                    mid_pos_ids = long_pos_ids.gather(1, mid_idx)
                    if neg_long_ids.dim() == 3:
                        idx_exp = mid_idx.unsqueeze(-1).expand(-1, -1, neg_long_ids.size(-1))
                        mid_neg_ids = neg_long_ids.gather(1, idx_exp)
                    else:
                        mid_neg_ids = neg_long_ids.gather(1, mid_idx)

                    mid_pos_ids = mid_pos_ids.masked_fill(~mid_mask, 0)
                    if mid_neg_ids.dim() == 3:
                        mid_neg_ids = mid_neg_ids.masked_fill(~mid_mask.unsqueeze(-1), 0)
                    else:
                        mid_neg_ids = mid_neg_ids.masked_fill(~mid_mask, 0)

                    mid_teacher_pos = pos_long_full.gather(1, mid_idx).detach()
                    if neg_long_full.dim() == 3:
                        mid_teacher_neg = neg_long_full.gather(1, idx_exp).detach()
                    else:
                        mid_teacher_neg = neg_long_full.gather(1, mid_idx).detach()

                    override = build_override(theta_names, theta_list)
                    params = {**base_params, **override, **base_buffers}
                    hidden_states = functional_call(
                        model,
                        params,
                        args=(),
                        kwargs={
                            "input_ids": short_input_ids,
                            "patch_params": eta_tensor,
                            "user_ids": user_ids,
                            "return_gating": False,
                            "use_patch": True,
                        },
                    )
                    patch_start = get_patch_token_start(config, short_input_ids.size(1))
                    patch_hidden = hidden_states[:, patch_start : patch_start + mid_samples, :]
                    head_params = _collect_head_params_from_params(model, params, config)
                    patch_proj = model.apply_head(patch_hidden, head_params=head_params)

                    pos_emb = _lookup_item_embeddings(model, mid_pos_ids, state=params)
                    pos_mid = _apply_similarity_logits(
                        patch_proj,
                        pos_emb,
                        config,
                        model=model,
                        item_ids=mid_pos_ids,
                        seq_lengths=short_seq_lengths,
                        state=params,
                    )
                    if mid_neg_ids.dim() == 3:
                        neg_emb = _lookup_item_embeddings(model, mid_neg_ids, state=params)
                        neg_mid = _apply_similarity_logits(
                            patch_proj,
                            neg_emb,
                            config,
                            model=model,
                            item_ids=mid_neg_ids,
                            seq_lengths=short_seq_lengths,
                            state=params,
                        )
                    else:
                        neg_emb = _lookup_item_embeddings(model, mid_neg_ids, state=params)
                        neg_mid = _apply_similarity_logits(
                            patch_proj,
                            neg_emb,
                            config,
                            model=model,
                            item_ids=mid_neg_ids,
                            seq_lengths=short_seq_lengths,
                            state=params,
                        )

                    mid_raw = _distill_logits(pos_mid, neg_mid, mid_teacher_pos, mid_teacher_neg)
                    mid_valid = mid_mask & (mid_pos_ids != 0)
                    if mid_valid.any():
                        mid_loss = mid_raw[mid_valid].mean()
                        loss = loss + mid_weight * mid_loss
                        last_outer_mid_loss = mid_loss.item()

                    if mid_rel_weight != 0.0 and mid_mask.any():
                        # Relational alignment between patch hidden and teacher mid hidden.
                        with torch.no_grad():
                            teacher_hidden = functional_call(
                                model,
                                {**base_params, **base_buffers},
                                args=(),
                                kwargs={
                                    "input_ids": long_input_ids,
                                    "patch_params": eta_tensor,
                                    "user_ids": user_ids,
                                    "return_gating": False,
                                    "use_patch": False,
                                },
                            )
                        teacher_mid = teacher_hidden.gather(
                            1, mid_idx.unsqueeze(-1).expand(-1, -1, teacher_hidden.size(-1))
                        )
                        valid_rows = mid_mask.sum(dim=1) == mid_samples
                        if valid_rows.any():
                            stud = patch_hidden[valid_rows]
                            teach = teacher_mid[valid_rows]
                            stud = F.normalize(stud, dim=-1)
                            teach = F.normalize(teach, dim=-1)
                            s_rel = stud @ stud.transpose(1, 2)
                            t_rel = teach @ teach.transpose(1, 2)
                            rel_loss = F.mse_loss(s_rel, t_rel)
                            loss = loss + mid_rel_weight * rel_loss
                            last_outer_mid_rel_loss = rel_loss.item()

        gt_weight = float(getattr(config, "outer_gt_weight", 1.0))
        if gt_weight != 0.0:
            pos_loss = bce_criterion(pos_short, torch.ones_like(pos_short))
            neg_loss = bce_criterion(neg_short, torch.zeros_like(neg_short))
            if neg_loss.dim() == 3:
                neg_loss = neg_loss.mean(dim=-1)
            gt_raw = pos_loss + neg_loss
            gt_valid = short_pos_ids != 0
            gt_mode = getattr(config, "outer_loss_mode", "all")
            gt_loss = _reduce_loss(gt_raw, gt_valid, gt_mode, config.outer_loss_decay)
            if isinstance(gt_loss, torch.Tensor):
                if gt_valid.any():
                    last_outer_gt_loss = gt_loss.item()
            loss = loss + gt_weight * gt_loss

        return loss

    grad_fn = get_fwdrev_grad_fn_eta(_inner_loss)
    grad_fn_meta = get_fwdrev_grad_fn_eta(_inner_loss_meta)

    model_param_dict = dict(model.named_parameters())
    outer_param_names = build_outer_param_names(model)
    outer_params = [
        model_param_dict[name]
        for name in outer_param_names
        if name in model_param_dict and model_param_dict[name].requires_grad
    ]
    if not outer_params:
        raise RuntimeError(
            "No outer-loop parameters selected; expected at least one active patch bank, shared prefix, or LoRA adapter."
        )
    logger.info(
        "Outer params: %s tensors, %s parameters",
        len(outer_params),
        sum(p.numel() for p in outer_params),
    )
    outer_opt = torch.optim.AdamW(outer_params, lr=config.outer_lr, weight_decay=config.outer_weight_decay)

    train_cfg = replace(config, max_seq_length=config.max_seq_length, batch_size=config.batch_size)
    outer_cfg = replace(config, max_seq_length=config.max_seq_length, batch_size=config.val_batch_size)
    train_data = LooTrainDataset(train_dataset, train_cfg)
    collate_fn = build_train_collate_fn(train_data, train_cfg)
    num_workers = max(0, int(config.num_workers))
    loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(config.pin_memory),
        "collate_fn": collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(config.persistent_workers)
    train_loader = DataLoader(train_data, **loader_kwargs)
    outer_loader = DataLoader(train_data, **{**loader_kwargs, "batch_size": outer_cfg.batch_size})
    outer_iter = iter(outer_loader)

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    pbar = tqdm(total=total_steps)

    recent_steps = deque(maxlen=config.meta_truncate_steps) if config.meta_truncate_steps > 0 else None

    global_step = 0
    last_outer_loss = None
    last_outer_tail_loss = None
    last_outer_mid_loss = None
    last_outer_mid_rel_loss = None
    last_outer_gt_loss = None
    best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}
    best_ckpt_path: Optional[Path] = None
    timing_enabled = bool(getattr(config, "enable_timing", False))
    timing_window = max(1, int(getattr(config, "timing_window", 50)))
    timing_accum = {
        "step_ms": 0.0,
        "fetch_ms": 0.0,
        "move_ms": 0.0,
        "extra_fetch_ms": 0.0,
        "extra_move_ms": 0.0,
        "inner_ms": 0.0,
        "outer_ms": 0.0,
        "log_ms": 0.0,
    }
    timing_steps = 0

    def _reset_inner_state() -> None:
        for i in range(len(theta)):
            theta[i] = theta_init[i].detach().clone().requires_grad_(True)
        inner_opt.params = theta
        inner_opt.m = [torch.zeros_like(p) for p in theta]
        if recent_steps is not None:
            recent_steps.clear()

    for epoch in range(config.num_epochs):
        model.train()

        train_iter = iter(train_loader)
        extra_iter = iter(train_loader)
        for _ in range(steps_per_epoch):
            global_step += 1
            if config.inner_reset_every and config.inner_reset_every > 0:
                if global_step % int(config.inner_reset_every) == 0:
                    _reset_inner_state()
                    logger.info("Reset inner parameters at global step %s", global_step)
            if timing_enabled:
                t_step_start = time.perf_counter()
                t_fetch_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            fetch_ms = (time.perf_counter() - t_fetch_start) * 1000.0 if timing_enabled else 0.0
            if timing_enabled:
                t_move_start = time.perf_counter()
            batch = _move_batch_to_device(batch, device_obj)
            move_ms = (time.perf_counter() - t_move_start) * 1000.0 if timing_enabled else 0.0
            extra_fetch_ms = 0.0
            extra_move_ms = 0.0
            step_batches = [batch]
            for _ in range(config.inner_steps - 1):
                if timing_enabled:
                    t_extra_fetch = time.perf_counter()
                try:
                    extra_batch = next(extra_iter)
                except StopIteration:
                    extra_iter = iter(train_loader)
                    extra_batch = next(extra_iter)
                if timing_enabled:
                    extra_fetch_ms += (time.perf_counter() - t_extra_fetch) * 1000.0
                    t_extra_move = time.perf_counter()
                step_batches.append(_move_batch_to_device(extra_batch, device_obj))
                if timing_enabled:
                    extra_move_ms += (time.perf_counter() - t_extra_move) * 1000.0

            if timing_enabled:
                t_inner_start = time.perf_counter()
            for step_batch in step_batches:
                if config.prefix_len and config.prefix_len > 0:
                    if config.inner_drop_prefix:
                        short_batch = _drop_prefix_from_batch(
                            step_batch,
                            config.prefix_len,
                            config.inner_seq_length,
                            train_dataset,
                        )
                    else:
                        short_batch = _build_prefix_tail_batch(
                            step_batch,
                            config.prefix_len,
                            config.inner_seq_length,
                            train_dataset,
                            supervise_prefix_targets=bool(getattr(config, "supervise_prefix_targets", False)),
                            prefix_source=str(getattr(config, "prefix_source", "head") or "head"),
                        )
                else:
                    short_batch = _slice_batch_tail(step_batch, config.inner_seq_length)
                if (not config.drop_unseen_items) and config.inner_unk_mask_prob > 0:
                    short_batch = {
                        **short_batch,
                        "input_ids": _mask_inputs_with_unk(
                            short_batch["input_ids"], config.inner_unk_mask_prob
                        ),
                    }
                if recent_steps is not None:
                    w_state, m_state = inner_opt.snapshot()
                    recent_steps.append(
                        (
                            w_state,
                            m_state,
                            {
                                "input_ids": short_batch["input_ids"],
                                "pos_ids": short_batch["pos_ids"],
                                "neg_ids": short_batch["neg_ids"],
                                "internal_user_ids": short_batch.get("internal_user_ids"),
                            },
                        )
                    )

                gflat = grad_fn(
                    theta,
                    eta,
                    short_batch["input_ids"],
                    short_batch["pos_ids"],
                    short_batch["neg_ids"],
                    short_batch.get("internal_user_ids"),
                )
                if config.inner_grad_clip and config.inner_grad_clip > 0:
                    gflat = torch.clamp(gflat, min=-config.inner_grad_clip, max=config.inner_grad_clip)
                inner_opt.step(gflat)
            inner_ms = (time.perf_counter() - t_inner_start) * 1000.0 if timing_enabled else 0.0

            do_outer = (
                config.outer_update_every > 0
                and global_step % config.outer_update_every == 0
                and (recent_steps is None or len(recent_steps) >= config.meta_truncate_steps)
            )
            if timing_enabled:
                t_outer_start = time.perf_counter()
            if do_outer:
                try:
                    batch_long = next(outer_iter)
                except StopIteration:
                    outer_iter = iter(outer_loader)
                    batch_long = next(outer_iter)
                batch_long = _move_batch_to_device(batch_long, device_obj)
                if config.prefix_len and config.prefix_len > 0:
                    if config.inner_drop_prefix:
                        batch_short = _drop_prefix_from_batch(
                            batch_long,
                            config.prefix_len,
                            config.inner_seq_length,
                            train_dataset,
                        )
                    else:
                        batch_short = _build_prefix_tail_batch(
                            batch_long,
                            config.prefix_len,
                            config.inner_seq_length,
                            train_dataset,
                            supervise_prefix_targets=bool(getattr(config, "supervise_prefix_targets", False)),
                            prefix_source=str(getattr(config, "prefix_source", "head") or "head"),
                        )
                else:
                    batch_short = _slice_batch_tail(batch_long, config.inner_seq_length)

                latest_w, latest_m = inner_opt.snapshot()
                if recent_steps is not None and config.meta_truncate_steps > 0:
                    start_w, start_m, _ = recent_steps[0]
                    inner_opt.restore(start_w, start_m)
                    for _, _, step_batch in recent_steps:
                        gflat = grad_fn_meta(
                            theta,
                            eta,
                            step_batch["input_ids"],
                            step_batch["pos_ids"],
                            step_batch["neg_ids"],
                            step_batch.get("internal_user_ids"),
                        )
                        if config.inner_grad_clip and config.inner_grad_clip > 0:
                            gflat = torch.clamp(gflat, min=-config.inner_grad_clip, max=config.inner_grad_clip)
                        inner_opt.step(gflat)

                outer_opt.zero_grad(set_to_none=True)
                model.eval()
                loss_outer = _outer_loss(
                    theta,
                    eta,
                    batch_short["input_ids"],
                    batch_short["pos_ids"],
                    batch_short["neg_ids"],
                    batch_long["input_ids"],
                    batch_long["pos_ids"],
                    batch_long["neg_ids"],
                    batch_long["internal_user_ids"],
                )
                model.train()
                loss_outer.backward()
                if config.outer_grad_clip and config.outer_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(outer_params, config.outer_grad_clip)
                outer_opt.step()
                last_outer_loss = loss_outer.item()

                inner_opt.restore(latest_w, latest_m)
            outer_ms = (time.perf_counter() - t_outer_start) * 1000.0 if timing_enabled else 0.0

            log_ms = 0.0
            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                if timing_enabled:
                    t_log_start = time.perf_counter()
                if config.prefix_len and config.prefix_len > 0:
                    if config.inner_drop_prefix:
                        log_batch = _drop_prefix_from_batch(
                            step_batches[-1],
                            config.prefix_len,
                            config.inner_seq_length,
                            train_dataset,
                        )
                    else:
                        log_batch = _build_prefix_tail_batch(
                            step_batches[-1],
                            config.prefix_len,
                            config.inner_seq_length,
                            train_dataset,
                            supervise_prefix_targets=bool(getattr(config, "supervise_prefix_targets", False)),
                            prefix_source=str(getattr(config, "prefix_source", "head") or "head"),
                        )
                else:
                    log_batch = _slice_batch_tail(step_batches[-1], config.inner_seq_length)
                avg_weights_list = None
                with torch.no_grad():
                    pos_logits, neg_logits, gating = sasrec_training_step_stateless(
                        model,
                        base_params,
                        base_buffers,
                        theta_names,
                        theta,
                        eta,
                        log_batch["input_ids"],
                        log_batch["pos_ids"],
                        log_batch["neg_ids"],
                        log_batch.get("internal_user_ids"),
                        return_gating=True,
                        use_patch=True,
                    )
                    pos_loss = bce_criterion(pos_logits, torch.ones_like(pos_logits))
                    neg_loss = bce_criterion(neg_logits, torch.zeros_like(neg_logits))
                    raw_loss = pos_loss + neg_loss
                    valid_mask = log_batch["pos_ids"] != 0
                    mode = getattr(config, "inner_loss_mode", "match_outer")
                    if mode == "match_outer":
                        mode = getattr(config, "outer_loss_mode", "all")
                    inner_loss = _reduce_loss(raw_loss, valid_mask, mode, config.outer_loss_decay)
                    if valid_mask.any():
                        log_metrics({"meta/inner_loss": inner_loss.item()})
                    if gating is not None and gating.numel() > 0:
                        weights = gating.detach().float().cpu()
                        avg_weights = weights.mean(dim=0)
                        avg_weights_list = avg_weights.tolist()
                        log_dict = {
                            f"gating/avg_weight_{i}": avg_weights[i].item()
                            for i in range(avg_weights.numel())
                        }
                        top1 = weights.argmax(dim=1)
                        for i in range(avg_weights.numel()):
                            log_dict[f"gating/top1_frac_{i}"] = (top1 == i).float().mean().item()

                        lengths = (log_batch["input_ids"] != 0).sum(dim=1).detach().cpu()
                        if lengths.numel() > 0:
                            median = lengths.median()
                            short_mask = lengths <= median
                            long_mask = lengths > median
                            max_log = min(avg_weights.numel(), 8)
                            if short_mask.any():
                                short_avg = weights[short_mask].mean(dim=0)
                                for i in range(max_log):
                                    log_dict[f"gating/avg_weight_short_{i}"] = short_avg[i].item()
                            if long_mask.any():
                                long_avg = weights[long_mask].mean(dim=0)
                                for i in range(max_log):
                                    log_dict[f"gating/avg_weight_long_{i}"] = long_avg[i].item()

                        log_metrics(log_dict)
                        log_metrics({"gating/weight_hist": wandb.Histogram(weights.numpy())})

                if last_outer_loss is not None:
                    gating_summary = ""
                    if avg_weights_list is not None:
                        preview = ", ".join(f"{w:.3f}" for w in avg_weights_list[: min(8, len(avg_weights_list))])
                        gating_summary = f" | GatingAvg[:8]: {preview}"
                    logger.info(
                        "Step %06d | Epoch %03d/%03d | InnerLoss: %.4f | OuterLoss: %.4f%s",
                        global_step,
                        epoch + 1,
                        config.num_epochs,
                        inner_loss.item() if valid_mask.any() else 0.0,
                        last_outer_loss,
                        gating_summary,
                    )
                    outer_log = {
                        "meta/outer_loss": last_outer_loss,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                    if last_outer_tail_loss is not None:
                        outer_log["meta/outer_tail_loss"] = last_outer_tail_loss
                    if last_outer_mid_loss is not None:
                        outer_log["meta/outer_mid_loss"] = last_outer_mid_loss
                    if last_outer_mid_rel_loss is not None:
                        outer_log["meta/outer_mid_rel_loss"] = last_outer_mid_rel_loss
                    if last_outer_gt_loss is not None:
                        outer_log["meta/outer_gt_loss"] = last_outer_gt_loss
                    log_metrics(outer_log)
                if timing_enabled:
                    log_ms = (time.perf_counter() - t_log_start) * 1000.0

            if timing_enabled:
                step_ms = (time.perf_counter() - t_step_start) * 1000.0
                timing_accum["step_ms"] += step_ms
                timing_accum["fetch_ms"] += fetch_ms
                timing_accum["move_ms"] += move_ms
                timing_accum["extra_fetch_ms"] += extra_fetch_ms
                timing_accum["extra_move_ms"] += extra_move_ms
                timing_accum["inner_ms"] += inner_ms
                timing_accum["outer_ms"] += outer_ms
                timing_accum["log_ms"] += log_ms
                timing_steps += 1
                if (
                    timing_steps >= timing_window
                    or global_step == 1
                    or global_step % config.steps_per_train_log == 0
                ):
                    avg = {k: v / max(1, timing_steps) for k, v in timing_accum.items()}
                    logger.info(
                        "Timing avg (%d steps) | step %.2f ms | fetch %.2f | move %.2f | "
                        "extra_fetch %.2f | extra_move %.2f | inner %.2f | outer %.2f | log %.2f",
                        timing_steps,
                        avg["step_ms"],
                        avg["fetch_ms"],
                        avg["move_ms"],
                        avg["extra_fetch_ms"],
                        avg["extra_move_ms"],
                        avg["inner_ms"],
                        avg["outer_ms"],
                        avg["log_ms"],
                    )
                    log_metrics(
                        {
                            "time/step_ms": avg["step_ms"],
                            "time/fetch_ms": avg["fetch_ms"],
                            "time/move_ms": avg["move_ms"],
                            "time/extra_fetch_ms": avg["extra_fetch_ms"],
                            "time/extra_move_ms": avg["extra_move_ms"],
                            "time/inner_ms": avg["inner_ms"],
                            "time/outer_ms": avg["outer_ms"],
                            "time/log_ms": avg["log_ms"],
                        }
                    )
                    timing_accum = {k: 0.0 for k in timing_accum}
                    timing_steps = 0

            pbar.update(1)

        if (
            val_dataset is not None
            and config.val_eval_every_epochs > 0
            and (epoch + 1) % config.val_eval_every_epochs == 0
        ):
            model.eval()
            val_metrics = evaluate(
                model,
                val_dataset,
                config=config,
                mode="val",
                device=str(device_obj),
                use_patch=True,
                use_head=True,
                max_seq_length=config.max_seq_length,
                truncate_len=config.eval_seq_length,
            )
            model.train()
            log_metrics(
                {
                    "val/meta_patch_ndcg@10": val_metrics["ndcg@10"],
                    "val/meta_patch_hr@10": val_metrics["hr@10"],
                    "progress/epoch": epoch + 1,
                }
            )
            logger.info(
                "Epoch %03d | Val Meta-Patch - NDCG@10: %.4f, HR@10: %.4f",
                epoch + 1,
                val_metrics["ndcg@10"],
                val_metrics["hr@10"],
            )
            if val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
                best_val_metrics = val_metrics
                if config.save_best_model:
                    best_ckpt_path = save_model_checkpoint(model, config)
                    logger.info("Saved best val checkpoint to %s", best_ckpt_path)

    pbar.close()

    # Copy learned theta back into the model for downstream evaluation.
    with torch.no_grad():
        param_dict = dict(model.named_parameters())
        for name, t in zip(theta_names, theta):
            if name in param_dict:
                param_dict[name].copy_(t.detach())

    if val_dataset is not None and config.eval_after_train:
        logger.info("Running validation evaluation after meta-training...")
        metrics = evaluate(
            model,
            val_dataset,
            config=config,
            mode="val",
            device=str(device_obj),
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
        )
        if metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
            best_val_metrics = metrics
            if config.save_best_model:
                best_ckpt_path = save_model_checkpoint(model, config)
                logger.info("Saved best val checkpoint to %s", best_ckpt_path)
    elif val_dataset is None:
        best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}

    return best_val_metrics, best_ckpt_path


def save_item_embeddings(model: nn.Module, dataset, config: SASRecConfig) -> Path:
    """Save item embedding matrix (excluding padding idx=0)."""
    emb = _all_item_embeddings(model).detach().cpu().numpy()
    emb = emb[2:]  # drop padding and UNK rows
    filename = f"{build_checkpoint_tag(config)}_item_embeddings_best.npy"
    out_path = config.checkpoint_dir / filename
    np.save(out_path, emb)
    logger.info(f"Saved item embeddings to {out_path}")
    logger.info("Item index mapping follows item2idx.json with +1 offset (PAD=0, UNK=1).")
    return out_path


def save_model_checkpoint(model: nn.Module, config: SASRecConfig, filename: Optional[str] = None) -> Path:
    if filename is None:
        filename = f"{build_checkpoint_tag(config)}_best.pt"
    out_path = config.checkpoint_dir / filename
    mode = str(getattr(config, "checkpoint_mode", "full")).lower()
    lightweight_hllm_text = (
        str(getattr(config, "backbone", "")).lower() in {"hllm_text", "hllmtext"}
        and bool(getattr(config, "hllm_text_lightweight_checkpoint", True))
        and hasattr(model, "checkpoint_state_dict")
    )
    if lightweight_hllm_text:
        mode = "delta"
    if mode not in {"full", "delta"}:
        raise ValueError(f"Unknown checkpoint_mode: {mode}")
    payload = {
        "checkpoint_mode": mode,
        "config": _serialize_config(config),
    }
    routing_state = _extract_routing_state(model)
    if routing_state:
        payload["routing_state"] = routing_state
    if lightweight_hllm_text:
        payload["state_dict"] = model.checkpoint_state_dict()
        payload["trainable_keys"] = list(payload["state_dict"].keys())
        payload["base_ckpt_path"] = None
    elif mode == "full":
        payload["state_dict"] = model.state_dict()
    else:
        delta_state, trainable_keys = _collect_trainable_state_dict(model)
        payload["state_dict"] = delta_state
        payload["trainable_keys"] = trainable_keys
        payload["base_ckpt_path"] = (
            str(config.pretrained_ckpt_path) if config.pretrained_ckpt_path else None
        )
    torch.save(payload, out_path)
    return out_path


def build_backbone(config: SASRecConfig, item_num: int) -> nn.Module:
    name = getattr(config, "backbone", "sasrec").lower()
    if name == "sasrec":
        return SASRec(config, item_num=item_num)
    if name == "hstu":
        return HSTU(config, item_num=item_num)
    if name in ("hstu_officialish", "hstu_official", "hstu_orig"):
        return HSTUOfficialish(config, item_num=item_num)
    if name in ("hstu_research_aligned", "hstu_research", "hstu_ra"):
        return HSTUResearchAligned(config, item_num=item_num)
    if name in ("hllm", "hllmonline"):
        from backbones.HLLM import HLLM

        return HLLM(config, item_num=item_num)
    if name in ("hllm_text", "hllmtext"):
        from backbones.HLLMText import HLLMText

        return HLLMText(config, item_num=item_num)
    if name == "longer":
        return LONGER(config, item_num=item_num)
    if name == "fmlp":
        return FMLP(config, item_num=item_num)
    if name == "linrec":
        return LinRec(config, item_num=item_num)
    if name == "lru":
        return LRU(config, item_num=item_num)
    if name in ("mamba4rec", "mamba"):
        return Mamba4Rec(config, item_num=item_num)
    if name in ("bert4rec", "bert"):
        return Bert4Rec(config, item_num=item_num)
    if name in ("gru4rec", "gru"):
        return GRU4Rec(config, item_num=item_num)
    raise ValueError(f"Unknown backbone: {config.backbone}")


def build_pure_backbone_eval_config(config: SASRecConfig) -> SASRecConfig:
    """Clone config for direct frozen-backbone evaluation with no PEFT or patch modules active."""
    pure_config = replace(config)
    pure_config.num_patches = 1
    pure_config.patch_len = 0
    pure_config.use_gating = False
    pure_config.enable_projection_head = False
    pure_config.head_residual = False
    pure_config.head_zero_init = False
    pure_config.head_use_gelu = False
    pure_config.head_use_ln = False
    pure_config.input_emb_lora_rank = 0
    pure_config.train_input_emb_lora = False
    pure_config.attn_lora_rank = 0
    pure_config.train_attn_lora = False
    return pure_config


def evaluate_pure_backbone_baseline(
    dataset: LooSequenceDataset,
    config: SASRecConfig,
    device: torch.device,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    mode: str = "test",
    streaming_last_k: int = 0,
) -> Dict[str, float]:
    """Evaluate the pretrained backbone directly, with no patch/head/adapters and no meta-test path."""
    pure_config = build_pure_backbone_eval_config(config)
    if state_dict is not None and "shared_prefix_tokens" not in state_dict:
        pure_config.shared_prefix_len = 0
    pure_model = build_backbone(pure_config, item_num=dataset.num_items)
    load_pretrained_backbone(pure_model, pure_config.pretrained_ckpt_path, state_dict=state_dict)
    for p in pure_model.parameters():
        p.requires_grad = False
    pure_model = pure_model.to(device)
    pure_model.eval()
    try:
        metrics = evaluate(
            pure_model,
            dataset,
            config=pure_config,
            mode=mode,
            device=device,
            use_patch=False,
            use_head=False,
            max_seq_length=pure_config.max_seq_length,
            truncate_len=pure_config.eval_seq_length,
            streaming_last_k=streaming_last_k,
        )
    finally:
        del pure_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    # Adjust hyperparameters by editing the SASRecConfig defaults above or overriding them here.
    config = SASRecConfig()

    parser = build_arg_parser()
    args = parser.parse_args()
    apply_overrides_from_args(config, args)

    project_name = os.getenv("WANDB_PROJECT") or f"gating_patch_long_short-{config.dataset}"
    run = wandb.init(project=project_name, config=config.__dict__)
    if run is not None:
        apply_overrides_from_dict(config, dict(run.config))

    resolve_dataset_config(config)
    resolve_eval_protocol_config(config)
    resolve_shared_prefix_config(config)
    set_global_seed(config.seed, config.deterministic)

    inferred_state = None
    inferred_ckpt = None
    if config.pretrained_ckpt_path and Path(config.pretrained_ckpt_path).exists():
        inferred_ckpt = load_checkpoint(config.pretrained_ckpt_path, trust_pickle=True)
        config = apply_config_from_checkpoint_payload(config, inferred_ckpt)
        inferred_state = _strip_module_prefix(_extract_state_dict(inferred_ckpt))
        inferred_state = _maybe_strip_prefix(inferred_state, config.ckpt_prefix_to_strip)
        config = infer_config_from_state_dict(inferred_state, config)
        apply_overrides_from_args(config, args)
        resolve_eval_protocol_config(config)
        resolve_shared_prefix_config(config)
    else:
        logger.warning("Pretrained checkpoint not found; proceeding without inference.")

    if run is not None:
        run.config.update(config.__dict__, allow_val_change=True)

    if str(getattr(config, "checkpoint_mode", "full")).lower() == "delta":
        if not (config.pretrained_ckpt_path and Path(config.pretrained_ckpt_path).exists()):
            raise FileNotFoundError(
                f"checkpoint_mode=delta requires a valid pretrained_ckpt_path, but got: {config.pretrained_ckpt_path}"
            )
    if getattr(config, "full_finetune", False):
        raise NotImplementedError(
            "full_finetune is not supported in train_gating_patch_long_short.py. "
            "Use train_patch_first_order.py or train_patch_frozen_teacher.py for full-model online distillation."
        )

    device_manager = DeviceManager(logger, preferred_device=config.device, gpu_id=None)
    device = device_manager.device

    run_name = _build_run_name(config)
    if run is not None:
        run.name = run_name

    base_ckpt_dir = Path(config.checkpoint_dir)
    if base_ckpt_dir.name != "gating_patch_long_short":
        base_ckpt_dir = base_ckpt_dir / "gating_patch_long_short"
    config.run_tag = _build_run_tag(config, run)
    config.checkpoint_dir = base_ckpt_dir / str(config.run_tag)
    if run is not None:
        run.config.update(
            {"checkpoint_dir": str(config.checkpoint_dir), "run_tag": config.run_tag},
            allow_val_change=True,
        )

    LOCAL_METRICS_LOGGER = LocalMetricsLogger(
        log_dir=str(config.checkpoint_dir / "logs"),
        run_name=(run.name if run is not None else run_name),
    )
    config.log_config()
    if LOCAL_METRICS_LOGGER is not None and LOCAL_METRICS_LOGGER.jsonl_path is not None:
        logger.info("Local metrics JSONL: %s", LOCAL_METRICS_LOGGER.jsonl_path)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config, run_name, sys.argv)

    train_dataset = LooSequenceDataset(config.data_txt_path, config, logger=logger)
    meta_valid_dataset = train_dataset
    test_dataset = train_dataset
    item_num = train_dataset.num_items

    model = build_backbone(config, item_num=item_num)
    if inferred_state is not None:
        has_optional_adapters = (
            int(getattr(config, "input_emb_lora_rank", 0) or 0) > 0
            or int(getattr(config, "attn_lora_rank", 0) or 0) > 0
        )
        if config.strict_load_pretrained and not has_optional_adapters:
            logger.info("Loading full checkpoint with strict=True...")
            model.load_state_dict(inferred_state, strict=True)
        else:
            if config.strict_load_pretrained and has_optional_adapters:
                logger.info("Optional adapters enabled; falling back to non-strict backbone loading.")
            load_pretrained_backbone(model, config.pretrained_ckpt_path, state_dict=inferred_state)
        if inferred_ckpt is not None:
            _restore_routing_state(model, inferred_ckpt)
    model = model.to(device)
    if config.patch_routing in {"kmeans", "user_table"}:
        centers = build_kmeans_centers(train_dataset, model, config)
        model.meta_patch.set_kmeans_centers(centers)
        logger.info("KMeans routing centers set: %s patches.", centers.size(0))
        if config.patch_routing == "user_table":
            user_to_patch = build_user_patch_table(train_dataset, model, centers, config)
            model.meta_patch.set_user_table(user_to_patch)
            logger.info("User routing table set for %s users.", len(user_to_patch))
    initialize_head_as_identity(model)
    train_bias, train_layernorm = resolve_inner_bitfit_flags(config)
    apply_bitfit_freeze(
        model,
        enable_bias=train_bias,
        enable_layernorm=train_layernorm,
        enable_head=config.inner_train_head and config.enable_projection_head,
        enable_score_head=bool(getattr(config, "inner_train_score_head", True)),
        enable_input_emb_lora=config.train_input_emb_lora,
        enable_attn_lora=config.train_attn_lora,
    )
    theta_names = build_bitfit_param_names(
        model,
        enable_bias=train_bias,
        enable_layernorm=train_layernorm,
        enable_head=config.inner_train_head and config.enable_projection_head,
        enable_score_head=bool(getattr(config, "inner_train_score_head", True)),
    )
    bitfit_init_state = _snapshot_params_by_name(model, theta_names)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if config.eval_before_train:
        logger.info("Running pre-train backbone baseline on val (pure pretrained backbone, direct eval)...")
        val_baseline = evaluate_pure_backbone_baseline(
            meta_valid_dataset,
            config=config,
            mode="val",
            device=device,
            state_dict=inferred_state,
        )
        logger.info(
            "Val Backbone Baseline - NDCG@10: %.4f, HR@10: %.4f",
            val_baseline["ndcg@10"],
            val_baseline["hr@10"],
        )
        log_metrics(
            {
                "val/baseline_ndcg@10": val_baseline["ndcg@10"],
                "val/baseline_hr@10": val_baseline["hr@10"],
                "progress/epoch": 0,
                "progress/step": 0,
            }
        )

        logger.info("Running pre-train meta-patch on val (patch + head)...")
        val_meta_patch = evaluate(
            model,
            meta_valid_dataset,
            config=config,
            mode="val",
            device=device,
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=theta_names,
            bitfit_init_state=bitfit_init_state,
        )
        logger.info(
            "Val Meta-Patch (pre-train) - NDCG@10: %.4f, HR@10: %.4f",
            val_meta_patch["ndcg@10"],
            val_meta_patch["hr@10"],
        )
        log_metrics(
            {
                "val/pre_meta_patch_ndcg@10": val_meta_patch["ndcg@10"],
                "val/pre_meta_patch_hr@10": val_meta_patch["hr@10"],
                "progress/epoch": 0,
                "progress/step": 0,
            }
        )

    best_metrics, best_ckpt_path = train_sasrec_meta(
        model=model,
        train_dataset=train_dataset,
        config=config,
        device=device,
        val_dataset=meta_valid_dataset,
    )

    if best_ckpt_path is not None and Path(best_ckpt_path).exists():
        logger.info("Loading best val checkpoint for test: %s", best_ckpt_path)
        best_ckpt = load_checkpoint(str(best_ckpt_path), trust_pickle=True)
        best_state = _strip_module_prefix(_extract_state_dict(best_ckpt))
        ckpt_mode = str(best_ckpt.get("checkpoint_mode", "full")).lower() if isinstance(best_ckpt, dict) else "full"
        if ckpt_mode == "delta":
            base_path = best_ckpt.get("base_ckpt_path") if isinstance(best_ckpt, dict) else None
            if base_path and config.pretrained_ckpt_path and str(base_path) != str(config.pretrained_ckpt_path):
                logger.warning(
                    "Delta checkpoint base_ckpt_path (%s) differs from current pretrained_ckpt_path (%s).",
                    base_path,
                    config.pretrained_ckpt_path,
                )
            model.load_state_dict(best_state, strict=False)
        else:
            model.load_state_dict(best_state, strict=True)
        _restore_routing_state(model, best_ckpt)

    logger.info("Running final test evaluation (baseline: pure pretrained backbone, direct test)...")
    baseline_metrics = evaluate_pure_backbone_baseline(
        test_dataset,
        config=config,
        mode="test",
        device=device,
        state_dict=inferred_state,
    )
    logger.info(
        "Backbone Baseline Test - NDCG@10: %.4f, HR@10: %.4f",
        baseline_metrics["ndcg@10"],
        baseline_metrics["hr@10"],
    )

    logger.info("Running final test evaluation (trained short seq, no patch, meta-test)...")
    trained_short_metrics = evaluate(
        model,
        test_dataset,
        config=config,
        mode="meta-test",
        device=device,
        use_patch=False,
        use_head=True,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
        theta_names=theta_names,
        bitfit_init_state=bitfit_init_state,
    )
    logger.info(
        "Trained Short No-Patch Test - NDCG@10: %.4f, HR@10: %.4f",
        trained_short_metrics["ndcg@10"],
        trained_short_metrics["hr@10"],
    )

    logger.info("Running final test evaluation (short seq + patch, meta-test)...")
    meta_metrics = evaluate(
        model,
        test_dataset,
        config=config,
        mode="meta-test",
        device=device,
        use_patch=True,
        use_head=True,
        max_seq_length=config.max_seq_length,
        truncate_len=config.eval_seq_length,
        theta_names=theta_names,
        bitfit_init_state=bitfit_init_state,
    )
    logger.info(
        "Meta-Patch Test - NDCG@10: %.4f, HR@10: %.4f",
        meta_metrics["ndcg@10"],
        meta_metrics["hr@10"],
    )

    log_metrics(
        {
            "test/baseline_ndcg@10": baseline_metrics["ndcg@10"],
            "test/baseline_hr@10": baseline_metrics["hr@10"],
            "test/trained_short_no_patch_ndcg@10": trained_short_metrics["ndcg@10"],
            "test/trained_short_no_patch_hr@10": trained_short_metrics["hr@10"],
            "test/meta_patch_ndcg@10": meta_metrics["ndcg@10"],
            "test/meta_patch_hr@10": meta_metrics["hr@10"],
        }
    )
    log_metrics(
        {
            "best/val_ndcg@10": best_metrics["ndcg@10"],
            "best/val_hr@10": best_metrics["hr@10"],
        }
    )

    if int(getattr(config, "streaming_eval_last_k", 0) or 0) > 1:
        stream_last_k = int(config.streaming_eval_last_k)
        logger.info(
            "Running additional streaming test evaluation over the last %s targets...",
            stream_last_k,
        )
        baseline_stream_metrics = evaluate_pure_backbone_baseline(
            test_dataset,
            config=config,
            mode="test",
            device=device,
            state_dict=inferred_state,
            streaming_last_k=stream_last_k,
        )
        trained_short_stream_metrics = evaluate(
            model,
            test_dataset,
            config=config,
            mode="meta-test",
            device=device,
            use_patch=False,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=theta_names,
            bitfit_init_state=bitfit_init_state,
            streaming_last_k=stream_last_k,
        )
        meta_stream_metrics = evaluate(
            model,
            test_dataset,
            config=config,
            mode="meta-test",
            device=device,
            use_patch=True,
            use_head=True,
            max_seq_length=config.max_seq_length,
            truncate_len=config.eval_seq_length,
            theta_names=theta_names,
            bitfit_init_state=bitfit_init_state,
            streaming_last_k=stream_last_k,
        )
        log_metrics(
            {
                **flatten_streaming_eval_metrics("test_stream/baseline", baseline_stream_metrics),
                **flatten_streaming_eval_metrics("test_stream/trained_short_no_patch", trained_short_stream_metrics),
                **flatten_streaming_eval_metrics("test_stream/meta_patch", meta_stream_metrics),
                **flatten_streaming_eval_test_aliases("baseline", baseline_stream_metrics),
                **flatten_streaming_eval_test_aliases(
                    "trained_short_no_patch",
                    trained_short_stream_metrics,
                ),
                **flatten_streaming_eval_test_aliases("meta_patch", meta_stream_metrics),
            }
        )

    if run is not None and best_ckpt_path is not None and Path(best_ckpt_path).exists():
        run.save(str(best_ckpt_path))

    if config.save_item_embeddings:
        emb_path = save_item_embeddings(model, train_dataset, config)
        if run is not None and emb_path is not None:
            run.save(str(emb_path))

    metrics_jsonl = None
    metrics_csv = None
    if LOCAL_METRICS_LOGGER is not None:
        metrics_jsonl = LOCAL_METRICS_LOGGER.jsonl_path
        csv_path = LOCAL_METRICS_LOGGER.export_csv()
        if csv_path is not None:
            metrics_csv = csv_path
            logger.info("Local metrics CSV: %s", csv_path)
        LOCAL_METRICS_LOGGER.close()

    save_run_summary(
        config,
        run_name,
        best_metrics,
        baseline_metrics,
        meta_metrics,
        best_ckpt_path,
        metrics_jsonl=metrics_jsonl,
        metrics_csv=metrics_csv,
    )

    wandb.finish()
    logger.info("Training complete!")
