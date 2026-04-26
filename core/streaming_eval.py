"""Helpers for protocol-aware sequential evaluation."""

from __future__ import annotations

import math
from typing import Any, Dict, List


LEGACY_LOO_PROTOCOL = "legacy_loo"
HOLDOUT_ANCHOR_PROTOCOL = "holdout_anchor"


def normalize_eval_protocol(value: Any) -> str:
    """Normalize legacy/new evaluation protocol names."""
    if value is None:
        return LEGACY_LOO_PROTOCOL
    norm = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "loo": LEGACY_LOO_PROTOCOL,
        "legacy": LEGACY_LOO_PROTOCOL,
        "legacyloo": LEGACY_LOO_PROTOCOL,
        "legacy_loo": LEGACY_LOO_PROTOCOL,
        "old": LEGACY_LOO_PROTOCOL,
        "leave_last_two_out": LEGACY_LOO_PROTOCOL,
        "holdout": HOLDOUT_ANCHOR_PROTOCOL,
        "anchor": HOLDOUT_ANCHOR_PROTOCOL,
        "holdout_anchor": HOLDOUT_ANCHOR_PROTOCOL,
        "persrec": HOLDOUT_ANCHOR_PROTOCOL,
        "persrec_anchor": HOLDOUT_ANCHOR_PROTOCOL,
    }
    norm = aliases.get(norm, norm)
    if norm not in {LEGACY_LOO_PROTOCOL, HOLDOUT_ANCHOR_PROTOCOL}:
        raise ValueError(f"Unsupported eval_protocol: {value}")
    return norm


def resolve_protocol_holdout_last_k(
    eval_protocol: Any = LEGACY_LOO_PROTOCOL,
    last_k_eval_test: int = 0,
) -> int:
    """Return the train/eval holdout span implied by the protocol."""
    protocol = normalize_eval_protocol(eval_protocol)
    if protocol == LEGACY_LOO_PROTOCOL:
        return 2

    holdout_k = int(last_k_eval_test or 0)
    if holdout_k < 2:
        raise ValueError(
            f"holdout_anchor protocol requires last_k_eval_test >= 2, got {last_k_eval_test}."
        )
    return holdout_k


def resolve_train_cutoff(
    seq_len: int,
    eval_protocol: Any = LEGACY_LOO_PROTOCOL,
    last_k_eval_test: int = 0,
) -> int:
    """Return the exclusive end index of the training-visible prefix."""
    return max(0, int(seq_len) - resolve_protocol_holdout_last_k(eval_protocol, last_k_eval_test))


def resolve_eval_target_positions(
    seq_len: int,
    mode: str,
    streaming_last_k: int = 0,
    min_context_len: int = 1,
    eval_protocol: Any = LEGACY_LOO_PROTOCOL,
    last_k_eval_test: int = 0,
) -> List[int]:
    """Return target indices to evaluate for a sequence."""
    protocol = normalize_eval_protocol(eval_protocol)
    holdout_k = resolve_protocol_holdout_last_k(protocol, last_k_eval_test)

    if mode == "val":
        target_idx = seq_len - holdout_k
        if target_idx < min_context_len:
            return []
        return [target_idx]

    if mode not in {"test", "meta-test"}:
        raise ValueError(f"Unsupported eval mode: {mode}")

    if seq_len - 1 < min_context_len:
        return []

    if int(streaming_last_k or 0) > 1:
        start = max(min_context_len, seq_len - int(streaming_last_k))
        return list(range(start, seq_len))

    if protocol == LEGACY_LOO_PROTOCOL:
        target_idx = seq_len - 1
    else:
        # holdout_anchor keeps the last_k_eval_test items fully hidden from training, then evaluates at the
        # midpoint anchor inside that hidden block. Example: last_k_eval_test=10 -> test target is seq[-5].
        # This is intentionally different from legacy_loo's final-item target.
        target_idx = seq_len - (holdout_k // 2)
    if target_idx < min_context_len:
        return []
    return [target_idx]


def update_rank_metrics(
    per_position: Dict[int, Dict[str, float]],
    rel_from_end: int,
    rank: int,
) -> None:
    bucket = per_position.setdefault(
        int(rel_from_end),
        {
            "ndcg5_sum": 0.0,
            "ndcg10_sum": 0.0,
            "hr10_sum": 0.0,
            "hr20_sum": 0.0,
            # Backward-compatible aliases for older consumers that only know @10.
            "ndcg_sum": 0.0,
            "hr_sum": 0.0,
            "count": 0.0,
        },
    )
    bucket["count"] += 1.0
    if rank <= 5:
        bucket["ndcg5_sum"] += 1.0 / math.log2(rank + 1)
    if rank <= 10:
        ndcg_gain = 1.0 / math.log2(rank + 1)
        bucket["hr10_sum"] += 1.0
        bucket["ndcg10_sum"] += ndcg_gain
        bucket["hr_sum"] += 1.0
        bucket["ndcg_sum"] += ndcg_gain
    if rank <= 20:
        bucket["hr20_sum"] += 1.0


def finalize_eval_metrics(
    ndcg_sum: float,
    hr_sum: float,
    num_examples: int,
    per_position: Dict[int, Dict[str, float]],
    streaming_last_k: int = 0,
) -> Dict[str, Any]:
    total_ndcg5_sum = 0.0
    total_ndcg10_sum = float(ndcg_sum or 0.0)
    total_hr10_sum = float(hr_sum or 0.0)
    total_hr20_sum = 0.0

    if per_position:
        total_ndcg5_sum = sum(float(bucket.get("ndcg5_sum", 0.0) or 0.0) for bucket in per_position.values())
        total_ndcg10_sum = sum(
            float(bucket.get("ndcg10_sum", bucket.get("ndcg_sum", 0.0)) or 0.0)
            for bucket in per_position.values()
        )
        total_hr10_sum = sum(
            float(bucket.get("hr10_sum", bucket.get("hr_sum", 0.0)) or 0.0)
            for bucket in per_position.values()
        )
        total_hr20_sum = sum(float(bucket.get("hr20_sum", 0.0) or 0.0) for bucket in per_position.values())

    metrics: Dict[str, Any] = {
        "ndcg@5": total_ndcg5_sum / num_examples if num_examples > 0 else 0.0,
        "ndcg@10": total_ndcg10_sum / num_examples if num_examples > 0 else 0.0,
        "hr@10": total_hr10_sum / num_examples if num_examples > 0 else 0.0,
        "hr@20": total_hr20_sum / num_examples if num_examples > 0 else 0.0,
        "num_examples": int(num_examples),
        "streaming_last_k": int(streaming_last_k or 0),
    }
    if per_position:
        metrics["per_position"] = {
            int(rel): {
                "ndcg@5": bucket.get("ndcg5_sum", 0.0) / bucket["count"] if bucket["count"] > 0 else 0.0,
                "ndcg@10": bucket.get("ndcg10_sum", bucket.get("ndcg_sum", 0.0)) / bucket["count"]
                if bucket["count"] > 0
                else 0.0,
                "hr@10": bucket.get("hr10_sum", bucket.get("hr_sum", 0.0)) / bucket["count"]
                if bucket["count"] > 0
                else 0.0,
                "hr@20": bucket.get("hr20_sum", 0.0) / bucket["count"] if bucket["count"] > 0 else 0.0,
                "count": int(bucket["count"]),
            }
            for rel, bucket in sorted(per_position.items())
        }
    return metrics


def flatten_streaming_eval_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float | int]:
    streaming_last_k = int(metrics.get("streaming_last_k", 0) or 0)
    if streaming_last_k <= 1:
        return {}

    flat: Dict[str, float | int] = {
        f"{prefix}/ndcg@5": float(metrics.get("ndcg@5", 0.0) or 0.0),
        f"{prefix}/ndcg@10": float(metrics.get("ndcg@10", 0.0) or 0.0),
        f"{prefix}/hr@10": float(metrics.get("hr@10", 0.0) or 0.0),
        f"{prefix}/hr@20": float(metrics.get("hr@20", 0.0) or 0.0),
        f"{prefix}/num_examples": int(metrics.get("num_examples", 0) or 0),
        f"{prefix}/last_k": streaming_last_k,
    }
    per_position = metrics.get("per_position")
    if isinstance(per_position, dict):
        for rel, stats in sorted(per_position.items()):
            if not isinstance(stats, dict):
                continue
            flat[f"{prefix}/last{int(rel)}_ndcg@5"] = float(stats.get("ndcg@5", 0.0) or 0.0)
            flat[f"{prefix}/last{int(rel)}_ndcg@10"] = float(stats.get("ndcg@10", 0.0) or 0.0)
            flat[f"{prefix}/last{int(rel)}_hr@10"] = float(stats.get("hr@10", 0.0) or 0.0)
            flat[f"{prefix}/last{int(rel)}_hr@20"] = float(stats.get("hr@20", 0.0) or 0.0)
            flat[f"{prefix}/last{int(rel)}_count"] = int(stats.get("count", 0) or 0)
    return flat


def flatten_streaming_eval_test_aliases(name: str, metrics: Dict[str, Any]) -> Dict[str, float | int]:
    """Expose rolling metrics under test/* aliases for easier W&B comparison."""
    streaming_last_k = int(metrics.get("streaming_last_k", 0) or 0)
    if streaming_last_k <= 1:
        return {}

    prefix = f"test/rolling_{name}"
    flat: Dict[str, float | int] = {
        f"{prefix}_ndcg@5": float(metrics.get("ndcg@5", 0.0) or 0.0),
        f"{prefix}_ndcg@10": float(metrics.get("ndcg@10", 0.0) or 0.0),
        f"{prefix}_hr@10": float(metrics.get("hr@10", 0.0) or 0.0),
        f"{prefix}_hr@20": float(metrics.get("hr@20", 0.0) or 0.0),
        f"{prefix}_num_examples": int(metrics.get("num_examples", 0) or 0),
        f"{prefix}_last_k": streaming_last_k,
    }
    per_position = metrics.get("per_position")
    if isinstance(per_position, dict):
        for rel, stats in sorted(per_position.items()):
            if not isinstance(stats, dict):
                continue
            flat[f"{prefix}_last{int(rel)}_ndcg@5"] = float(stats.get("ndcg@5", 0.0) or 0.0)
            flat[f"{prefix}_last{int(rel)}_ndcg@10"] = float(stats.get("ndcg@10", 0.0) or 0.0)
            flat[f"{prefix}_last{int(rel)}_hr@10"] = float(stats.get("hr@10", 0.0) or 0.0)
            flat[f"{prefix}_last{int(rel)}_hr@20"] = float(stats.get("hr@20", 0.0) or 0.0)
            flat[f"{prefix}_last{int(rel)}_count"] = int(stats.get("count", 0) or 0)
    return flat
