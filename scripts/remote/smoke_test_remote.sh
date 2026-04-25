#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

SMOKE_GPU="${SMOKE_GPU:-0}"
SMOKE_LOG="${SMOKE_LOG:-$LOG_DIR/smoke_test_remote.log}"
SMOKE_TAG="${SMOKE_TAG:-sasrec_ml10m_remote_smoke_$(date '+%Y%m%d_%H%M%S')}"

sync_branch_if_requested
ensure_ready

cd "$ROOT"

echo "[$(timestamp)] import sanity check"
PYTHONPATH="$ROOT:$ROOT/id_patch:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "train_backbone_standard",
    "run_sasrec_taobao_standard",
    "run_hstu_taobao_standard",
    "run_lru_taobao_standard",
    "id_patch.train_patch_first_order",
    "id_patch.train_Persrec",
]

for module_name in modules:
    importlib.import_module(module_name)
    print(f"[ok] import {module_name}")
PY

echo "[$(timestamp)] argparse sanity check"
"$PYTHON_BIN" "$ROOT/id_patch/train_patch_first_order.py" --help >/dev/null
"$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" --help >/dev/null

echo "[$(timestamp)] tiny end-to-end SASRec smoke run on cuda:${SMOKE_GPU}"
env WANDB_MODE=disabled "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_standard.py" \
  --dataset ml10m_loo202 \
  --device "cuda:${SMOKE_GPU}" \
  --max-seq-length 20 \
  --batch-size 64 \
  --num-epochs 1 \
  --hidden-units 64 \
  --num-blocks 1 \
  --num-heads 1 \
  --dropout-rate 0.1 \
  --max-learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 0 \
  --grad-clip 0 \
  --steps-per-train-log 1 \
  --steps-per-val-log 20 \
  --early-stop-patience 1 \
  --eval-sample-size 100 \
  --wandb-mode disabled \
  --run-tag "$SMOKE_TAG" \
  >"$SMOKE_LOG" 2>&1

echo "[$(timestamp)] smoke test complete"
echo "log: $SMOKE_LOG"
tail -n 20 "$SMOKE_LOG"
