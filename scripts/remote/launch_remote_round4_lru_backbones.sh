#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

LRU_ML10M_FULL_GPU="${LRU_ML10M_FULL_GPU:-0}"
LRU_ML10M_SHORT_GPU="${LRU_ML10M_SHORT_GPU:-1}"
LRU_XLONG_FULL_GPU="${LRU_XLONG_FULL_GPU:-2}"
LRU_XLONG_SHORT_GPU="${LRU_XLONG_SHORT_GPU:-3}"

sync_branch_if_requested
ensure_ready
cd "$ROOT"

launch_job \
  "lru_ml10m_full_remote" \
  "$LOG_DIR/lru_ml10m_full_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_lru_taobao_standard.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${LRU_ML10M_FULL_GPU}" \
    --max-seq-length 200 \
    --batch-size 512 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.2 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 20 \
    --steps-per-val-log 220 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine \
    --wandb-mode "$WANDB_MODE" \
    --run-tag lru_ml10m_full_remote

launch_job \
  "lru_ml10m_short_remote" \
  "$LOG_DIR/lru_ml10m_short_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_lru_taobao_standard.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${LRU_ML10M_SHORT_GPU}" \
    --max-seq-length 20 \
    --batch-size 512 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.2 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 20 \
    --steps-per-val-log 220 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine \
    --wandb-mode "$WANDB_MODE" \
    --run-tag lru_ml10m_short_remote

launch_job \
  "lru_xlong_full_remote" \
  "$LOG_DIR/lru_xlong_full_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_lru_taobao_standard.py" \
    --dataset xlong_loo402 \
    --device "cuda:${LRU_XLONG_FULL_GPU}" \
    --max-seq-length 402 \
    --batch-size 128 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.2 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 100 \
    --steps-per-val-log 2000 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine \
    --wandb-mode "$WANDB_MODE" \
    --run-tag lru_xlong_full_remote

launch_job \
  "lru_xlong_short_remote" \
  "$LOG_DIR/lru_xlong_short_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_lru_taobao_standard.py" \
    --dataset xlong_loo402 \
    --device "cuda:${LRU_XLONG_SHORT_GPU}" \
    --max-seq-length 20 \
    --batch-size 256 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.2 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 100 \
    --steps-per-val-log 2000 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine \
    --wandb-mode "$WANDB_MODE" \
    --run-tag lru_xlong_short_remote

echo
echo "launched round4 logs under: $LOG_DIR"
