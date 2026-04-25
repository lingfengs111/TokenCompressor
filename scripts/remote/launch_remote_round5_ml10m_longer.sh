#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

LONGER_ML10M_GPU="${LONGER_ML10M_GPU:-1}"

sync_branch_if_requested
ensure_ready
cd "$ROOT"

launch_job \
  "longer_ml10m_sampledsoftmax_sim_remote" \
  "$LOG_DIR/longer_ml10m_sampledsoftmax_sim_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_longer_taobao_sample_softmax.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${LONGER_ML10M_GPU}" \
    --max-seq-length 202 \
    --batch-size 512 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.1 \
    --longer-global-tokens 4 \
    --longer-merge-size 4 \
    --longer-merge-pool last \
    --longer-inner-num-layers 1 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --num-negatives 128 \
    --sampled-softmax-chunk-size 1024 \
    --steps-per-train-log 20 \
    --steps-per-val-log 220 \
    --early-stop-patience 8 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --eval-sample-size 1000 \
    --eval-protocol legacy_loo \
    --last-k-eval-test 10 \
    --streaming-eval-last-k 0 \
    --selection-metric ndcg@10 \
    --temperature 0.07 \
    --user-embedding-norm l2_norm \
    --scheduler-type cosine \
    --wandb-mode "$WANDB_MODE" \
    --run-tag longer_ml10m_sampledsoftmax_sim_remote

echo
echo "launched round5 logs under: $LOG_DIR"
