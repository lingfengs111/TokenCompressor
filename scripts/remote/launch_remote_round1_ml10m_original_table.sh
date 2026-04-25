#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

SASREC_FULL_GPU="${SASREC_FULL_GPU:-0}"
HSTU_FULL_GPU="${HSTU_FULL_GPU:-1}"
SASREC_SHORT_GPU="${SASREC_SHORT_GPU:-2}"
HSTU_SHORT_GPU="${HSTU_SHORT_GPU:-3}"

sync_branch_if_requested
ensure_ready
cd "$ROOT"

launch_job \
  "sasrec_ml10m_original_full_dotbce_remote" \
  "$LOG_DIR/sasrec_ml10m_original_full_dotbce_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_standard.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${SASREC_FULL_GPU}" \
    --max-seq-length 200 \
    --batch-size 512 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.1 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 20 \
    --steps-per-val-log 220 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine \
    --sasrec-attention-norm softmax \
    --wandb-mode "$WANDB_MODE" \
    --run-tag sasrec_ml10m_original_full_dotbce_remote

launch_job \
  "hstu_ml10m_original_full_dotbce_remote" \
  "$LOG_DIR/hstu_ml10m_original_full_dotbce_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_hstu_taobao_standard.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${HSTU_FULL_GPU}" \
    --max-seq-length 202 \
    --batch-size 128 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 4 \
    --num-heads 4 \
    --dropout-rate 0.2 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 100 \
    --steps-per-val-log 900 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine_with_warmup \
    --warmup-steps 100 \
    --hstu-linear-dim 32 \
    --hstu-attention-dim 32 \
    --hstu-attn-dropout 0.0 \
    --hstu-normalization softmax1_rel_bias \
    --wandb-mode "$WANDB_MODE" \
    --run-tag hstu_ml10m_original_full_dotbce_remote

launch_job \
  "sasrec_ml10m_original_short_dotbce_remote" \
  "$LOG_DIR/sasrec_ml10m_original_short_dotbce_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_standard.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${SASREC_SHORT_GPU}" \
    --max-seq-length 20 \
    --batch-size 512 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.1 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 20 \
    --steps-per-val-log 220 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine \
    --sasrec-attention-norm softmax \
    --wandb-mode "$WANDB_MODE" \
    --run-tag sasrec_ml10m_original_short_dotbce_remote

launch_job \
  "hstu_ml10m_original_short_dotbce_remote" \
  "$LOG_DIR/hstu_ml10m_original_short_dotbce_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_hstu_taobao_standard.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${HSTU_SHORT_GPU}" \
    --max-seq-length 20 \
    --batch-size 128 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 4 \
    --num-heads 4 \
    --dropout-rate 0.2 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --steps-per-train-log 100 \
    --steps-per-val-log 900 \
    --early-stop-patience 8 \
    --eval-sample-size 1000 \
    --scheduler-type cosine_with_warmup \
    --warmup-steps 100 \
    --hstu-linear-dim 32 \
    --hstu-attention-dim 32 \
    --hstu-attn-dropout 0.0 \
    --hstu-normalization softmax1_rel_bias \
    --wandb-mode "$WANDB_MODE" \
    --run-tag hstu_ml10m_original_short_dotbce_remote

echo
echo "launched round1 logs under: $LOG_DIR"
