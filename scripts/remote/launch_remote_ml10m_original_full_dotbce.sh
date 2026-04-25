#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/u/lshi8/TokenCompressor}"
PYTHON_BIN="${PYTHON_BIN:-/u/lshi8/miniconda3/envs/py313/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/tmp_logs_remote}"
SASREC_GPU="${SASREC_GPU:-0}"
HSTU_GPU="${HSTU_GPU:-1}"
BRANCH="${BRANCH:-}"

mkdir -p "$LOG_DIR"

if [ -n "$BRANCH" ]; then
  git -C "$ROOT" fetch origin "$BRANCH"
  git -C "$ROOT" checkout "$BRANCH"
  git -C "$ROOT" pull --ff-only origin "$BRANCH"
fi

bash "$ROOT/scripts/remote/check_remote_ready.sh"

echo "[$(date '+%F %T')] launch sasrec_ml10m_original_full_dotbce_remote on cuda:${SASREC_GPU}"
nohup "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_standard.py" \
  --dataset ml10m_loo202 \
  --device "cuda:${SASREC_GPU}" \
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
  --run-tag sasrec_ml10m_original_full_dotbce_remote \
  > "$LOG_DIR/sasrec_ml10m_original_full_dotbce_remote.log" 2>&1 &
echo $! > "$LOG_DIR/sasrec_ml10m_original_full_dotbce_remote.pid"

echo "[$(date '+%F %T')] launch hstu_ml10m_original_full_dotbce_remote on cuda:${HSTU_GPU}"
nohup "$PYTHON_BIN" "$ROOT/run_hstu_taobao_standard.py" \
  --dataset ml10m_loo202 \
  --device "cuda:${HSTU_GPU}" \
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
  --run-tag hstu_ml10m_original_full_dotbce_remote \
  > "$LOG_DIR/hstu_ml10m_original_full_dotbce_remote.log" 2>&1 &
echo $! > "$LOG_DIR/hstu_ml10m_original_full_dotbce_remote.pid"

echo
echo "launched logs:"
echo "  $LOG_DIR/sasrec_ml10m_original_full_dotbce_remote.log"
echo "  $LOG_DIR/hstu_ml10m_original_full_dotbce_remote.log"
