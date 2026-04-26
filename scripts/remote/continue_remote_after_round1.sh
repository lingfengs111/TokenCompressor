#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/continue_remote_after_round1.log}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$QUEUE_LOG"
}

wait_for_pidfile_exit() {
  local pidfile="$1"
  until [ -f "$pidfile" ]; do
    log "waiting for pidfile $(basename "$pidfile")"
    sleep 60
  done

  local pid=""
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ -z "$pid" ]; then
    log "empty pidfile: $pidfile"
    exit 1
  fi

  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
  done
  log "finished $(basename "$pidfile") pid=$pid"
}

sync_branch_if_requested
ensure_ready
cd "$ROOT"

SASREC_FULL_CKPT="${SASREC_FULL_CKPT:-$ROOT/checkpoints/sasrec_loo_standard/sasrec_ml10m_original_full_dotbce_remote/sasrec_ml10m_loo202_seq200_dim128_L2_H1_best.pt}"
HSTU_FULL_CKPT="${HSTU_FULL_CKPT:-$ROOT/checkpoints/hstu_loo_standard/hstu_ml10m_original_full_dotbce_remote/hstu_ml10m_loo202_seq202_dim128_L4_H4_best.pt}"
SASREC_WARMSTART_CKPT="${SASREC_WARMSTART_CKPT:-$ROOT/checkpoints/sasrec_loo_sample_softmax/sasrec_ml10m_oldbest_recipe_softmax_sample_softmax_bs512_evalseed2026/sasrec_ml10m_loo202_seq200_dim128_L2_H1_best.pt}"
HSTU_WARMSTART_CKPT="${HSTU_WARMSTART_CKPT:-$ROOT/checkpoints/hstu_loo_sample_softmax/hstu_true_mh_ml10m_sm1_sampledsoftmax_backbone_20260419/hstu_ml10m_loo202_seq202_dim128_L4_H4_best.pt}"

ROUND2_SASREC_PID="$LOG_DIR/sasrec_ml10m_original_prefix_p10_remote.pid"
ROUND2_HSTU_PID="$LOG_DIR/hstu_ml10m_original_prefix_p10_remote.pid"
ROUND3_SASREC_PID="$LOG_DIR/persrec_sasrec_ml10m_warmstart_legacyloo_remote.pid"
ROUND3_HSTU_PID="$LOG_DIR/persrec_hstu_ml10m_warmstart_legacyloo_remote.pid"
ROUND4_LRU_SHORT_PID="$LOG_DIR/lru_ml10m_short_remote.pid"

log "continue_remote_after_round1 started"

require_file "$SASREC_FULL_CKPT"
require_file "$HSTU_FULL_CKPT"

log "launch round2 on GPU0/1"
SASREC_PREFIX_GPU="${SASREC_PREFIX_GPU:-0}" \
HSTU_PREFIX_GPU="${HSTU_PREFIX_GPU:-1}" \
bash "$ROOT/scripts/remote/launch_remote_round2_ml10m_original_prefix.sh" >>"$QUEUE_LOG" 2>&1

if [ -f "$SASREC_WARMSTART_CKPT" ] && [ -f "$HSTU_WARMSTART_CKPT" ]; then
  log "launch round3 on GPU2/3"
  SASREC_PERSREC_GPU="${SASREC_PERSREC_GPU:-2}" \
  HSTU_PERSREC_GPU="${HSTU_PERSREC_GPU:-3}" \
  SASREC_WARMSTART_CKPT="$SASREC_WARMSTART_CKPT" \
  HSTU_WARMSTART_CKPT="$HSTU_WARMSTART_CKPT" \
  bash "$ROOT/scripts/remote/launch_remote_round3_ml10m_persrec.sh" >>"$QUEUE_LOG" 2>&1
else
  log "skip round3: missing sampled-softmax warmstart ckpts"
fi

wait_for_pidfile_exit "$ROUND2_SASREC_PID"
wait_for_pidfile_exit "$ROUND2_HSTU_PID"

if [ -f "$ROUND3_SASREC_PID" ]; then
  wait_for_pidfile_exit "$ROUND3_SASREC_PID"
fi
if [ -f "$ROUND3_HSTU_PID" ]; then
  wait_for_pidfile_exit "$ROUND3_HSTU_PID"
fi

log "launch round4 on GPU0/1/2/3"
bash "$ROOT/scripts/remote/launch_remote_round4_lru_backbones.sh" >>"$QUEUE_LOG" 2>&1

log "queue round5 after lru_ml10m_short_remote finishes"
WAIT_PIDFILE="$ROUND4_LRU_SHORT_PID" \
LONGER_ML10M_GPU="${LONGER_ML10M_GPU:-1}" \
bash "$ROOT/scripts/remote/queue_remote_round5_ml10m_longer_after_lru_ml10m_short.sh" >>"$QUEUE_LOG" 2>&1 &

log "continue_remote_after_round1 finished submitting downstream jobs"
