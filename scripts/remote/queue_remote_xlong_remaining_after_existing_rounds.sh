#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/queue_remote_xlong_remaining_after_existing_rounds.log}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$QUEUE_LOG"
}

pidfiles=(
  "$LOG_DIR/sasrec_ml10m_original_prefix_p10_remote.pid"
  "$LOG_DIR/hstu_ml10m_original_prefix_p10_remote.pid"
  "$LOG_DIR/persrec_sasrec_ml10m_warmstart_legacyloo_remote.pid"
  "$LOG_DIR/persrec_hstu_ml10m_warmstart_legacyloo_remote.pid"
  "$LOG_DIR/lru_ml10m_full_remote.pid"
  "$LOG_DIR/lru_ml10m_short_remote.pid"
  "$LOG_DIR/lru_xlong_full_remote.pid"
  "$LOG_DIR/lru_xlong_short_remote.pid"
  "$LOG_DIR/longer_ml10m_sampledsoftmax_sim_remote.pid"
)

sync_branch_if_requested
ensure_ready
cd "$ROOT"

log "xlong remaining watcher started"

while true; do
  alive=0
  for pidfile in "${pidfiles[@]}"; do
    if [ ! -f "$pidfile" ]; then
      continue
    fi
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      alive=1
      log "waiting for $(basename "$pidfile") pid=$pid"
      break
    fi
  done
  if [ "$alive" -eq 0 ]; then
    break
  fi
  sleep 180
done

log "launch xlong remaining on all 4 GPUs"
bash "$ROOT/scripts/remote/launch_remote_xlong_remaining.sh" >>"$QUEUE_LOG" 2>&1
log "xlong remaining watcher finished"
