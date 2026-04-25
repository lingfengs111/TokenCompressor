#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

WAIT_PIDFILE="${WAIT_PIDFILE:-$LOG_DIR/lru_ml10m_short_remote.pid}"
LONGER_ML10M_GPU="${LONGER_ML10M_GPU:-1}"
QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/queue_remote_round5_ml10m_longer_after_lru_ml10m_short.log}"

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

log "round5 queue watcher started"
wait_for_pidfile_exit "$WAIT_PIDFILE"
log "launch round5 on GPU${LONGER_ML10M_GPU}"
LONGER_ML10M_GPU="$LONGER_ML10M_GPU" \
  bash "$ROOT/scripts/remote/launch_remote_round5_ml10m_longer.sh" >>"$QUEUE_LOG" 2>&1
log "round5 queue watcher finished"
