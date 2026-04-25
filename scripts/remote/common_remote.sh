#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/u/lshi8/TokenCompressor}"
PYTHON_BIN="${PYTHON_BIN:-/u/lshi8/miniconda3/envs/py313/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/tmp_logs_remote}"
WANDB_MODE="${WANDB_MODE:-online}"
BRANCH="${BRANCH:-}"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%F %T'
}

sync_branch_if_requested() {
  if [ -z "$BRANCH" ]; then
    return 0
  fi
  git -C "$ROOT" fetch origin "$BRANCH"
  git -C "$ROOT" checkout "$BRANCH"
  git -C "$ROOT" pull --ff-only origin "$BRANCH"
}

ensure_ready() {
  bash "$ROOT/scripts/remote/check_remote_ready.sh"
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "[missing] required file: $path" >&2
    exit 1
  fi
}

launch_job() {
  local name="$1"
  local log_file="$2"
  shift 2

  echo "[$(timestamp)] launch $name"
  nohup "$@" >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" >"${log_file%.log}.pid"
  echo "[$(timestamp)] pid=$pid log=$log_file"
}
