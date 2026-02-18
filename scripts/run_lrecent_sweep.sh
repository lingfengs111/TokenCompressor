#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run three L_recent settings in parallel on three GPUs.
# Edit the L_RECENTS or GPUS arrays to change sweep values or GPU mapping.

L_RECENTS=(50 100 150)
GPUS=(0 1 2)

PYTHON="python"
ENTRY="id_mixflow/train_mixflow.py"

for idx in "${!L_RECENTS[@]}"; do
  lr_recent="${L_RECENTS[$idx]}"
  gpu="${GPUS[$idx]:-}"
  if [[ -z "${gpu}" ]]; then
    echo "No GPU specified for index ${idx}; skipping."
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
  ${PYTHON} "${ENTRY}" \
    dataset=xlong2018 \
    dataset.L_recent="${lr_recent}" \
    device="cuda:0" \
    wandb_run_name="lrecent${lr_recent}" \
    "$@" &
done

wait
