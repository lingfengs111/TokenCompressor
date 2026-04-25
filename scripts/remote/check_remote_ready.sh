#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/u/lshi8/TokenCompressor}"
PYTHON_BIN="${PYTHON_BIN:-/u/lshi8/miniconda3/envs/py313/bin/python}"

required_files=(
  "$ROOT/run_sasrec_taobao_standard.py"
  "$ROOT/run_hstu_taobao_standard.py"
  "$ROOT/run_lru_taobao_standard.py"
  "$ROOT/train_backbone_standard.py"
  "$ROOT/core/loo_dataset.py"
  "$ROOT/core/mixflow.py"
  "$ROOT/core/streaming_eval.py"
  "$ROOT/backbones/SASRec.py"
  "$ROOT/backbones/HSTU.py"
  "$ROOT/backbones/HSTUOfficialish.py"
  "$ROOT/backbones/HSTUResearchAligned.py"
  "$ROOT/backbones/LONGER.py"
  "$ROOT/backbones/Mamba4Rec.py"
  "$ROOT/backbones/modules.py"
  "$ROOT/backbones/patch.py"
  "$ROOT/id_patch/train_patch_first_order.py"
  "$ROOT/id_patch/train_Persrec.py"
)

required_dirs=(
  "$ROOT/data/ml-10m/loo_202"
  "$ROOT/data/xlong2018/loo_402"
)

echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "HOST=$(hostname)"
echo

missing=0

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[missing] python executable: $PYTHON_BIN"
  missing=1
else
  "$PYTHON_BIN" --version
fi

echo
for path in "${required_files[@]}"; do
  if [ -f "$path" ]; then
    echo "[ok] $path"
  else
    echo "[missing] $path"
    missing=1
  fi
done

echo
for path in "${required_dirs[@]}"; do
  if [ -d "$path" ]; then
    echo "[ok] $path"
  else
    echo "[missing] $path"
    missing=1
  fi
done

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
else
  echo "[missing] nvidia-smi"
  missing=1
fi

echo
if [ "$missing" -ne 0 ]; then
  echo "remote tree is NOT ready"
  exit 1
fi

echo "remote tree is ready"
