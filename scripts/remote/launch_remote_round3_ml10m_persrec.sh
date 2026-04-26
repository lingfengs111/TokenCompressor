#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

SASREC_PERSREC_GPU="${SASREC_PERSREC_GPU:-0}"
HSTU_PERSREC_GPU="${HSTU_PERSREC_GPU:-1}"

SASREC_WARMSTART_CKPT="${SASREC_WARMSTART_CKPT:-$ROOT/checkpoints/sasrec_loo_sample_softmax/sasrec_ml10m_oldbest_recipe_softmax_sample_softmax_bs512_evalseed2026/sasrec_ml10m_loo202_seq200_dim128_L2_H1_best.pt}"
HSTU_WARMSTART_CKPT="${HSTU_WARMSTART_CKPT:-$ROOT/checkpoints/hstu_loo_sample_softmax/hstu_true_mh_ml10m_sm1_sampledsoftmax_backbone_20260419/hstu_ml10m_loo202_seq202_dim128_L4_H4_best.pt}"

sync_branch_if_requested
ensure_ready
require_file "$SASREC_WARMSTART_CKPT"
require_file "$HSTU_WARMSTART_CKPT"
cd "$ROOT"

launch_job \
  "persrec_sasrec_ml10m_warmstart_legacyloo_remote" \
  "$LOG_DIR/persrec_sasrec_ml10m_warmstart_legacyloo_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${SASREC_PERSREC_GPU}" \
    --backbone sasrec \
    --max_seq_length 200 \
    --hidden_units 128 \
    --num_blocks 2 \
    --num_heads 1 \
    --dropout_rate 0.1 \
    --right_align_positions true \
    --sasrec_attention_norm softmax \
    --use_flash_attention true \
    --persrec_enable true \
    --persrec_num_tokens 8 \
    --persrec_pretrain_len 180 \
    --persrec_recent_len 20 \
    --persrec_eval_use_full_seq true \
    --persrec_train_mode full \
    --eval_seq_length 20 \
    --eval_protocol legacy_loo \
    --last_k_eval_test 0 \
    --streaming_eval_last_k 0 \
    --batch_size 128 \
    --num_epochs 50 \
    --max_learning_rate 2.0360441936032465e-05 \
    --min_learning_rate 1e-06 \
    --scheduler_type cosine \
    --checkpoint_dir "$ROOT/checkpoints/persrec_sasrec_ml10m_warmstart_legacyloo_remote" \
    --pretrained_ckpt_path "$SASREC_WARMSTART_CKPT"

launch_job \
  "persrec_hstu_ml10m_warmstart_legacyloo_remote" \
  "$LOG_DIR/persrec_hstu_ml10m_warmstart_legacyloo_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${HSTU_PERSREC_GPU}" \
    --backbone hstu \
    --max_seq_length 202 \
    --hidden_units 128 \
    --num_blocks 4 \
    --num_heads 4 \
    --dropout_rate 0.2 \
    --hstu_linear_dim 32 \
    --hstu_attention_dim 32 \
    --hstu_attn_dropout 0.0 \
    --hstu_normalization softmax1_rel_bias \
    --persrec_enable true \
    --persrec_num_tokens 8 \
    --persrec_pretrain_len 182 \
    --persrec_recent_len 20 \
    --persrec_eval_use_full_seq true \
    --persrec_train_mode full \
    --eval_seq_length 20 \
    --eval_protocol legacy_loo \
    --last_k_eval_test 0 \
    --streaming_eval_last_k 0 \
    --batch_size 64 \
    --num_epochs 50 \
    --max_learning_rate 2.0360441936032465e-05 \
    --min_learning_rate 1e-06 \
    --scheduler_type cosine \
    --checkpoint_dir "$ROOT/checkpoints/persrec_hstu_ml10m_warmstart_legacyloo_remote" \
    --pretrained_ckpt_path "$HSTU_WARMSTART_CKPT"

echo
echo "launched round3 logs under: $LOG_DIR"
