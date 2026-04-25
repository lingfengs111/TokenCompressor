#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

SASREC_PREFIX_GPU="${SASREC_PREFIX_GPU:-0}"
HSTU_PREFIX_GPU="${HSTU_PREFIX_GPU:-1}"

SASREC_FULL_TAG="${SASREC_FULL_TAG:-sasrec_ml10m_original_full_dotbce_remote}"
HSTU_FULL_TAG="${HSTU_FULL_TAG:-hstu_ml10m_original_full_dotbce_remote}"

SASREC_CKPT="${SASREC_CKPT:-$ROOT/checkpoints/sasrec_loo_standard/$SASREC_FULL_TAG/sasrec_ml10m_loo202_seq200_dim128_L2_H1_best.pt}"
HSTU_CKPT="${HSTU_CKPT:-$ROOT/checkpoints/hstu_loo_standard/$HSTU_FULL_TAG/hstu_ml10m_loo202_seq202_dim128_L4_H4_best.pt}"

sync_branch_if_requested
ensure_ready
require_file "$SASREC_CKPT"
require_file "$HSTU_CKPT"
cd "$ROOT"

launch_job \
  "sasrec_ml10m_original_prefix_p10_remote" \
  "$LOG_DIR/sasrec_ml10m_original_prefix_p10_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/id_patch/train_patch_first_order.py" \
    --dataset ml10m_loo202 \
    --device "cuda:${SASREC_PREFIX_GPU}" \
    --backbone sasrec \
    --checkpoint_mode full \
    --seed 2026 \
    --num_epochs 120 \
    --eval_after_train false \
    --eval_sample_size 1000 \
    --val_eval_every_epochs 4 \
    --early_stop_patience 6 \
    --num_workers 1 \
    --pin_memory true \
    --persistent_workers true \
    --batch_size 128 \
    --short_seq_length 20 \
    --short_eval_length 20 \
    --prefix_len 10 \
    --prefix_source head \
    --patch_after_prefix true \
    --prefix_tail_positions true \
    --short_drop_prefix false \
    --use_gating true \
    --gating_hidden_dim 64 \
    --gating_temperature 0.5 \
    --gating_noise_std 0.01 \
    --patch_routing learned \
    --input_emb_lora_alpha 8.0 \
    --train_input_emb_lora true \
    --attn_lora_alpha 8.0 \
    --train_attn_lora true \
    --distill_temperature 1.0 \
    --distill_loss_mode decay \
    --distill_loss_decay 0.9 \
    --distill_neg_samples 1 \
    --distill_tail_weight 1.0 \
    --distill_mid_samples 0 \
    --num_patches 16 \
    --patch_len 4 \
    --input_emb_lora_rank 4 \
    --attn_lora_rank 8 \
    --attn_lora_blocks all \
    --distill_type kl \
    --distill_mid_weight 1.0 \
    --student_gt_weight 1.0 \
    --distill_mid_rel_weight 1.0 \
    --distill_lr 0.0001 \
    --pretrained_ckpt_path "$SASREC_CKPT" \
    --checkpoint_dir "$ROOT/checkpoints/sasrec_patch_prefix_ml10m_original_dotbce_p10_remote" \
    --sasrec_attention_norm softmax \
    --enable_projection_head true \
    --train_head true \
    --run_tag sasrec_ml10m_original_prefix_p10_remote

launch_job \
  "hstu_ml10m_original_prefix_p10_remote" \
  "$LOG_DIR/hstu_ml10m_original_prefix_p10_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/id_patch/train_patch_first_order.py" \
    --dataset ml10m_loo202 \
    --backbone hstu \
    --device "cuda:${HSTU_PREFIX_GPU}" \
    --pretrained_ckpt_path "$HSTU_CKPT" \
    --checkpoint_dir "$ROOT/checkpoints/hstu_patch_prefix_ml10m_original_dotbce_p10_remote" \
    --batch_size 32 \
    --val_batch_size 32 \
    --num_epochs 120 \
    --seed 2026 \
    --use_gradient_checkpointing true \
    --eval_after_train false \
    --eval_sample_size 1000 \
    --val_eval_every_epochs 4 \
    --early_stop_patience 6 \
    --num_workers 1 \
    --pin_memory true \
    --persistent_workers true \
    --short_seq_length 20 \
    --short_eval_length 20 \
    --prefix_len 10 \
    --shared_token_len 0 \
    --patch_after_prefix true \
    --prefix_tail_positions true \
    --num_patches 16 \
    --patch_len 4 \
    --use_gating true \
    --gating_hidden_dim 64 \
    --gating_temperature 0.5 \
    --gating_noise_std 0.01 \
    --patch_routing learned \
    --train_adapter true \
    --input_emb_lora_rank 8 \
    --input_emb_lora_alpha 8.0 \
    --train_input_emb_lora true \
    --attn_lora_rank 4 \
    --attn_lora_alpha 8.0 \
    --attn_lora_blocks all \
    --train_attn_lora true \
    --distill_type kl \
    --distill_temperature 1.0 \
    --distill_loss_mode decay \
    --distill_loss_decay 0.9 \
    --distill_neg_samples 1 \
    --distill_tail_weight 1.0 \
    --distill_mid_weight 1.0 \
    --distill_mid_samples 0 \
    --student_gt_weight 1.0 \
    --distill_mid_rel_weight 1.0 \
    --distill_lr 0.0001 \
    --distill_update_every 20 \
    --distill_grad_clip 1.0 \
    --hstu_normalization softmax1_rel_bias \
    --run_tag hstu_ml10m_original_prefix_p10_remote

echo
echo "launched round2 logs under: $LOG_DIR"
