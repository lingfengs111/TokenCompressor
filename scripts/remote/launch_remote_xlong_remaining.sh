#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

SASREC_ORIG_GPU="${SASREC_ORIG_GPU:-0}"
HSTU_ORIG_GPU="${HSTU_ORIG_GPU:-1}"
XLONG_OVERFLOW_GPU="${XLONG_OVERFLOW_GPU:-2}"
HSTU_PERSREC_GPU="${HSTU_PERSREC_GPU:-3}"
XLONG_SASREC_WARMSTART_CKPT="${XLONG_SASREC_WARMSTART_CKPT:-$ROOT/checkpoints/sasrec_loo_sample_softmax/sasrec_xlong402_softmax_sampledsoftmax_backbone_20260418/sasrec_xlong_loo402_seq400_dim128_L2_H1_best.pt}"
XLONG_HSTU_WARMSTART_CKPT="${XLONG_HSTU_WARMSTART_CKPT:-$ROOT/checkpoints/hstu_loo_sample_softmax/hstu_true_mh_xlong402_sm1_sampledsoftmax_backbone_20260419/hstu_xlong_loo402_seq402_dim128_L4_H4_best.pt}"

sync_branch_if_requested
ensure_ready
cd "$ROOT"

launch_job \
  "queue_xlong_sasrec_original_remote" \
  "$LOG_DIR/queue_xlong_sasrec_original_remote.log" \
  env WANDB_MODE="$WANDB_MODE" ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" GPU_ID="$SASREC_ORIG_GPU" \
  bash -lc '
set -euo pipefail
run_cmd() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "[$(date "+%F %T")] start $name" | tee -a "$LOG_FILE"
  if "$@" >"$log_file" 2>&1; then
    echo "[$(date "+%F %T")] done  $name" | tee -a "$LOG_FILE"
    return 0
  fi
  local status=$?
  echo "[$(date "+%F %T")] fail  $name exit=$status log=$log_file" | tee -a "$LOG_FILE"
  return $status
}
LOG_FILE="$ROOT/tmp_logs_remote/queue_xlong_sasrec_original_remote.status.log"
run_cmd "sasrec_xlong402_original_full_dotbce_remote" "$ROOT/tmp_logs_remote/sasrec_xlong402_original_full_dotbce_remote.log" \
  "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_standard.py" \
    --dataset xlong_loo402 --device "cuda:${GPU_ID}" --max-seq-length 400 --batch-size 64 --num-epochs 200 \
    --hidden-units 128 --num-blocks 2 --num-heads 1 --dropout-rate 0.1 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 \
    --early-stop-patience 8 --eval-sample-size 1000 --scheduler-type cosine --sasrec-attention-norm softmax \
    --run-tag sasrec_xlong402_original_full_dotbce_remote
run_cmd "sasrec_xlong402_original_short_dotbce_remote" "$ROOT/tmp_logs_remote/sasrec_xlong402_original_short_dotbce_remote.log" \
  "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_standard.py" \
    --dataset xlong_loo402 --device "cuda:${GPU_ID}" --max-seq-length 20 --batch-size 256 --num-epochs 200 \
    --hidden-units 128 --num-blocks 2 --num-heads 1 --dropout-rate 0.1 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 \
    --early-stop-patience 8 --eval-sample-size 1000 --scheduler-type cosine --sasrec-attention-norm softmax \
    --run-tag sasrec_xlong402_original_short_dotbce_remote
run_cmd "sasrec_xlong402_original_prefix_p10_remote" "$ROOT/tmp_logs_remote/sasrec_xlong402_original_prefix_p10_remote.log" \
  "$PYTHON_BIN" "$ROOT/id_patch/train_patch_first_order.py" \
    --dataset xlong_loo402 --device "cuda:${GPU_ID}" --backbone sasrec --checkpoint_mode full --seed 2026 \
    --num_epochs 200 --eval_after_train false --eval_sample_size 1000 --val_eval_every_epochs 4 \
    --num_workers 1 --pin_memory true --persistent_workers true --batch_size 32 --short_seq_length 20 \
    --short_eval_length 20 --prefix_len 10 --prefix_source head --patch_after_prefix true \
    --prefix_tail_positions true --short_drop_prefix false --use_gating true --gating_hidden_dim 64 \
    --gating_temperature 0.5 --gating_noise_std 0.01 --patch_routing learned --input_emb_lora_alpha 8.0 \
    --train_input_emb_lora true --attn_lora_alpha 8.0 --train_attn_lora true --distill_temperature 1.0 \
    --distill_loss_mode decay --distill_loss_decay 0.9 --distill_neg_samples 1 --distill_tail_weight 1.0 \
    --distill_mid_samples 0 --num_patches 16 --patch_len 4 --input_emb_lora_rank 4 --attn_lora_rank 8 \
    --attn_lora_blocks all --distill_type kl --distill_mid_weight 1.0 --student_gt_weight 1.0 \
    --distill_mid_rel_weight 1.0 --distill_lr 0.00010612713466290138 \
    --pretrained_ckpt_path "$ROOT/checkpoints/sasrec_loo_standard/sasrec_xlong402_original_full_dotbce_remote/sasrec_xlong_loo402_seq400_dim128_L2_H1_best.pt" \
    --checkpoint_dir "$ROOT/checkpoints/sasrec_patch_prefix_xlong402_original_dotbce_p10_remote" \
    --sasrec_attention_norm softmax --enable_projection_head true --train_head true
'

launch_job \
  "queue_xlong_hstu_original_remote" \
  "$LOG_DIR/queue_xlong_hstu_original_remote.log" \
  env WANDB_MODE="$WANDB_MODE" ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" GPU_ID="$HSTU_ORIG_GPU" \
  bash -lc '
set -euo pipefail
run_cmd() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "[$(date "+%F %T")] start $name" | tee -a "$LOG_FILE"
  if "$@" >"$log_file" 2>&1; then
    echo "[$(date "+%F %T")] done  $name" | tee -a "$LOG_FILE"
    return 0
  fi
  local status=$?
  echo "[$(date "+%F %T")] fail  $name exit=$status log=$log_file" | tee -a "$LOG_FILE"
  return $status
}
LOG_FILE="$ROOT/tmp_logs_remote/queue_xlong_hstu_original_remote.status.log"
run_cmd "hstu_xlong402_original_full_dotbce_remote" "$ROOT/tmp_logs_remote/hstu_xlong402_original_full_dotbce_remote.log" \
  "$PYTHON_BIN" "$ROOT/run_hstu_taobao_standard.py" \
    --dataset xlong_loo402 --device "cuda:${GPU_ID}" --batch-size 32 --num-epochs 200 --hidden-units 128 \
    --num-blocks 4 --num-heads 4 --dropout-rate 0.2 --max-learning-rate 1e-3 --min-learning-rate 1e-5 \
    --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 --early-stop-patience 8 \
    --eval-sample-size 1000 --scheduler-type cosine_with_warmup --warmup-steps 100 --hstu-linear-dim 32 \
    --hstu-attention-dim 32 --hstu-attn-dropout 0.0 --hstu-normalization softmax1_rel_bias \
    --run-tag hstu_xlong402_original_full_dotbce_remote
run_cmd "hstu_xlong402_original_short_dotbce_remote" "$ROOT/tmp_logs_remote/hstu_xlong402_original_short_dotbce_remote.log" \
  "$PYTHON_BIN" "$ROOT/run_hstu_taobao_standard.py" \
    --dataset xlong_loo402 --device "cuda:${GPU_ID}" --max-seq-length 20 --batch-size 128 --num-epochs 200 \
    --hidden-units 128 --num-blocks 4 --num-heads 4 --dropout-rate 0.2 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 \
    --early-stop-patience 8 --eval-sample-size 1000 --scheduler-type cosine_with_warmup --warmup-steps 100 \
    --hstu-linear-dim 32 --hstu-attention-dim 32 --hstu-attn-dropout 0.0 --hstu-normalization softmax1_rel_bias \
    --run-tag hstu_xlong402_original_short_dotbce_remote
run_cmd "hstu_xlong402_original_prefix_p10_remote" "$ROOT/tmp_logs_remote/hstu_xlong402_original_prefix_p10_remote.log" \
  "$PYTHON_BIN" "$ROOT/id_patch/train_patch_first_order.py" \
    --dataset xlong_loo402 --backbone hstu --device "cuda:${GPU_ID}" \
    --pretrained_ckpt_path "$ROOT/checkpoints/hstu_loo_standard/hstu_xlong402_original_full_dotbce_remote/hstu_xlong_loo402_seq402_dim128_L4_H4_best.pt" \
    --checkpoint_dir "$ROOT/checkpoints/hstu_patch_prefix_xlong402_original_dotbce_p10_remote" \
    --batch_size 16 --val_batch_size 16 --num_epochs 120 --seed 2026 --use_gradient_checkpointing true \
    --eval_after_train false --eval_sample_size 1000 --val_eval_every_epochs 4 --early_stop_patience 6 \
    --num_workers 1 --pin_memory true --persistent_workers true --short_seq_length 20 --short_eval_length 20 \
    --prefix_len 10 --shared_token_len 0 --patch_after_prefix true --prefix_tail_positions true --num_patches 16 \
    --patch_len 4 --use_gating true --gating_hidden_dim 64 --gating_temperature 0.5 --gating_noise_std 0.01 \
    --patch_routing learned --train_adapter true --input_emb_lora_rank 8 --input_emb_lora_alpha 8.0 \
    --train_input_emb_lora true --attn_lora_rank 4 --attn_lora_alpha 8.0 --attn_lora_blocks all \
    --train_attn_lora true --distill_type kl --distill_temperature 1.0 --distill_loss_mode decay \
    --distill_loss_decay 0.9 --distill_neg_samples 1 --distill_tail_weight 1.0 --distill_mid_weight 1.0 \
    --distill_mid_samples 0 --student_gt_weight 1.0 --distill_mid_rel_weight 1.0 --distill_lr 0.0001 \
    --distill_update_every 20 --distill_grad_clip 1.0 --hstu_normalization softmax1_rel_bias
'

launch_job \
  "queue_xlong_overflow_remote" \
  "$LOG_DIR/queue_xlong_overflow_remote.log" \
  env WANDB_MODE="$WANDB_MODE" ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" GPU_ID="$XLONG_OVERFLOW_GPU" XLONG_SASREC_WARMSTART_CKPT="$XLONG_SASREC_WARMSTART_CKPT" \
  bash -lc '
set -euo pipefail
run_cmd() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "[$(date "+%F %T")] start $name" | tee -a "$LOG_FILE"
  if "$@" >"$log_file" 2>&1; then
    echo "[$(date "+%F %T")] done  $name" | tee -a "$LOG_FILE"
    return 0
  fi
  local status=$?
  echo "[$(date "+%F %T")] fail  $name exit=$status log=$log_file" | tee -a "$LOG_FILE"
  return $status
}
LOG_FILE="$ROOT/tmp_logs_remote/queue_xlong_overflow_remote.status.log"
run_cmd "longer_xlong402_sampledsoftmax_sim_remote" "$ROOT/tmp_logs_remote/longer_xlong402_sampledsoftmax_sim_remote.log" \
  "$PYTHON_BIN" "$ROOT/run_longer_taobao_sample_softmax.py" \
    --dataset xlong_loo402 --device "cuda:${GPU_ID}" --max-seq-length 402 --batch-size 128 --num-epochs 200 \
    --hidden-units 128 --num-blocks 2 --num-heads 1 --dropout-rate 0.1 --longer-global-tokens 4 \
    --longer-merge-size 4 --longer-merge-pool last --longer-inner-num-layers 1 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --num-negatives 128 --sampled-softmax-chunk-size 1024 \
    --steps-per-train-log 100 --steps-per-val-log 2000 --early-stop-patience 8 --num-workers 4 --prefetch-factor 2 \
    --eval-sample-size 1000 --eval-protocol legacy_loo --last-k-eval-test 0 --streaming-eval-last-k 0 \
    --selection-metric ndcg@10 --temperature 0.07 --user-embedding-norm l2_norm --scheduler-type cosine \
    --run-tag longer_xlong402_sampledsoftmax_sim_remote
if [ -f "$XLONG_SASREC_WARMSTART_CKPT" ]; then
  run_cmd "persrec_sasrec_xlong402_warmstart_legacyloo_remote" "$ROOT/tmp_logs_remote/persrec_sasrec_xlong402_warmstart_legacyloo_remote.log" \
    "$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" \
      --dataset xlong_loo402 --device "cuda:${GPU_ID}" --backbone sasrec --max_seq_length 400 --hidden_units 128 \
      --num_blocks 2 --num_heads 1 --dropout_rate 0.1 --right_align_positions true --sasrec_attention_norm softmax \
      --use_flash_attention true --persrec_enable true --persrec_num_tokens 8 --persrec_pretrain_len 380 \
      --persrec_recent_len 20 --persrec_eval_use_full_seq true --persrec_train_mode full --eval_seq_length 20 \
      --eval_protocol legacy_loo --last_k_eval_test 0 --streaming_eval_last_k 0 --batch_size 128 --num_epochs 50 \
      --max_learning_rate 2.0360441936032465e-05 --min_learning_rate 1e-06 --scheduler_type cosine \
      --checkpoint_dir "$ROOT/checkpoints/persrec_sasrec_xlong402_warmstart_legacyloo_remote" \
      --pretrained_ckpt_path "$XLONG_SASREC_WARMSTART_CKPT"
else
  echo "[$(date "+%F %T")] skip persrec_sasrec_xlong402_warmstart_legacyloo_remote missing ckpt=$XLONG_SASREC_WARMSTART_CKPT" | tee -a "$LOG_FILE"
fi
'

launch_job \
  "queue_xlong_hstu_persrec_legacyloo_remote" \
  "$LOG_DIR/queue_xlong_hstu_persrec_legacyloo_remote.log" \
  env WANDB_MODE="$WANDB_MODE" ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" GPU_ID="$HSTU_PERSREC_GPU" XLONG_HSTU_WARMSTART_CKPT="$XLONG_HSTU_WARMSTART_CKPT" \
  bash -lc '
set -euo pipefail
run_cmd() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "[$(date "+%F %T")] start $name" | tee -a "$LOG_FILE"
  if "$@" >"$log_file" 2>&1; then
    echo "[$(date "+%F %T")] done  $name" | tee -a "$LOG_FILE"
    return 0
  fi
  local status=$?
  echo "[$(date "+%F %T")] fail  $name exit=$status log=$log_file" | tee -a "$LOG_FILE"
  return $status
}
LOG_FILE="$ROOT/tmp_logs_remote/queue_xlong_hstu_persrec_legacyloo_remote.status.log"
if [ -f "$XLONG_HSTU_WARMSTART_CKPT" ]; then
  run_cmd "persrec_hstu_xlong402_warmstart_legacyloo_remote" "$ROOT/tmp_logs_remote/persrec_hstu_xlong402_warmstart_legacyloo_remote.log" \
    "$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" \
      --dataset xlong_loo402 --device "cuda:${GPU_ID}" --backbone hstu --max_seq_length 402 --hidden_units 128 \
      --num_blocks 4 --num_heads 4 --dropout_rate 0.2 --hstu_linear_dim 32 --hstu_attention_dim 32 \
      --hstu_attn_dropout 0.0 --hstu_normalization softmax1_rel_bias --persrec_enable true --persrec_num_tokens 8 \
      --persrec_pretrain_len 382 --persrec_recent_len 20 --persrec_eval_use_full_seq true --persrec_train_mode full \
      --eval_seq_length 20 --eval_protocol legacy_loo --last_k_eval_test 0 --streaming_eval_last_k 0 \
      --batch_size 64 --num_epochs 50 --max_learning_rate 2.0360441936032465e-05 --min_learning_rate 1e-06 \
      --scheduler_type cosine --checkpoint_dir "$ROOT/checkpoints/persrec_hstu_xlong402_warmstart_legacyloo_remote" \
      --pretrained_ckpt_path "$XLONG_HSTU_WARMSTART_CKPT"
else
  echo "[$(date "+%F %T")] skip persrec_hstu_xlong402_warmstart_legacyloo_remote missing ckpt=$XLONG_HSTU_WARMSTART_CKPT" | tee -a "$LOG_FILE"
fi
'

echo
echo "launched xlong remaining logs under: $LOG_DIR"
