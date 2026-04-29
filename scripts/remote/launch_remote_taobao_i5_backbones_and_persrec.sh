#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_remote.sh"

SASREC_SOFTMAX_GPU="${SASREC_SOFTMAX_GPU:-0}"
HSTU_SOFTMAX_GPU="${HSTU_SOFTMAX_GPU:-1}"
PERSREC_SASREC_OURS_GPU="${PERSREC_SASREC_OURS_GPU:-2}"
PERSREC_HSTU_OURS_GPU="${PERSREC_HSTU_OURS_GPU:-3}"
PERSREC_SASREC_ORIG_GPU="${PERSREC_SASREC_ORIG_GPU:-$SASREC_SOFTMAX_GPU}"

BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/temp_persrec_taobao_i5_bundle_20260428}"
SASREC_OURS_CKPT="${SASREC_OURS_CKPT:-$BUNDLE_DIR/sasrec_taobao_i5_ours_sm1_sampledsoftmax_best.pt}"
HSTU_OURS_CKPT="${HSTU_OURS_CKPT:-$BUNDLE_DIR/hstu_taobao_i5_ours_true_mh_sm1_best.pt}"
SASREC_ORIG_CKPT="${SASREC_ORIG_CKPT:-$BUNDLE_DIR/sasrec_taobao_i5_original_dotbce_best.pt}"

SASREC_OURS_ATTN_NORM="${SASREC_OURS_ATTN_NORM:-softmax1}"
HSTU_OURS_NORMALIZATION="${HSTU_OURS_NORMALIZATION:-softmax1_rel_bias}"

QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/queue_remote_taobao_i5_backbones_and_persrec.log}"
QUEUE_NAME="queue_persrec_sasrec_taobao_i5_original_dotbce_cosnce_remote"
QUEUE_PIDFILE="$LOG_DIR/sasrec_taobao_i5_softmax_sampledsoftmax_backbone_remote.pid"

sync_branch_if_requested
ensure_ready
require_file "$SASREC_OURS_CKPT"
require_file "$HSTU_OURS_CKPT"
require_file "$SASREC_ORIG_CKPT"
cd "$ROOT"

launch_job \
  "sasrec_taobao_i5_softmax_sampledsoftmax_backbone_remote" \
  "$LOG_DIR/sasrec_taobao_i5_softmax_sampledsoftmax_backbone_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_sasrec_taobao_sample_softmax.py" \
    --dataset taobao_loo202_i5 \
    --device "cuda:${SASREC_SOFTMAX_GPU}" \
    --max-seq-length 200 \
    --sasrec-attention-norm softmax \
    --batch-size 128 \
    --num-epochs 200 \
    --hidden-units 128 \
    --num-blocks 2 \
    --num-heads 1 \
    --dropout-rate 0.1 \
    --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --weight-decay 0 \
    --grad-clip 0 \
    --num-negatives 128 \
    --sampled-softmax-chunk-size 1024 \
    --steps-per-train-log 100 \
    --steps-per-val-log 900 \
    --early-stop-patience 0 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --eval-sample-size 1000 \
    --selection-metric ndcg@10 \
    --temperature 0.07 \
    --user-embedding-norm l2_norm \
    --scheduler-type cosine \
    --warmup-steps 0 \
    --warmup-start-lr 5e-7 \
    --disable-gradient-checkpointing \
    --run-tag sasrec_taobao_i5_oldbest_recipe_softmax_sample_softmax_remote

launch_job \
  "hstu_taobao_i5_true_mh_softmax_sampledsoftmax_backbone_remote" \
  "$LOG_DIR/hstu_taobao_i5_true_mh_softmax_sampledsoftmax_backbone_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/run_hstu_taobao_sample_softmax.py" \
    --dataset taobao_loo202_i5 \
    --device "cuda:${HSTU_SOFTMAX_GPU}" \
    --max-seq-length 202 \
    --backbone-name hstu \
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
    --num-negatives 128 \
    --sampled-softmax-chunk-size 1024 \
    --steps-per-train-log 20 \
    --steps-per-val-log 200 \
    --early-stop-patience 16 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --eval-sample-size 1000 \
    --selection-metric hr@10 \
    --temperature 0.07 \
    --user-embedding-norm l2_norm \
    --scheduler-type cosine \
    --warmup-steps 0 \
    --warmup-start-lr 5e-7 \
    --disable-gradient-checkpointing \
    --hstu-linear-dim 32 \
    --hstu-attention-dim 32 \
    --hstu-attn-dropout 0.0 \
    --hstu-normalization softmax_rel_bias \
    --run-tag hstu_taobao_i5_true_mh_softmax_sampledsoftmax_backbone_remote

launch_job \
  "persrec_sasrec_taobao_i5_ours_cosnce_remote" \
  "$LOG_DIR/persrec_sasrec_taobao_i5_ours_cosnce_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" \
    --dataset taobao_loo202_i5 \
    --device "cuda:${PERSREC_SASREC_OURS_GPU}" \
    --backbone sasrec \
    --max_seq_length 200 \
    --hidden_units 128 \
    --num_blocks 2 \
    --num_heads 1 \
    --dropout_rate 0.1 \
    --right_align_positions true \
    --sasrec_attention_norm "$SASREC_OURS_ATTN_NORM" \
    --use_flash_attention true \
    --use_gradient_checkpointing false \
    --persrec_enable true \
    --persrec_num_tokens 8 \
    --persrec_pretrain_len 180 \
    --persrec_recent_len 20 \
    --persrec_loss nce \
    --persrec_num_negatives 4 \
    --persrec_nce_thres 0.99 \
    --persrec_nce_temperature 0.05 \
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
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --eval_sample_size 1000 \
    --steps_per_train_log 100 \
    --steps_per_val_log 1000 \
    --early_stop_patience 5 \
    --infer_ckpt_config false \
    --checkpoint_dir "$ROOT/checkpoints/persrec_sasrec_taobao_i5_ours_cosnce_remote" \
    --pretrained_ckpt_path "$SASREC_OURS_CKPT"

launch_job \
  "persrec_hstu_taobao_i5_ours_cosnce_remote" \
  "$LOG_DIR/persrec_hstu_taobao_i5_ours_cosnce_remote.log" \
  env WANDB_MODE="$WANDB_MODE" "$PYTHON_BIN" "$ROOT/id_patch/train_Persrec.py" \
    --dataset taobao_loo202_i5 \
    --device "cuda:${PERSREC_HSTU_OURS_GPU}" \
    --backbone hstu \
    --max_seq_length 202 \
    --hidden_units 128 \
    --num_blocks 4 \
    --num_heads 4 \
    --dropout_rate 0.2 \
    --right_align_positions true \
    --hstu_linear_dim 32 \
    --hstu_attention_dim 32 \
    --hstu_linear_activation silu \
    --hstu_attn_dropout 0.0 \
    --hstu_enable_relative_attention_bias true \
    --hstu_normalization "$HSTU_OURS_NORMALIZATION" \
    --hstu_concat_ua false \
    --hstu_epsilon 1e-6 \
    --use_gradient_checkpointing true \
    --persrec_enable true \
    --persrec_num_tokens 8 \
    --persrec_pretrain_len 182 \
    --persrec_recent_len 20 \
    --persrec_loss nce \
    --persrec_num_negatives 4 \
    --persrec_nce_thres 0.99 \
    --persrec_nce_temperature 0.05 \
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
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    --eval_sample_size 1000 \
    --steps_per_train_log 100 \
    --steps_per_val_log 1000 \
    --early_stop_patience 5 \
    --infer_ckpt_config false \
    --checkpoint_dir "$ROOT/checkpoints/persrec_hstu_taobao_i5_ours_cosnce_remote" \
    --pretrained_ckpt_path "$HSTU_OURS_CKPT"

launch_job \
  "$QUEUE_NAME" \
  "$LOG_DIR/${QUEUE_NAME}.log" \
  bash -lc "
set -euo pipefail
timestamp() { date '+%F %T'; }
log() { echo \"[\$(timestamp)] \$*\" | tee -a '$QUEUE_LOG'; }

until [ -f '$QUEUE_PIDFILE' ]; do
  log 'waiting for pidfile $(basename "$QUEUE_PIDFILE")'
  sleep 60
done

pid=\"\$(cat '$QUEUE_PIDFILE' 2>/dev/null || true)\"
if [ -z \"\$pid\" ]; then
  log 'empty pidfile for sasrec softmax backbone'
  exit 1
fi

while kill -0 \"\$pid\" 2>/dev/null; do
  log 'waiting for sasrec softmax backbone pid='\"\$pid\"
  sleep 120
done

log 'launch persrec_sasrec_taobao_i5_original_dotbce_cosnce_remote on GPU${PERSREC_SASREC_ORIG_GPU}'
env WANDB_MODE='$WANDB_MODE' '$PYTHON_BIN' '$ROOT/id_patch/train_Persrec.py' \
  --dataset taobao_loo202_i5 \
  --device 'cuda:${PERSREC_SASREC_ORIG_GPU}' \
  --backbone sasrec \
  --max_seq_length 202 \
  --hidden_units 128 \
  --num_blocks 2 \
  --num_heads 1 \
  --dropout_rate 0.1 \
  --right_align_positions true \
  --sasrec_attention_norm softmax \
  --use_flash_attention true \
  --use_gradient_checkpointing false \
  --persrec_enable true \
  --persrec_num_tokens 8 \
  --persrec_pretrain_len 182 \
  --persrec_recent_len 20 \
  --persrec_loss nce \
  --persrec_num_negatives 4 \
  --persrec_nce_thres 0.99 \
  --persrec_nce_temperature 0.05 \
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
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --eval_sample_size 1000 \
  --steps_per_train_log 100 \
  --steps_per_val_log 1000 \
  --early_stop_patience 5 \
  --infer_ckpt_config false \
  --checkpoint_dir '$ROOT/checkpoints/persrec_sasrec_taobao_i5_original_dotbce_cosnce_remote' \
  --pretrained_ckpt_path '$SASREC_ORIG_CKPT' \
  > '$LOG_DIR/persrec_sasrec_taobao_i5_original_dotbce_cosnce_remote.log' 2>&1
"

echo
echo "launched taobao_i5 remote backbone/PersRec runs under: $LOG_DIR"
