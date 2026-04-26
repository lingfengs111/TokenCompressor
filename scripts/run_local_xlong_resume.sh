#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lingfengs111/codes/soft_patch_training"
LOG_DIR="$ROOT/tmp_logs"
CONDA_SH="/home/lingfengs111/miniconda3/etc/profile.d/conda.sh"

mkdir -p "$LOG_DIR"

launch_queue() {
  local name="$1"
  local body="$2"
  local log_file="$LOG_DIR/${name}.launcher.log"
  nohup bash -lc "set -euo pipefail; source '$CONDA_SH'; conda activate py313; cd '$ROOT'; $body" \
    >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" >"$LOG_DIR/${name}.pid"
  echo "[$(date '+%F %T')] launched $name pid=$pid log=$log_file"
}

run_cmd_fn='
run_cmd() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "[$(date '\''+%F %T'\'')] start $name" | tee -a "$QUEUE_LOG"
  if "$@" >"$log_file" 2>&1; then
    echo "[$(date '\''+%F %T'\'')] done  $name" | tee -a "$QUEUE_LOG"
    return 0
  fi
  local status=$?
  echo "[$(date '\''+%F %T'\'')] fail  $name exit=$status log=$log_file" | tee -a "$QUEUE_LOG"
  return $status
}
'

gpu0_body='
QUEUE_LOG="$LOG_DIR/queue_gpu0_xlong_table_sasrec_original_rerun.log"
GPU_ID=0
'"$run_cmd_fn"'
run_cmd "sasrec_xlong402_original_full_dotbce_rerun" "$LOG_DIR/sasrec_xlong402_original_full_dotbce_rerun.log" \
  python run_sasrec_taobao_standard.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --max-seq-length 400 --batch-size 64 --num-epochs 200 \
    --hidden-units 128 --num-blocks 2 --num-heads 1 --dropout-rate 0.1 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 \
    --early-stop-patience 8 --eval-sample-size 1000 --scheduler-type cosine --sasrec-attention-norm softmax \
    --run-tag sasrec_xlong402_original_full_dotbce_rerun
run_cmd "sasrec_xlong402_original_short_dotbce_rerun" "$LOG_DIR/sasrec_xlong402_original_short_dotbce_rerun.log" \
  python run_sasrec_taobao_standard.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --max-seq-length 20 --batch-size 256 --num-epochs 200 \
    --hidden-units 128 --num-blocks 2 --num-heads 1 --dropout-rate 0.1 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 \
    --early-stop-patience 8 --eval-sample-size 1000 --scheduler-type cosine --sasrec-attention-norm softmax \
    --run-tag sasrec_xlong402_original_short_dotbce_rerun
run_cmd "sasrec_xlong402_original_prefix_p10_rerun" "$LOG_DIR/sasrec_xlong402_original_prefix_p10_rerun.log" \
  python id_patch/train_patch_first_order.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --backbone sasrec --checkpoint_mode full --seed 2026 \
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
    --pretrained_ckpt_path "$ROOT/checkpoints/sasrec_loo_standard/sasrec_xlong402_original_full_dotbce_rerun/sasrec_xlong_loo402_seq400_dim128_L2_H1_best.pt" \
    --checkpoint_dir "$ROOT/checkpoints/sasrec_patch_prefix_xlong402_original_dotbce_p10_rerun" \
    --sasrec_attention_norm softmax --enable_projection_head true --train_head true
'

gpu2_body='
QUEUE_LOG="$LOG_DIR/queue_gpu2_xlong_table_hstu_original_rerun.log"
GPU_ID=2
'"$run_cmd_fn"'
run_cmd "hstu_xlong402_original_full_dotbce_rerun" "$LOG_DIR/hstu_xlong402_original_full_dotbce_rerun.log" \
  python run_hstu_taobao_standard.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --batch-size 32 --num-epochs 200 --hidden-units 128 \
    --num-blocks 4 --num-heads 4 --dropout-rate 0.2 --max-learning-rate 1e-3 --min-learning-rate 1e-5 \
    --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 --early-stop-patience 8 \
    --eval-sample-size 1000 --scheduler-type cosine_with_warmup --warmup-steps 100 --hstu-linear-dim 32 \
    --hstu-attention-dim 32 --hstu-attn-dropout 0.0 --hstu-normalization softmax1_rel_bias \
    --run-tag hstu_xlong402_original_full_dotbce_rerun
run_cmd "hstu_xlong402_original_short_dotbce_rerun" "$LOG_DIR/hstu_xlong402_original_short_dotbce_rerun.log" \
  python run_hstu_taobao_standard.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --max-seq-length 20 --batch-size 128 --num-epochs 200 \
    --hidden-units 128 --num-blocks 4 --num-heads 4 --dropout-rate 0.2 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --steps-per-train-log 100 --steps-per-val-log 2000 \
    --early-stop-patience 8 --eval-sample-size 1000 --scheduler-type cosine_with_warmup --warmup-steps 100 \
    --hstu-linear-dim 32 --hstu-attention-dim 32 --hstu-attn-dropout 0.0 --hstu-normalization softmax1_rel_bias \
    --run-tag hstu_xlong402_original_short_dotbce_rerun
run_cmd "hstu_xlong402_original_prefix_p10_rerun" "$LOG_DIR/hstu_xlong402_original_prefix_p10_rerun.log" \
  python id_patch/train_patch_first_order.py \
    --dataset xlong_loo402 --backbone hstu --device cuda:${GPU_ID} \
    --pretrained_ckpt_path "$ROOT/checkpoints/hstu_loo_standard/hstu_xlong402_original_full_dotbce_rerun/hstu_xlong_loo402_seq402_dim128_L4_H4_best.pt" \
    --checkpoint_dir "$ROOT/checkpoints/hstu_patch_prefix_xlong402_original_dotbce_p10_rerun" \
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

gpu1_body='
QUEUE_LOG="$LOG_DIR/queue_gpu1_xlong_table_sasrec_overflow_rerun.log"
GPU_ID=1
'"$run_cmd_fn"'
run_cmd "longer_xlong402_sampledsoftmax_sim_rerun" "$LOG_DIR/longer_xlong402_sampledsoftmax_sim_rerun.log" \
  python run_longer_taobao_sample_softmax.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --max-seq-length 402 --batch-size 128 --num-epochs 200 \
    --hidden-units 128 --num-blocks 2 --num-heads 1 --dropout-rate 0.1 --longer-global-tokens 4 \
    --longer-merge-size 4 --longer-merge-pool last --longer-inner-num-layers 1 --max-learning-rate 1e-3 \
    --min-learning-rate 1e-5 --weight-decay 0 --grad-clip 0 --num-negatives 128 --sampled-softmax-chunk-size 1024 \
    --steps-per-train-log 100 --steps-per-val-log 2000 --early-stop-patience 8 --num-workers 4 --prefetch-factor 2 \
    --eval-sample-size 1000 --eval-protocol legacy_loo --last-k-eval-test 0 --streaming-eval-last-k 0 \
    --selection-metric ndcg@10 --temperature 0.07 --user-embedding-norm l2_norm --scheduler-type cosine \
    --run-tag longer_xlong402_sampledsoftmax_sim_rerun
run_cmd "persrec_sasrec_xlong402_warmstart_legacyloo" "$LOG_DIR/persrec_sasrec_xlong402_warmstart_legacyloo.log" \
  python id_patch/train_Persrec.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --backbone sasrec --max_seq_length 400 --hidden_units 128 \
    --num_blocks 2 --num_heads 1 --dropout_rate 0.1 --right_align_positions true --sasrec_attention_norm softmax \
    --use_flash_attention true --persrec_enable true --persrec_num_tokens 8 --persrec_pretrain_len 380 \
    --persrec_recent_len 20 --persrec_eval_use_full_seq true --persrec_train_mode full --eval_seq_length 20 \
    --eval_protocol legacy_loo --last_k_eval_test 0 --streaming_eval_last_k 0 --batch_size 128 --num_epochs 50 \
    --max_learning_rate 2.0360441936032465e-05 --min_learning_rate 1e-06 --scheduler_type cosine \
    --checkpoint_dir "$ROOT/checkpoints/persrec_sasrec_xlong402_warmstart_legacyloo_20260425" \
    --pretrained_ckpt_path "$ROOT/checkpoints/sasrec_loo_sample_softmax/sasrec_xlong402_softmax_sampledsoftmax_backbone_20260418/sasrec_xlong_loo402_seq400_dim128_L2_H1_best.pt"
'

gpu3_body='
QUEUE_LOG="$LOG_DIR/queue_gpu3_xlong_table_hstu_persrec_legacyloo.log"
GPU_ID=3
'"$run_cmd_fn"'
run_cmd "persrec_hstu_xlong402_warmstart_legacyloo" "$LOG_DIR/persrec_hstu_xlong402_warmstart_legacyloo.log" \
  python id_patch/train_Persrec.py \
    --dataset xlong_loo402 --device cuda:${GPU_ID} --backbone hstu --max_seq_length 402 --hidden_units 128 \
    --num_blocks 4 --num_heads 4 --dropout_rate 0.2 --hstu_linear_dim 32 --hstu_attention_dim 32 \
    --hstu_attn_dropout 0.0 --hstu_normalization softmax1_rel_bias --persrec_enable true --persrec_num_tokens 8 \
    --persrec_pretrain_len 382 --persrec_recent_len 20 --persrec_eval_use_full_seq true --persrec_train_mode full \
    --eval_seq_length 20 --eval_protocol legacy_loo --last_k_eval_test 0 --streaming_eval_last_k 0 \
    --batch_size 64 --num_epochs 50 --max_learning_rate 2.0360441936032465e-05 --min_learning_rate 1e-06 \
    --scheduler_type cosine --checkpoint_dir "$ROOT/checkpoints/persrec_hstu_xlong402_warmstart_legacyloo_20260425" \
    --pretrained_ckpt_path "$ROOT/checkpoints/hstu_loo_sample_softmax/hstu_true_mh_xlong402_sm1_sampledsoftmax_backbone_20260419/hstu_xlong_loo402_seq402_dim128_L4_H4_best.pt"
'

launch_queue "queue_gpu0_xlong_table_sasrec_original_rerun" "$gpu0_body"
launch_queue "queue_gpu2_xlong_table_hstu_original_rerun" "$gpu2_body"
launch_queue "queue_gpu1_xlong_table_sasrec_overflow_rerun" "$gpu1_body"
launch_queue "queue_gpu3_xlong_table_hstu_persrec_legacyloo" "$gpu3_body"

echo
echo "local xlong rerun queues launched under: $LOG_DIR"
