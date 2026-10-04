#!/bin/bash
# Batch size 1 vs 8 on the same videos: stage timings and answer agreement (random 10%, timeline prompt).
cd /home/chieu.nguyen/HERMES
echo host=$(hostname); nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY=/l/users/chieu.nguyen/venvs/hermes-qwen/bin/python3
NV=${NV:-12}
COMMON="--model qwen3_vl_8b --sample_fps 1.0 --frame_sampling uniform --anno_path /nfs-stor/chieu.nguyen/s-ember/sember_grounding.jsonl --debug false --num_chunks 1 --chunk_idx 0 --kv_size 6000 --streaming true --encode_chunk_size 16 --max_new_tokens 384 --repetition_penalty 1.1 --recency_weight_start 0.75 --recency_weight_decay 0.6 --reindex_margin 1024 --use_history false --verbose_token_trace false --min_tokens_per_frame 0 --frame_summary_strategy mean --frame_summary_temperature 0.1 --keep_time_tokens surviving --model_path /l/users/chieu.nguyen/models/Qwen3-VL-8B-Instruct --offline_keep_ratio 0.1 --prune_score random --uniform_num_frames 64 --dataset_adapter sember_grounding --video_root /nfs-stor/chieu.nguyen/s-ember/videos --max_videos $NV --grounding_prompt timeline --question_categories time_duration counting_objects_events location_trace"
PCD="--contrastive_mode blind --contrastive_alpha 1.0 --contrastive_adaptive true --contrastive_scope time"
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
OUT=/home/chieu.nguyen/HERMES/logs/batching/out
rm -rf $OUT; mkdir -p $OUT
for name in base pcd; do
  extra=""; [ $name = pcd ] && extra="$PCD"
  for bs in 1 8; do
    echo "### $name bs=$bs"
    $PY logs/batching/stage_timer.py $COMMON $extra --batch_size $bs --save_dir $OUT/$name-bs$bs > $OUT/$name-bs$bs.log 2>&1
    grep -A12 "=== PROFILE" $OUT/$name-bs$bs.log || tail -30 $OUT/$name-bs$bs.log
  done
done
$PY logs/batching/compare.py $OUT/base-bs1 $OUT/base-bs8
$PY logs/batching/compare.py $OUT/pcd-bs1 $OUT/pcd-bs8
