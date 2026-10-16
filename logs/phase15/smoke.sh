#!/bin/bash
# 2-video smoke of the phase 15 diagnostic hooks (Qwen3-VL-8B, uniform 64, S-EMBER grounding).
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
BASE="experiment=sember_grounding_uniform model=qwen3_vl_8b run.uniform_num_frames=64 run.keep_time_tokens=surviving dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=128"
i=0
for extra in "run.question_attention=true run.retention_snapshot=true run.offline_keep_ratio=0.1 run.prune_score=oracle" \
             "run.retention_snapshot=true run.offline_keep_ratio=0.1 run.prune_score=hermes_exact" \
             "run.time_offset=200" "run.drop_timestamps=true" "run.uniform_start_frac=0.5"; do
  i=$((i+1))
  .venv/hermes/bin/python3 scripts/run.py $BASE $extra paths.save_dir=$R/smoke15-$i \
    hydra.sweep.dir=outputs/hydra/smoke15-$i hydra.sweep.subdir=0 > logs/phase15/smoke-$i.out 2>&1
  echo "case $i exit=$? ($extra)"
done
