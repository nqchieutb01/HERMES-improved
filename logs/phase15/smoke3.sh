#!/bin/bash
# 2-video smoke of the evidence-window oracle (64 frames inside the gold interval, stamps-only outside).
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
BASE="experiment=sember_grounding_uniform model=qwen3_vl_8b run.uniform_num_frames=64 run.keep_time_tokens=surviving dataset.max_videos=2 run.num_chunks=1 run.oracle_window=true"
i=20
for extra in "run.max_new_tokens=128" "dataset.grounding_prompt=timeline run.max_new_tokens=384"; do
  i=$((i+1))
  .venv/hermes/bin/python3 scripts/run.py $BASE $extra paths.save_dir=$R/smoke15-$i \
    hydra.sweep.dir=outputs/hydra/smoke15-$i hydra.sweep.subdir=0 > logs/phase15/smoke-$i.out 2>&1
  echo "case $i exit=$? ($extra)"
done
