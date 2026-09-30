#!/bin/bash
# 2-video smoke of the phase 15b hooks: answer attention, blind, frame/timestamp shuffles.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
BASE="experiment=sember_grounding_uniform model=qwen3_vl_8b run.uniform_num_frames=64 run.keep_time_tokens=surviving dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=384"
i=10
for extra in "run.answer_attention=true dataset.grounding_prompt=timeline" "run.blind=true" "run.shuffle_mode=frames" "run.shuffle_mode=stamps"; do
  i=$((i+1))
  .venv/hermes/bin/python3 scripts/run.py $BASE $extra paths.save_dir=$R/smoke15-$i \
    hydra.sweep.dir=outputs/hydra/smoke15-$i hydra.sweep.subdir=0 > logs/phase15/smoke-$i.out 2>&1
  echo "case $i exit=$? ($extra)"
done
