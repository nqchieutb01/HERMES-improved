#!/bin/bash
# Smoke: contrastive decoding against a counterfactual memory (stamps / blind), random 10%, timeline prompt.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
BASE="experiment=sember_grounding_uniform model=qwen3_vl_8b run.uniform_num_frames=64 run.keep_time_tokens=surviving dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=384 dataset.grounding_prompt=timeline run.offline_keep_ratio=0.1 run.prune_score=random"
i=40
for extra in "run.contrastive_mode=stamps" "run.contrastive_mode=blind"; do
  i=$((i+1))
  .venv/hermes/bin/python3 scripts/run.py $BASE $extra paths.save_dir=$R/smoke17-$i hydra.sweep.dir=outputs/hydra/smoke17-$i hydra.sweep.subdir=0 > logs/phase17/smoke-$i.out 2>&1
  echo "case $i exit=$? ($extra)"
done
