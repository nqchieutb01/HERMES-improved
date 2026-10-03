#!/bin/bash
# 1-video check of the /l environment and weights: random 10% + prior-contrastive decoding, timeline prompt.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
[ -x /l/users/chieu.nguyen/venvs/hermes-qwen/bin/python3 ] || { echo "no /l on $(hostname)"; exit 3; }
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
.venv/hermes/bin/python3 scripts/run.py experiment=sember_grounding_uniform model=qwen3_vl_8b run.uniform_num_frames=64 \
  run.keep_time_tokens=surviving dataset.max_videos=1 run.num_chunks=1 run.max_new_tokens=384 dataset.grounding_prompt=timeline \
  run.offline_keep_ratio=0.1 run.prune_score=random run.contrastive_mode=blind run.contrastive_alpha=0.5 \
  paths.save_dir=$R/smoke-lstorage hydra.sweep.dir=outputs/hydra/smoke-lstorage hydra.sweep.subdir=0 > logs/phase18/smoke_lstorage.out 2>&1
echo "exit=$?"
