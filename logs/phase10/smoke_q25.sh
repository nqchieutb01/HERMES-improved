#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
PY=/home/chieu.nguyen/HERMES/.venv/hermes/bin/python3
R=/home/chieu.nguyen/HERMES/results/qwen2.5_vl_7b/sember_grounding
common="model=qwen2.5_vl_7b dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=128 experiment=sember_grounding_uniform run.uniform_num_frames=64"
$PY scripts/run.py $common paths.save_dir=$R/smoke10-none hydra.sweep.dir=outputs/hydra/q25a hydra.sweep.subdir=0 > /dev/null 2>&1; echo "a exit=$?"
$PY scripts/run.py $common run.offline_keep_ratio=0.25 run.prune_score=stratified paths.save_dir=$R/smoke10-strat \
  hydra.sweep.dir=outputs/hydra/q25b hydra.sweep.subdir=0 > /dev/null 2>&1; echo "b exit=$?"
