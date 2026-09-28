#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
PY=/home/chieu.nguyen/HERMES/.venv/hermes/bin/python3
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
common="model=qwen3_vl_8b dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=128 run.keep_time_tokens=surviving"
$PY scripts/run.py experiment=sember_grounding_time_count_location $common run.sample_fps=0.2 run.kv_size=6000 \
  run.min_tokens_per_frame=0 run.frame_summary_strategy=attention_weighted run.retention_snapshot=true \
  paths.save_dir=$R/smoke9-retention hydra.sweep.dir=outputs/hydra/s9a hydra.sweep.subdir=0 > /dev/null 2>&1; echo "a exit=$?"
$PY scripts/run.py experiment=sember_grounding_uniform $common run.uniform_num_frames=64 run.offline_keep_ratio=0.25 \
  paths.save_dir=$R/smoke9-offline-hermes hydra.sweep.dir=outputs/hydra/s9b hydra.sweep.subdir=0 > /dev/null 2>&1; echo "b exit=$?"
$PY scripts/run.py experiment=sember_grounding_uniform $common run.uniform_num_frames=64 run.offline_keep_ratio=0.25 run.prune_score=random \
  paths.save_dir=$R/smoke9-offline-random hydra.sweep.dir=outputs/hydra/s9c hydra.sweep.subdir=0 > /dev/null 2>&1; echo "c exit=$?"
