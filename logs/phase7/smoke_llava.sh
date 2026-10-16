#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname) start=$(date)
PY=/home/chieu.nguyen/HERMES/.venv/hermes/bin/python3
R=/home/chieu.nguyen/HERMES/results
common="dataset.max_videos=2 run.num_chunks=1 run.kv_size=6000 run.max_new_tokens=128 run.sample_fps=0.2"
$PY scripts/run.py experiment=sember_grounding_time_count_location model=llava_ov_7b $common \
  run.min_tokens_per_frame=0 run.frame_summary_strategy=attention_weighted "run.sample_schedule='grid:32:8'" \
  paths.save_dir=$R/llava_ov_7b/sember_grounding/smoke-grid hydra.sweep.dir=outputs/hydra/smoke-b hydra.sweep.subdir=0 > /dev/null 2>&1; echo "b exit=$?"
$PY scripts/run.py experiment=sember_grounding_time_count_location model=llava_ov_7b $common \
  run.min_tokens_per_frame=1 run.frame_summary_strategy=top_attention_patch "run.sample_schedule='grid:32:8'" \
  paths.save_dir=$R/llava_ov_7b/sember_grounding/smoke-grid-k1 hydra.sweep.dir=outputs/hydra/smoke-c hydra.sweep.subdir=0 > /dev/null 2>&1; echo "c exit=$?"
echo end=$(date)
