#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname) start=$(date)
HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2 /home/chieu.nguyen/HERMES/.venv/hermes/bin/python3 scripts/run.py \
  experiment=sember_grounding_time_count_location model=qwen3_vl_8b dataset.max_videos=2 \
  run.sample_fps=0.2 "run.sample_schedule='grid:32:8'" run.kv_size=6000 run.min_tokens_per_frame=1 \
  run.frame_summary_strategy=top_attention_patch run.max_new_tokens=128 run.keep_time_tokens=surviving run.num_chunks=1 \
  paths.save_dir=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/smoke-grid-k1 \
  hydra.sweep.dir=outputs/hydra/smoke-grid-k1 hydra.sweep.subdir=0
echo exit=$? end=$(date)
