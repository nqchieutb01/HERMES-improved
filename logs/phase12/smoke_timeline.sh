#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2 .venv/hermes/bin/python3 scripts/run.py \
  experiment=sember_grounding_time_count_location model=qwen3_vl_8b dataset.max_videos=4 run.num_chunks=1 \
  run.sample_fps=0.2 run.kv_size=6000 run.min_tokens_per_frame=0 run.frame_summary_strategy=attention_weighted \
  run.keep_time_tokens=surviving run.max_new_tokens=384 dataset.grounding_prompt=timeline \
  paths.save_dir=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding/smoke-timeline \
  hydra.sweep.dir=outputs/hydra/smoke-timeline hydra.sweep.subdir=0 > /dev/null 2>&1; echo "exit=$?"
