#!/bin/bash
# Smoke: prior-contrastive decoding in HERMES streaming (0.2 fps, 6k budget), 2 videos, timeline prompt.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
.venv/hermes/bin/python3 scripts/run.py experiment=sember_grounding_time_count_location model=qwen3_vl_8b dataset.max_videos=2 \
  run.num_chunks=1 run.sample_fps=0.2 run.kv_size=6000 run.min_tokens_per_frame=0 run.frame_summary_strategy=attention_weighted \
  run.keep_time_tokens=surviving run.max_new_tokens=384 dataset.grounding_prompt=timeline run.contrastive_mode=blind \
  run.contrastive_alpha=0.5 paths.save_dir=$R/smoke18-stream hydra.sweep.dir=outputs/hydra/smoke18-stream hydra.sweep.subdir=0 \
  > logs/phase18/smoke_stream.out 2>&1
echo "exit=$?"
