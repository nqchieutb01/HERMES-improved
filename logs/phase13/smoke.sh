#!/bin/bash
# 2-video smoke: timeline prompt with Qwen2.5-VL stratified pruning, and with Qwen3 k=1 streaming.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
P=.venv/hermes/bin/python3
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results
$P scripts/run.py experiment=sember_grounding_uniform model=qwen2.5_vl_7b run.uniform_num_frames=64 \
  run.offline_keep_ratio=0.1 run.prune_score=stratified run.keep_time_tokens=surviving dataset.max_videos=2 \
  run.num_chunks=1 run.max_new_tokens=384 dataset.grounding_prompt=timeline \
  paths.save_dir=$R/qwen2.5_vl_7b/sember_grounding/smoke-tl13 hydra.sweep.dir=outputs/hydra/smoke-tl13a \
  hydra.sweep.subdir=0 > logs/phase13/smoke_q25.out 2>&1; echo "q25 exit=$?"
$P scripts/run.py experiment=sember_grounding_time_count_location model=qwen3_vl_8b run.sample_fps=0.2 \
  run.kv_size=4000 run.min_tokens_per_frame=1 run.frame_summary_strategy=top_attention_patch \
  run.keep_time_tokens=surviving dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=384 \
  dataset.grounding_prompt=timeline paths.save_dir=$R/qwen3_vl_8b/sember_grounding/smoke-tl13 \
  hydra.sweep.dir=outputs/hydra/smoke-tl13b hydra.sweep.subdir=0 > logs/phase13/smoke_q3.out 2>&1; echo "q3 exit=$?"
