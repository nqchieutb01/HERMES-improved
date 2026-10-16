#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
export HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2
R=/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding
.venv/hermes/bin/python3 scripts/run.py experiment=sember_grounding_uniform model=qwen3_vl_8b run.uniform_num_frames=64 \
  run.keep_time_tokens=surviving dataset.max_videos=2 run.num_chunks=1 run.max_new_tokens=384 dataset.grounding_prompt=timeline \
  run.offline_keep_ratio=0.1 run.prune_score=zoom run.zoom_windows=/home/chieu.nguyen/HERMES/logs/phase17/win_self_random10.json \
  run.retention_snapshot=true paths.save_dir=$R/smoke17 hydra.sweep.dir=outputs/hydra/smoke17 hydra.sweep.subdir=0 > logs/phase17/smoke.out 2>&1
echo "exit=$?"
