#!/usr/bin/env bash
set -euo pipefail

repo_dir="/l/users/chieu.nguyen/HERMES"
results_path="$repo_dir/results/llava_ov_0.5b/rvs_ego/sweep-20260915-long-1gpu/17-adaptive/qwen_judge/judgments.csv"
output_dir="$repo_dir/results/llava_ov_0.5b/rvs_ego/sweep-20260915-long-1gpu/17-adaptive/gpt_oss_20b_judge_mxfp4"

cd "$repo_dir"
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_OFFLINE=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$output_dir"
exec /nfs-stor/chieu.nguyen/gpt-oss-eval-env/bin/python -u scripts/judge_rvs_ego.py \
  --model /nfs-stor/chieu.nguyen/gpt-oss-20b \
  --backend transformers \
  --reasoning-effort low \
  --tensor-parallel-size 1 \
  --batch-size 1 \
  --results-path "$results_path" \
  --output-dir "$output_dir"
