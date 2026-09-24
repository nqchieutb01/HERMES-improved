#!/usr/bin/env bash
#SBATCH --job-name=gpt-oss-rvs-judge
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --qos=gpu-debug-qos

set -euo pipefail
repo_dir="${HERMES_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$repo_dir"
export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
export HF_HUB_OFFLINE=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/nfs-stor/chieu.nguyen/.cache/vllm-gpt-oss}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/nfs-stor/chieu.nguyen/.cache/triton-gpt-oss}"
python_bin="${GPT_OSS_PYTHON:-/nfs-stor/chieu.nguyen/gpt-oss-eval-env/bin/python}"
model_path="${GPT_OSS_MODEL:-/nfs-stor/chieu.nguyen/gpt-oss-20b}"
exec "$python_bin" -u eval/rvs/judge.py \
  --model "$model_path" \
  --reasoning-effort low \
  --backend transformers \
  --tensor-parallel-size 1 --batch-size 1 \
  --results-path "$1" --output-dir "$2"
