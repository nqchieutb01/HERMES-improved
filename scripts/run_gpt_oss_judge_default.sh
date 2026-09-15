#!/usr/bin/env bash
#SBATCH --job-name=gpt-oss-rvs-judge
#SBATCH --partition=long
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
repo_dir="/l/users/chieu.nguyen/HERMES"
cd "$repo_dir"
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="/usr/local/cuda"
export VLLM_CACHE_ROOT=/nfs-stor/chieu.nguyen/.cache/vllm-gpt-oss
export TRITON_CACHE_DIR=/nfs-stor/chieu.nguyen/.cache/triton-gpt-oss
exec "/nfs-stor/chieu.nguyen/gpt-oss-eval-env/bin/python" -u scripts/judge_rvs_ego.py \
  --model /nfs-stor/chieu.nguyen/gpt-oss-20b \
  --reasoning-effort low \
  --tensor-parallel-size 1 --batch-size 32 \
  --results-path "$1" --output-dir "$2"
