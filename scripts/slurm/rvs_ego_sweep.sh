#!/usr/bin/env bash
#SBATCH --job-name=hermes-rvs-sweep
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-24,gpu-62
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -euo pipefail
repo_dir="${HERMES_SWEEP_SOURCE:?Set HERMES_SWEEP_SOURCE to the prepared source snapshot}"
export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
cd "$repo_dir"
nvidia-smi
exec "$repo_dir/.venv/hermes/bin/python" -u scripts/sweeps/rvs_ego.py "$@"
