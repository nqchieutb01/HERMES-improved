#!/usr/bin/env bash
#SBATCH --job-name=rvs-ego-judge
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
repo_dir="/l/users/chieu.nguyen/HERMES"
cd "$repo_dir"
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=/nfs-stor/chieu.nguyen/.cache/uv
export VLLM_CACHE_ROOT=/nfs-stor/chieu.nguyen/.cache/vllm
export TRITON_CACHE_DIR=/nfs-stor/chieu.nguyen/.cache/triton
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$repo_dir/.venv/judge/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$repo_dir/.venv/cuda-compat-13/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
"$repo_dir/.venv/judge/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "visible GPUs", torch.cuda.device_count(), flush=True)'
"$repo_dir/.venv/judge/bin/python" -u eval/rvs/judge.py "$@"
