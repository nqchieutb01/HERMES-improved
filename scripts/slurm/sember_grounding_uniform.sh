#!/usr/bin/env bash
#SBATCH --job-name=sember-uniform
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --output=slurm-sember-uniform-%j.out
#SBATCH --exclude=gpu-05

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1" 

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python3}"
exec "$python_bin" scripts/run.py "$@" experiment=sember_grounding_uniform
