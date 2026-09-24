#!/usr/bin/env bash
#SBATCH --job-name=sember-mcq-uniform
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-24
#SBATCH --output=slurm-sember-mcq-uniform-%j.out

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python3}"
launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  launcher=(srun --ntasks=1)
fi

exec "${launcher[@]}" "$python_bin" scripts/run.py "$@" \
  experiment=sember_mcq_uniform
