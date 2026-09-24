#!/usr/bin/env bash
#SBATCH --job-name=sember-tcl-smoke
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=41G
#SBATCH --output=slurm-sember-tcl-smoke-%j.out

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python}"
exec "$python_bin" scripts/run.py \
  experiment=sember_mcq_time_count_location_smoke "$@"
