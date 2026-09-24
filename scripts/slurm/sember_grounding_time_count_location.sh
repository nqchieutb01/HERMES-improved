#!/usr/bin/env bash
#SBATCH --job-name=sember-grounding-tcl
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm-sember-grounding-tcl-%j.out
#SBATCH --exclude=gpu-15,gpu-24
#SBATCH --time=3:00:00

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python3}"
launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Start a job step so Slurm exposes the GPU assigned to this allocation.
  launcher=(srun --ntasks=1)
fi
# Forward Hydra flags first: argparse requires global options such as -m/--multirun
# to appear before configuration overrides.
exec "${launcher[@]}" "$python_bin" scripts/run.py "$@" \
  experiment=sember_grounding_time_count_location
