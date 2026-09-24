#!/usr/bin/env bash
#SBATCH --job-name=sember-mcq-tcl
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-54,gpu-05
#SBATCH --output=slurm-sember-mcq-tcl-%j.out

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python3}"
launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # One Slurm step exposes both allocated GPUs; the Hydra runner assigns one
  # GPU to each of the two inference chunks.
  launcher=(srun --ntasks=1)
fi
# Forward Hydra flags first, consistent with the other S-EMBER launchers.
exec "${launcher[@]}" "$python_bin" scripts/run.py "$@" \
  experiment=sember_mcq_time_count_location
