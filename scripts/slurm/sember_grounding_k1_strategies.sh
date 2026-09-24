#!/usr/bin/env bash
#SBATCH --job-name=sember-k1-strategy
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
## #SBATCH --exclude=gpu-05
#SBATCH --output=slurm-sember-k1-strategies-%j.out

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python3}"
launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Start one job step for the entire Hydra sweep so every strategy reuses it.
  launcher=(srun --ntasks=1)
fi
# Forward Hydra flags first; the experiment profile defines the five-strategy
# multirun and creates a separate output directory for each strategy.
exec "${launcher[@]}" "$python_bin" scripts/run.py "$@" \
  experiment=sember_grounding_k1_strategies
