#!/usr/bin/env bash
#SBATCH --job-name=sember-grounding-k1
#SBATCH --partition=long
#SBATCH --qos=gpu-debug-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-15,gpu-24
#SBATCH --output=slurm-sember-grounding-k1-%j.out

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python3}"
sample_fps="${SAMPLE_FPS:-0.2}"
kv_size="${KV_SIZE:-6000}"

cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

launcher=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Let Slurm bind the allocated GPU instead of setting CUDA_VISIBLE_DEVICES.
  launcher=(srun --ntasks=1)
fi

exec "${launcher[@]}" "$python_bin" scripts/run.py \
  hydra.mode=RUN \
  experiment=sember_grounding_time_count_location \
  run.sample_fps="$sample_fps" \
  run.kv_size="$kv_size" \
  run.min_tokens_per_frame=1 \
  "$@"
