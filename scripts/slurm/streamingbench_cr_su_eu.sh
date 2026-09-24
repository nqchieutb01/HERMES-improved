#!/usr/bin/env bash
#SBATCH --job-name=streamingbench-cr-su-eu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=41G
#SBATCH --output=slurm-streamingbench-cr-su-eu-%j.out

set -euo pipefail

repo_dir="${HERMES_ROOT:-${SLURM_SUBMIT_DIR:-/l/users/chieu.nguyen/HERMES}}"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python}"
dataset_root="${STREAMINGBENCH_ROOT:-/nfs-stor/chieu.nguyen/StreamingBench}"
annotation="$dataset_root/streamingbench_realtime_cr_su_eu.json"

if [[ "${STREAMINGBENCH_SKIP_DOWNLOAD:-true}" != "true" ]]; then
    "$python_bin" scripts/data/download_streamingbench_tasks.py \
        --tasks CR SU EU --output "$dataset_root"
fi

exec "$python_bin" scripts/run.py -m \
    experiment=streamingbench_cr_su_eu \
    dataset.annotation="$annotation" \
    run.min_tokens_per_frame=1,0 "$@"
