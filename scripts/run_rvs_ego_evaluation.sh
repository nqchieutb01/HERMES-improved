#!/usr/bin/env bash
#SBATCH --job-name=rvs-ego-kv6072
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
repo_dir="${HERMES_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_dir"
export HF_HOME="${HF_HOME:-$repo_dir/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
nvidia-smi
bash scripts/run_streamingbench_smoke.sh
results_path="${RVS_EGO_RESULTS_PATH:-$repo_dir/results/llava_ov_0.5b/rvs_ego/fps0.5-kv1001/results.csv}"
python_bin="${HERMES_PYTHON:-$repo_dir/.venv/hermes/bin/python}"
"$python_bin" scripts/test_rvs_ego_results.py --results-path "$results_path"
"$python_bin" scripts/eval_rvs_ego_yes_no.py --results-path "$results_path"
