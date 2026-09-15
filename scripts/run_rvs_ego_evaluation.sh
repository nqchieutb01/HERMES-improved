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
cd /l/users/chieu.nguyen/HERMES
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
nvidia-smi
bash scripts/run_streamingbench_smoke.sh
results_path=results/llava_ov_0.5b/rvs_ego/fps0.5-kv6072/results.csv
.venv/hermes/bin/python scripts/test_rvs_ego_results.py --results-path "$results_path"
.venv/hermes/bin/python scripts/eval_rvs_ego_yes_no.py --results-path "$results_path"
