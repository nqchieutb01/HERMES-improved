#!/usr/bin/env bash
#SBATCH --job-name=rvs-ego-continuous
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
repo_dir="/l/users/chieu.nguyen/HERMES"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
save_dir="$repo_dir/results/llava_ov_0.5b/rvs_ego/continuous-fps0.5-kv6000"
python_bin="$repo_dir/.venv/hermes/bin/python"
if [[ -e "$save_dir/results.csv" || -e "$save_dir/2_0.csv" || -e "$save_dir/2_1.csv" ]]; then
    echo "Existing predictions found; refusing to overwrite $save_dir" >&2
    exit 1
fi
"$python_bin" scripts/data/prepare_rvs_ego_continuous.py --output-dir "$save_dir"
nvidia-smi
"$python_bin" -c 'import torch; assert torch.cuda.device_count() == 2, "Expected two visible allocated GPUs"'
"$python_bin" scripts/run.py experiment=rvs_ego_continuous
bash scripts/slurm/rvs_ego_judge.sh --results-path "$save_dir/results.csv"
"$python_bin" eval/rvs/compare_continuous.py
