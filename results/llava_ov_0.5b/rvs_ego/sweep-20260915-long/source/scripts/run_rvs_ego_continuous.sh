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
"$python_bin" scripts/prepare_rvs_ego_continuous.py --output-dir "$save_dir"
nvidia-smi
"$python_bin" -c 'import torch; assert torch.cuda.device_count() == 2, "Expected two visible allocated GPUs"'
IFS=',' read -r -a assigned_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm must assign two GPUs}"
[[ ${#assigned_gpus[@]} -eq 2 ]]
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
for chunk in 0 1; do
    CUDA_VISIBLE_DEVICES="${assigned_gpus[$chunk]}" "$python_bin" -u video_qa/hermes_vqa.py \
        --model llava_ov_0.5b --anno_path "$save_dir/annotations.json" \
        --save_dir "$save_dir" --sample_fps 0.5 --kv_size 6000 \
        --streaming true --num_chunks 2 --chunk_idx "$chunk" --debug false \
        > "$save_dir/inference-$chunk.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]]
"$python_bin" - "$save_dir" <<'PY'
import csv
from pathlib import Path
import sys
root = Path(sys.argv[1])
rows = []
for chunk in range(2):
    with (root / f"2_{chunk}.csv").open(newline="") as source:
        rows.extend(csv.DictReader(source))
with (root / "results.csv").open("w", newline="") as target:
    writer = csv.DictWriter(target, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY
"$python_bin" scripts/test_rvs_ego_results.py --results-path "$save_dir/results.csv" --anno-path "$save_dir/annotations.json"
"$python_bin" scripts/eval_rvs_ego_yes_no.py --results-path "$save_dir/results.csv"
bash scripts/run_rvs_ego_judge.sh --results-path "$save_dir/results.csv"
"$python_bin" scripts/compare_rvs_ego_continuous.py
