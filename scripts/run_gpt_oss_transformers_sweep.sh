#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_root="${1:-$repo_dir/results/llava_ov_0.5b/rvs_ego/sweep-20260915-long-1gpu}"
model="${GPT_OSS_MODEL:-$repo_dir/models/gpt-oss-20b}"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/gpt-oss/bin/python}"
output_name="${GPT_OSS_OUTPUT_NAME:-gpt_oss_20b_judge_transformers}"
batch_size="${GPT_OSS_BATCH_SIZE:-16}"
device_map="${GPT_OSS_DEVICE_MAP:-auto}"
dequantize="${GPT_OSS_DEQUANTIZE:-0}"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing evaluation Python: $python_bin" >&2
  echo "Create it with: uv venv $repo_dir/.venv/gpt-oss --python 3.12" >&2
  exit 2
fi

if [[ ! -d "$results_root" ]]; then
  echo "Results directory does not exist: $results_root" >&2
  exit 2
fi

if [[ ("$model" == /* || "$model" == ./*) && ! -e "$model" ]]; then
  echo "GPT-OSS checkpoint is not available: $model" >&2
  echo "Set GPT_OSS_MODEL to the mounted checkpoint path before running." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$repo_dir/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

mapfile -t result_files < <(find "$results_root" -mindepth 2 -maxdepth 2 -type f -name results.csv -print | sort)
if (( ${#result_files[@]} == 0 )); then
  echo "No trial results.csv files found under: $results_root" >&2
  exit 2
fi

judge_args=(
  --model "$model"
  --reasoning-effort low
  --backend transformers
  --device-map "$device_map"
  --tensor-parallel-size 1
  --batch-size "$batch_size"
)
if [[ "$dequantize" == "1" ]]; then
  judge_args+=(--dequantize)
fi

for results_path in "${result_files[@]}"; do
  trial_dir="$(dirname "$results_path")"
  output_dir="$trial_dir/$output_name"
  echo "Evaluating $results_path -> $output_dir"
  "$python_bin" -u "$repo_dir/scripts/judge_rvs_ego.py" \
    "${judge_args[@]}" \
    --results-path "$results_path" \
    --output-dir "$output_dir"
done
