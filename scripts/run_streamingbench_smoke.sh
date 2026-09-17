#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="$repo_dir/.venv/hermes/bin/python"
dataset_root="${VSTREAM_QA_ROOT:-/nfs-stor/chieu.nguyen/VStream-QA}"
anno_path="$dataset_root/vstream-realtime/rvs_ego_hermes.json"
save_dir="$repo_dir/results/llava_ov_0.5b/rvs_ego/fps0.5-kv1001"
mkdir -p "$save_dir"

# Validate the extracted 1-FPS frames and regenerate portable local annotations.
"$python_bin" scripts/prepare_rvs_ego.py --dataset-root "$dataset_root"

"$python_bin" -u video_qa/hermes_vqa.py \
    --model llava_ov_0.5b \
    --anno_path "$anno_path" \
    --save_dir "$save_dir" \
    --sample_fps 0.5 --kv_size 1001 --streaming true \
    --num_chunks 1 --chunk_idx 0 --debug false \
    2>&1 | tee "$save_dir/inference.log"

cp "$save_dir/1_0.csv" "$save_dir/results.csv"
echo "RVS-Ego predictions saved to $save_dir/results.csv"
echo "Run scripts/run_rvs_ego_judge.sh --results-path $save_dir/results.csv for local Qwen judging."
