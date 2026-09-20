#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
source "$repo_dir/hermes/bin/activate"

gpu="${STREAMINGBENCH_GPU:-1}"
annotation="$repo_dir/data/streamingbench/realtime_cr_su_eu/streamingbench_realtime_cr_su_eu.json"
save_dir="$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/frame-summary-strategies-20260919/softmax_score_weighted/fps0.5-kv6000/min-k1-retry1"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

if [[ -e "$save_dir/1_0.csv" || -e "$save_dir/results.csv" || -e "$save_dir/inference.log" ]]; then
    echo "Refusing to overwrite retry output: $save_dir" >&2
    exit 1
fi
mkdir -p "$save_dir"

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] GPU $gpu retrying softmax_score_weighted KV=6000"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=2024 \
    python -u video_qa/hermes_vqa.py \
    --model llava_ov_0.5b \
    --anno_path "$annotation" \
    --save_dir "$save_dir" \
    --sample_fps 0.5 \
    --kv_size 6000 \
    --streaming true \
    --num_chunks 1 \
    --chunk_idx 0 \
    --encode_chunk_size 16 \
    --min_tokens_per_frame 1 \
    --frame_summary_strategy softmax_score_weighted \
    --frame_summary_temperature 0.1 \
    --seed 2024 \
    --debug false \
    > "$save_dir/inference.log" 2>&1

cp "$save_dir/1_0.csv" "$save_dir/results.csv"
python eval/eval_multiple_choice.py general \
    --results_path "$save_dir/results.csv" \
    > "$save_dir/evaluation.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] softmax_score_weighted KV=6000 retry complete"
