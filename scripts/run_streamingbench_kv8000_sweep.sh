#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
source "$repo_dir/hermes/bin/activate"

gpu="${STREAMINGBENCH_GPU:-1}"
annotation="$repo_dir/data/streamingbench/realtime_cr_su_eu/streamingbench_realtime_cr_su_eu.json"
output_root="${STREAMINGBENCH_KV8000_SAVE_DIR:-$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/frame-summary-kv8000-20260919}"
temperature="${STREAMINGBENCH_FRAME_SUMMARY_TEMPERATURE:-0.1}"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

mkdir -p "$output_root"
git rev-parse HEAD > "$output_root/base_commit.txt"
git diff -- \
    inference/abstract_hermes.py \
    inference/llavaov_hermes.py \
    inference/qwenvl_hermes.py \
    video_qa/base.py \
    scripts/test_llava_attention.py \
    > "$output_root/source.patch"

run_one() {
    local label="$1"
    local minimum="$2"
    local strategy="$3"
    local save_dir="$output_root/$label/fps0.5-kv8000/min-k${minimum}"

    if [[ -s "$save_dir/results.csv" && -s "$save_dir/eval_results.txt" ]]; then
        echo "[$(timestamp)] $label already complete; skipping"
        return 0
    fi
    if [[ -e "$save_dir/1_0.csv" || -e "$save_dir/results.csv" || -e "$save_dir/inference.log" ]]; then
        echo "Refusing to overwrite incomplete output: $save_dir" >&2
        return 1
    fi
    mkdir -p "$save_dir"

    echo "[$(timestamp)] GPU $gpu starting $label (min-k=$minimum, strategy=$strategy)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=2024 \
        python -u video_qa/hermes_vqa.py \
        --model llava_ov_0.5b \
        --anno_path "$annotation" \
        --save_dir "$save_dir" \
        --sample_fps 0.5 \
        --kv_size 8000 \
        --streaming true \
        --num_chunks 1 \
        --chunk_idx 0 \
        --encode_chunk_size 16 \
        --min_tokens_per_frame "$minimum" \
        --frame_summary_strategy "$strategy" \
        --frame_summary_temperature "$temperature" \
        --seed 2024 \
        --debug false \
        > "$save_dir/inference.log" 2>&1

    cp "$save_dir/1_0.csv" "$save_dir/results.csv"
    python eval/eval_multiple_choice.py general \
        --results_path "$save_dir/results.csv" \
        > "$save_dir/evaluation.log" 2>&1
    echo "[$(timestamp)] GPU $gpu completed $label"
}

run_lane() {
    local status=0
    while (( $# >= 3 )); do
        if ! run_one "$1" "$2" "$3"; then
            status=1
        fi
        shift 3
    done
    return "$status"
}

echo "[$(timestamp)] Starting KV=8000 sweep on physical GPU $gpu"
echo "Output: $output_root"
echo "Parallel lanes: [k0, top_attention_patch], [mean, attention_weighted], [top_patch, softmax_score_weighted]"

run_lane k0 0 mean top_attention_patch 1 top_attention_patch &
pids=("$!")
run_lane mean 1 mean attention_weighted 1 attention_weighted &
pids+=("$!")
run_lane top_patch 1 top_patch softmax_score_weighted 1 softmax_score_weighted &
pids+=("$!")

terminate_workers() {
    kill "${pids[@]}" 2>/dev/null || true
}
trap terminate_workers INT TERM

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "[$(timestamp)] One or more KV=8000 workers failed" >&2
    exit "$status"
fi

python - "$output_root" <<'PY'
import csv
import pathlib
import sys

import pandas as pd

root = pathlib.Path(sys.argv[1])
configs = (
    ("k0", 0),
    ("mean", 1),
    ("top_patch", 1),
    ("top_attention_patch", 1),
    ("attention_weighted", 1),
    ("softmax_score_weighted", 1),
)
frames = {
    label: pd.read_csv(root / label / f"fps0.5-kv8000/min-k{minimum}/results.csv")
    for label, minimum in configs
}
reference = frames["k0"]
keys = ["video_id", "question", "answer", "task"]
rows = []
task_rows = []
for label, minimum in configs:
    frame = frames[label]
    if len(frame) != len(reference) or not frame[keys].equals(reference[keys]):
        raise ValueError(f"Sample order mismatch for {label}")
    accuracy = float(frame["qa_acc"].mean())
    rows.append({
        "strategy": label,
        "kv_size": 8000,
        "min_tokens_per_frame": minimum,
        "samples": len(frame),
        "accuracy": f"{accuracy:.4f}",
        "delta_vs_k0": "" if label == "k0" else f"{accuracy - reference['qa_acc'].mean():.4f}",
        "different_predictions_vs_k0": int((frame["pred_choice"] != reference["pred_choice"]).sum()),
    })
    for task, group in frame.groupby("task"):
        task_rows.append({
            "strategy": label,
            "task": task,
            "samples": len(group),
            "accuracy": f"{group['qa_acc'].mean():.4f}",
        })

for filename, records in (
    ("comparison.csv", rows),
    ("comparison_by_task.csv", task_rows),
):
    with (root / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
for row in rows:
    print(row)
PY

echo "[$(timestamp)] KV=8000 sweep completed"
