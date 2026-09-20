#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
source "$repo_dir/hermes/bin/activate"

annotation="$repo_dir/data/streamingbench/realtime_cr_su_eu/streamingbench_realtime_cr_su_eu.json"
output_root="${STREAMINGBENCH_TOP_ATTN_K2_SAVE_DIR:-$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/top-attention-k2-20260920}"
baseline_root="$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/postfix-rope-budget-20260918"
kv8000_root="$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/frame-summary-kv8000-20260919"

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
    local gpu="$1"
    local kv_size="$2"
    local save_dir="$output_root/fps0.5-kv${kv_size}/min-k2"

    if [[ -s "$save_dir/results.csv" && -s "$save_dir/eval_results.txt" ]]; then
        echo "[$(timestamp)] KV=$kv_size already complete; skipping"
        return 0
    fi
    if [[ -e "$save_dir/1_0.csv" || -e "$save_dir/results.csv" || -e "$save_dir/inference.log" ]]; then
        echo "Refusing to overwrite incomplete output: $save_dir" >&2
        return 1
    fi
    mkdir -p "$save_dir"

    echo "[$(timestamp)] GPU $gpu starting top_attention_patch KV=$kv_size k=2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=2024 \
        python -u video_qa/hermes_vqa.py \
        --model llava_ov_0.5b \
        --anno_path "$annotation" \
        --save_dir "$save_dir" \
        --sample_fps 0.5 \
        --kv_size "$kv_size" \
        --streaming true \
        --num_chunks 1 \
        --chunk_idx 0 \
        --encode_chunk_size 16 \
        --min_tokens_per_frame 2 \
        --frame_summary_strategy top_attention_patch \
        --seed 2024 \
        --debug false \
        > "$save_dir/inference.log" 2>&1

    cp "$save_dir/1_0.csv" "$save_dir/results.csv"
    python eval/eval_multiple_choice.py general \
        --results_path "$save_dir/results.csv" \
        > "$save_dir/evaluation.log" 2>&1
    echo "[$(timestamp)] GPU $gpu completed top_attention_patch KV=$kv_size k=2"
}

echo "[$(timestamp)] Starting top_attention_patch k=2 sweep"
run_one 0 4000 &
pids=("$!")
run_one 1 6000 &
pids+=("$!")
run_one 0 8000 &
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
    echo "[$(timestamp)] One or more k=2 workers failed" >&2
    exit "$status"
fi

python - "$output_root" "$baseline_root" "$kv8000_root" <<'PY'
import csv
import pathlib
import sys

import pandas as pd

output = pathlib.Path(sys.argv[1])
baseline = pathlib.Path(sys.argv[2])
kv8000 = pathlib.Path(sys.argv[3])
rows = []
task_rows = []
for kv_size in (4000, 6000, 8000):
    frame = pd.read_csv(output / f"fps0.5-kv{kv_size}/min-k2/results.csv")
    if kv_size == 8000:
        k0_path = kv8000 / "k0/fps0.5-kv8000/min-k0/results.csv"
        k1_path = kv8000 / "top_attention_patch/fps0.5-kv8000/min-k1/results.csv"
    else:
        k0_path = baseline / f"fps0.5-kv{kv_size}/min-k0/results.csv"
        k1_path = (
            output.parent
            / "frame-summary-strategies-20260919"
            / "top_attention_patch"
            / f"fps0.5-kv{kv_size}/min-k1/results.csv"
        )
    k0 = pd.read_csv(k0_path)
    k1 = pd.read_csv(k1_path)
    keys = ["video_id", "question", "answer", "task"]
    if not frame[keys].equals(k0[keys]) or not frame[keys].equals(k1[keys]):
        raise ValueError(f"Sample order mismatch for KV={kv_size}")
    accuracy = float(frame["qa_acc"].mean())
    rows.append({
        "strategy": "top_attention_patch",
        "kv_size": kv_size,
        "min_tokens_per_frame": 2,
        "samples": len(frame),
        "accuracy": f"{accuracy:.4f}",
        "delta_vs_k0": f"{accuracy - k0['qa_acc'].mean():.4f}",
        "delta_vs_k1_top_attention": f"{accuracy - k1['qa_acc'].mean():.4f}",
        "different_predictions_vs_k0": int(
            (frame["pred_choice"] != k0["pred_choice"]).sum()
        ),
    })
    for task, group in frame.groupby("task"):
        task_rows.append({
            "strategy": "top_attention_patch",
            "kv_size": kv_size,
            "min_tokens_per_frame": 2,
            "task": task,
            "samples": len(group),
            "accuracy": f"{group['qa_acc'].mean():.4f}",
        })

for filename, records in (
    ("comparison.csv", rows),
    ("comparison_by_task.csv", task_rows),
):
    with (output / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
for row in rows:
    print(row)
PY

echo "[$(timestamp)] top_attention_patch k=2 sweep completed"
