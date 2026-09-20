#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="$repo_dir/hermes/bin/python"
annotation="$repo_dir/data/streamingbench/realtime_cr_su_eu/streamingbench_realtime_cr_su_eu.json"
output_root="$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/frame-summary-strategies-20260919"
baseline_root="$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/postfix-rope-budget-20260918"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

run_one() {
    local gpu="$1"
    local kv_size="$2"
    local save_dir="$output_root/top_attention_patch/fps0.5-kv${kv_size}/min-k1"
    local chunk_path="$save_dir/1_0.csv"
    local results_path="$save_dir/results.csv"

    if [[ -e "$chunk_path" || -e "$results_path" || -e "$save_dir/inference.log" ]]; then
        echo "Refusing to overwrite existing output: $save_dir" >&2
        return 1
    fi
    mkdir -p "$save_dir"
    echo "[$(timestamp)] GPU $gpu starting top_attention_patch KV=$kv_size"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=2024 \
        "$python_bin" -u video_qa/hermes_vqa.py \
        --model llava_ov_0.5b \
        --anno_path "$annotation" \
        --save_dir "$save_dir" \
        --sample_fps 0.5 \
        --kv_size "$kv_size" \
        --streaming true \
        --num_chunks 1 \
        --chunk_idx 0 \
        --encode_chunk_size 16 \
        --min_tokens_per_frame 1 \
        --frame_summary_strategy top_attention_patch \
        --seed 2024 \
        --debug false \
        > "$save_dir/inference.log" 2>&1

    cp "$chunk_path" "$results_path"
    "$python_bin" eval/eval_multiple_choice.py general \
        --results_path "$results_path" \
        > "$save_dir/evaluation.log" 2>&1
    echo "[$(timestamp)] GPU $gpu completed top_attention_patch KV=$kv_size"
}

git diff -- \
    inference/abstract_hermes.py \
    inference/llavaov_hermes.py \
    inference/qwenvl_hermes.py \
    video_qa/base.py \
    scripts/test_llava_attention.py \
    > "$output_root/source_with_top_attention.patch"

run_one 0 4000 &
pid_4000=$!
run_one 1 6000 &
pid_6000=$!
trap 'kill "$pid_4000" "$pid_6000" 2>/dev/null || true' INT TERM

status=0
wait "$pid_4000" || status=1
wait "$pid_6000" || status=1
if (( status != 0 )); then
    exit "$status"
fi

"$python_bin" - "$output_root" "$baseline_root" <<'PY'
import csv
import pathlib
import sys

import pandas as pd

output = pathlib.Path(sys.argv[1])
baseline = pathlib.Path(sys.argv[2])
rows = []
for kv_size in (4000, 6000):
    frame = pd.read_csv(
        output / f"top_attention_patch/fps0.5-kv{kv_size}/min-k1/results.csv"
    )
    k0 = pd.read_csv(baseline / f"fps0.5-kv{kv_size}/min-k0/results.csv")
    mean = pd.read_csv(baseline / f"fps0.5-kv{kv_size}/min-k1/results.csv")
    keys = ["video_id", "question", "answer", "task"]
    if not frame[keys].equals(mean[keys]):
        raise ValueError(f"Sample order mismatch for KV={kv_size}")
    accuracy = frame["qa_acc"].mean()
    rows.append({
        "strategy": "top_attention_patch",
        "kv_size": kv_size,
        "samples": len(frame),
        "accuracy": f"{accuracy:.4f}",
        "delta_vs_k0": f"{accuracy - k0['qa_acc'].mean():.4f}",
        "delta_vs_mean": f"{accuracy - mean['qa_acc'].mean():.4f}",
        "different_predictions_vs_mean": int(
            (frame["pred_answer"] != mean["pred_answer"]).sum()
        ),
    })

with (output / "comparison_top_attention.csv").open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
PY

echo "[$(timestamp)] top_attention_patch evaluations completed"
