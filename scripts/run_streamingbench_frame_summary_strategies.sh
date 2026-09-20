#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -x "$repo_dir/hermes/bin/python" ]]; then
    echo "Hermes Python environment not found" >&2
    exit 1
fi

python_bin="$repo_dir/hermes/bin/python"
annotation="$repo_dir/data/streamingbench/realtime_cr_su_eu/streamingbench_realtime_cr_su_eu.json"
output_root="${STREAMINGBENCH_STRATEGY_SAVE_DIR:-$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/frame-summary-strategies-20260919}"
baseline_root="$repo_dir/results/llava_ov_0.5b/streamingbench_cr_su_eu/postfix-rope-budget-20260918"
temperature="${STREAMINGBENCH_FRAME_SUMMARY_TEMPERATURE:-0.1}"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

if [[ ! -s "$annotation" ]]; then
    echo "Annotation file is missing: $annotation" >&2
    exit 1
fi
mkdir -p "$output_root"

git rev-parse HEAD > "$output_root/base_commit.txt"
git diff -- \
    inference/abstract_hermes.py \
    inference/llavaov_hermes.py \
    inference/qwenvl_hermes.py \
    video_qa/base.py \
    scripts/test_llava_attention.py \
    scripts/run_streamingbench_frame_summary_strategies.sh \
    > "$output_root/source.patch"

echo "[$(timestamp)] Starting frame-summary strategy sweep"
echo "Output: $output_root"
echo "Strategies: top_patch attention_weighted softmax_score_weighted"
echo "KV sizes: 4000 6000"
echo "Softmax score temperature: $temperature"

run_one() {
    local gpu="$1"
    local strategy="$2"
    local kv_size="$3"
    local save_dir="$output_root/$strategy/fps0.5-kv${kv_size}/min-k1"
    local chunk_path="$save_dir/1_0.csv"
    local results_path="$save_dir/results.csv"

    if [[ -s "$results_path" && -s "$save_dir/eval_results.txt" ]]; then
        echo "[$(timestamp)] GPU $gpu $strategy KV=$kv_size already complete; skipping"
        return 0
    fi
    if [[ -e "$chunk_path" || -e "$results_path" || -e "$save_dir/inference.log" ]]; then
        echo "Refusing to overwrite incomplete output: $save_dir" >&2
        return 1
    fi
    mkdir -p "$save_dir"

    echo "[$(timestamp)] GPU $gpu starting $strategy KV=$kv_size"
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
        --frame_summary_strategy "$strategy" \
        --frame_summary_temperature "$temperature" \
        --seed 2024 \
        --debug false \
        > "$save_dir/inference.log" 2>&1

    if [[ ! -s "$chunk_path" ]]; then
        echo "Inference did not create $chunk_path" >&2
        return 1
    fi
    cp "$chunk_path" "$results_path"
    "$python_bin" eval/eval_multiple_choice.py general \
        --results_path "$results_path" \
        > "$save_dir/evaluation.log" 2>&1
    echo "[$(timestamp)] GPU $gpu completed $strategy KV=$kv_size"
}

# GPU 0 already hosts a large external process and one baseline evaluation, so
# add only one worker there. GPU 1 has enough room for the other five models.
run_one 0 top_patch 4000 &
pids=("$!")
labels=("top_patch-kv4000")
for config in \
    "top_patch 6000" \
    "attention_weighted 4000" \
    "attention_weighted 6000" \
    "softmax_score_weighted 4000" \
    "softmax_score_weighted 6000"
do
    read -r strategy kv_size <<< "$config"
    run_one 1 "$strategy" "$kv_size" &
    pids+=("$!")
    labels+=("${strategy}-kv${kv_size}")
done

terminate_workers() {
    kill "${pids[@]}" 2>/dev/null || true
}
trap terminate_workers INT TERM

status=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "[$(timestamp)] Worker failed: ${labels[$index]}" >&2
        status=1
    fi
done
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
task_rows = []
for kv_size in (4000, 6000):
    references = {}
    for minimum in (0, 1):
        path = baseline / f"fps0.5-kv{kv_size}/min-k{minimum}/results.csv"
        references[minimum] = pd.read_csv(path)

    for strategy in (
        "top_patch",
        "attention_weighted",
        "softmax_score_weighted",
    ):
        path = output / strategy / f"fps0.5-kv{kv_size}/min-k1/results.csv"
        frame = pd.read_csv(path)
        keys = ["video_id", "question", "answer", "task"]
        if not frame[keys].equals(references[1][keys]):
            raise ValueError(
                f"Sample order mismatch for {strategy}, KV={kv_size}"
            )
        accuracy = float(frame["qa_acc"].mean())
        rows.append({
            "strategy": strategy,
            "kv_size": kv_size,
            "samples": len(frame),
            "accuracy": f"{accuracy:.4f}",
            "delta_vs_k0": f"{accuracy - references[0]['qa_acc'].mean():.4f}",
            "delta_vs_mean": f"{accuracy - references[1]['qa_acc'].mean():.4f}",
            "different_predictions_vs_mean": int(
                (frame["pred_answer"] != references[1]["pred_answer"]).sum()
            ),
        })
        for task, group in frame.groupby("task"):
            task_rows.append({
                "strategy": strategy,
                "kv_size": kv_size,
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

echo "[$(timestamp)] Strategy sweep and comparison completed"
