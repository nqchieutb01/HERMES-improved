#!/usr/bin/env bash
set -euo pipefail

# Launch the post-fix k=1 runs concurrently with the already-running k=0
# baselines. GPU 0 handles KV=4000 and GPU 1 handles KV=6000.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -f "$repo_dir/hermes/bin/activate" ]]; then
    echo "Hermes environment not found: $repo_dir/hermes/bin/activate" >&2
    exit 1
fi
source "$repo_dir/hermes/bin/activate"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${STREAMINGBENCH_PYTHON:-python}"
model="${STREAMINGBENCH_MODEL:-llava_ov_0.5b}"
sample_fps="${STREAMINGBENCH_SAMPLE_FPS:-0.5}"
encode_chunk_size="${STREAMINGBENCH_ENCODE_CHUNK_SIZE:-16}"
dataset_root="${STREAMINGBENCH_ROOT:-$repo_dir/data/streamingbench/realtime_cr_su_eu}"
anno_path="$dataset_root/streamingbench_realtime_cr_su_eu.json"
base_dir="${STREAMINGBENCH_AB_SAVE_DIR:-$repo_dir/results/$model/streamingbench_cr_su_eu/postfix-rope-budget-20260918}"

mkdir -p "$base_dir"
controller_log="$base_dir/controller_k1_concurrent.log"
exec > >(tee -a "$controller_log") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

echo "[$(timestamp)] Starting concurrent k=1 runs"
echo "Output: $base_dir"

if [[ ! -s "$anno_path" ]]; then
    echo "Annotation file not found or empty: $anno_path" >&2
    exit 1
fi

run_k1() {
    local gpu="$1"
    local kv_size="$2"
    local save_dir="$base_dir/fps${sample_fps}-kv${kv_size}/min-k1"
    local chunk_file="$save_dir/1_0.csv"
    local results_path="$save_dir/results.csv"
    local inference_log="$save_dir/inference.log"
    local evaluation_log="$save_dir/evaluation.log"

    if [[ -s "$results_path" && -s "$save_dir/eval_results.txt" ]]; then
        echo "[$(timestamp)] GPU $gpu KV=$kv_size k=1 already complete; skipping"
        return 0
    fi
    if [[ -e "$results_path" || -e "$chunk_file" || -e "$inference_log" ]]; then
        echo "Refusing to overwrite incomplete output: $save_dir" >&2
        return 1
    fi

    mkdir -p "$save_dir"
    echo "[$(timestamp)] GPU $gpu starting KV=$kv_size k=1"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u video_qa/hermes_vqa.py \
        --model "$model" \
        --anno_path "$anno_path" \
        --save_dir "$save_dir" \
        --sample_fps "$sample_fps" \
        --kv_size "$kv_size" \
        --streaming true \
        --num_chunks 1 \
        --chunk_idx 0 \
        --encode_chunk_size "$encode_chunk_size" \
        --min_tokens_per_frame 1 \
        --debug false \
        > "$inference_log" 2>&1

    if [[ ! -s "$chunk_file" ]]; then
        echo "Inference did not create a non-empty result: $chunk_file" >&2
        return 1
    fi
    cp "$chunk_file" "$results_path"
    rm "$chunk_file"

    "$python_bin" eval/eval_multiple_choice.py general \
        --results_path "$results_path" \
        > "$evaluation_log" 2>&1

    echo "[$(timestamp)] GPU $gpu completed KV=$kv_size k=1"
    tail -n 18 "$save_dir/eval_results.txt"
}

run_k1 0 4000 &
pid_4000=$!
run_k1 1 6000 &
pid_6000=$!

terminate_workers() {
    kill "$pid_4000" "$pid_6000" 2>/dev/null || true
}
trap terminate_workers INT TERM

status=0
if ! wait "$pid_4000"; then
    echo "[$(timestamp)] KV=4000 k=1 worker failed" >&2
    status=1
fi
if ! wait "$pid_6000"; then
    echo "[$(timestamp)] KV=6000 k=1 worker failed" >&2
    status=1
fi
if (( status != 0 )); then
    exit "$status"
fi

# k=0 was started by the original controller. Wait for both baseline
# evaluations so this independent controller can always produce the final A/B
# tables, even if the original controller exits after finding k=1 in progress.
while true; do
    missing=0
    for kv_size in 4000 6000; do
        baseline_dir="$base_dir/fps${sample_fps}-kv${kv_size}/min-k0"
        if [[ ! -s "$baseline_dir/results.csv" || ! -s "$baseline_dir/eval_results.txt" ]]; then
            missing=1
        fi
    done
    if (( missing == 0 )); then
        break
    fi
    echo "[$(timestamp)] k=1 complete; waiting for k=0 evaluation(s)"
    sleep 60
done

"$python_bin" - "$base_dir" "$sample_fps" <<'PY'
import ast
import csv
import sys
from pathlib import Path

base = Path(sys.argv[1])
sample_fps = sys.argv[2]
rows = []
task_rows = []
for kv_size in (4000, 6000):
    baseline_accuracy = None
    for min_tokens in (0, 1):
        path = base / f"fps{sample_fps}-kv{kv_size}" / f"min-k{min_tokens}" / "results.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
        accuracy = sum(float(row["qa_acc"]) for row in results) / len(results)
        if min_tokens == 0:
            baseline_accuracy = accuracy
        errors = 0
        for row in results:
            choices = ast.literal_eval(row["choices"])
            valid = {chr(ord("A") + index) for index in range(len(choices))}
            if not row["pred_answer"].strip() or row["pred_choice"].strip() not in valid:
                errors += 1
        rows.append({
            "kv_size": kv_size,
            "min_tokens_per_frame": min_tokens,
            "samples": len(results),
            "accuracy": f"{accuracy:.4f}",
            "invalid_predictions": errors,
            "delta_vs_k0": "" if min_tokens == 0 else f"{accuracy - baseline_accuracy:.4f}",
        })

        for task in sorted({row["task"] for row in results}):
            selected = [row for row in results if row["task"] == task]
            task_accuracy = sum(float(row["qa_acc"]) for row in selected) / len(selected)
            task_rows.append({
                "kv_size": kv_size,
                "min_tokens_per_frame": min_tokens,
                "task": task,
                "samples": len(selected),
                "accuracy": f"{task_accuracy:.4f}",
            })

with (base / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with (base / "comparison_by_task.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=task_rows[0].keys())
    writer.writeheader()
    writer.writerows(task_rows)

print("Final comparison:")
for row in rows:
    print(row)
PY

echo "[$(timestamp)] Concurrent k=1 runs and final comparison completed"
