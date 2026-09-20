#!/usr/bin/env bash
set -euo pipefail

# Post-fix A/B validation for the LLaVA-OneVision k=1 frame floor.
# GPU 0 runs KV=4000 and GPU 1 runs KV=6000 concurrently. Each worker runs
# k=0 first and then k=1, using identical inference settings and no token-trace
# CSV (the trace is not needed for accuracy and is close to 1 GB per run).

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
gpu_wait_seconds="${STREAMINGBENCH_GPU_WAIT_SECONDS:-60}"
gpu_idle_memory_mb="${STREAMINGBENCH_GPU_IDLE_MEMORY_MB:-1024}"
wait_for_idle="${STREAMINGBENCH_WAIT_FOR_IDLE:-true}"

mkdir -p "$base_dir"
controller_log="$base_dir/controller.log"
exec > >(tee -a "$controller_log") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

echo "[$(timestamp)] Starting StreamingBench k=0/k=1 post-fix validation"
echo "Repository: $repo_dir"
echo "Output: $base_dir"
echo "Model: $model; sample_fps=$sample_fps; encode_chunk_size=$encode_chunk_size"
echo "Wait for idle GPUs: $wait_for_idle"
echo "Git revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python interpreter not found after activating hermes: $python_bin" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for GPU coordination" >&2
    exit 1
fi
if [[ ! -s "$anno_path" ]]; then
    echo "Annotation file not found or empty: $anno_path" >&2
    exit 1
fi
if [[ "$wait_for_idle" != "true" && "$wait_for_idle" != "false" ]]; then
    echo "STREAMINGBENCH_WAIT_FOR_IDLE must be true or false" >&2
    exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( gpu_count < 2 )); then
    echo "This runner requires two visible GPUs; found $gpu_count" >&2
    exit 1
fi

wait_for_idle_gpu() {
    local gpu="$1"
    local checks=0
    local memory_used utilization

    while true; do
        memory_used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
        utilization="$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
        if [[ "$memory_used" =~ ^[0-9]+$ ]] && (( memory_used < gpu_idle_memory_mb )); then
            # Require two consecutive idle samples to avoid grabbing a GPU in
            # a short gap between another process's teardown/startup steps.
            sleep 5
            memory_used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
            if [[ "$memory_used" =~ ^[0-9]+$ ]] && (( memory_used < gpu_idle_memory_mb )); then
                echo "[$(timestamp)] GPU $gpu is idle (${memory_used} MiB used); starting its sweep"
                return 0
            fi
        fi

        if (( checks % 10 == 0 )); then
            echo "[$(timestamp)] GPU $gpu busy: ${memory_used} MiB, ${utilization}% utilization; waiting"
        fi
        checks=$((checks + 1))
        sleep "$gpu_wait_seconds"
    done
}

run_one() {
    local gpu="$1"
    local kv_size="$2"
    local min_tokens="$3"
    local save_dir="$base_dir/fps${sample_fps}-kv${kv_size}/min-k${min_tokens}"
    local chunk_file="$save_dir/1_0.csv"
    local results_path="$save_dir/results.csv"
    local inference_log="$save_dir/inference.log"
    local evaluation_log="$save_dir/evaluation.log"

    if [[ -s "$results_path" && -s "$save_dir/eval_results.txt" ]]; then
        echo "[$(timestamp)] GPU $gpu KV=$kv_size k=$min_tokens already complete; skipping"
        return 0
    fi
    if [[ -e "$results_path" || -e "$chunk_file" || -e "$inference_log" ]]; then
        echo "Refusing to overwrite incomplete output: $save_dir" >&2
        return 1
    fi

    mkdir -p "$save_dir"
    echo "[$(timestamp)] GPU $gpu starting KV=$kv_size k=$min_tokens"
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
        --min_tokens_per_frame "$min_tokens" \
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

    echo "[$(timestamp)] GPU $gpu completed KV=$kv_size k=$min_tokens"
    tail -n 18 "$save_dir/eval_results.txt"
}

run_kv_sweep() {
    local gpu="$1"
    local kv_size="$2"
    if [[ "$wait_for_idle" == "true" ]]; then
        wait_for_idle_gpu "$gpu"
    else
        echo "[$(timestamp)] GPU $gpu idle wait disabled; starting despite existing utilization"
    fi
    run_one "$gpu" "$kv_size" 0
    run_one "$gpu" "$kv_size" 1
}

run_kv_sweep 0 4000 &
pid_4000=$!
run_kv_sweep 1 6000 &
pid_6000=$!

terminate_workers() {
    kill "$pid_4000" "$pid_6000" 2>/dev/null || true
}
trap terminate_workers INT TERM

status=0
if ! wait "$pid_4000"; then
    echo "[$(timestamp)] KV=4000 worker failed" >&2
    status=1
fi
if ! wait "$pid_6000"; then
    echo "[$(timestamp)] KV=6000 worker failed" >&2
    status=1
fi
if (( status != 0 )); then
    exit "$status"
fi

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
    by_k = {}
    for min_tokens in (0, 1):
        path = base / f"fps{sample_fps}-kv{kv_size}" / f"min-k{min_tokens}" / "results.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
        accuracy = sum(float(row["qa_acc"]) for row in results) / len(results)
        errors = 0
        for row in results:
            choices = ast.literal_eval(row["choices"])
            valid = {chr(ord("A") + index) for index in range(len(choices))}
            if not row["pred_answer"].strip() or row["pred_choice"].strip() not in valid:
                errors += 1
        by_k[min_tokens] = accuracy
        rows.append({
            "kv_size": kv_size,
            "min_tokens_per_frame": min_tokens,
            "samples": len(results),
            "accuracy": f"{accuracy:.4f}",
            "invalid_predictions": errors,
            "delta_vs_k0": "" if min_tokens == 0 else f"{accuracy - by_k[0]:.4f}",
        })

        tasks = sorted({row["task"] for row in results})
        for task in tasks:
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

echo "[$(timestamp)] All runs and comparisons completed successfully"
