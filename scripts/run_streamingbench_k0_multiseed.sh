#!/usr/bin/env bash
set -euo pipefail

# Run two additional k=0 seeds for KV=4000 and KV=6000. All four jobs launch
# concurrently, with one job of each KV size assigned to each GPU.

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
base_dir="${STREAMINGBENCH_SEED_SAVE_DIR:-$repo_dir/results/$model/streamingbench_cr_su_eu/postfix-rope-budget-k0-multiseed-20260918}"
seed_2024_dir="${STREAMINGBENCH_SEED_2024_DIR:-$repo_dir/results/$model/streamingbench_cr_su_eu/postfix-rope-budget-20260918}"

mkdir -p "$base_dir"
controller_log="$base_dir/controller.log"
exec > >(tee -a "$controller_log") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

echo "[$(timestamp)] Starting four concurrent k=0 seeded runs"
echo "Seeds: 2025 2026"
echo "KV sizes: 4000 6000"
echo "Output: $base_dir"

if [[ ! -s "$anno_path" ]]; then
    echo "Annotation file not found or empty: $anno_path" >&2
    exit 1
fi
for kv_size in 4000 6000; do
    baseline="$seed_2024_dir/fps${sample_fps}-kv${kv_size}/min-k0/results.csv"
    if [[ ! -s "$baseline" ]]; then
        echo "Completed seed-2024 baseline not found: $baseline" >&2
        exit 1
    fi
done

run_one() {
    local gpu="$1"
    local kv_size="$2"
    local seed="$3"
    local save_dir="$base_dir/seed-${seed}/fps${sample_fps}-kv${kv_size}/min-k0"
    local chunk_file="$save_dir/1_0.csv"
    local results_path="$save_dir/results.csv"
    local inference_log="$save_dir/inference.log"
    local evaluation_log="$save_dir/evaluation.log"

    if [[ -s "$results_path" && -s "$save_dir/eval_results.txt" ]]; then
        echo "[$(timestamp)] GPU $gpu KV=$kv_size seed=$seed already complete; skipping"
        return 0
    fi
    if [[ -e "$results_path" || -e "$chunk_file" || -e "$inference_log" ]]; then
        echo "Refusing to overwrite incomplete output: $save_dir" >&2
        return 1
    fi

    mkdir -p "$save_dir"
    echo "[$(timestamp)] GPU $gpu starting KV=$kv_size k=0 seed=$seed"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED="$seed" \
        "$python_bin" -u video_qa/hermes_vqa.py \
        --model "$model" \
        --anno_path "$anno_path" \
        --save_dir "$save_dir" \
        --sample_fps "$sample_fps" \
        --kv_size "$kv_size" \
        --streaming true \
        --num_chunks 1 \
        --chunk_idx 0 \
        --encode_chunk_size "$encode_chunk_size" \
        --min_tokens_per_frame 0 \
        --seed "$seed" \
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

    echo "[$(timestamp)] GPU $gpu completed KV=$kv_size k=0 seed=$seed"
    tail -n 18 "$save_dir/eval_results.txt"
}

# Balance the KV sizes across GPUs while launching all configurations at once.
run_one 0 4000 2025 &
pid_4000_2025=$!
run_one 1 6000 2025 &
pid_6000_2025=$!
run_one 1 4000 2026 &
pid_4000_2026=$!
run_one 0 6000 2026 &
pid_6000_2026=$!

pids=("$pid_4000_2025" "$pid_6000_2025" "$pid_4000_2026" "$pid_6000_2026")
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
    echo "[$(timestamp)] At least one seeded run failed" >&2
    exit "$status"
fi

"$python_bin" - "$base_dir" "$seed_2024_dir" "$sample_fps" <<'PY'
import ast
import csv
import hashlib
import statistics
import sys
from pathlib import Path

base = Path(sys.argv[1])
seed_2024_base = Path(sys.argv[2])
sample_fps = sys.argv[3]
run_rows = []
task_rows = []

for kv_size in (4000, 6000):
    for seed in (2024, 2025, 2026):
        if seed == 2024:
            path = seed_2024_base / f"fps{sample_fps}-kv{kv_size}" / "min-k0" / "results.csv"
        else:
            path = base / f"seed-{seed}" / f"fps{sample_fps}-kv{kv_size}" / "min-k0" / "results.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))

        accuracy = sum(float(row["qa_acc"]) for row in results) / len(results)
        errors = 0
        predictions = []
        for row in results:
            choices = ast.literal_eval(row["choices"])
            valid = {chr(ord("A") + index) for index in range(len(choices))}
            prediction = row["pred_choice"].strip()
            predictions.append(prediction)
            if not row["pred_answer"].strip() or prediction not in valid:
                errors += 1
        prediction_hash = hashlib.sha256("\n".join(predictions).encode()).hexdigest()
        run_rows.append({
            "kv_size": kv_size,
            "seed": seed,
            "samples": len(results),
            "accuracy": f"{accuracy:.4f}",
            "invalid_predictions": errors,
            "prediction_sha256": prediction_hash,
            "results_path": str(path),
        })

        for task in sorted({row["task"] for row in results}):
            selected = [row for row in results if row["task"] == task]
            task_accuracy = sum(float(row["qa_acc"]) for row in selected) / len(selected)
            task_rows.append({
                "kv_size": kv_size,
                "seed": seed,
                "task": task,
                "samples": len(selected),
                "accuracy": f"{task_accuracy:.4f}",
            })

summary_rows = []
for kv_size in (4000, 6000):
    selected = [row for row in run_rows if row["kv_size"] == kv_size]
    accuracies = [float(row["accuracy"]) for row in selected]
    hashes = {row["prediction_sha256"] for row in selected}
    summary_rows.append({
        "kv_size": kv_size,
        "seeds": "2024,2025,2026",
        "mean_accuracy": f"{statistics.mean(accuracies):.4f}",
        "sample_std_accuracy": f"{statistics.stdev(accuracies):.4f}",
        "min_accuracy": f"{min(accuracies):.4f}",
        "max_accuracy": f"{max(accuracies):.4f}",
        "identical_predictions_across_seeds": int(len(hashes) == 1),
    })

outputs = (
    ("seed_results.csv", run_rows),
    ("seed_task_results.csv", task_rows),
    ("seed_summary.csv", summary_rows),
)
for filename, rows in outputs:
    with (base / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print("Seeded-run summary:")
for row in summary_rows:
    print(row)
PY

echo "[$(timestamp)] All four seeded runs and three-seed summary completed"
