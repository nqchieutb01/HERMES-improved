#!/usr/bin/env bash
#SBATCH --job-name=streamingbench-subset
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=41G
# The one allocated GPU is reused for every k value in the sweep.
#SBATCH --output=slurm-streamingbench-subset-%j.out

set -euo pipefail

# sbatch can execute a copied script from a Slurm spool directory. Prefer the
# submitted repository, then the known workspace path, and finally the script
# location for local execution.
if [[ -n "${STREAMINGBENCH_REPO_DIR:-}" ]]; then
    repo_dir="$STREAMINGBENCH_REPO_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/video_qa/hermes_vqa.py" ]]; then
    repo_dir="$SLURM_SUBMIT_DIR"
elif [[ -f "/l/users/chieu.nguyen/HERMES/video_qa/hermes_vqa.py" ]]; then
    repo_dir="/l/users/chieu.nguyen/HERMES"
else
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_dir"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${STREAMINGBENCH_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in "$repo_dir/.venv/hermes/bin/python" "$repo_dir/.venv/gpt-oss/bin/python"; do
        if [[ -x "$candidate" ]]; then
            python_bin="$candidate"
            break
        fi
    done
fi
python_bin="${python_bin:-$(command -v python3 || true)}"
subset_root="${STREAMINGBENCH_SUBSET_ROOT:-$repo_dir/data/streamingbench/subset}"
anno_path="${STREAMINGBENCH_ANNO_PATH:-$subset_root/streamingbench_realtime_subset.json}"
model="${STREAMINGBENCH_MODEL:-llava_ov_0.5b}"
sample_fps="${STREAMINGBENCH_SAMPLE_FPS:-0.5}"
kv_size="${STREAMINGBENCH_KV_SIZE:-4000}"
debug="${STREAMINGBENCH_DEBUG:-false}"
encode_chunk_size="${STREAMINGBENCH_ENCODE_CHUNK_SIZE:-16}"
min_tokens_sweep="${STREAMINGBENCH_MIN_TOKENS_SWEEP:-1,0}"

# Set STREAMINGBENCH_MIN_TOKENS_PER_FRAME to run one k value instead of the
# sweep. Otherwise, every value in MIN_TOKENS_SWEEP runs sequentially inside
# this single Slurm allocation.
if [[ -n "${STREAMINGBENCH_MIN_TOKENS_PER_FRAME:-}" ]]; then
    min_tokens_sweep="$STREAMINGBENCH_MIN_TOKENS_PER_FRAME"
fi

IFS=',' read -r -a min_tokens_values <<< "$min_tokens_sweep"
if (( ${#min_tokens_values[@]} == 0 )); then
    echo "Minimum-token sweep is empty." >&2
    exit 1
fi

base_save_dir="${STREAMINGBENCH_SAVE_DIR:-$repo_dir/results/$model/streamingbench_subset/fps${sample_fps}-kv${kv_size}}"

for min_tokens_per_frame in "${min_tokens_values[@]}"; do
    if ! [[ "$min_tokens_per_frame" =~ ^[0-9]+$ ]]; then
        echo "Minimum tokens per frame must be a nonnegative integer: $min_tokens_per_frame" >&2
        exit 1
    fi
done

if [[ ! -x "$python_bin" ]]; then
    echo "Python interpreter not found or not executable: $python_bin" >&2
    exit 1
fi
if [[ ! -s "$anno_path" ]]; then
    echo "Subset annotation not found or empty: $anno_path" >&2
    echo "Set STREAMINGBENCH_SUBSET_ROOT or STREAMINGBENCH_ANNO_PATH." >&2
    exit 1
fi

"$python_bin" - "$anno_path" <<'PY'
import json
import pathlib
import sys

annotation_path = pathlib.Path(sys.argv[1])
records = json.loads(annotation_path.read_text())
if not records:
    raise SystemExit(f"Annotation is empty: {annotation_path}")

missing = []
questions = 0
for record in records:
    questions += len(record.get("conversations", []))
    video_path = pathlib.Path(record["video_path"])
    if not video_path.is_file():
        missing.append(str(video_path))

if missing:
    raise SystemExit("Missing subset video(s):\n" + "\n".join(missing))

print(
    f"Validated StreamingBench subset: {len(records)} video(s), "
    f"{questions} question(s)"
)
PY

for min_tokens_per_frame in "${min_tokens_values[@]}"; do
    save_dir="$base_save_dir/min-k${min_tokens_per_frame}"
    results_path="$save_dir/results.csv"
    token_trace_path="${STREAMINGBENCH_TOKEN_TRACE_PATH:-$save_dir/token_retention.csv}"
    chunk_file="$save_dir/1_0.csv"
    log_path="$save_dir/inference.log"

    mkdir -p "$save_dir"
    echo "Running $model on $(basename "$anno_path")"
    echo "  sample_fps=$sample_fps kv_size=$kv_size encode_chunk_size=$encode_chunk_size"
    echo "  min_tokens_per_frame=$min_tokens_per_frame"
    echo "  output=$save_dir"

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
        --token_trace_path "$token_trace_path" \
        --min_tokens_per_frame "$min_tokens_per_frame" \
        --debug "$debug" \
        2>&1 | tee "$log_path"

    if [[ ! -s "$chunk_file" ]]; then
        echo "Inference did not create a non-empty result: $chunk_file" >&2
        exit 1
    fi

    cp "$chunk_file" "$results_path"
    "$python_bin" eval/eval_multiple_choice.py general \
        --results_path "$results_path" \
        > "$save_dir/evaluation.log"

    echo "StreamingBench k=$min_tokens_per_frame results saved to $results_path"
done
