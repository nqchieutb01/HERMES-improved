#!/usr/bin/env bash
#SBATCH --job-name=streamingbench-cr-su-eu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=41G
## #SBATCH --time=06:00:00
#SBATCH --output=slurm-streamingbench-cr-su-eu-%j.out
set -euo pipefail

if [[ -n "${STREAMINGBENCH_REPO_DIR:-}" ]]; then
    repo_dir="$STREAMINGBENCH_REPO_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/video_qa/hermes_vqa.py" ]]; then
    # sbatch may execute a copied script from /var/lib/slurm-llnl/...;
    # SLURM_SUBMIT_DIR still points to the submitted repository.
    repo_dir="$SLURM_SUBMIT_DIR"
elif [[ -f "/l/users/chieu.nguyen/HERMES/video_qa/hermes_vqa.py" ]]; then
    repo_dir="/l/users/chieu.nguyen/HERMES"
else
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$repo_dir"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${STREAMINGBENCH_PYTHON:-$repo_dir/.venv/hermes/bin/python}"
dataset_root="${STREAMINGBENCH_ROOT:-/nfs-stor/chieu.nguyen/StreamingBench}"
model="${STREAMINGBENCH_MODEL:-llava_ov_0.5b}"
sample_fps="${STREAMINGBENCH_SAMPLE_FPS:-0.5}"
kv_size="${STREAMINGBENCH_KV_SIZE:-4000}"
num_chunks="${STREAMINGBENCH_NUM_CHUNKS:-1}"
debug="${STREAMINGBENCH_DEBUG:-false}"
encode_chunk_size="${STREAMINGBENCH_ENCODE_CHUNK_SIZE:-16}"
min_tokens_sweep="${STREAMINGBENCH_MIN_TOKENS_SWEEP:-1,0}"
if [[ -n "${STREAMINGBENCH_MIN_TOKENS_PER_FRAME:-}" ]]; then
    min_tokens_sweep="$STREAMINGBENCH_MIN_TOKENS_PER_FRAME"
fi

IFS=',' read -r -a min_tokens_values <<< "$min_tokens_sweep"
if (( ${#min_tokens_values[@]} == 0 )); then
    echo "Minimum-token sweep is empty." >&2
    exit 1
fi
for min_tokens_per_frame in "${min_tokens_values[@]}"; do
    if ! [[ "$min_tokens_per_frame" =~ ^[0-9]+$ ]]; then
        echo "Minimum tokens per frame must be a nonnegative integer: $min_tokens_per_frame" >&2
        exit 1
    fi
done

base_save_dir="${STREAMINGBENCH_SAVE_DIR:-$repo_dir/results/$model/streamingbench_cr_su_eu/fps${sample_fps}-kv${kv_size}}"
anno_path="$dataset_root/streamingbench_realtime_cr_su_eu.json"
chunk_idx=0

if [[ ! -x "$python_bin" ]]; then
    echo "Python interpreter not found or not executable: $python_bin" >&2
    exit 1
fi
if [[ "$num_chunks" != 1 ]]; then
    echo "This one-GPU Slurm script requires STREAMINGBENCH_NUM_CHUNKS=1" >&2
    exit 1
fi

if [[ ! -s "$anno_path" ]]; then
    echo "CR/SU/EU annotation not found or empty: $anno_path" >&2
    exit 1
fi

if [[ "${STREAMINGBENCH_SKIP_DOWNLOAD:-true}" == "true" ]]; then
    echo "Skipping dataset preparation because STREAMINGBENCH_SKIP_DOWNLOAD=true"
else
    echo "Preparing StreamingBench CR/SU/EU data under $dataset_root"
    "$python_bin" scripts/download_streamingbench_tasks.py \
        --tasks CR SU EU \
        --output "$dataset_root"
fi

echo "Running $model on $(basename "$anno_path") with $num_chunks chunk(s)"
echo "Sequential k sweep: ${min_tokens_values[*]}"

for min_tokens_per_frame in "${min_tokens_values[@]}"; do
    save_dir="$base_save_dir/min-k${min_tokens_per_frame}"
    results_path="$save_dir/results.csv"
    token_trace_path="${STREAMINGBENCH_TOKEN_TRACE_PATH:-$save_dir/token_retention.csv}"
    log_path="$save_dir/inference_${num_chunks}_${chunk_idx}.log"
    first_chunk="$save_dir/${num_chunks}_0.csv"

    mkdir -p "$save_dir"
    echo "Starting k=$min_tokens_per_frame; log: $log_path"
    "$python_bin" -u video_qa/hermes_vqa.py \
        --model "$model" \
        --anno_path "$anno_path" \
        --save_dir "$save_dir" \
        --sample_fps "$sample_fps" \
        --kv_size "$kv_size" \
        --streaming true \
        --num_chunks "$num_chunks" \
        --chunk_idx "$chunk_idx" \
        --encode_chunk_size "$encode_chunk_size" \
        --token_trace_path "$token_trace_path" \
        --min_tokens_per_frame "$min_tokens_per_frame" \
        --debug "$debug" \
        > "$log_path" 2>&1

    if [[ ! -s "$first_chunk" ]]; then
        echo "Inference did not create a non-empty result: $first_chunk" >&2
        exit 1
    fi
    cp "$first_chunk" "$results_path"
    rm "$first_chunk"

    "$python_bin" eval/eval_multiple_choice.py general \
        --results_path "$results_path" \
        > "$save_dir/evaluation.log" 2>&1

    echo "StreamingBench CR/SU/EU k=$min_tokens_per_frame results saved to $results_path"
done
