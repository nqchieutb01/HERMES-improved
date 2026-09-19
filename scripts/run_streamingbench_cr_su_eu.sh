#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${STREAMINGBENCH_REPO_DIR:-}" ]]; then
    repo_dir="$STREAMINGBENCH_REPO_DIR"
else
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_dir"

hermes_env="$repo_dir/hermes"
if [[ ! -f "$hermes_env/bin/activate" ]]; then
    echo "Hermes environment not found: $hermes_env/bin/activate" >&2
    exit 1
fi
# This launcher runs directly on the local machine; it does not require Slurm.
source "$hermes_env/bin/activate"

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

python_bin="${STREAMINGBENCH_PYTHON:-python}"
dataset_root="${STREAMINGBENCH_ROOT:-$repo_dir/data/streamingbench/realtime_cr_su_eu}"
model="${STREAMINGBENCH_MODEL:-llava_ov_0.5b}"
sample_fps="${STREAMINGBENCH_SAMPLE_FPS:-0.5}"
kv_size="${STREAMINGBENCH_KV_SIZE:-6000}"
num_chunks="${STREAMINGBENCH_NUM_CHUNKS:-1}"
debug="${STREAMINGBENCH_DEBUG:-false}"
encode_chunk_size="${STREAMINGBENCH_ENCODE_CHUNK_SIZE:-16}"
min_tokens_sweep="${STREAMINGBENCH_MIN_TOKENS_SWEEP:-1}"
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
skip_download="${STREAMINGBENCH_SKIP_DOWNLOAD:-}"
if [[ -z "$skip_download" ]]; then
    if [[ -s "$anno_path" ]]; then
        skip_download=true
    else
        skip_download=false
    fi
fi
chunk_idx=0

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python interpreter not found in the hermes environment: $python_bin" >&2
    exit 1
fi
if [[ "$num_chunks" != 1 ]]; then
    echo "This local one-GPU script requires STREAMINGBENCH_NUM_CHUNKS=1" >&2
    exit 1
fi

case "$model" in
    llava_ov_0.5b)
        model_path="$repo_dir/models/llava-onevision-qwen2-0.5b-ov-hf"
        ;;
    llava_ov_7b)
        model_path="$repo_dir/models/llava-onevision-qwen2-7b-ov-hf"
        ;;
    llava_ov_72b)
        model_path="$repo_dir/models/llava-onevision-qwen2-72b-ov-hf"
        ;;
    qwen2.5_vl_3b)
        model_path="$repo_dir/models/Qwen2.5-VL-3B-Instruct"
        ;;
    qwen2.5_vl_7b)
        model_path="$repo_dir/models/Qwen2.5-VL-7B-Instruct"
        ;;
    qwen2.5_vl_32b)
        model_path="$repo_dir/models/Qwen2.5-VL-32B-Instruct"
        ;;
    *)
        echo "Unsupported StreamingBench model: $model" >&2
        echo "Use one of the model names supported by video_qa/base.py." >&2
        exit 1
        ;;
esac
if [[ ! -d "$model_path" ]]; then
    echo "Model directory not found: $model_path" >&2
    echo "Download the selected model into models/ before running inference." >&2
    exit 1
fi

if [[ "$skip_download" == "true" ]]; then
    echo "Skipping dataset preparation because STREAMINGBENCH_SKIP_DOWNLOAD=true"
else
    echo "Preparing StreamingBench CR/SU/EU data under $dataset_root"
    "$python_bin" scripts/download_streamingbench_tasks.py \
        --tasks CR SU EU \
        --output "$dataset_root"
fi
if [[ ! -s "$anno_path" ]]; then
    echo "CR/SU/EU annotation not found or empty after preparation: $anno_path" >&2
    exit 1
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
