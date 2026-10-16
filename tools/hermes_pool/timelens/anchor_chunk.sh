#!/bin/bash
# One chunk of TimeLens' own evaluation (evaluation/eval_dataloader.py, unmodified) on the shard-01 subset.
# Args: <tag> <fps> <total_tokens> <min_tokens> <chunks> <index> [model path, default Qwen3-VL-8B-Instruct]. Output: runs/<tag>/qvhighlights-timelens_<index>.jsonl
set -e
TAG=$1 FPS=$2 TOTAL=$3 MIN=$4 CHUNKS=$5 IDX=$6 MODEL=${7:-/nfs-stor/chieu.nguyen/models/Qwen3-VL-8B-Instruct}
T=/home/chieu.nguyen/hermes_pool/timelens
cd $T/repo
export PYTHONPATH=/nfs-stor/chieu.nguyen/venvs/timelens-extra:$T/repo HF_HUB_OFFLINE=1
mkdir -p $T/runs/$TAG
/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3 -u evaluation/eval_dataloader.py --dataset qvhighlights-timelens \
  --pred_path $T/runs/$TAG/qvhighlights-timelens --model_path $MODEL \
  --min_tokens $MIN --total_tokens $TOTAL --fps $FPS --chunk $CHUNKS --index $IDX
