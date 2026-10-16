#!/bin/bash
# One chunk of TimeLens' own evaluation (unmodified) on a full TimeLens-Bench dataset.
# Args: <model tag> <model path> <dataset> <run tag> <fps> <total_tokens> <min_tokens> <chunks> <index>
# Output: runs_full/<model tag>/<dataset>/<run tag>/<dataset>-timelens_<index>.jsonl
set -e
MT=$1 MODEL=$2 DS=$3 TAG=$4 FPS=$5 TOTAL=$6 MIN=$7 CHUNKS=$8 IDX=$9
T=/home/chieu.nguyen/hermes_pool/timelens
OUT=$T/runs_full/$MT/$DS/$TAG
cd $T/repo_full
export PYTHONPATH=/nfs-stor/chieu.nguyen/venvs/timelens-extra:$T/repo_full HF_HUB_OFFLINE=1
# Their 10 loader workers decoding long ActivityNet videos at native resolution exceed 48 GB; 4 changes only loading.
export TL_NUM_WORKERS=4
mkdir -p $OUT
/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3 -u evaluation/eval_dataloader.py --dataset $DS-timelens \
  --pred_path $OUT/$DS-timelens --model_path $MODEL \
  --min_tokens $MIN --total_tokens $TOTAL --fps $FPS --chunk $CHUNKS --index $IDX
