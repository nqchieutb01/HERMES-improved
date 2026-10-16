#!/bin/bash
cd /home/chieu.nguyen/HERMES
echo host=$(hostname)
HERMES_QWEN3_ATTENTION_BACKEND=flash_attention_2 /nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3 logs/phase10/dump_frame_embeddings.py "$@"
echo exit=$?
