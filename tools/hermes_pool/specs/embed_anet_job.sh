#!/bin/bash
cd /home/chieu.nguyen/HERMES
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface HF_HUB_OFFLINE=1
/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3 -u /home/chieu.nguyen/hermes_pool/specs/embed_anet.py
