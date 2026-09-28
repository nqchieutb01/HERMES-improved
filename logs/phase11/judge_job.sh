#!/bin/bash
# Official-protocol S-EMBER answer judge (logs/phase11/judge_grounding.py) on a 2-GPU node.
# Environment copied from scripts/slurm/rvs_ego_judge.sh: the judge venv's torch is built for CUDA 13,
# the nodes' driver supports CUDA 12.8, so NVIDIA's forward-compatibility libraries are preloaded.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname) start=$(date)
J=/l/users/chieu.nguyen/HERMES/.venv
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet" TRANSFORMERS_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=/nfs-stor/chieu.nguyen/.cache/vllm TRITON_CACHE_DIR=/nfs-stor/chieu.nguyen/.cache/triton
export OMP_NUM_THREADS=8 VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM tensor-parallel workers must be spawned, not forked, once CUDA is initialized.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME="$J/judge/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$J/cuda-compat-13/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$J/judge/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPUs", torch.cuda.device_count(), flush=True)'
"$J/judge/bin/python" -u logs/phase11/judge_grounding.py --runs logs/phase11/runs.txt "$@"
echo exit=$? end=$(date)
