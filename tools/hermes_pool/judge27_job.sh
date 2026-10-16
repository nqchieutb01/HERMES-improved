#!/bin/bash
# Phase 27 fresh judge (copy of logs/phase27/judge_fresh_job.sh) with node-local Triton/vLLM caches: on 2026-10-13
# vLLM's profiling compiled one kernel every 10.5 min against the NFS cache (2 h, job hit its limit). The NFS cache is
# copied to local disk first, and new kernels are copied back at the end.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname) start=$(date)
J=/nfs-stor/chieu.nguyen/HERMES/.venv
[ -x "$J/judge/bin/python" ] || { echo "judge venv missing on $(hostname)"; exit 3; }
NFS_TRITON=/nfs-stor/chieu.nguyen/.cache/triton
LOCAL=${TMPDIR:-/tmp}/judge27-$SLURM_JOB_ID
mkdir -p $LOCAL/triton $LOCAL/vllm
cp -r $NFS_TRITON/. $LOCAL/triton/
# Triton's __grp__ index files store absolute paths; old entries pointed into /l/users, which hangs (autofs) on some
# nodes for ~10 min per lookup. Point them at the local copy.
grep -rl "/.cache/triton" $LOCAL/triton | xargs -r sed -i -E "s#/(l/users|nfs-stor)/chieu.nguyen/.cache/triton#$LOCAL/triton#g"
echo "triton cache copied: $(ls $LOCAL/triton | wc -l) entries $(date +%T)"
export HF_HOME=/nfs-stor/chieu.nguyen/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet" TRANSFORMERS_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=$LOCAL/vllm TRITON_CACHE_DIR=$LOCAL/triton
export OMP_NUM_THREADS=8 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME="$J/judge/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$J/cuda-compat-13/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$J/judge/bin/python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPUs", torch.cuda.device_count(), flush=True)'
"$J/judge/bin/python" -u logs/phase22/judge_fresh.py --runs logs/phase27/judge_runs_fresh.txt --out-name answer_judgments_fresh27.jsonl
rc=$?
grep -rl "$LOCAL/triton" $LOCAL/triton | xargs -r sed -i "s#$LOCAL/triton#$NFS_TRITON#g"
cp -rn $LOCAL/triton/. $NFS_TRITON/
rm -rf $LOCAL
echo exit=$rc end=$(date)
