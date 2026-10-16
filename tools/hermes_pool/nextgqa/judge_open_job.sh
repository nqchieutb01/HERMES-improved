#!/bin/bash
# Open-ended NExT-GQA judge (judge_open.py; JOBS = jobs file), same environment as judge_job.sh:
# vLLM's profiling compiled one kernel every 10.5 min against the NFS cache (2 h, job hit its limit). The NFS cache is
# copied to local disk first, and new kernels are copied back at the end.
cd /home/chieu.nguyen/HERMES
echo host=$(hostname) start=$(date)
J=/nfs-stor/chieu.nguyen/HERMES/.venv
[ -x "$J/judge/bin/python" ] || { echo "judge venv missing on $(hostname)"; exit 3; }
NFS_TRITON=/nfs-stor/chieu.nguyen/.cache/triton
LOCAL=${TMPDIR:-/tmp}/judge-$SLURM_JOB_ID
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
"$J/judge/bin/python" -u /home/chieu.nguyen/hermes_pool/nextgqa/judge_open.py --jobs "$JOBS" --tp ${TP:-2}
rc=$?
grep -rl "$LOCAL/triton" $LOCAL/triton | xargs -r sed -i "s#$LOCAL/triton#$NFS_TRITON#g"
cp -rn $LOCAL/triton/. $NFS_TRITON/
rm -rf $LOCAL
echo exit=$rc end=$(date)
