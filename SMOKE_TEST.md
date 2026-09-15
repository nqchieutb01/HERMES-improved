# StreamingBench smoke test

This setup uses the smallest supported LLaVA-OneVision checkpoint, three original
StreamingBench videos, and six timestamped multiple-choice questions. It tests
streaming ingestion, KV compression, answer generation, and local evaluation.

## Environment on gpu-52

```bash
cd /nfs-stor/chieu.nguyen/HERMES
source .venv/hermes/bin/activate
export UV_CACHE_DIR=/nfs-stor/chieu.nguyen/.cache/uv
```

The uv environment is named `hermes` and uses Python 3.11.15 with
`requirements_llava.txt`, including the pinned Transformers source commit.
The shared cache avoids the home-directory quota. The managed Python interpreter
is under `/home/chieu.nguyen/.local/share/uv/python` on gpu-52.

## Inputs

- Model: `models/llava-onevision-qwen2-0.5b-ov-hf`, downloaded from
  `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (revision
  `74dd0bf867a4cda7950c17663794267c60cf4b40`).
- Annotations: `data/streamingbench/subset/streamingbench_realtime_subset.json`.
- Videos: `data/streamingbench/subset/videos/` (113,084,708 bytes total).
- Dataset provenance and archive checksums: `data/streamingbench/subset/manifest.json`.
- Re-download selected videos with `python scripts/download_streamingbench_subset.py`.

Original benchmark annotations and inference source files are preserved.

## Run

Within a Slurm allocation with one GPU visible:

```bash
cd /nfs-stor/chieu.nguyen/HERMES
bash scripts/run_streamingbench_smoke.sh
```

From SSH, join your existing allocation explicitly (replace the job ID as needed):

```bash
srun --jobid=238303 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gres=gpu:1 \
  bash /nfs-stor/chieu.nguyen/HERMES/scripts/run_streamingbench_smoke.sh
```

The runner uses 0.5 sampled FPS and a 1,024-token visual-memory budget. It saves
`results.csv`, `inference.log`, `evaluation.log`, and `eval_results.txt` under
`results/llava_ov_0.5b/streamingbench_subset/fps0.5-kv1024/`.
Re-running replaces this smoke-test output. Accuracy on six selected questions
is only a smoke-test observation, not a benchmark estimate.

## Verified result (2026-09-13)

- Ran on gpu-52 in Slurm job 238303, using its assigned A100 GPU 0.
- All three videos decoded at the selected timestamps; durations are 90.0,
  149.56, and 152.48 seconds. See `data/streamingbench/subset/decode_validation.json`.
- Inference and evaluation exited successfully: six predictions, six correct
  answers, zero malformed answer letters (100% on this tiny selected subset).
- Repeated compression retained 1,037 tokens per layer: 13 initial prompt tokens
  plus the 1,024-token visual budget. The longest selected prefix crossed
  multiple 16-frame chunks.
- Reported peak PyTorch GPU allocation was approximately 4.64 GiB.
- Dependency imports and processor loading passed. FlashAttention was not
  installed; the PyTorch SDPA path worked.
- `uv pip check` reports an upstream Decord wheel metadata warning: its embedded
  tag says `cp36-cp36m-manylinux2010_x86_64`. Decord imports and real video decoding
  passed under Python 3.11. One video also emitted H.264 decoder warnings during
  sampling, without interrupting the run.

The result directory includes `validation.json` with the output consistency
checks and run configuration. No existing inference source code was changed.
