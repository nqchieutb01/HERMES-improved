# StreamingBench smoke test

The smoke test uses the smallest supported LLaVA-OneVision checkpoint, three
original StreamingBench videos, and six timestamped multiple-choice questions.
It covers streaming ingestion, KV compression, answer generation, CSV merging,
and local evaluation.

## Inputs

- Model: `models/llava-onevision-qwen2-0.5b-ov-hf`
- Annotations: `data/streamingbench/subset/streamingbench_realtime_subset.json`
- Videos: `data/streamingbench/subset/videos/`
- Provenance: `data/streamingbench/subset/manifest.json`

Re-download the selected videos with:

```bash
python scripts/data/download_streamingbench_subset.py
```

Large datasets and checkpoints should remain under `/nfs-stor/chieu.nguyen/`;
the local `data/` and `models/` paths may be symlinks.

## Run

Activate the LLaVA environment, then run from the repository root inside a GPU
allocation:

```bash
source .venv/hermes/bin/activate
python scripts/run.py experiment=streamingbench_smoke
```

The profile uses 0.5 sampled FPS and a 1,024-token visual-memory budget. It
writes predictions and evaluation under
`results/llava_ov_0.5b/streamingbench_subset/smoke-fps0.5-kv1024/`.

Inspect the fully resolved test configuration without using a GPU:

```bash
python scripts/run.py experiment=streamingbench_smoke --cfg job --resolve
```

To evaluate an existing smoke-test CSV without rerunning inference:

```bash
python scripts/run.py experiment=streamingbench_smoke run.mode=evaluate
```

Run CPU regression tests separately:

```bash
python -m unittest discover -s tests
```

Accuracy on this six-question subset is a pipeline check, not a benchmark
estimate.

## S-EMBER grounding smoke test

The S-EMBER smoke profile uses the first video referenced by the official
grounding JSONL and retains all three of its questions. The direct adapter
sorts those questions by their query timestamps before inference.

Inputs:

- Annotations: `/nfs-stor/chieu.nguyen/s-ember/sember_grounding.jsonl`
- Videos: `/nfs-stor/chieu.nguyen/s-ember/videos/`
- Model: `models/llava-onevision-qwen2-0.5b-ov-hf`

Run locally in a GPU allocation:

```bash
source .venv/hermes/bin/activate
python scripts/run.py experiment=sember_smoke
```

Or submit the provided single-GPU job:

```bash
sbatch scripts/slurm/sember_smoke.sh
```

Expected artifacts under
`results/llava_ov_0.5b/sember_grounding/smoke-fps0.2-kv1024/` are
`results.csv`, `sember_grounding_metrics.json`,
`sember_grounding_scored.jsonl`, and captured inference/evaluation logs.

## S-EMBER MCQ smoke test

The MCQ smoke profile reads the same videos and
`/nfs-stor/chieu.nguyen/s-ember/sember_mcq.jsonl`. It selects the first video
and both of its questions, asks for one A-E letter using the official prompt,
and computes deterministic MCQ accuracy.

Run locally in a GPU allocation:

```bash
source .venv/hermes/bin/activate
python scripts/run.py experiment=sember_mcq_smoke
```

Or submit the single-GPU job:

```bash
sbatch scripts/slurm/sember_mcq_smoke.sh
```

Expected artifacts under
`results/llava_ov_0.5b/sember_mcq/smoke-fps0.2-kv1024/` are `results.csv`,
`sember_mcq_metrics.json`, `sember_mcq_scored.jsonl`, and captured logs.

## S-EMBER focused MCQ smoke test

The focused profile filters the official MCQ JSONL to `time_duration`,
`counting_objects_events`, and `location_trace` before applying its 12-video
limit. The downloaded dataset yields 22 smoke questions and includes every
selected category.

Run locally in a GPU allocation:

```bash
source .venv/hermes/bin/activate
python scripts/run.py experiment=sember_mcq_time_count_location_smoke
```

Or submit the provided single-GPU job:

```bash
sbatch scripts/slurm/sember_mcq_time_count_location_smoke.sh
```

Expected artifacts are written under
`results/llava_ov_0.5b/sember_mcq/time-count-location-smoke-fps0.2-kv1024/`.
