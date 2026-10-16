# hermes_pool: experiment runner, evaluation scripts and reports

Snapshot of `/home/chieu.nguyen/hermes_pool` (kept outside the repo while experiments run). Code only: queue state,
logs, cloned benchmark repos, model outputs and caches are not included. Paths inside the scripts are absolute
(cluster layout).

| Path | What |
| --- | --- |
| `pool.py`, `slots.json`, `resume_all.sh` | Slurm GPU pool: keeper submits 1/2-GPU jobs, workers pull chunks from a queue, managers (`pool.py run <spec>.json`) merge and score runs |
| `specs/` | Run specs (S-EMBER phases 30-36, 43 gold-time upper bounds), judge chains, data preparation (QVHighlights, ActivityNet) |
| `judge_job.sh`, `judge27_job.sh` | S-EMBER answer judge (local Qwen3.8-27B-FP8, official S-EMBER prompt) |
| `timelens/` | TimeLens-Bench: data preparation, TimeLens' own script runner (`anchor_full.sh`), scorer for our runs (`score.py`), phase 37/38/40 specs |
| `timelens/patches/` | Changes needed to run TimeLens' evaluation here: data-loader workers (memory), linear patch embedding equivalent to torch 2.9's slow bf16 Conv3d (`sitecustomize.py`) |
| `nextgqa/` | NExT-GQA: subset and prompts (`prepare*.py`), official-eval scorer (`score.py`: MCQ, LOVER joint, open-ended), open-ended LLM judge (`judge_open.py`, Video-ChatGPT/TOGA protocol), paired bootstrap (`paired.py`, `paired_open.py`, `ground_open.py`), phase 39/41/42 specs |
| `reports/` | Run log (`PLAN_20261015.md`), summaries and result tables |
