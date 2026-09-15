# Full RVS-Ego parameter sweep

Partition: `long`; 2 GPUs. Every scored trial uses all 1,465 questions.
Fixed judge: Qwen3.8-27B-FP8, original HERMES semantic rubric.
Targets: accuracy >=53% AND mean score >=3.8.
Stop on both targets, 24 trials, or three consecutive failed trials.
The first eight trials isolate parameters; later trials combine changes around the best result.
This is exploratory benchmark tuning; judge differences remain.

`status.json` records live progress; `leaderboard.json` and `leaderboard.csv` retain all trials.
`source/` snapshots code; `plan.json` pins code hashes and judge settings.

Scheduler: `long` with `gpu-debug-qos` and `cscc-users` accepts three-hour allocations. The controller submits continuations after two hours at trial boundaries. Explicit `gpu-12` was rejected by Slurm.
