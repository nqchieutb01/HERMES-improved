# Continuous RVS-Ego rerun

This run restores the original HERMES evaluation grouping: 10 continuous video
streams, 1,465 questions, and the repository's exact question order and timestamps.
Each stream uses its longest available extracted frame prefix. KV memory and
conversation history persist between question timestamps in the same video.

Inference uses LLaVA-OneVision-0.5B, 0.5 FPS, a 6,000-token visual KV budget,
16-frame encoding chunks, FP16, and greedy decoding. Two GPU workers process
five independent streams each. Qwen3.8-27B-FP8 judging uses the same checkpoint,
prompt, and settings as the previous local evaluations.

The paper uses GPT-3.5-turbo-0125 and original video inputs. This run retains
extracted 1-FPS JPEG inputs and the local Qwen evaluator, so it is still not an
exact reproduction of the paper's numerical scoring protocol.

Submitted Slurm job: **240240**. Submission is not evidence of successful
completion; check the Slurm log and output reports.

The batch pipeline runs inference, validates coverage, evaluates binary answers,
runs the Qwen judge, and writes `comparison_vs_prefix_runs.json`. That final
comparison also verifies that conversation-conditioned compression activated
and the judge configuration matches both previous runs.

Files available before inference:

- `annotations.json`: exact original grouping/order/timestamps, mapped to local frames.
- `run_manifest.json`: configuration and frame-source provenance.
- `execution_manifest.json`: submitted job and source-code hashes.
- `frame_validation.json`: successful boundary-frame decoding checks.

Completion outputs:

- `results.csv`: all 1,465 predictions.
- `validation.json`: coverage and nonempty-prediction checks.
- `yes_no_evaluation.json`: exact yes/no subset evaluation.
- `qwen3.8_27b_fp8_judge/summary.json`: aggregate and per-task Qwen scores.
- `comparison_vs_prefix_runs.json`: comparison with prior KV1024/KV6072 runs.

Runner: `scripts/run_rvs_ego_continuous.sh`. It refuses to overwrite existing
prediction files. Earlier prefix-run results are preserved.
