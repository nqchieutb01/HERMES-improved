# TimeLens-Bench protocol, QVHighlights-TimeLens subset (Qwen3-VL-8B)

Setup follows https://github.com/TencentARC/TimeLens: TimeLens-Bench annotations and videos, their prompt
("Please find the visual event described by the sentence '...' ... 'The event happens in <start time> - <end time>
seconds'"), greedy decoding, 512 new tokens, 2 fps sampling with a 14,336-token video budget (min 64 tokens per frame),
their `extract_time` parser, 1 s rounding and `compute_metrics.py`. Longest dataset by mean duration:
QVHighlights-TimeLens (149.6 s). Subset: every query whose video is in video shard 01 — 513 queries, 506 videos
(no sampling). Native resolution 534x300; the protocol resizes frames to 416x224 (91 tokens per frame pair).

Validation: TimeLens' own script (unmodified) gives 60.5 mIoU on the subset (they report 59.4 on the full set);
our pipeline, unpruned, under the same protocol gives 60.5 (paired difference -0.0 [-1.3, +1.3]).

| Budget | Method | Visual tokens | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | ΔmIoU vs budget baseline |
|---|---|---|---|---|---|---|---|
| full | Qwen3-VL-8B, TimeLens script (anchor) | 13,650 | 75.8 | 64.7 | 50.7 | 60.5 | – |
| full | Ours, unpruned (bridge) | 13,650 | 77.0 | 64.9 | 49.5 | 60.5 | -0.0 [-1.3, +1.3] |
| ~4.2k | TimeLens script 2 fps, lower resolution | 4,200 | 74.5 | 62.0 | 47.0 | 58.0 | – |
| ~4.2k | TimeLens script 1 fps | 4,500 | 72.1 | 58.7 | 41.3 | 54.5 | -3.5 [-6.1, -1.0] * |
| ~4.2k | Random, 300 frames | 3,271 | 74.7 | 60.8 | 47.0 | 58.2 | +0.1 [-2.0, +2.4] |
| ~4.2k | SemVID, 300 frames | 4,232 | 75.6 | 62.4 | 48.3 | 58.8 | +0.8 [-1.5, +3.1] |
| ~4.2k | **Cover, 300 frames** | 3,271 | 78.0 | 66.7 | 50.9 | **61.7** | +3.7 [+1.3, +6.0] * |
| ~4.2k | Cover + PCD, 300 frames | 3,271 | 79.1 | 66.9 | 50.9 | 61.6 | +3.6 [+1.1, +6.0] * |
| ~4.2k | Retrieve + random, 128 frames | 3,876 | 76.4 | 66.1 | 47.4 | 59.7 | +1.7 [-0.7, +4.3] |
| ~4.2k | **Retrieve + Cover, 128 frames** | 3,876 | 78.4 | 68.0 | 49.5 | **61.5** | +3.5 [+1.1, +6.0] * |
| ~4.2k | Retrieve + Cover + PCD, 128 frames | 3,876 | 80.1 | 68.4 | 51.5 | 61.2 | +3.2 [+0.7, +5.8] * |
| ~1.1k | TimeLens script 1 fps, lower resolution | 1,125 | 64.7 | 53.0 | 37.4 | 49.5 | – |
| ~1.1k | Random, 300 frames (timestamps fill the budget) | ~0 | 4.5 | 1.8 | 1.2 | 4.2 | -45.3 [-48.6, -42.2] * |
| ~1.1k | Retrieve + random, 64 frames | 897 | 73.5 | 60.8 | 43.1 | 55.5 | +6.0 [+3.4, +8.4] * |
| ~1.1k | **Retrieve + Cover, 64 frames** | 897 | 76.6 | 63.7 | 45.4 | **58.3** | +8.8 [+6.3, +11.3] * |
| ~1.1k | Retrieve + Cover + PCD, 64 frames | 897 | 78.2 | 64.7 | 45.0 | 57.8 | +8.3 [+5.7, +11.0] * |

Paired, same frames and budget:

| Comparison | ΔmIoU |
|---|---|
| 300 frames: Cover vs Random | +3.5 [+1.6, +5.4] * |
| 300 frames: Cover vs SemVID | +2.9 [+0.9, +4.9] * |
| 128 frames: Retrieve + Cover vs Retrieve + random | +1.8 [+0.6, +3.1] * |
| 64 frames: Retrieve + Cover vs Retrieve + random | +2.8 [+1.2, +4.5] * |
| PCD on Cover, 300 frames | -0.1 [-2.1, +1.9] |
| PCD on Retrieve + Cover, 128 frames | -0.3 [-2.0, +1.5] |

## Findings

1. Question-aware token selection (Cover) is the component that transfers: with 24% of the unpruned model's visual
   tokens it beats the unpruned model (61.7 vs 60.5) and the best simple budget baseline (feeding a lower-resolution
   video, 58.0) by +3.7*; it beats random (+3.5*) and SemVID (+2.9*) on identical input.
2. At a tight budget the gap widens: 64 retrieval frames + Cover keep 897 visual tokens (6.6% of unpruned, 80% of the
   baseline's) and reach 58.3 mIoU, +8.8* over the 1 fps low-resolution baseline (49.5) and 2.2 below unpruned.
3. PCD does not help under this prompt (-0.1, -0.3, -0.5; n.s.). It is active (it changes 76-89% of answers) but
   only the decimal of a time ("81.0" -> "81.2"): with a one-line answer the model's uncertainty sits in the decimal
   place, and TimeLens' 1 s rounding removes the change. Its earlier gains came with the timeline prompt, where it
   corrects listed moment times — so PCD's benefit depends on the answer format.
4. Budget accounting: Qwen3-VL writes ~9 timestamp / frame-boundary tokens per frame pair. At 300 frames and 8% of the
   tokens they fill the whole budget (random 8%: 4.2 mIoU), so frame count must shrink with the budget (as in
   phase 36). TimeLens' token budget does not count these tokens; ours does, so our rows use fewer visual tokens than
   the baselines they are compared with.

## Notes

- Retrieval index: SigLIP 2 at 1 fps on the 506 subset videos. With 300 protocol frames (0.5 s apart) retrieval is
  off (coverage gate); retrieval runs use 128 / 64 frames at native resolution (288x544, 153 tokens per pair).
- Environment: TimeLens pins torch 2.6; ours is torch 2.9, whose bf16 Conv3d made their script ~100x slower
  (Qwen3-VL patch embedding). `/nfs-stor/chieu.nguyen/venvs/timelens-extra/sitecustomize.py` swaps in the exactly
  equivalent linear layer (max abs difference 2e-6), as our own pipeline does; their code is otherwise unmodified
  (plus qwen-vl-utils 0.0.14 and a 3-function nncore stand-in on PYTHONPATH).
- Code: branch `worktree-p31-gate-pcdfix` commit 8b05267 (`grounding_prompt=timelens`, `frame_max_pixels`,
  PCD time slot for "happens in"). Runs, scripts and scores: `/home/chieu.nguyen/hermes_pool/timelens/`
  (`compare.py`, `score.py`, `runs/`), results `results/qwen3_vl_8b/timelens_qvh/`.

## Phase 38 (added 18:45): TimeLens-8B, and PCD under our timeline prompt

TimeLens-8B (fine-tuned Qwen3-VL-8B; reported 65.5 mIoU on full QVHighlights-TimeLens), same subset and protocol:

| TimeLens-8B | Visual tokens | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|---|---|---|---|---|---|
| TimeLens script (anchor) | 13,650 | 80.3 | 71.9 | 56.1 | 66.1 |
| Ours, unpruned | 13,650 | 81.7 | 73.1 | 57.5 | 67.1 |
| TimeLens script 2 fps, lower resolution | 4,200 | 78.9 | 68.8 | 54.0 | 64.3 |
| Random 31%, 300 frames | 3,271 | 79.7 | 70.4 | 55.4 | 65.2 |
| **Cover 31%, 300 frames** | 3,271 | 81.5 | 73.9 | 57.1 | **67.3** |
| TimeLens script 1 fps, lower resolution | 1,125 | 71.7 | 61.8 | 47.4 | 57.5 |
| **Retrieve + Cover 23%, 64 frames** | 897 | 82.5 | 71.2 | 53.2 | **64.3** |

PCD with our timeline prompt (Qwen3-VL-8B, 300 frames, 31% of tokens; scored on the final "Time:" line):

| Prompt / method | mIoU |
|---|---|
| timeline, unpruned | 30.4 |
| timeline, Random / + PCD + reserve | 33.4 / 43.9 |
| timeline, Cover / + PCD + reserve | 30.7 / 43.9 |
| TimeLens prompt, Cover (phase 37) | 61.7 |

Paired comparisons:

| Comparison | ΔmIoU |
|---|---|
| TimeLens-8B: our unpruned vs their anchor | +1.0 [-0.2, +2.3] |
| TimeLens-8B ~4.2k: Cover 31% vs TimeLens 2 fps | +3.0 [+1.1, +4.9] * |
| TimeLens-8B ~4.2k: Cover vs Random (same input) | +2.0 [+0.7, +3.4] * |
| TimeLens-8B ~4.2k: Cover 31% vs our unpruned | +0.2 [-0.8, +1.1] |
| TimeLens-8B ~1.1k: Retrieve + Cover 64 frames vs TimeLens 1 fps | +6.8 [+4.4, +9.6] * |
| Timeline prompt: Cover + PCD + reserve vs Cover | +13.2 [+10.2, +16.3] * |
| Timeline prompt: Random + PCD + reserve vs Random | +10.4 [+7.8, +13.2] * |
| Cover + PCD + reserve (timeline) vs Cover (TimeLens prompt) | -17.8 [-20.8, -14.7] * |

Findings:
5. Cover transfers to the strongest open VTG model: on TimeLens-8B it keeps full accuracy with 24% of the visual
   tokens (67.3 vs 67.1 unpruned; +3.0* over the low-resolution baseline at the same budget), and with 897 visual
   tokens it loses 2.8 points (64.3; +6.8* over the 1 fps baseline).
6. PCD + reserve repairs the timeline prompt (+10 to +13*: run-on "Seen:" lists with no final interval) but the
   timeline prompt is far worse for single-moment grounding than a direct answer (43.9 vs 61.7). On standard VTG,
   PCD is not a contribution; its gains (S-EMBER, our Charades/QVH/ANet copies) are tied to answers that list moments
   before the interval.
