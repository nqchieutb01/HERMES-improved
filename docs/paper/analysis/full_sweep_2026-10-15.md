# Full TimeLens-Bench sweep: Cover 31% vs Random 31% (one-line TimeLens prompt), scored 2026-10-15 ~15:40

Script: `score_full.py` (this dir); raw numbers: `full_sweep.json`. The scoring is copied from
`docs/paper/analysis/collect_2026-10-15.py`. It applies TimeLens `convert()` and `compute_metrics` to the full reply: first pair, 1 s rounding, and a failed parse becomes [dur+10, dur+20]. The unpruned reference is the TL-script anchor `hermes_pool/timelens/runs_full/qwen3/<ds>/anchor/all.jsonl`.
Only finished chunk CSVs (`<n>_<i>.csv`) were read. Rows were deduplicated by `question_id`: there were **0 duplicates** in every cell, and distinct TimeLens keys equal the distinct question ids. Bootstrap: paired, 2,000 resamples over queries, `default_rng(0)`, 95% percentile CI, `*` = CI excludes 0.

Settings come from the inference logs. They are identical across all chunks of a cell, and Cover and Random differ only in `prune_score` (core_topk vs random):
input 2 fps, `protocol_total_tokens=14336`, min 64 tokens/frame, `frame_max_pixels=0`, `grounding_prompt=timelens`, `offline_keep_ratio=0.31`, `answer_reserve=0`, no contrastive decoding, `keep_time_tokens=surviving`, `prune_seed=0`.

## 1. Cell status and absolute scores (Qwen3-VL-8B)

| Dataset | Selection | Distinct q / expected | Chunks done | Complete | R@0.3 | R@0.5 | R@0.7 | mIoU | Unparsed | Batch | Visual tokens kept / total (mean per query) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Charades | Cover 31 | 3,363 / 3,363 | 12/12 | **yes** | 69.82 | 53.97 | 27.89 | 48.65 | 0 | 8 | 3,348 / 10,803 (31.0%; min 508, max 4,515) |
| Charades | Random 31 | 3,363 / 3,363 | 12/12 | **yes** | 68.42 | 52.33 | 28.31 | 48.19 | 0 | 8 | 3,348 / 10,803 (31.0%) |
| QVH | Cover 31 | 1,541 / 1,541 | 6/6 | **yes** | 73.91 | 64.50 | 49.38 | 59.68 | 0 | 8 | 4,652 / 15,009 (31.0%; min 4,057, max 4,693) |
| QVH | Random 31 | 1,030 / 1,541 | 4/6 (6_0–6_3; 6_4, 6_5 running) | no (66.8%) | 72.23 | 63.20 | 49.13 | 58.87 | 0 | 8 | 4,651 / 15,006 (31.0%) |
| ActivityNet | Cover 31 | 219 / 4,500 | 1/18 (18_0) | no (4.9%) | 71.69 | 58.45 | 41.55 | 53.82 | 0 | 4 | 4,056 / 13,086 (31.0%; min 1,411, max 4,527) |
| ActivityNet | Random 31 | 0 / 4,500 | 0/18 | no | – | – | – | – | – | – | – |

- The TimeLens-8B dirs (`results/timelens_8b/timelens_*/full-t8-*`) exist for all 6 cells, but **all are empty** (0 chunks).
- Cross-check: the run directories' own `timelens_pred.log` files give the same numbers to rounding, for example Charades Cover mIoU 48.648 and Random 48.191, and QVH Cover 59.683.
- Token counts come from the per-query `Offline pruning: keeping X of Y visual tokens` lines. The QVH Random token stats include the two chunks still running (1,364 lines).
- Charades videos are short, so the mean pre-prune count is about 10.8k, under the 14,336 cap.

Unpruned TL-script anchor over the full set:

| Dataset | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---|---|---|
| Charades (3,363) | 69.43 | 54.03 | 27.74 | 48.51 |
| QVH (1,541) | 74.11 | 64.50 | 49.84 | 59.63 |
| ActivityNet (4,500) | 62.53 | 51.53 | 34.62 | 47.05 |

The anchor on the 219 ActivityNet Cover queries scores 71.69 / 60.27 / 42.01 / 54.97. ActivityNet Cover minus the anchor on mIoU is −1.1 [−2.8, +0.4], from §2d of the earlier summary; this does not change.

## 2. Paired comparison on the query intersection (Qwen3-VL-8B)

### Charades (n = 3,363, complete)

| | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---|---|---|
| Cover 31 | 69.82 | 53.97 | 27.89 | 48.65 |
| Random 31 | 68.42 | 52.33 | 28.31 | 48.19 |
| Unpruned (TL-script anchor) | 69.43 | 54.03 | 27.74 | 48.51 |
| Cover − Random | +1.40 [+0.56, +2.20]* | +1.64 [+0.77, +2.47]* | −0.42 [−1.13, +0.27] | +0.46 [+0.03, +0.88]* |
| Cover − Unpruned | +0.39 [−0.39, +1.16] | −0.06 [−0.86, +0.74] | +0.15 [−0.48, +0.83] | +0.13 [−0.23, +0.49] |
| Random − Unpruned | −1.01 [−1.90, −0.09]* | −1.69 [−2.59, −0.80]* | +0.56 [−0.18, +1.34] | −0.32 [−0.74, +0.10] |

### QVHighlights (n = 1,030 = Random chunks 0–3; Cover complete, Random 4/6)

| | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---|---|---|---|
| Cover 31 | 74.17 | 65.24 | 50.78 | 60.45 |
| Random 31 | 72.23 | 63.20 | 49.13 | 58.87 |
| Unpruned (TL-script anchor) | 74.37 | 65.44 | 51.65 | 60.59 |
| Cover − Random | +1.94 [+0.29, +3.50]* | +2.04 [+0.39, +3.79]* | +1.65 [+0.10, +3.40]* | +1.57 [+0.42, +2.84]* |
| Cover − Unpruned | −0.19 [−1.75, +1.26] | −0.19 [−1.65, +1.36] | −0.87 [−2.52, +0.58] | −0.14 [−1.31, +0.98] |
| Random − Unpruned | −2.14 [−4.08, −0.39]* | −2.23 [−4.17, −0.39]* | −2.52 [−4.47, −0.78]* | −1.72 [−3.10, −0.40]* |

### ActivityNet

There is no paired comparison: Random has 0 chunks and Cover has 1/18.

## 3. Reading

- **Charades is complete in both cells.** Cover is above Random at R@0.3 (+1.4), R@0.5 (+1.6) and mIoU (+0.46), and all three CIs exclude 0, though the mIoU CI only barely does. R@0.7 shows no difference.
  Cover is indistinguishable from the unpruned anchor on every metric, with all |Δ| ≤ 0.4. Random falls below unpruned at R@0.3 and R@0.5 but not significantly on mIoU.
- **QVH is partial (Random 4/6), so the result is provisional.** On the 1,030 shared queries, Cover beats Random by +1.6 mIoU and +2.0 R@0.5, and every metric is significant. Cover is within 0.2 mIoU of unpruned (n.s.), while Random is significantly below unpruned on every metric.
  The intersection is easier than the full QVH set: the anchor scores 60.59 here vs 59.63 on all 1,541. The full Cover cell's 59.68 should therefore not be compared with Random's 58.87 directly.
- The effect sizes are small. The project's noise note puts batch-size decoding noise at around ±2 pts on Acc. That note concerns S-EMBER accuracy, but it is a reason to read a +0.46 mIoU gap cautiously. Cover and Random here run at the same batch size (8) and the same configuration, so the paired comparison itself is matched.
