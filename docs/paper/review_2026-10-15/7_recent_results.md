# Results newer than the paper draft (as of 2026-10-15 ~11:50)

**What counts as "already in the draft".** The 10-14 00:00–04:12 runs are in the draft. I checked this: rescoring `qwen3_vl_8b/qvhighlights/random10-pcd-ar64` and `rc64-qtopk10-pcd-ar64` with the m:ss-aware parser gives 37.8/39.2 and 48.8/54.8, which match the draft's QVH rows. That batch covers the charades_sta, qvhighlights, activitynet and S-EMBER variants. Everything below is phase 37 or later (10-14 ~18:00 onward). None of it is in `6_results.tex`, which never mentions TimeLens or NExT-GQA.

**Sources.** Numbers come from the metric files in each run directory, unless a row is marked *(log)*. *(log)* means the number is copied from `/home/chieu.nguyen/hermes_pool/reports/{OVERNIGHT,PLAN}_20261015.md` and I did not re-score it. The paired bootstrap CIs (* = 95% CI excludes 0) come from those reports (`nextgqa/paired.py`). They are reported values; I did not re-run them.

---

## 1. NExT-GQA (test, 1,005 questions on 191 videos; 64 uniform frames; two-query protocol: VideoMind MCQ prompt plus a separate TimeLens-style grounding prompt)

Scorer: `hermes_pool/nextgqa/score.py`, using the official `eval_ground.py`. Spans are parsed with TimeLens `extract_time`, which is **m:ss-aware**: it reads HH:MM:SS and MM:SS before plain numbers (`timelens/repo/timelens/utils.py`). The videos are ~40 s, and every grounding reply I spot-checked was in "X - Y seconds" form. The joint-prompt parser (`parse_span`) reads plain numbers only, so "0:06" becomes "0.00 to 0.07", which is part of why the joint runs score low. Metrics come from `nextgqa_eval.json`. Ignore `sember_grounding_metrics.json` in these directories: it shows mIoU 0, parseable 0/2010, because the S-EMBER parser cannot read this format.

| Model | Method (keep) | n | Acc@QA | Acc&GQA | mIoP | IoP@0.3 | IoP@0.5 | mIoU | IoU@0.3 | IoU@0.5 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | Unpruned (bs 2) | 1005 | 79.7 | 38.9 | 48.0 | 62.1 | 48.6 | 37.7 | 54.5 | 38.5 | complete |
| Qwen3-VL-8B | Cover 30% | 1005 | **79.8** | **39.4** | 47.9 | 62.0 | 48.8 | 37.6 | 54.6 | 37.8 | complete |
| Qwen3-VL-8B | Random 30% | 1005 | 78.6 | 38.6 | 47.5 | 61.1 | 47.7 | 37.3 | 54.3 | 37.2 | complete |
| Qwen3-VL-8B | SemVID 30% | 1005 | 78.9 | 38.5 | 46.2 | 60.5 | 47.4 | 35.8 | 53.5 | 35.3 | complete |
| Qwen3-VL-8B | Cover 10% | 1005 | 78.4 | 36.1 | 44.6 | 57.9 | 45.2 | 34.8 | 50.9 | 34.6 | complete |
| Qwen3-VL-8B | Random 10% | 1005 | 75.3 | 34.9 | 45.2 | 58.4 | 45.6 | 35.3 | 51.4 | 35.1 | complete (1 unanswered) |
| Qwen3-VL-8B | Unpruned, joint LOVER prompt (bs 2) | 1005 | 70.2 | 22.4 | 29.7 | 36.7 | 27.6 | 23.3 | 32.1 | 19.9 | complete (58 unanswered, 46 ungrounded) |
| TimeLens-8B | Unpruned (bs 2) | 1005 | 76.7 | **41.5** | **52.1** | 65.1 | 53.4 | **39.8** | 57.6 | 39.5 | complete |
| TimeLens-8B | Cover 10% | 1005 | 75.4 | 38.5 | 49.3 | 61.7 | 50.1 | 37.8 | 54.5 | 37.6 | complete |
| TimeLens-8B | Random 10% | 1005 | 72.3 | 36.9 | 48.9 | 61.9 | 49.4 | 37.7 | 55.8 | 37.9 | complete |
| TimeLens-8B | Unpruned, joint LOVER prompt (bs 2) | 1005 | 69.5 | 32.3 | 44.1 | 55.2 | 43.8 | 33.3 | 48.6 | 32.2 | complete (14 unanswered, 13 ungrounded) |

Config, read from the logs: no run uses PCD (`contrastive_mode=none`, `answer_reserve=0`) or retrieval. Cover = `core_topk`, equal per-frame allocation. Pruned runs use bs 8; unpruned and joint runs use bs 2. `frame_max_pixels` is 0 (default) for Qwen3 and 1,572,864 for TimeLens-8B, so I am not sure the two models saw the same resolution. Treat cross-model comparisons with care; comparisons within one model are clean. Per the log, Cover 10% keeps about 760 of about 7,600 visual tokens.

Paired differences (reported): Cover10 − Random10 on Acc@QA is +3.1* for both models (Qwen3 [+1.0, +5.1]; TimeLens-8B [+1.2, +4.9]). On Acc&GQA it is +1.2 and +1.6 (not significant); on mIoU, −0.5 and +0.1 (not significant). Cover30 − Random30 is all within noise. Cover30 − SemVID30: mIoU +1.8* [+0.6, +3.0], mIoP +1.7*. Joint minus two-query, Qwen3: Acc&GQA −16.5*. This reproduces LOVER's Qwen3-VL-4B row (22.1 / 32.1 / 19.4), so the high absolute numbers come from the protocol, not from a scoring error.

## 2. S-EMBER with TimeLens-8B (475 questions, 64 frames, timeline prompt, 10% budget, bs 8; unpruned uses bs 1)

| Method (10%) | TimeLens-8B mIoU / R@0.3 / R@0.5 / R@0.7 | unparsed | Qwen3-VL-8B mIoU / R@0.5 (paper draft) |
|---|---|---|---|
| Unpruned | 33.1 / 45.3 / 34.1 / 20.2 | 1 | 28.0 / 26.7 |
| Random | 28.2 / 39.8 / 26.7 / 13.5 | 0 | 25.6 / 24.6 |
| Random + PCD + reserve 64 | 38.4 / 51.8 / 39.4 / 24.0 | 1 | 35.0 / 33.9 |
| Retrieve + Coverage (rc64-qtopk10) | 37.2 / 51.6 / 37.9 / 23.4 | 0 | 32.5 / 34.1 |
| **CoCo (Retrieve + Coverage + PCD + reserve 64)** | **43.9 / 60.2 / 46.9 / 28.8** | 0 | **39.3 / 40.8** |

- All 5 runs are complete (475/475). Answers have **not** been judged, so there is no Acc or GQ@0.5 yet (no `answer_judgments_*.jsonl`).
- Parser: I scored TimeLens-8B with both the official parser and the m:ss-aware one, and they agree exactly. TimeLens-8B writes seconds, so the parser choice does not matter here. The Qwen3 column is the draft's m:ss-aware numbers. I reproduced 35.0 and 39.3 that way; the official parser gives 34.4 and 38.6.
- TimeLens-8B used `frame_max_pixels=1572864`. The Qwen3 logs do not record this key, so whether resolution matches is assumed, not verified.
- Per category, TimeLens-8B with CoCo vs unpruned: time_duration 53.1 vs 42.0, counting 47.1 vs 30.8, location_trace 20.3 vs 20.6 (no gain on location).

## 3. TimeLens-Bench (TimeLens prompt "The event happens in …", 2 fps protocol input, TimeLens `extract_time` + `compute_metrics`)

The S-EMBER `evaluation.log` in these directories reports mIoU 0, and that number is invalid (wrong parser). The scores below come from `timelens_pred.log` or the anchors' `all.log`.

### 3a. Checks against published numbers (validation, not method results)

| Model | Run | n | R@0.3 | R@0.5 | R@0.7 | mIoU | Published |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TimeLens script, Charades (full) | 3,363 | 69.4 | 54.0 | 27.7 | 48.5 | 48.3 |
| Qwen3-VL-8B | TimeLens script, QVH (full) | 1,541 | 74.1 | 64.5 | 49.8 | 59.6 | 59.4 |
| Qwen3-VL-8B | TimeLens script, ActivityNet (full) | 4,500 | 62.5 | 51.5 | 34.6 | 47.1 | 46.8 |
| TimeLens-8B | TimeLens script, Charades (full) | 3,363 | 76.7 | 63.1 | 35.8 | 55.4 | 55.2 |
| TimeLens-8B | TimeLens script, QVH (full) | 1,541 | 80.3 | 71.8 | 55.9 | 65.7 | 65.5 |
| TimeLens-8B | TimeLens script, ActivityNet | – | – | – | – | – | 53.2. **Incomplete**: shard 5 missing (gpu-02 hang 11:30, requeued) |
| Qwen3-VL-8B | Our pipeline, unpruned, bridge-charades | 400 | 71.8 | 51.5 | 26.0 | 48.0 | TL script on the same 400 q: 47.5 *(log)* |
| Qwen3-VL-8B | Our pipeline, unpruned, bridge-activitynet | 400 | 62.3 | 53.3 | 33.3 | 47.1 | 46.8 (full set) |

### 3b. Full-set sweep (Cover 31% vs Random 31%, both models): about 1.5% done (2 of 144 chunk tasks), in progress

`p40_sweep.status`: Qwen3 QVH cover31 has 1/6 chunks done (264 q; 59.9 mIoU *(log, smoke)*). Qwen3 ANet cover31 has 1/18 done (219 q; 53.8 *(log, smoke)*). Qwen3 Charades cover31 has 1 chunk claimed and running. **Every random31 directory and every TimeLens-8B (`full-t8-*`) directory is empty.** The matched-budget TimeLens-script baselines (tl-2fps-3500/4500/4800) have not started. The ETA in the overnight report is about 1.5 days on 6 GPUs. No usable full-set method comparison exists yet.

### 3c. QVHighlights-TimeLens subset, 513 queries (phase 37, 10-14 before 18:00, and phase 38, 10-14 18:00–18:35; complete; both predate the draft's last edit, but the draft references neither). Source: `TIMELENS_QVH_20261014.md`

| Model | Method (visual tokens) | mIoU | R@0.5 |
|---|---|---|---|
| Qwen3-VL-8B | TL-script anchor / our unpruned (13,650) | 60.5 / 60.5 | 64.7 / 64.9 |
| Qwen3-VL-8B | TL script 2 fps low-res (4,200) | 58.0 | 62.0 |
| Qwen3-VL-8B | Random 31% / SemVID 31% / **Cover 31%** (3,271; SemVID 4,232) | 58.2 / 58.8 / **61.7** | 60.8 / 62.4 / 66.7 |
| Qwen3-VL-8B | Cover 31% + PCD (no reserve) | 61.6 | 66.9 |
| Qwen3-VL-8B | Retrieve + Cover 64 frames (897) / + PCD | 58.3 / 57.8 | 63.7 / 64.7 |
| TimeLens-8B | TL-script anchor / our unpruned | 66.1 / 67.1 | 71.9 / 73.1 |
| TimeLens-8B | TL script 2 fps low-res (4,200) | 64.3 | 68.8 |
| TimeLens-8B | Random 31% / **Cover 31%** (3,271) | 65.2 / **67.3** | 70.4 / 73.9 |
| TimeLens-8B | Retrieve + Cover 23%, 64 frames (897) vs TL 1 fps (1,125) | 64.3 vs 57.5 | 71.2 vs 61.8 |

Paired results (reported): Cover − Random is +3.5* (Qwen3) and +2.0* (TimeLens-8B). PCD on Cover under the TimeLens prompt is −0.1 and −0.3 (not significant).

---

## Observations

1. **Coverage transfers to both new settings; PCD has not been tested on them.** On NExT-GQA at 10%, Cover beats Random on answering by +3.1* Acc@QA for both Qwen3-VL-8B and TimeLens-8B, with the same pattern on both models. On S-EMBER with TimeLens-8B, Retrieve + Coverage beats random by +9.0 mIoU (37.2 vs 28.2). On the QVH-TimeLens subset, Cover 31% matches or beats unpruned on both models.
2. **PCD transfers only when the answer lists moments before the interval.** On S-EMBER with TimeLens-8B (timeline prompt), PCD + reserve adds +10.2 to random and +6.7 to Retrieve + Coverage. Full CoCo reaches 43.9 / 46.9: +10.8 mIoU over unpruned TimeLens-8B and +15.7 over random. Under the one-line TimeLens prompt, PCD has no effect (QVH: −0.1 / −0.3, not significant). The report's explanation: PCD only changes the decimal of the time, and TimeLens rounds to 1 s. The paper needs to state this format dependence.
2b. **The timeline prompt itself is a problem for single-moment grounding (phase 38, QVH subset, Qwen3).** Under the timeline prompt, Cover scores 30.7 mIoU and Cover + PCD + reserve 43.9. Under the one-line TimeLens prompt, Cover alone scores 61.7 (paired −17.8* for timeline + PCD vs TimeLens prompt). PCD's +10–13* under the timeline prompt repairs a format the model handles badly ("Seen:" lists with no final interval). The draft's QVH, Charades and ActivityNet tables use the timeline prompt, so this challenges how those tables are framed.
3. **The ordering holds on a grounding-finetuned model.** On S-EMBER, both TimeLens-8B and Qwen3-VL-8B order the methods random < unpruned < Coverage < Random+PCD < CoCo. TimeLens-8B is 2.6–5.1 mIoU higher row by row, and its m:ss parsing issue is gone (0–1 unparsed under either parser).
4. **NExT-GQA at 10% regresses vs unpruned, and Cover does not help grounding there.** Qwen3 Acc&GQA drops 38.9 → 36.1, mIoP 48.0 → 44.6, mIoU 37.7 → 34.8. TimeLens-8B drops 41.5 → 38.5, 52.1 → 49.3, 39.8 → 37.8. Cover vs Random on mIoU and IoU@0.5 is a wash (Qwen3 Cover10 is slightly below Random10 on mIoP, mIoU and IoU@0.3/0.5; not significant). The likely reason: NExT-QA videos are short (~40 s) and densely covered by 64 frames. At 30%, Cover is lossless (39.4 Acc&GQA vs 38.9 unpruned), but Random 30% is too, so 30% does not separate the methods.
5. **SemVID is again worst on grounding**: Cover30 − SemVID30 on mIoU is +1.8*, and SemVID 30% is below Random 30%. This matches the S-EMBER, QVH and Charades findings.
6. **Absolute NExT-GQA numbers depend on the protocol.** With the joint LOVER prompt, Qwen3 falls to 22.4 Acc&GQA, which reproduces the published zero-shot rows. About 26% of its spans come out as minutes.seconds ("0.00 to 0.07"). The paper must state that it uses the two-query protocol and compare methods only within it. Grounding finetuning trades answering for grounding: TimeLens-8B vs Qwen3 is +4.1* mIoP and −3.0* Acc@QA. This comparison is possibly confounded by the resolution setting and the bs 2 vs bs 8 split.
7. **Batch-size caveat:** NExT-GQA unpruned and joint runs used bs 2, pruned runs bs 8; S-EMBER TimeLens-8B unpruned used bs 1, pruned runs bs 8. Per the earlier note, batch size moves about half the outputs and ±2 points is noise. The 3-point drops in observation 4 sit near that level.
8. **Not yet run:**
   - **PCD (+ reserve) on NExT-GQA**, for both models. The two-query grounding reply is a one-line TimeLens-style answer, so per observation 2 a null result is likely; this is a prediction, not a measurement.
   - Judging of the TimeLens-8B S-EMBER answers (no Acc or GQ@0.5).
   - On NExT-GQA: SemVID 10%, any retrieval variant, a 5% budget (planned to find where grounding separates), and TimeLens-8B at 30%.
   - The full TimeLens-Bench sweep (about 98% outstanding: 142 of 144 chunk tasks, including all random31, all TimeLens-8B runs and the matched TL-script baselines).
   - TimeLens-8B ActivityNet anchor (shard 5).
   - Retrieve + Cover on full TimeLens-Bench, which needs SigLIP indexes that do not exist yet.
