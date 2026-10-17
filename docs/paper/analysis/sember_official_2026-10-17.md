# S-EMBER dev, official S-EMBER prompt: Phase 52 results (Qwen3-VL-8B, 10%), collected 2026-10-17 10:25 (+04)

## 0. Sources and method

- **Mirror.** I rsynced hermes-new read-only into `collect5/hn/` at 10:25:09–10:25:12:
  - `/data/chieu/results/` went to `hn/results/`.
  - `/data/chieu/code/HERMES/tools/hermes_pool/{reports,timelens,nextgqa,specs,state}` went to `hn/pool/`.
- **Nothing was written to hermes-new.** I ran no git and did not start, stop or modify any job.
- **Local writes.** Only under `collect5/`. A find for files newer than 10:20 under `collect4/`, `HERMES/results` and `HERMES/tools/hermes_pool` returns only `collect4/drift.md` (10:27:43). It was not written by this session.
- **Scorer.** `collect5/score_p52.py` → `sember_p52.json` and `score_p52.out`. It uses the same parsers and bootstrap as `collect4/sember/score_sember.py`:
  - **Headline:** the m:ss-aware parser (`report_all.py --mss`). An unparsed answer gets IoU 0.
  - The official-parser mIoU, R@0.5 and GQ, and the unparsed counts, are given alongside.
- **Judge.** Acc and GQ@0.5 come from `answer_judgments_new.jsonl`.
  - Judge model: Qwen3.8-27B-FP8, session **"new"**.
  - All 7 official-prompt runs were judged in one batch: `specs/judge_p52off.out` (7 runs, 1,315 unique prompts, FRESH DONE at 10:20).
  - GQ@0.5 = judged correct AND IoU ≥ 0.5.
- **Bootstrap.** Paired over questions, 2,000 resamples, numpy `default_rng(0)`, 95% percentile CI. `*` means the CI excludes 0.
- **Video-clustered check.** I also ran a video-clustered bootstrap (hermes-new's `specs/paired_sember.py` convention) for the key pairs in §3.
- **Reproduction.** I ran hermes-new's own `paired_sember.py` on the mirror. It reproduces NEWSERVER_20261016.md (10-17 p52 official-prompt entry) exactly: all Acc / GQ / mIoU / R@0.5 values and every CI.
  - NEWSERVER uses the **official parser** (via `sember_grounding_scored.jsonl`) and the **video-clustered** CI.
- **Batch size.** Every pruned row is bs 8. Unpruned is **bs 4**: bs 8 ran out of memory, per the `p52_spec.py` docstring. Every "− Unpruned" delta is therefore also a bs 4 vs bs 8 comparison (†). Batch size alone moves grounding ±1–2.

### Data-integrity warning: p52-dev-off-unpruned

- The first attempt at chunks 2 and 3 was logged as "failed" at 09:50, kept running, and finished at 10:03. The pool requeued both chunks, so they ran a second time.
- `results.csv` (merged 10:15) and the judgments (10:20) use the **first-attempt** replies.
- The second attempts then overwrote `4_2.csv` (10:17) and `4_3.csv` (10:22). Against `results.csv`, 40 of 118 replies differ in 4_2 and 46 of 120 in 4_3; chunks 0 and 1 are identical.
- **Headline row = `results.csv`, the judged version.** The rerun version is shown only as a noise estimate. Its mIoU is −0.4 (CI [−0.8, −0.0]), and 389 of 475 replies are identical.
- **Re-merging or re-evaluating this directory on hermes-new would desync 86 judgments from their replies.**

## 1. Completeness (Phase 52 S-EMBER, as of 10:25)

| Run | Prompt | Status | n | Judged |
|---|---|---|---|---|
| dev-off: unpruned, random10, cover10, qtopk10, snapkv10, retcover10, final10 | official | **all complete, 4/4 chunks** | 475 each | yes, session "new", 475 each |
| dev-tim-cover10 (Cover on uniform frames) | timeline | partial: chunks 0–1 done, 2–3 claimed | 237 | no |
| dev-tim-cover10-pcd | timeline | 1 chunk claimed, 0 done | 0 | – |
| dev-tim-unpruned (bs 4), retrand10, retrand10-pcd, qtopk10, snapkv10 | timeline | queued | 0 | – |
| held-out (tim: unpruned, random10, retrand10-pcd, final10; off: unpruned, final10) | both | all queued, 0/6 chunks | 0 | – |

Phase 52 never queued some official-prompt cells:
- Retrieve + Random;
- Cover + PCD;
- Random + PCD.

The only official-prompt PCD on/off pair is therefore **Full − Retrieve+Cover**.

## 2. Main table (paste-ready). S-EMBER dev, official prompt, Qwen3-VL-8B, 64 uniform frames, 10% KV budget, 475 q

Columns:
- mIoU and R@ are m:ss-parser values.
- "Off. mIoU" is the official parser.
- "Unparsed" is m:ss / official.
- GQ@0.5 is m:ss, with the official-parser value in parentheses.

| Method | bs | mIoU | R@0.3 | R@0.5 | R@0.7 | Acc | GQ@0.5 | Off. mIoU | Unparsed |
|---|---|---|---|---|---|---|---|---|---|
| Unpruned (judged `results.csv`) | 4 | 29.1 | 40.8 | 28.4 | 15.2 | 13.7 | 4.8 (4.4) | 27.4 | 0 / 20 |
| *Unpruned, duplicate attempt (chunk CSVs, c2/c3 rerun)* | 4 | *28.8* | *39.8* | *28.2* | *14.9* | – | – | *27.0* | *0 / 19* |
| Random | 8 | 27.9 | 37.3 | 26.1 | 14.1 | 11.2 | 3.6 (3.4) | 26.4 | 1 / 17 |
| Cover (uniform frames) | 8 | 30.1 | 41.7 | 29.3 | 15.2 | 13.7 | 5.5 (5.3) | 28.8 | 0 / 13 |
| Global question top-k, no quota | 8 | 30.0 | 41.9 | 29.3 | 15.2 | 13.9 | 5.1 (4.8) | 28.5 | 0 / 17 |
| SnapKV | 8 | 30.9 | 42.5 | 29.5 | 16.0 | 13.1 | 5.1 (4.8) | 29.6 | 0 / 16 |
| Retrieve (BOLT) + Cover | 8 | 31.9 | 45.3 | 32.2 | 15.6 | 17.1 | 5.9 (5.5) | 30.2 | 2 / 23 |
| Full (BOLT + Cover + PCD) | 8 | **34.6** | **48.0** | **34.3** | **16.2** | 16.6 | **6.3** (5.5) | 29.5 | 0 / **73** |

**Clock-time answers.** PCD moves answers into m:ss format (`Time: [2:49, 4:00]`). About 73 Full replies have a clock-time span, against about 21 for Retrieve+Cover and about 19 for unpruned. The official parser fails on these.

This is why Full's grounding looks different under the two parsers:

| Comparison | m:ss parser | Official parser |
|---|---|---|
| Full mIoU | 34.6 | 29.5 |
| Full − Retrieve+Cover (PCD on − off), mIoU | +2.8* | −0.8 |

## 3. Paired statistics (B − A, n = 475, question-level bootstrap)

The "clustered" column is the video-clustered CI on m:ss mIoU / Acc, given for the key pairs only.

| Pair (B − A) | Identical replies | ΔmIoU | ΔR@0.3 | ΔR@0.5 | ΔR@0.7 | ΔAcc | ΔGQ@0.5 | ΔmIoU (official parser) | Clustered ΔmIoU / ΔAcc |
|---|---|---|---|---|---|---|---|---|---|
| Random − Unpruned † | 48 | −1.3 [−3.7, +1.1] | −3.6 [−7.8, +0.4] | −2.3 [−6.1, +1.5] | −1.1 [−3.8, +1.9] | −2.5 [−4.8, −0.4]* | −1.3 [−2.9, +0.4] | −1.0 [−3.5, +1.3] | |
| Cover − Unpruned † | 141 | +1.0 [−0.6, +2.5] | +0.8 [−2.1, +3.8] | +0.8 [−1.7, +3.4] | +0.0 [−2.3, +2.3] | +0.0 [−1.9, +1.7] | +0.6 [−0.6, +1.9] | +1.4 [−0.1, +2.9] | |
| Top-k (no quota) − Unpruned † | 140 | +0.8 [−0.8, +2.4] | +1.1 [−1.9, +4.0] | +0.8 [−2.1, +3.6] | +0.0 [−2.1, +2.1] | +0.2 [−1.5, +1.9] | +0.2 [−1.1, +1.5] | +1.1 [−0.4, +2.7] | |
| SnapKV − Unpruned † | 120 | +1.8 [+0.1, +3.3]* | +1.7 [−1.5, +4.6] | +1.1 [−1.7, +3.6] | +0.8 [−1.5, +3.2] | −0.6 [−2.5, +1.3] | +0.2 [−0.8, +1.3] | +2.2 [+0.6, +3.7]* | +1.8 [+0.1, +3.4]* / −0.6 n.s. |
| **Retrieve+Cover − Unpruned** † | 14 | +2.7 [+0.2, +5.3]* | +4.4 [+0.2, +8.6]* | +3.8 [−0.4, +8.0] | +0.4 [−2.7, +4.0] | **+3.4 [+0.6, +6.3]*** | +1.1 [−0.8, +3.2] | +2.9 [+0.5, +5.5]* | +2.7 [+0.3, +5.1]* / +3.4 [+0.4, +6.4]* |
| **Full − Unpruned** † | 3 | **+5.5 [+2.7, +8.3]*** | +7.2 [+2.3, +12.2]* | +5.9 [+1.3, +10.5]* | +1.1 [−2.5, +4.6] | +2.9 [+0.2, +6.1]* | +1.5 [−0.6, +3.6] | +2.1 [−0.6, +4.9] | +5.5 [+2.6, +8.1]* / **+2.9 [−0.2, +6.3] n.s.** |
| **Cover − Random** | 56 | +2.2 [−0.0, +4.5] | +4.4 [+0.8, +8.2]* | +3.2 [−0.4, +6.7] | +1.1 [−1.7, +4.0] | +2.5 [+0.2, +5.1]* | +1.9 [+0.2, +3.8]* | +2.4 [+0.2, +4.6]* | +2.2 [−0.1, +4.5] / +2.5 [+0.2, +4.9]* |
| **Cover − Top-k (no quota)** | 229 | +0.1 [−1.3, +1.5] | −0.2 [−2.9, +2.3] | +0.0 [−2.5, +2.5] | +0.0 [−1.7, +1.7] | −0.2 [−1.3, +0.6] | +0.4 [−0.6, +1.5] | +0.3 [−1.2, +1.6] | +0.1 / −0.2, both n.s. |
| **Cover − SnapKV** | 205 | −0.8 [−2.2, +0.6] | −0.8 [−3.6, +1.7] | −0.2 [−3.2, +2.3] | −0.8 [−2.9, +1.3] | +0.6 [−0.6, +1.9] | +0.4 [−0.6, +1.5] | −0.8 [−2.3, +0.6] | −0.8 / +0.6, both n.s. |
| Top-k (no quota) − Random | 57 | +2.1 [−0.3, +4.5] | +4.6 [+0.8, +8.6]* | +3.2 [−0.6, +6.9] | +1.1 [−1.9, +3.8] | +2.7 [+0.4, +5.1]* | +1.5 [+0.0, +3.2] | +2.2 [−0.2, +4.5] | |
| SnapKV − Random | 51 | +3.0 [+0.7, +5.4]* | +5.3 [+1.5, +9.1]* | +3.4 [−0.4, +7.2] | +1.9 [−1.1, +4.8] | +1.9 [−0.4, +4.4] | +1.5 [−0.2, +3.2] | +3.2 [+0.8, +5.6]* | |
| Retrieve+Cover − Cover (retrieval step) | 17 | +1.7 [−0.7, +4.3] | +3.6 [−0.4, +7.8] | +2.9 [−1.1, +6.9] | +0.4 [−2.9, +3.8] | +3.4 [+0.6, +6.3]* | +0.4 [−1.3, +2.1] | +1.5 [−0.9, +4.0] | +1.7 n.s. / +3.4 [+0.6, +6.2]* |
| **PCD on − off: Full − Retrieve+Cover** | 28 | **+2.8 [+0.7, +5.0]*** | +2.7 [−1.1, +6.7] | +2.1 [−1.3, +5.9] | +0.6 [−2.3, +3.6] | −0.4 [−1.5, +0.6] | +0.4 [−1.1, +1.9] | −0.8 [−2.9, +1.4] | +2.8 [+0.8, +4.8]* / −0.4 n.s. |
| Full − Cover | 2 | +4.5 [+2.0, +7.2]* | +6.3 [+1.7, +11.2]* | +5.1 [+0.8, +9.5]* | +1.1 [−2.3, +4.4] | +2.9 [+0.2, +5.9]* | +0.8 [−1.1, +2.7] | +0.7 [−1.9, +3.4] | +4.5 [+1.9, +7.0]* / +2.9 [+0.2, +5.9]* |
| Full − Random | 0 | +6.8 [+4.0, +9.6]* | +10.7 [+5.9, +15.4]* | +8.2 [+3.6, +12.8]* | +2.1 [−1.7, +5.9] | +5.5 [+2.7, +8.4]* | +2.7 [+0.8, +4.8]* | +3.1 [+0.3, +5.9]* | |
| *Unpruned duplicate attempt − judged (noise)* | 389 | −0.4 [−0.8, −0.0]* | −1.1 [−2.1, +0.0] | −0.2 [−1.1, +0.4] | −0.2 [−1.1, +0.4] | – | – | −0.4 [−0.8, −0.0]* | |

† Unpruned is bs 4; the other row is bs 8.

Official-parser ΔR@0.5 and ΔGQ for the key pairs:

| Pair | ΔR@0.5 (official) | ΔGQ (official) |
|---|---|---|
| Full − Unpruned | +2.5 [−1.9, +6.9] | +1.1 n.s. |
| PCD on − off | −1.7 [−4.8, +1.9] | +0.0 n.s. |
| Retrieve+Cover − Unpruned | +4.2 [+0.0, +8.4] | +1.1 n.s. |
| Cover − Random | +3.2 n.s. | +1.9 [+0.2, +3.8]* |

## 4. Timeline-prompt Phase 52 checks (partial; grounding only, not judged)

| Comparison | n | Result |
|---|---|---|
| p52 timeline Cover (uniform, bs 8, hermes-new, chunks 0–1) | 237 | mIoU 32.5, R@0.3/0.5/0.7 45.1 / 31.2 / 19.0, 0 unparsed |
| p52 timeline Cover − old-cluster p24 timeline Cover (same config; cross-cluster reproduction) | 237 | ΔmIoU −1.1 [−2.9, +0.4], 128/237 identical |
| Cover: official − timeline prompt (same cluster, same 237 q) | 237 | ΔmIoU −2.9 [−7.3, +1.3], ΔR@0.7 −5.5 [−11.0, −0.4]* (official parser: −4.2 [−8.5, +0.3]) |
| Unpruned: official (hermes-new bs 4) − timeline (old cluster, older run) | 475 | ΔmIoU +1.1 [−2.1, +4.0] (official parser −0.7 n.s.). Cross-cluster and cross-bs. |

- None of the batch-matched timeline checks has a result yet. These are:
  - unpruned at bs 4;
  - Cover + PCD;
  - Retrieve + Random ± PCD;
  - top-k;
  - SnapKV.
- No held-out split run has started.

## 5. Check of `memory/timeline-prompt-confound.md` (2026-10-17 update)

The update was evidently written from NEWSERVER, which uses the official parser and a video-clustered CI. Every statement reproduces under that convention. Two of them change under the m:ss parser, which the paper states it uses for every method and `clock-time-answers.md` mandates.

| Memory statement | Official parser + clustered CI (NEWSERVER) | m:ss parser (paper's stated parser) | Verdict |
|---|---|---|---|
| "Final vs unpruned is n.s. on every metric" | True: Acc +2.9, GQ +1.1, mIoU +2.1, R@0.5 +2.5, all n.s. | **False for grounding:** mIoU +5.5*, R@0.3 +7.2*, R@0.5 +5.9* (* under both bootstrap units; clustered R@0.3 +7.2 [+2.2, +11.9]*). Acc +2.9 is borderline: question-level [+0.2, +6.1]*, video-clustered [−0.2, +6.3] n.s. GQ +1.5 n.s. All with the bs 4 vs bs 8 caveat. | **Correct only for the official parser.** Rephrase: "Final beats unpruned on m:ss grounding (+5.5 mIoU*). Its answer gain (+2.9 Acc) is borderline and, unlike Retrieve+Cover's, not robust to video clustering." |
| "PCD is null" | True: Full − Retrieve+Cover mIoU −0.8, R@0.5 −1.7, Acc −0.4, GQ 0.0, all n.s. | Acc −0.4 n.s. and R@0.5 +2.1 n.s., but **mIoU +2.8* [+0.7, +5.0]** (clustered +2.8*). PCD also roughly triples clock-time spans (73 vs 23 official-unparsed), so official-parser PCD numbers are penalised for format. | **Partly correct.** PCD is null for answers (Acc / GQ) and for R@0.5. It is not null for m:ss mIoU. This is the only PCD on/off pair under the official prompt; there is no Cover±PCD or Random±PCD. |
| "The per-frame quota is not distinguishable from global top-k or SnapKV" | True | True: Cover − top-k +0.1 mIoU / −0.2 Acc (229/475 identical replies); Cover − SnapKV −0.8 / +0.6, all n.s. | **Confirmed** under both parsers and both bootstrap units. |
| "Retrieval is the main gain: Retrieve+Cover vs unpruned Acc +3.4*, mIoU +2.9*" | True (official-parser mIoU +2.9 [+0.4, +5.3]*) | m:ss mIoU +2.7 [+0.2, +5.3]*, Acc +3.4* | **Numbers confirmed.** "Main gain" holds for answers: the retrieval step is Acc +3.4* and the PCD step −0.4. For m:ss grounding, the PCD step (+2.8*) is larger than the retrieval step (Retrieve+Cover − Cover +1.7 n.s.). |
| "So 'beats uncompressed' and the PCD gains are specific to the timeline prompt" | Holds | Not supported as stated. Under the official prompt with the m:ss parser, Full still beats unpruned on grounding (+5.5*) and PCD still adds mIoU (+2.8*). What disappears is the PCD gain on answers and a robust answer gain over unpruned. | **Needs scoping.** Only answer-side gains and official-parser grounding gains are timeline-specific. |

Also worth recording:
- Question-aware selection beats Random: Cover − Random Acc +2.5*, GQ +1.9*, R@0.3 +4.4*.
- Random is the only pruned method significantly below unpruned on Acc (−2.5*). SnapKV is also below, but n.s. (−0.6).

## 6. Files

- `collect5/sember_official.md` (this file);
- `collect5/score_p52.py`, `score_p52.out`, `sember_p52.json`;
- `collect5/check_consistency.py`, `check_chunks.py` (the unpruned provenance checks);
- `collect5/hn/` (the mirror).
