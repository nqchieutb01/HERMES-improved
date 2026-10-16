# Overnight 2026-10-14 23:00 → 10-15 09:00: results and next plan

Detailed log with every check: `PLAN_20261015.md`. Code: branch `worktree-p31-gate-pcdfix` (no new code commits;
new scripts outside the repo in `hermes_pool/`). All scores below are from the official scorers
(`timelens/score.py --subset`, TimeLens `compute_metrics.py`, `nextgqa/score.py`); the managers' "RESULT mIoU 0.0"
lines for TimeLens-prompt / NExT-GQA runs come from the built-in S-EMBER parser and are not results.
Paired differences: bootstrap over the same questions, 95% CI; * = CI excludes 0.

## 1. Checks against published numbers (all pass)

| Check | Ours | Published / reference |
|---|---|---|
| TimeLens script, Qwen3-VL-8B, Charades-TimeLens (3,363 q) | 48.5 mIoU | 48.3 |
| TimeLens script, Qwen3-VL-8B, QVHighlights-TimeLens (1,541 q) | 59.6 | 59.4 |
| TimeLens script, TimeLens-8B, Charades | 55.4 | 55.2 |
| TimeLens script, TimeLens-8B, QVHighlights | 65.7 | 65.5 |
| TimeLens script, Qwen3-VL-8B, ActivityNet-TimeLens (4,500 q) | 47.1 | 46.8 |
| Our pipeline (per-video protocol input) vs TimeLens script, same 400 Charades queries | 48.0 vs 47.5 (288/400 identical) | – |
| Our pipeline, 400 ActivityNet queries | 47.1 | 46.8 (full set) |
| Full-sweep smoke chunks: QVH Cover 31% vs phase-37 Cover 31% on 89 shared queries | 67.9 vs 67.7 (64/89 identical) | – |

TimeLens-8B ActivityNet with the TimeLens script is running (09:25).

## 2. S-EMBER (475 q, 64 frames, timeline prompt), TimeLens-8B vs Qwen3-VL-8B, mIoU (R@0.5)

| Method (pruned rows: 10% of visual tokens) | TimeLens-8B | Qwen3-VL-8B |
|---|---|---|
| Unpruned | 33.1 (34.1) | 28.0 (26.7) |
| Random | 28.2 (26.7) | 25.6 (24.6) |
| Random + PCD + reserve | 38.4 (39.4) | 35.0 (33.9) |
| Retrieve + Cover | 37.2 (37.9) | 32.5 (34.1) (phase 25) |
| **Retrieve + Cover + PCD + reserve (final)** | **43.9 (46.9)** | **39.3 (40.8)** |

The method transfers to a grounding-tuned model: same ordering, final +10.8 over unpruned and +15.7 over random.
Answers (judge) not scored yet.

## 3. NExT-GQA (1,005 test questions on 191 videos; VideoMind MCQ prompt + TimeLens grounding prompt; 64 frames)

| Model / method | Acc@QA | Acc&GQA | mIoP | mIoU |
|---|---|---|---|---|
| Qwen3-VL-8B unpruned | 79.7 | 38.9 | 48.0 | 37.7 |
| Cover 30% | 79.8 | 39.4 | 47.9 | 37.6 |
| Random 30% | 78.6 | 38.6 | 47.5 | 37.3 |
| SemVID 30% | 78.9 | 38.5 | 46.2 | 35.8 |
| Cover 10% | 78.4 | 36.1 | 44.6 | 34.8 |
| Random 10% | 75.3 | 34.9 | 45.2 | 35.3 |
| TimeLens-8B unpruned | 76.7 | 41.5 | 52.1 | 39.8 |
| TimeLens-8B Cover 10% | 75.4 | 38.5 | 49.3 | 37.8 |
| TimeLens-8B Random 10% | 72.3 | 36.9 | 48.9 | 37.7 |

- Cover − Random at 10%: Acc@QA +3.1* on both models (Qwen3 [+1.0, +5.1], TimeLens-8B [+1.2, +4.9]); Acc&GQA
  +1.2 / +1.6 (n.s.); mIoU ±0.5 (n.s.). At 30% everything is within noise (random 30% is already lossless).
- Cover − SemVID at 30%: mIoU +1.8* [+0.6, +3.0], mIoP +1.7*; SemVID is below random on grounding, as on S-EMBER /
  QVH / Charades. Budgets checked: SemVID keeps 2,354 tokens per sample on average, Cover 2,436 (of 8,122; Cover's
  count includes the timestamp tokens, so its visual tokens are slightly fewer).
- TimeLens-8B − Qwen3-VL (unpruned): mIoP +4.1*, mIoU +2.1*, Acc@QA −3.0* (grounding tuning trades answering).
- Reading: NExT-QA videos are short (~40 s) and 64 frames cover them densely, so grounding barely suffers from
  pruning; Cover's gain shows in answering. Grounding gains of Cover need longer videos (S-EMBER, QVH).
- Caveat for the paper: our absolute Acc&GQA (38.9) is far above published zero-shot rows (Qwen3-VL-4B 22.1 in LOVER,
  arXiv 2609.15224; specialised 7B models ~28-32). Checked: official scorer, no leak, trivial baselines 12.9-19.1
  IoP@0.5, predicted spans not short. **Resolved (phase 41, 10:04):** with LOVER's joint prompt (one reply,
  <answer>s to e</answer><choice>X</choice>) the same model on the same 1,005 questions gives Acc&GQA 22.4 / mIoP 29.7
  / mIoU 23.3 / Acc@QA 70.2, i.e. LOVER's Qwen3-VL row (22.1 / 32.1 / 19.4). The gap is the protocol: under the joint
  prompt 26% of spans are written as minutes.seconds ("0:06" -> "0.00 to 0.07"), 4% copy the prompt's example, 6% of
  replies lack a choice, and answering drops 9.5 points. Report NExT-GQA with our two-query protocol (state it, cite
  this comparison) and compare methods within it. TimeLens-8B under the joint prompt: Acc&GQA 32.3 / mIoU 33.3 /
  Acc@QA 69.5 (-9.2* / -6.5* / -7.3* vs its two-query run); the grounding-tuned model loses less (seconds format).

## 4. Still running (queue: ~200 tasks, 4-6 GPUs)

- TimeLens script on ActivityNet (both models).
- Full TimeLens-Bench sweep, both models: Cover 31% and Random 31% (protocol input), plus the TimeLens script at the
  matched budget (2 fps, min 16, total 3,500 / 4,500 / 4,800 → 3.2k / 4.2k / 4.0k visual tokens). Estimate: ~1.5 days
  at 6 GPUs. Priority: Qwen3 Charades/QVH → TimeLens-8B Charades/QVH → ActivityNet.
- Matching: budgets are matched on Cover's kept count, which includes the ~9 timestamp tokens per frame pair (QVH smoke:
  4,641 kept of 14,971 vs 13,641 visual), so Cover has fewer visual tokens than the TimeLens-script baseline: the
  comparison favours the baseline. Cover vs random is the clean comparison (same input, same batch size); Cover
  (batch 8 / 4) vs the TimeLens script (batch 1) also crosses the batch-size noise (~±1-2 points).

## 5. Proposed next plan

0. ~~**NExT-GQA joint prompt first**~~ done (phase 41, see section 3). (~20 min per run, before the long sweep): one reply with the option and the time
   span (LOVER-style), Qwen3-VL-8B unpruned. This separates "8B is stronger" from "the two-query protocol inflates
   grounding". If Acc&GQA drops toward the low 20s, NExT-GQA goes in the paper as a within-protocol pruning
   comparison only. Needs a joint prompt + a parser for letter and span from one reply (small code; check on the
   synthetic perfect / wrong predictions as for score.py). Not started: it is new code and needs your OK on the prompt.
1. **Let the full TimeLens-Bench sweep finish** (main table: Cover vs random vs TimeLens-script low-res at the same
   budget, 3 datasets × 2 models). If time matters, drop TimeLens-8B ActivityNet (largest, ~25% of the queue).
2. **Judge S-EMBER TimeLens-8B answers** (one judging session, reuse `judge27_job.sh`), to give Acc / GQ@0.5 next to
   the mIoU above.
3. **Retrieve + Cover on full TimeLens-Bench**: needs SigLIP 2 1-fps indexes for Charades / ActivityNet / QVH
   videos (one embedding job per dataset), then the same 31% runs. Retrieval was +0.8 to +1.2 on top of Cover so far.
4. **NExT-GQA at 5%**: Cover / random, to find where grounding starts to separate on short videos.
5. Skip: PCD under the TimeLens prompt (n.s. on QVH, phase 38); more input-level pruners (all below random).
