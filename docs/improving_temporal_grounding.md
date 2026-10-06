# Improving temporal grounding under visual-token pruning: what we did and what we found

Model: Qwen3-VL-8B. Data: S-EMBER grounded QA, 300 videos, 475 questions (duration 191, counting 184, location 100).
Everything is training-free. Every number below comes from the same 475 questions; Δ is a paired difference against
the same visual memory, with a 95% bootstrap interval; **\*** means the interval excludes 0 (significant).

Detailed tables: `logs/phase18/report.md` (PCD and its ablations), `logs/phase20/early.md` (refinements and the
early-evidence analysis). Method write-up for the paper: `docs/method_prior_contrastive.md`. Diagnosis:
`docs/phase15_diagnosis.md`.

---

## 1. Summary in five points

1. **The problem is not only which tokens pruning keeps.** When the video memory is pruned, the model fills the gaps
   with a *language prior about time*: without any video it says the evidence is at the start of the video
   (90% of its intervals start in the first 5% of the window). This prior leaks into answers with video.
2. **Our method, prior-contrastive decoding (PCD), removes that prior while the answer is written.** At each word it
   compares "what the model says with the video" against "what it would say with no video" and prefers what the video
   adds. No training, no change to the pruning, no post-processing of the output.
3. **PCD improves grounding significantly in every setting we tested**: mIoU +3.5 to +6.3 and GQ@0.5 +2.5 to +4.2,
   across budgets (5% to unpruned), selectors (HERMES, random, stratified) and HERMES streaming. With 10–25% of the
   visual tokens it beats the unpruned model. Answer accuracy is never significantly hurt.
4. **Two refinements make it better and safer.** *Time-scoped* PCD (contrast only the time values) keeps the grounding
   gain and disturbs the answer text less. *Confidence-adaptive* PCD (weaker contrast where the model is already sure)
   recovers the questions whose evidence really is at the start of the video, which plain PCD hurt when the memory is
   rich (unpruned: accuracy on those questions 16.7 → 23.1, GQ@0.5 12.8 → 19.2, both \* vs PCD). **Combined
   (adaptive strength, contrast only on time values) they give the best overall result**: averaged over the 8 memories,
   mIoU 24.0 → 30.7 and GQ@0.5 6.1 → 10.0 (plain PCD: 29.2 / 9.3), significant against the baseline in all 8 for mIoU
   and in 7 of 8 for GQ@0.5 (section 6.3a).
5. **What did not work is informative too**: other counterfactuals (noised frames as in VCD, permuted timestamps), a
   one-sided "evidence-against" rule, and re-allocating frames (zoom, relevance sampling, attention boosting) give
   little or nothing. The prior is the right thing to contrast against, and decoding cannot recover evidence the
   memory does not contain.

---

## 2. Task and metrics

The model sees a video memory (possibly pruned to a token budget) and a question, and must give an **answer** and the
**time interval** of its evidence. We use the *timeline prompt*: the model first lists the moments it saw
(`Seen: 12 seconds ...`), then writes `Answer: ...` and `Time: [start, end]`.

| Metric | Meaning |
|---|---|
| Acc. | answer judged correct (official S-EMBER judge prompt, local Qwen3.8-27B judge) |
| mIoU | mean overlap between predicted and gold interval |
| R@0.5 | share of intervals with overlap ≥ 0.5 |
| GQ@0.5 | answer correct **and** overlap ≥ 0.5 (grounded QA: both must be right) |
| Early start | share of predicted intervals starting in the first 5% of the window (a symptom of the prior) |

Memories tested: offline pruning to 5 / 10 / 25% of the visual tokens with a random, stratified or HERMES selector,
the unpruned model, and HERMES streaming memory with a 4k or 6k KV budget.

---

## 3. The problem we found: a temporal prior leaks into the answers

| Evidence | Numbers |
|---|---|
| With no video at all, the model still answers with times | 89% of first `Seen` moments are 0 s; 90% of intervals start in the first 5% |
| With video, the same habit appears | 38% of intervals start in the first 5% at 10% random pruning, 56% with HERMES pruning; only 16% of gold intervals do |
| It grows when the evidence is missing | removing the frames that show the evidence raises it from 37% to 51% (unpruned) and from 31% to 46% (random 10%); see 6.7 |
| It does not depend on the token budget | about 30% at random 5, 10, 25% and unpruned; higher when temporal coverage is lost (HERMES 49%, streaming 37–39%) |
| A better selector alone does not fix it | even a perfect (oracle) selector at 10% only matches the unpruned model |

So the model is a mixture of "what the video shows" and "what such questions usually look like". Wherever the
memory does not show the evidence (because it was never sampled, was pruned away, or the memory lost coverage of
that part of the video), the second part takes over.

---

## 4. The method: prior-contrastive decoding (PCD)

While generating, each next token is chosen by

    score(y) = (1 + α) · log p(y | video memory, question)  −  α · log p(y | no video, question)

among tokens that are reasonably likely with the video (p ≥ β · the top probability; β = 0.1). Default α = 0.5.

**Intuition.** Suppose the model must write the first time. With no video it is almost certain of "0". With video it
puts some weight on "0" and some on "48". The contrast lowers "0" (the prior explains it) and raises "48" (only the
video explains it).

**Properties.**
- Works on top of any memory: no change to the selector, the KV cache or the model.
- Cheap: the "no video" branch holds only the prompt (a few hundred tokens), so each step costs one small extra forward.
- Not post-processing: it changes the model's own token choices during generation, so the answer and the interval are
  produced together.

Code: `contrastive_answering` in `inference/qwen3vl_hermes.py`; options `run.contrastive_mode=blind`,
`run.contrastive_alpha`, `run.contrastive_beta`.

---

## 5. Main result: PCD

| Memory | Acc. | mIoU | GQ@0.5 | Early start |
|---|---|---|---|---|
| Random 5% → +PCD | 15.8 → 15.4 | 23.6 → 29.2 (+5.6\*) | 5.9 → 9.1 (+3.2\*) | 39% → 20% |
| Random 10% → +PCD | 15.2 → 16.2 | 25.8 → 32.1 (+6.2\*) | 7.2 → 10.1 (+2.9\*) | 38% → 19% |
| Random 25% → +PCD | 17.7 → 18.9 | 28.2 → 33.0 (+4.8\*) | 8.2 → 11.8 (+3.6\*) | 37% → 21% |
| Unpruned → +PCD | 18.5 → 19.8 | 28.0 → 31.5 (+3.5\*) | 8.6 → 11.2 (+2.5) | 41% → 23% |
| HERMES 10% → +PCD | 13.1 → 15.4 | 18.3 → 22.6 (+4.3\*) | 4.0 → 7.2 (+3.2\*) | 56% → 31% |
| Stratified 10% → +PCD | 14.1 → 14.5 | 23.7 → 29.9 (+6.3\*) | 6.1 → 8.8 (+2.7\*) | 36% → 17% |
| HERMES streaming 6k → +PCD | 14.3 → 14.7 | 22.6 → 28.1 (+5.6\*) | 4.8 → 7.6 (+2.7\*) | 46% → 24% |
| HERMES streaming 4k → +PCD | 13.5 → 17.1 | 21.6 → 27.4 (+5.8\*) | 4.0 → 8.2 (+4.2\*) | 45% → 24% |

What this shows:
- PCD roughly **halves the prior leak** everywhere (early starts to ~20%, below even the unpruned model's 41%).
- Grounding gains are **significant in all pruned settings**, and **larger when the memory is smaller** (+4.3 to +6.3
  mIoU pruned vs +3.5 unpruned), as expected if PCD removes what pruning lets in.
- **Random 10% + PCD beats the unpruned model** (mIoU 32.1 vs 28.0, GQ@0.5 10.1 vs 8.6) with a tenth of the tokens.
- **Robust**: α ∈ {0.25, 0.5, 1.0} gives mIoU +5.0 / +6.2 / +6.1; β = 0.2 gives +5.7.
- **Boundary**: with the official direct-answer prompt there is no gain (mIoU +0.4). The prior shows up when the model
  lists timestamped moments, so PCD and the timeline prompt work as a pair.

---

## 6. Refinements and what each one taught us

### 6.1 What should the model be compared against? The no-video input is the right choice

**How contrastive decoding works.** While the answer is written, the model runs twice in parallel on every word:

- **Main input**: the question and the real (pruned) video memory. This is the same in every row below.
- **Reference input**: the question and a *damaged* version of the memory. The model prefers words the main input
  supports *more than* the reference input does.

The reference input decides what gets removed: whatever the model still says with the damaged memory is treated as
"not coming from the video" and pushed down. This experiment only changes the **reference input**:

| Reference input (main input: random 10% memory in every row) | What the reference input contains | mIoU | GQ@0.5 | Early start |
|---|---|---|---|---|
| none: plain decoding, no contrast (baseline) | — | 25.8 | 7.2 | 38% |
| **no video (our PCD)** | the question only, no visual tokens at all | **32.1 (+6.2\*)** | **10.1 (+2.9\*)** | **19%** |
| noised frames (Visual Contrastive Decoding, CVPR 2024) | the same frames and timestamps, each frame blended 50/50 with random noise | 26.3 (+0.5) | 7.8 (+0.6) | 30% |
| permuted timestamps | the same frames, but the timestamps shuffled between frames | +0.9 (n.s.) | — | — |

Note: "no video" here is **not** the model answering without video. The model always answers *with* the video;
"no video" only describes the reference input it is compared against.

**Why the no-video reference works best.** With no video, the model can only use its habit, so the comparison
removes exactly the habit ("evidence is at the start"): early starts drop from 38% to 19%. With noised frames, the
damaged memory still produces the same habit *and* still shows some of the video, so the two runs disagree mainly about
video content, not about the habit; little of the habit is removed (early starts 30%) and grounding barely changes.
Shuffled timestamps likewise leave the habit in place. So the published VCD recipe (noised frames) does not fix this
failure, and the no-video reference does.

### 6.2 Time-scoped PCD: contrast only the time values

Error analysis of PCD (random 10%, `logs/phase19/error_analysis.py`) showed that contrasting *every* token also
changes words that have nothing to do with time: the answer text changed in 85% of questions, answers listing more
than 8 moments rose from 12 to 56, and over-counting rose from 14% to 34% of counting questions. The diagnosed
leak is in the time values, so we applied the contrast only while the model writes a `Seen:` time or the `Time: [...]`
interval, and decoded everything else normally.

| Memory | PCD (all tokens): Acc. / mIoU / GQ@0.5 | Time-scoped PCD: Acc. / mIoU / GQ@0.5 |
|---|---|---|
| Random 10% | 16.2 / 32.1 / 10.1 | 16.0 / 32.1 / 10.3 |
| Random 25% | 18.9 / 33.0 / 11.8 | 17.7 / 34.0 / 11.8 |
| Unpruned | 19.8 / 31.5 / 11.2 | **20.8 / 32.9 / 12.4** (Acc. +2.3, GQ +3.8\* vs baseline) |
| HERMES 10% | 15.4 / 22.6 / 7.2 | **14.7 / 24.3 / 7.4** |
| Streaming 4k | 17.1 / 27.4 / 8.2 | **16.2 / 29.2 / 9.3** (mIoU +7.5\*, GQ +5.3\* vs baseline) |
| Random 10%, official prompt | — | 10.3 / 27.2 / 3.2 (no gain, as before) |

Same or better grounding with a more targeted intervention; run-on lists drop (more than 8 moments: 56 → 41).
Answers still change often, because the answer is computed from the times the model wrote: fixing the times
legitimately changes the answer.

### 6.3 Confidence-adaptive PCD: recovering questions whose evidence really starts early

**The concern.** 78 of the 475 questions have gold evidence that truly starts in the first 5% of the window. PCD
pushes away from early times, so it could hurt exactly these questions. With plain PCD on the unpruned memory, their
accuracy fell 21.8 → 16.7 and GQ@0.5 16.7 → 12.8.

**The idea.** Make the contrast fade where the video memory is already confident:
α_t = α_max · (1 − max p(· | video)). If the video clearly supports "0", the model is confident and the contrast
nearly switches off; if the model is unsure (where the prior sneaks in), the contrast acts fully. Option:
`run.contrastive_adaptive=true`.

**Result on the 78 early-evidence questions** (Acc. / mIoU / GQ@0.5):

| Memory | Baseline | PCD | Adaptive PCD (α_max 1) |
|---|---|---|---|
| Unpruned | 21.8 / 49.2 / 16.7 | 16.7 / 55.3 / 12.8 | **23.1 / 56.9 / 19.2** (Acc. and GQ +6.4\* vs PCD) |
| Random 25% | 23.1 / 48.7 / 17.9 | 19.2 / 53.9 / 17.9 | 20.5 / 52.7 / 16.7 |
| Random 10% | 19.2 / 47.6 / 16.7 | 16.7 / 57.0 / 15.4 | 16.7 / 54.4 / 14.1 |
| Random 5% | 15.4 / 40.3 / 10.3 | 14.1 / 47.2 / 12.8 | 16.7 / **51.8** / 12.8 (mIoU +4.6\* vs PCD) |
| Streaming 6k | 11.5 / 38.4 / 7.7 | 12.8 / 46.6 / 11.5 | 15.4 / 46.0 / 12.8 |

**Result on all 475 questions** (Acc. / mIoU / GQ@0.5; Δ vs baseline for adaptive):

| Memory | Baseline | PCD | Adaptive PCD (α_max 1) |
|---|---|---|---|
| Random 5% | 15.8 / 23.6 / 5.9 | 15.4 / 29.2 / 9.1 | 14.9 / **30.6** (+7.1\*) / 9.1 (+3.2\*) |
| Random 10% | 15.2 / 25.8 / 7.2 | 16.2 / 32.1 / 10.1 | 15.4 / 32.0 (+6.2\*) / 9.7 (+2.5) |
| Random 25% | 17.7 / 28.2 / 8.2 | 18.9 / 33.0 / 11.8 | 19.2 / **33.8** (+5.6\*) / **14.1** (+5.9\*; +2.3\* vs PCD) |
| Unpruned | 18.5 / 28.0 / 8.6 | 19.8 / 31.5 / 11.2 | 19.8 / **32.7** (+4.7\*) / **12.2** (+3.6\*) |
| HERMES 10% | 13.1 / 18.3 / 4.0 | 15.4 / 22.6 / 7.2 | 14.3 / 23.0 (+4.6\*) / 6.5 (+2.5\*) |
| Stratified 10% | 14.1 / 23.7 / 6.1 | 14.5 / 29.9 / 8.8 | 14.7 / **30.4** (+6.7\*) / **9.7** (+3.6\*) |
| Streaming 6k | 14.3 / 22.6 / 4.8 | 14.7 / 28.1 / 7.6 | 15.6 / **28.6** (+6.1\*) / 7.6 (+2.7\*) |
| Streaming 4k | 13.5 / 21.6 / 4.0 | 17.1 / 27.4 / 8.2 | 16.0 / 27.2 (+5.6\*) / 6.9 (+2.9\*) |

- Adaptive PCD keeps all of PCD's significant gains over the baseline and is **never significantly worse than PCD**.
- It **recovers the early-evidence questions when the memory has enough evidence** (unpruned fully, 25% partly).
  Under heavy pruning the model is rarely confident, so the contrast stays on and nothing changes.
- α_max = 2 is not better (random 10%: 16.6 / 32.8 / 10.3; random 25%: 17.9 / 33.2 / 12.2; unpruned: 19.4 / 32.2 / 11.6).
- **Adaptive + time scope** gives the best unpruned result: **21.5 / 32.7 / 13.3** (GQ +4.6\* vs baseline). Its
  results on every memory are in 6.3a.

### 6.3a Final configuration on every memory: adaptive strength + time scope

Acc. / mIoU / GQ@0.5 on all 475 questions; Δ vs the baseline in brackets (\* significant). The last column compares
with plain PCD.

| Memory | Baseline | PCD | **Adaptive + time scope** | vs PCD (Acc. / mIoU / GQ) |
|---|---|---|---|---|
| Random 5% | 15.8 / 23.6 / 5.9 | 15.4 / 29.2 / 9.1 | **16.4 / 31.5 (+8.0\*) / 9.9 (+4.0\*)** | +1.1 / +2.3\* / +0.8 |
| Random 10% | 15.2 / 25.8 / 7.2 | 16.2 / 32.1 / 10.1 | **16.2 / 32.6 (+6.8\*) / 10.1 (+2.9)** | +0.0 / +0.6 / +0.0 |
| Random 25% | 17.7 / 28.2 / 8.2 | 18.9 / 33.0 / 11.8 | **18.9 / 34.9 (+6.7\*) / 13.3 (+5.1\*)** | +0.0 / +1.9\* / +1.5 |
| Unpruned | 18.5 / 28.0 / 8.6 | 19.8 / 31.5 / 11.2 | **21.5 / 32.7 (+4.6\*) / 13.3 (+4.6\*)** | +1.7 / +1.2 / +2.1 |
| HERMES 10% | 13.1 / 18.3 / 4.0 | 15.4 / 22.6 / 7.2 | 13.3 / **24.1 (+5.7\*)** / 5.9 (+1.9) | −2.1 / +1.5\* / −1.3 |
| Stratified 10% | 14.1 / 23.7 / 6.1 | 14.5 / 29.9 / 8.8 | **15.8 / 31.0 (+7.3\*) / 10.5 (+4.4\*)** | +1.3 / +1.0 / +1.7 |
| Streaming 4k | 13.5 / 21.6 / 4.0 | 17.1 / 27.4 / 8.2 | 15.4 / **28.6 (+6.9\*)** / 8.2 (+4.2\*) | −1.7 / +1.2 / +0.0 |
| Streaming 6k | 14.3 / 22.6 / 4.8 | 14.7 / 28.1 / 7.6 | **15.4 / 29.8 (+7.2\*) / 8.4 (+3.6\*)** | +0.6 / +1.7\* / +0.8 |
| **Mean of 8** | 15.3 / 24.0 / 6.1 | 16.5 / 29.2 / 9.3 | **16.6 / 30.7 / 10.0** | +0.1 / +1.5 / +0.7 |

- Higher mIoU than plain PCD on **all 8** memories (significantly on 4), higher or equal GQ@0.5 on 6 of 8, never
  significantly worse than PCD on any metric. Constant-strength time-scoped PCD has a slightly higher mIoU on three
  memories (unpruned 32.9, HERMES 10% 24.3, streaming 4k 29.2), within noise.
- The two exceptions are HERMES 10% and streaming 4k, the memories with the strongest prior leak (56% and 45% early
  starts): there constant-strength time-scoped PCD is a little better (14.7 / 24.3 / 7.4 and 16.2 / 29.2 / 9.3),
  consistent with the adaptive rule easing off when the model is (wrongly) confident about a leaked early time.
- On the early-evidence questions, the combination keeps most of the adaptive gain where the memory is rich (unpruned
  GQ@0.5 17.9 vs PCD 12.8; stratified 15.4 vs 11.5; streaming 4k 14.1 vs 11.5).

### 6.4 What "losing early starts" really means (key insight)

A natural way to measure the damage is "how often does the model start early on early-evidence questions". But the
model with **no video at all** starts early on 93.5% of them, and also on 89% of the other questions. A high score
there is mostly the prior, not evidence. The fair measure is **discrimination**: early starts on early-evidence
questions *minus* early starts on the others (J; higher = the model starts early *because* the evidence is early).

| Memory | No video | Baseline | PCD | Adaptive PCD |
|---|---|---|---|---|
| Random 10% | 4.2 | 49.4 | 51.1 | 50.1 |
| Random 25% | 4.2 | 49.5 | 52.1 | 51.6 |
| Unpruned | 4.2 | 49.6 | 53.2 | **56.4** |
| Random 5% | 4.2 | 41.9 | 48.8 | **51.8** |
| Streaming 4k | 4.2 | 44.9 | 39.8 | 45.7 |

PCD does not throw away evidence: it removes the prior's "free" early starts. Adaptive PCD gives the best
discrimination in most settings.

### 6.5 Is there a better decoding rule? No: PCD sits at the limit

We logged both probability distributions (with and without video) at the first time value of every answer and
replayed other decision rules offline (`logs/phase20/first_time_token.py`, 475 questions, random 10%):

| Rule | Early-evidence questions that start at "0" (want high) | Other questions that start at "0" (leak, want low) |
|---|---|---|
| no contrast | 69% | 16% |
| PCD α 0.5 | 46% | 6% |
| adaptive α_max 1 | 51% | 7% |
| evidence-against α 4 | 51% | 9% |
| skewed or tempered prior (best settings) | 50–55% | 8–11% |
| **best threshold chosen with the gold labels** | **46%** | **6%** |

Even a rule tuned on the answers cannot beat PCD's trade-off using the model's own probabilities. The remaining
early-start errors are cases where the video memory itself is unsure: they need better evidence near the start of
the video, not a cleverer decoding rule.

We also ran the most different rule end-to-end. **Evidence-against** penalises only tokens the video argues against
(log p_video < log p_no-video) and never rewards tokens for being unlikely without video. At random 10% it is
clearly worse than PCD (mIoU 26.5 / 27.3 for α = 1 / 2, i.e. −5.6\* / −4.8\* vs PCD), as the replay predicted. The
reason: with no video the model is ~99% sure of "0", so even a well-supported "0" looks like "evidence against".

### 6.6 Earlier ideas that did not work (same budget, random 10%)

| Method | Idea | Result |
|---|---|---|
| Visual-attention gain (PAI-style) | boost attention to visual tokens while decoding | mIoU +1.3, GQ +0.8 (n.s.); the prior stays (38% → 42% early starts) |
| Relevance-guided importance sampling | give more frames/tokens where grounding heads look | mIoU −1.5, Acc. −1.7 (n.s.); heads localise weakly |
| Coverage-then-zoom, gold window | extra frames inside the gold interval (an oracle) | mIoU +2.3 (n.s.): re-allocating a fixed budget is near its ceiling |
| Coverage-then-zoom, self window | extra frames around the model's first answer | Acc. +4.6\* in some settings but not robust; combined with PCD it is worse than PCD alone |

The common lesson: at a fixed budget, *where* tokens go matters less than the prior that fills the gaps.

### 6.7 Why PCD: a causal test and three supporting analyses

Scripts: `logs/phase21/motivation.py` (→ `motivation.md`) and the phase 21 runs. "Leak" = the interval starts in the
first 5% of the window although the evidence starts later.

**Causal test: remove the evidence from the memory.** We drop the uniform frames that fall inside the gold evidence
("gold"), or the same number of frames elsewhere ("control"), keep every other frame with its true timestamp, and
compare. Timeline prompt; Acc. / mIoU / GQ@0.5 and leak.

| Memory | Frames removed | Baseline | Leak | Final method | Leak |
|---|---|---|---|---|---|
| Random 10% | control | 16.6 / 28.8 / 8.8 | 31.2% | 15.4 / 34.6 / 10.5 | 11.1% |
| Random 10% | **evidence** | 7.8 / 8.6 / 1.7 | **45.6%** | 9.3 / 10.3 / 1.1 | **23.2%** |
| Unpruned | control | 19.6 / 29.4 / 10.1 | 37.3% | 18.1 / 35.9 / 12.6 | 15.4% |
| Unpruned | **evidence** | 9.1 / 9.4 / 1.9 | **50.6%** | 9.1 / 10.5 / 1.5 | **22.4%** |

- The leak is measured on questions whose evidence starts after the first 5% of the window (397 of 475); the 15
  questions whose evidence covers every sampled frame (no frame left after removal) are not among them.
- **Missing evidence causes the leak**: with the evidence removed, the baseline's answers move to the start of the
  video (+14 points of leak) instead of anywhere else. This is the prior taking over, measured directly.
- **PCD halves the fallback**, with and without the evidence. It cannot make up evidence that is not there (accuracy
  and GQ@0.5 stay near zero when the evidence is removed, for both methods), which is the expected behaviour of a
  decoding method that removes a bias rather than adding information.
- Caveat: the control condition removes frames chosen using the gold interval, so it also removes distractors; it
  scores higher than the unablated memory and is only a matched-size control for the leak.

**Supporting analyses on the existing runs.**

| Question | Finding |
|---|---|
| Is the leak a matter of token budget? | No: ~30% at every random budget including unpruned (33%); 49% for HERMES 10%, 37–39% for streaming, 88% with no video. Every PCD variant cuts it to about a third (e.g. 29.5% → 10.8% at random 10%). |
| Does the gain come from leaked answers? | Yes, disproportionately: on questions where the baseline leaked, mIoU rises by +8.9 [+6.5, +11.5] vs +5.8 [+4.0, +7.4] elsewhere; 25–41% of questions contribute 34–58% of the total gain. |
| Does writing what the prior would write predict errors? | Yes: the quarter of answers whose first time value has the lowest likelihood ratio log p(t \| video) − log p(t \| no video) (≤ 0, "the prior explains it") has the lowest mIoU (20.0 vs 23–29 for the other quarters) and leaks 60% of the time (9–15% for the others). |

### 6.8 Event-interval timeline: a better listing format

The error analysis (section 7) showed that the timeline lists single moments, so extents are too short. The `events`
prompt lists each distinct occurrence once, with its start and end (`Event: 12.0 - 15.5 seconds, …`); the contrast
applies to those start and end values. Random 10%; Acc. / mIoU / GQ@0.5.

| Prompt (token budget) | Baseline | Final method | Final vs baseline |
|---|---|---|---|
| timeline (384) | 15.2 / 25.8 / 7.2 | 16.2 / 32.6 / 10.1 | mIoU +6.8\*, GQ +2.9 |
| events (384) | 17.3 / 30.5 / 9.9 | 14.9 / 33.1 / 10.9 | |
| events (768) | 17.7 / **33.2** / 10.1 | 17.1 / **35.3** / 11.8 | mIoU +2.1, GQ +1.7 |
| events, merged instruction (768) | 15.8 / 31.7 / 9.9 | **17.9 / 35.0 / 12.6** | mIoU +3.3\*, GQ +2.7 |

- **The format alone is a large gain for the baseline**: mIoU +7.4\* and GQ@0.5 +2.9\* over the timeline, by
  stating extents. The best system, events (merged) + final method, reaches 17.9 / 35.0 / 12.6, i.e. +2.7 / +9.2 /
  +5.4 over the original timeline baseline.
- **The two fixes are complementary but overlap**: the format corrects extents (the baseline's leak is unchanged,
  31%), PCD corrects the leak (31% → 14%); on top of the events format PCD adds a smaller mIoU gain (+2.1 to +3.3)
  than on the timeline (+6.8).
- **Token budget**: at 384 tokens every run-on answer had hit the limit (an event line costs ~25 tokens; most are
  counting questions with many occurrences); 768 tokens removes most of them for the baseline (8.6% → 3.2%).
- **Fragmentation remains a PCD-specific failure**: with PCD, 12–13% of answers still run to the limit, and 84–86%
  of these split one continuing action into back-to-back fixed-length events with repeated descriptions. An explicit
  instruction to merge them does not help (13.3%), so the cause is in decoding, not in the instructions: the contrast
  on event boundaries rewards starting a new event where the previous one ended. This is the next thing to fix at the
  decoding level (for example by contrasting only the start of each event, not its end).

---

### 6.9 Comparison with upper bounds that use the gold time

Oracles that are given the gold interval when building the memory (timeline prompt, no PCD), against the final method
(no gold information). Acc. / mIoU / R@0.5 / GQ@0.5 and leak; Δ paired, 95% bootstrap.

| Memory | Acc. | mIoU | R@0.5 | GQ@0.5 | Leak |
|---|---|---|---|---|---|
| No video | 10.5 | 7.9 | 4.8 | 0.8 | 88.4% |
| Random 5%, baseline | 15.8 | 23.6 | 21.1 | 5.9 | 32.5% |
| Random 5%, **final method** | 16.4 | **31.5** | **31.6** | **9.9** | **12.1%** |
| Oracle selector 5% (gold frames' tokens kept first) | **17.1** | 27.0 | 24.6 | 7.8 | 34.8% |
| Random 10%, baseline | 15.2 | 25.8 | 24.2 | 7.2 | 29.5% |
| Random 10%, **final method** | 16.2 | **32.6** | **33.1** | **10.1** | **10.8%** |
| Oracle selector 10% | **17.5** | 27.9 | 25.7 | 8.4 | 32.5% |
| Gold-window zoom 10% (extra frames inside the gold) | 14.5 | 28.2 | 26.1 | 7.8 | 24.7% |
| Unpruned, baseline | 18.5 | 28.0 | 26.7 | 8.6 | 33.0% |
| Unpruned, **final method** | 21.5 | 32.7 | 34.1 | 13.3 | **13.1%** |
| Oracle window (all 64 frames inside the gold) | **26.3** | **35.3** | **36.2** | **13.9** | 45.1% |

- **At the same budget, the final method beats perfect token selection on grounding**: mIoU +4.7 [+1.9, +7.5]\* at
  10% and +4.5 [+1.7, +7.2]\* at 5% over the oracle selector, GQ@0.5 +1.7 / +2.1 (n.s.), accuracy −1.3 / −0.6 (n.s.).
  Knowing which tokens to keep does not stop the prior (the oracle selector still leaks 32.5%); removing the prior is
  worth more than selecting the evidence perfectly.
- **Against the strongest oracle**, which sees only the evidence, the final method (without gold information) closes
  most of the grounding gap: GQ@0.5 13.3 vs 13.9 (−0.6, n.s.), mIoU 32.7 vs 35.3 (−2.6, n.s.). The remaining gap is in
  answers: accuracy 21.5 vs 26.3 (−4.8\*), mostly duration (oracle mIoU 48.2 vs 40.3), where dense evidence frames
  give precise boundaries.
- **Even the oracle window leaks (45%)**: with every frame inside the evidence, the model still often starts its
  interval near 0 s, where it sees only timestamp text. More evidence alone does not remove the prior, which is what
  PCD targets; the two should be complementary.

## 7. Error analysis of the final method and what to do next

Script and full tables: `logs/phase21/error_analysis.py` → `logs/phase21/error_analysis.md`. All 8 memories pooled
(3,800 question instances); Δ with a cluster bootstrap over questions (\* significant).

### 7.1 Where answers fail, before and after

| Outcome of a question | Baseline | Final | Δ |
|---|---|---|---|
| IoU ≥ 0.5 and answer correct (GQ) | 6.1% | 9.9% | +3.8\* |
| IoU ≥ 0.5 but answer wrong | 16.3% | **20.7%** | +4.4\* |
| partial overlap (IoU < 0.5) | 17.0% | 15.7% | −1.3 |
| interval inside gold but too short | 22.2% | **18.5%** | −3.7\* |
| interval entirely **before** gold | **33.1%** | **18.9%** | **−14.2\*** |
| interval entirely after gold | 2.7% | 5.7% | +3.0\* |
| interval covers gold, too long | 2.2% | 2.5% | +0.3 |
| no `Answer:` line (list runs on) or unparsed interval | 0.5% | **8.1%** | +7.6\* |

- **The method works where it was aimed**: "before gold" (the prior pulling the interval to the start) drops from
  the largest error to a third of its size, and intervals starting early although the evidence is later fall from
  34.7% to 14.0%. It fixes 13.8% of instances (IoU < 0.5 → ≥ 0.5) and breaks 5.5%.
- **The bottleneck has moved from grounding to answering**: twice as many answers are well grounded but wrong (20.7%)
  as grounded and right (9.9%).

### 7.2 Which questions are hard

| Slice | mIoU base → final | GQ@0.5 base → final | Reading |
|---|---|---|---|
| Duration | 32.5 → 37.0 | 10.4 → **17.6** (+7.2\*) | the main beneficiary |
| Counting | 26.1 → **36.8** (+10.7\*) | 4.8 → 6.8 | grounding improves a lot, the count does not (accuracy 15.6 → 13.7) |
| Location | 3.9 → 7.2 | 0.4 → 1.1 | essentially unsolved |
| Evidence starts in the last third of the window | 13.4 → 16.3 | 1.2 → 4.4 | recent events are the hardest |
| Evidence shorter than 10% of the window | ≤ 12.7 | ≤ 2.8 | short events are rarely grounded |
| Fewer than 4 of the 64 sampled frames fall in the evidence | 6.0 → 8.4 | 0.5 → 1.6 | the evidence is barely in the input |
| ≥ 16 frames in the evidence | 33.1 → **43.2** | 10.4 → 14.8 | gains are largest when evidence is visible |
| Annotators disagree (pairwise IoU < 0.6, 98 questions) | ~8 → ~11 | ~2 → ~3 | label ambiguity caps these |
| HERMES 10% memory | — | — | most "before gold" errors (26.7%): pruning removed the evidence |

### 7.3 Five insights, each with a concrete next step

1. **The contrast fragments continuous events.** The new run-on failure (6.2% of answers, half of them counting
   questions) has one shape: a list of one ongoing activity at one-second steps ("5.5 s holds the pens, 6.5 s
   continues to hold…", median 22 lines; times non-decreasing in 96%, descriptions repeated in 96%). The no-video
   prior prefers coarse, round jumps; the contrast rewards the fine steps the prior finds unlikely. The same
   mechanism fits the rise in over-counting (14% → 35% of counting answers) and plausibly the extra "after gold"
   errors (2.7% → 5.7%).
   **Next step: an event-interval timeline**, where the model lists distinct events with start and end
   (`Seen: 5–15 s, …`) and the contrast acts on those boundaries. One line per event removes per-second
   enumeration, gives the count directly as the number of listed events, and states each event's extent, which also
   targets the next error.
2. **Extents are under-enumerated.** Intervals inside the gold but too short remain 18.5%: a median 19% of the gold
   length, 25% a single moment. The model finds the event but reports one moment of it. **Next step**: the same
   start–end format; beyond it, ask about each sampled timestamp near the found event whether the event is still
   ongoing (evidence-wise extent).
3. **The wrong occurrence is chosen for recent events.** 18.9% of intervals still end before the gold; 81% of these
   have their evidence in the second half of the window, with a median gap of 110 s. These questions refer to a
   specific instance ("the second time", "after I…", "until now"), and an earlier similar event is picked; evidence in
   the last third of the window has GQ@0.5 of only 4.4. **Next step**: list all occurrences first, then resolve the
   reference among them; the analysis gives the selection target (latest vs ordinal instance) per question.
4. **Short or sparsely sampled evidence is a perception limit, not a decoding one.** With fewer than 4 of 64 frames
   inside the evidence, GQ@0.5 is about 1–2% with or without the method; with 16 or more it is 15%. No decoding
   rule recovers evidence that is not in the input (section 6.5 shows the same for early starts). **Next step**: an
   evidence-adaptive second pass that adds frames around the model's own candidate moments for short events. The
   earlier global zoom (6.6) re-allocated a fixed budget across all questions; this would target only short evidence.
   HERMES pruning, which loses the most evidence (26.7% "before gold"), would benefit from a coverage floor as in
   stratified selection (31.0 vs 24.1 mIoU with the same method).
5. **Answers, not intervals, now limit GQ@0.5.** For counting, 31.5% of answers have a good interval and a wrong count
   (61% under, 39% over): the count is not taken from the events the model itself listed. For duration, half of the
   well-grounded but wrong answers are within 25% of the gold duration, and the stated duration matches the model's
   own interval 97% of the time. So duration errors follow from interval precision (an IoU of 0.5 still allows a 2×
   length error), and counting errors from enumeration. **Next step**: the event-interval format makes both the
   count and the duration follow from the listed events inside the model's own answer. Rule-based rewriting of
   answers is excluded by design.

### 7.4 Evaluation caveats found along the way

- **Label ambiguity**: on the 98 questions where annotators' intervals overlap little (mean pairwise IoU < 0.6), every
  method scores near zero (GQ@0.5 ≤ 3%). 62 of them are location questions, i.e. 63% of all location questions, which
  largely explains why location stays unsolved. We report agreement-stratified results.
- **Decoding noise**: the same configuration decoded with another matrix shape moves accuracy by up to ~2 points and
  GQ@0.5 by ~1 point, with mIoU stable to ±0.3 (section 8), so mIoU is the most reliable metric for small differences.

---

## 8. Recommended configuration (final)

**Timeline prompt + PCD against the no-video prior, with confidence-adaptive strength (α_max = 1, β = 0.1), applied
only to time values** (`run.contrastive_mode=blind run.contrastive_alpha=1.0 run.contrastive_adaptive=true
run.contrastive_scope=time`).

- One configuration for every memory: higher mIoU than plain PCD on all 8, mean mIoU +6.7 and GQ@0.5 +3.9 over the
  baseline (6.3a).
- Each part has its own evidence: the no-video reference removes the prior (6.1), the time scope keeps the answer
  text intact (6.2), the adaptive strength protects evidence-supported early starts (6.3–6.4).
- Plain PCD (α = 0.5, all tokens) remains the simplest version and keeps most of the gain (mean mIoU 29.2, GQ@0.5 9.3);
  it is the natural ablation baseline in the paper.
- For memories with a very strong leak (HERMES 10%, streaming 4k) constant-strength time-scoped PCD is marginally
  better; the difference is within noise.

Inference speed: `run.batch_size=8 run.prefetch_videos=4` decodes the answers of 8 questions together and loads video
frames in background threads (2.8× faster for greedy decoding, 4.0× for PCD on an A100-40GB; `logs/batching/`).
Batch size 8 vs 1 on all 475 questions (Acc. / mIoU / GQ@0.5): plain decoding 15.8 / 25.6 / 6.5 vs 15.2 / 25.8 / 7.2;
adaptive time-scoped PCD 18.1 / 32.6 / 10.9 vs 16.2 / 32.6 / 10.1; no paired difference is significant. Answers
differ only where the two best tokens are within bf16 rounding (near-ties), which still changes the wording of 38%
(greedy) and 52% (PCD) of the answers.

**Noise floor.** The same configuration decoded with a different matrix shape moves accuracy by up to ~2 points and
GQ@0.5 by ~1 point, with mIoU stable to ±0.3. Differences between methods of that size in accuracy or GQ@0.5 should be
read as decoding noise; mIoU differences are much more reliable. This floor covers numerical noise only: rewording the
prompt is a larger effect (the two event prompts differ by 1.5 mIoU for the baseline, 6.8), so prompt variants should
be compared with that in mind.

---

## 9. Reproduce

| Step | Command / file |
|---|---|
| Run a configuration | `python3 scripts/run.py model=qwen3_vl_8b experiment=sember_grounding_uniform dataset.max_videos=300 run.uniform_num_frames=64 run.keep_time_tokens=surviving dataset.grounding_prompt=timeline run.max_new_tokens=384 run.offline_keep_ratio=0.1 run.prune_score=random run.contrastive_mode=blind run.contrastive_alpha=1.0 run.contrastive_adaptive=true run.contrastive_scope=time` |
| All options | `run.contrastive_mode` (none / blind / noise / stamps), `run.contrastive_alpha`, `run.contrastive_beta`, `run.contrastive_scope` (all / time / time+answer), `run.contrastive_adaptive`, `run.contrastive_rule` (pmi / against), `run.contrastive_trace` |
| Experiment specs (Slurm, chunked) | `logs/phase18/`, `logs/phase19/`, `logs/phase20/make_spec*.py` with `logs/sched.py` |
| Result tables | `python3 logs/phase18/report.py` → `logs/phase18/report.md`; `python3 logs/phase20/early.py` → `logs/phase20/early.md` |
| Decode-rule replay | `python3 logs/phase20/first_time_token.py` |
| Paper tables | `python3 tables/make_pcd_tables.py` → `tables/pcd_main.tex`, `tables/pcd_ablation.tex` |

Limitations: one model and one benchmark subset by design; the judge is a local model with the official judge prompt
(not the official Gemini judge); PCD needs the reasoning-style timeline prompt.
