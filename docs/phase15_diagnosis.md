# Why grounded video QA fails under token pruning: diagnosis (phases 15–16)

This report asks where, when, how and why Qwen3-VL-8B fails at grounded question answering on S-EMBER, and
how much of that failure comes from token pruning versus the model itself. Each experiment below lists its
setup, its result and what we learn from it. A summary comes first; implications for methods come last.

## Summary

1. **Pruning is not the main bottleneck; the model's reading of the evidence is.** A perfect selector at 10%
   of the tokens only matches the unpruned model, and 37% of questions are never answered correctly by any of
   100 runs. About half of the correct answers can be produced without any video.
2. **Denser evidence helps.** Showing all 64 frames inside the evidence interval (and nothing else) raises
   accuracy by +7.4 to +7.8 points and mIoU by about +7 points. How densely the evidence is seen matters; how
   much of the rest of the video is kept does not.
3. **HERMES pruning collapses because of recency.** At 10% and 5% of tokens it puts about 70% of its budget on
   the last few frames; the rest of the timeline is starved and predicted intervals fall back to the video
   start. No selector (HERMES, random, stratified, pooling) keeps more evidence than chance.
4. **The model tells time by reading the timestamp text, and it has a strong early prior.** Without the text
   stamps grounding collapses (mIoU 26.9 → 5.0); shifted stamps shift its answers; it invents "0.0 s" moments
   and starts intervals before any frame it was shown.
5. **Attention is U-shaped over the video (primacy and recency), and the model writes timestamps mostly from
   text.** A few "grounding heads" (layers 20–21) do look at the evidence, but they do so about equally when
   the output is right or wrong.
6. **Labels are reliable except for location questions**, where annotators agree only moderately.

## Common setup

- **Model and data.** Qwen3-VL-8B, S-EMBER grounded QA: 475 questions (duration 191, counting 184, location
  100) on 300 videos. Each question is asked at a moment in the video (the question time); the answer must
  come from the video before that moment.
- **Default input ("unpruned").** 64 frames sampled uniformly from 0 s to the question time, about 21.4k
  visual tokens. Qwen3 encodes frames in pairs; each pair is preceded by its timestamp as text,
  `<t seconds>`.
- **Prompts.** *Official*: the S-EMBER prompt (answer, then `Time: [start, end]`). *Timeline*: the model
  first lists up to 8 moments as `Seen: <t> seconds, …`, then answers and gives the interval from the first
  to the last moment.
- **Metrics.** Acc.: answer judged correct with the official S-EMBER judge prompt (local Qwen judge).
  mIoU and R@0.5: overlap of the predicted interval with the gold interval. GQ@0.5: answer correct *and*
  IoU ≥ 0.5 (the official joint metric).
- **Gold frames.** Sampled frames whose time lies inside the gold interval (first annotator). On average
  they are about a third of the 64 frames.
- **Significance.** Differences marked `*` have a 95% paired bootstrap interval over questions that excludes
  zero. One question is worth about 0.2 points.

---

## Experiment 1: error decomposition of existing runs (no GPU)

**Setup.** Existing outputs of about 100 judged runs (unpruned, pruned, streaming, both prompts) were broken
down by question category, position of the evidence, interval error type and answer type.

**Results.**

| Breakdown (unpruned, official → timeline prompt) | Acc. | mIoU | GQ@0.5 |
|---|---|---|---|
| Duration | 8.4 → 22.5 | 26.4 → 37.6 | 3.1 → 14.7 |
| Counting | 15.8 → 16.3 | 33.1 → 30.1 | 6.0 → 6.5 |
| Location | 19.0 → 15.0 | 16.3 → 5.9 | 4.0 → 1.0 |

| Evidence start (as a fraction of the question time) | 0–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 |
|---|---|---|---|---|
| mIoU, unpruned (official) | 40.4 | 28.8 | 18.0 | 15.2 |
| mIoU, HERMES 5% (official) | 26.6 | 12.9 | 4.4 | 9.0 |

- **Interval errors** (unpruned, official): 34% of predictions lie inside the gold interval but are too
  short, 25% end before the gold interval starts, 23% overlap partially, 8% cover it, 7% lie after it.
  Predicted intervals are half the gold length (median 14 s vs 43 s). With HERMES at 5% of tokens,
  "entirely before the gold interval" rises to 49% (official) and 60% (timeline).
- **Stock answers.** With the official prompt, 26% of stated durations are exactly "10 seconds" and 18% "90
  seconds".
- **Counting.** The model undercounts in 69% of counting questions (median answer 3 vs gold 5). With the
  timeline prompt the count equals the number of `Seen` lines in 79% of answers; the median list has 3
  entries, and 16% of questions have a gold count above 8 (the list limit), of which 97% are undercounted.
- **Location with the timeline prompt.** 55% of location intervals collapse to a single point (the model lists
  one moment only).
- **Hard core.** 36.6% of questions are never answered correctly by any of the 100 runs; 61.7% never reach
  GQ@0.5. About 20% of questions get IoU ≥ 0.5 with a wrong answer.

**Insight.** Failures concentrate where the evidence lies late in the window, on counting (undercounting) and
on location (point intervals under the timeline prompt). Many failures are shared by every configuration,
which points to the model rather than to any memory setting.

## Experiment 2: blind baseline (no video)

**Setup.** Same questions and prompts, but no frame is encoded (`run.blind=true`).

**Results.** Accuracy 6.9% (official) and 10.5% (timeline), against 13.5% and 18.5% with video. Grounding
collapses (mIoU 3.4 official). The blind model gives the same stock answers ("1 minute and 30 seconds").

**Insight.** About half of the correct answers come from language priors (plausible durations, common
locations). Grounding does need the video.

## Experiment 3: label audit

**Setup.** Each question has up to three annotator answers with their own interval. We measured their
agreement and re-scored predictions against the best-matching annotator.

**Results.** Mean pairwise interval IoU between annotators is 0.76 (duration 0.86, counting 0.81, location
0.50); duration answers differ by a median of 7%. Scoring against the best-matching annotator raises the
unpruned mIoU from 26.7 to 30.8 and R@0.5 from 23.6 to 28.6.

**Insight.** Labels are consistent, so the gap to the human-agreement ceiling (IoU ≈ 0.76) is real. Location
labels are ambiguous, which explains part of the location failure.

## Experiment 4: oracle pruning (perfect selection at the same budget)

**Setup.** Offline pruning of the 64-frame input to 10% or 5% of the visual tokens, but tokens of gold frames
are kept first (`run.prune_score=oracle`). Tokens are spread evenly across gold frames; only if the budget
exceeds the gold frames does anything else survive. In practice 75% (10%) and 92% (5%) of the other frames
keep no token. Timestamps survive only for frames that keep tokens, so this also reveals where the evidence
is. This uses the labels and is a diagnostic, not a method.

**Results (timeline prompt).**

| | Acc. | mIoU | GQ@0.5 |
|---|---|---|---|
| Unpruned | 18.5 | 28.0 | 8.6 |
| Oracle 10% / 5% | 17.5 / 17.1 | 27.9 / 27.0 | 8.4 / 7.8 |
| Random 10% | 15.2 | 25.8 | 7.2 |
| Stratified 10% | 14.1 | 23.7 | 6.1 |
| HERMES 10% | 13.1 | 18.3 | 4.0 |

Oracle 10% vs HERMES 10%: GQ@0.5 +4.4*, mIoU +9.6*. Vs random 10%: +1.3 GQ@0.5, +2.1 mIoU (not
significant). Vs unpruned: no significant difference.

**Insight.** Perfect selection at 5–10% of the tokens recovers the unpruned result but not more. The pruning
loss is what the selector throws away, and a simple selector that covers the whole video (random) already
recovers most of it. HERMES is the exception.

## Experiment 5: evidence-window oracle (upper bound on memory)

**Setup.** 64 frames sampled uniformly *inside* the gold interval only (clipped to the question time, at least
1 s). Outside it, the model sees only timestamp text every 5 s (0.2 fps) from 0 s to the question time, so the
clock is complete but only the evidence has pixels. No pruning (`run.oracle_window=true`). Compared with
Experiment 4, the evidence is seen much more densely (all 64 frames instead of the ~15–20 that fall inside it
under uniform sampling) and there are no distractor frames.

**Results.**

| | Acc. | mIoU | R@0.5 | GQ@0.5 |
|---|---|---|---|---|
| Unpruned, official / timeline | 13.5 / 18.5 | 26.9 / 28.0 | 25.3 / 26.7 | 4.4 / 8.6 |
| Evidence window, official | 20.8 (+7.4*) | 34.3 (+7.4*) | 34.3 | 7.6 (+3.2*) |
| Evidence window, timeline | 26.3 (+7.8*) | 35.3 (+7.2*) | 36.2 | 13.9 (+5.3*) |

By category (timeline): duration GQ@0.5 14.7 → 24.1, counting 6.5 → 9.8, location 1.0 → 2.0. Even so, 43% of
the predicted intervals extend outside the only region that has frames, and 27–40% start more than 5 s before
it.

**Insight.** Seeing the evidence densely raises the ceiling substantially, so a memory that keeps the evidence
at high temporal resolution would pay off. But with ideal evidence the model still scores only 26% accuracy
and 35 mIoU and places intervals where it saw no frame: it writes times from the text timeline and its prior,
not only from what it sees.

## Experiment 6: how much evidence each selector keeps

**Setup.** For each selector and budget, the share of kept visual tokens that come from gold frames, from
per-frame retention logs (`run.retention_snapshot=true`) or, for uniform frame dropping and pooling, from the
sampled frame times.

**Results.**

| Selector (budget) | % of kept tokens in gold frames | % of gold-frame tokens kept | Evidence lost |
|---|---|---|---|
| Unpruned | 33.5 | 100 | – |
| Spatial pooling ×0.7 / ×0.5 / ×0.35 | 33.5 | 49 / 25 / 12 | – |
| Uniform frames 32 / 16 / 6 / 4 | 33.1 / 32.5 / 30.0 / 27.0 | 50 / 24 / 9 / 5 | no gold frame at all in 2 / 8 / 23 / 35% of questions |
| Random 10% | 33.4 | 8.6 | – |
| Stratified 10% | 33.5 | 8.6 | – |
| HERMES 10% / 5% | 32.3 / 28.8 | 13.4 / 5.9 | 0.1 / 5.8% of gold frames get no token |
| Oracle 10% / 5% | 89.0 / 97.4 | 52.6 / 34.0 | – |

**Insight.** No real selector is evidence-aware: all keep evidence at the rate it occurs in the video (about a
third). Dropping whole frames loses the evidence entirely for a quarter to a third of questions at small
budgets, which token-level selectors never do.

## Experiment 7: where HERMES spends its budget

**Setup.** Retention logs of offline pruning, grouped by the position of each frame in the video (tenths).
We also tested `hermes_exact`, which scores with the probe questions properly propagated through all layers
(the default HERMES scorer reuses the input embeddings as queries at every layer).

**Results.**

| Pruner | Kept tokens in the last 10% of frames | Median predicted start / question time | mIoU (official) |
|---|---|---|---|
| Stratified 10% | 9% (even) | 0.36 | 25.9 |
| HERMES 10% | 68% | 0.10 | 23.3 |
| HERMES 5% | 75% | 0.04 | 14.8 |
| hermes_exact 10% / 5% | 74% / 81% | 0.01 / 0.00 | 12.0 (−11.3*) / 8.3 (−6.5*) |

**Insight.** HERMES's recency term (pure recency in early layers) and attention's recency bias spend about 70%
of the budget on the final few frames. The rest of the timeline keeps a sliver, and the model then puts its
intervals at the video start. Exact scoring makes the concentration stronger and the results worse, so the
embedding shortcut is not the cause; the recency bias is. This is the positional bias described for attention
pruning in the literature.

## Experiment 8: how the model reads time (timestamp counterfactuals)

**Setup.** On the unpruned input: (a) remove the timestamp text (`run.drop_timestamps=true`); (b) add +200 s
to every timestamp (`run.time_offset=200`); (c) permute the timestamps across frame pairs while keeping the
frames in order (`run.shuffle_mode=stamps`); (d) permute frame pairs together with their timestamps
(`run.shuffle_mode=frames`).

**Results.**

| Test | Result |
|---|---|
| (a) No timestamp text | mIoU 26.9 → 5.0, GQ@0.5 4.4 → 0.6; accuracy 13.5 → 10.9 |
| (b) Timestamps +200 s | 56% (official) / 47% (timeline) of predictions follow the shift; mIoU after undoing it 20.1 / 17.0 |
| (c) Timestamps permuted | 27% of predictions land at the (wrong) timestamps shown on gold frames, 7% at the true time |
| (d) Frames and timestamps permuted together | Accuracy unchanged; mIoU −3.9* (official), −7.6* (timeline) |

In the question-attention run (Experiment 10), early layers give 45% of their video attention to timestamp text
(about 1% of the video tokens); late layers 8%. In `Seen` lists, only 12–23% of the times copy a shown
timestamp exactly (median 1.4 s off), and a "0.0 s" moment that was never shown opens 18% of answers unpruned
and 41% with HERMES at 5%.

**Insight.** The timestamp text is the model's clock: without it the model cannot ground, and when it is
wrong the model follows it. The frame order also matters (d), and a large share of predictions ignore the
shift (b), falling back to a prior over times.

## Experiment 9: primacy (what the model does when the start is not shown)

**Setup.** Frames sampled only from half the question time onward (`run.uniform_start_frac=0.5`); the first
half of the video is not shown and has no timestamps.

**Results.** 59% (official) and 65% (timeline) of predicted intervals start in the first 10% of the shown
window, and 38–41% start before it, at times the model never saw. For the 13% of questions whose evidence lies
entirely before the window, accuracy is 13% (official) and 8% (timeline).

**Insight.** The model has a strong prior for early times: it anchors intervals at the earliest time it can see,
or earlier. This explains the early, short intervals of Experiment 1 and why they get worse when pruning
empties the middle of the video.

## Experiment 10: where the question looks (question attention)

**Setup.** After encoding the 64 frames, the question text alone is run through all layers (exact
propagation) and its attention to each frame and to the timestamp text is recorded
(`run.question_attention=true`). **Lift** = attention share on gold frames ÷ gold frames' share of frames
(1 = no preference for the evidence).

**Results.** Median lift 1.0 (early layers), 1.16 (middle), 1.09 (late). Late-layer attention by position in
the video is U-shaped: first tenth 14%, last tenth 16%, each middle tenth about 7%. The first frame pair (3% of
frames) takes 4–7% of the attention. Lift is the same whether the answer is right or wrong (1.37 vs 1.40).

**Insight.** Averaged over the model, the question barely targets the evidence, and attention favours the
start and the end of the video ("lost in the middle" over frames). This matches the drop in grounding for
evidence in the middle-to-late part of the window.

## Experiment 11: perception vs generation (grounding heads)

**Setup.** Following *MLLMs Know When Before Speaking* (arXiv 2605.21954). After the model answers, the cache
is cropped back to its pre-answer state and the prompt plus the model's own answer are replayed; for every
head (36 layers × 32 heads) we record the share of visual attention on gold frames in three groups of tokens:
reading the prompt, writing the answer text, and writing `Time: [...]` (`run.answer_attention=true`). The 10
heads with the highest lift while writing timestamps are chosen on half of the videos and evaluated on the other
half (234 questions). "Right" = output IoU ≥ 0.5; "wrong" = IoU < 0.1.

**Results.**

| Measure | Result |
|---|---|
| Grounding heads | Layers 20–21 (one in 24); lift 2.4 while writing timestamps vs 1.2 for all heads |
| Same heads, output right vs wrong | 2.38 vs 2.13 (timestamp tokens); 1.41 vs 1.42 (prompt tokens) |
| Wrong outputs whose heads still favour gold (lift > 1.5) | 28% |
| Share of attention on video | 9% while reading the prompt, 5.7% while writing timestamps |
| Same heads after pruning to 10% | Lift 1.55 (HERMES), 1.38 (stratified) vs 2.42 unpruned |

**Insight.** Qwen3 has a small set of heads that locate the evidence, but whether they do barely predicts
whether the written interval is right; in 28% of wrong outputs they looked at the right place. While writing
timestamps the model attends mostly to text (prompt, its own `Seen` lines, timestamp text). This is a
perception–generation gap, smaller than in the source paper. Pruning weakens these heads' signal.
(Caveats: attention shows where the model looks, not what it uses; the lift under HERMES partly reflects where
tokens survived.)

---

## Synthesis

| Question | Answer |
|---|---|
| **Where** does it fail? | Evidence in the middle or late part of the window; counting (undercounting); location (ambiguous labels, point intervals); long events (intervals too short) |
| **When** in the pipeline? | Mostly at reading and writing, not at memory: oracle selection only matches unpruned, blind answers are half as good as seeing, and wrong outputs often had the evidence in view. Memory matters in two ways: HERMES starves the timeline, and dense evidence (Experiment 5) raises the ceiling |
| **How** does it fail? | Early and short intervals; "0.0 s" and intervals outside what was shown; stock durations ("10 s", "90 s"); counts limited by how many moments it lists |
| **What** information is lost? | Under HERMES, the whole timeline except the last few frames; under uniform frame dropping, the evidence itself at small budgets; under random and stratified, detail only |
| **Why**? | (1) Recency bias in HERMES scoring; (2) a primacy prior over time and U-shaped attention over frames; (3) time is read from the timestamp text and written from text context rather than from the frames; (4) language priors for answers |

## Implications for methods

1. **Fix the selector's position bias first.** Debiasing recency (or simply keeping every part of the video
   covered, as random and stratified do) removes the HERMES collapse; this is where pruning loses most.
2. **Keep the evidence dense, not just present.** The evidence window beats the oracle at the same evidence
   location by +7 mIoU: a memory that concentrates frames or tokens on relevant moments has room to gain.
3. **Timestamps are the only clock.** Protect them under pruning, and counter the early/0 s prior in
   prompting or decoding (for example, only allow times that were shown).
4. **Read then regenerate.** The grounding heads (layers 20–21) carry a localisation signal the output does not
   fully use; cropping to where they look and asking again is a training-free follow-up to test.
5. **The largest headroom is reasoning.** Counting, stock durations and language priors limit accuracy even
   with perfect evidence.

## Reproduction

| Experiment | Runs (results/qwen3_vl_8b/sember_grounding/…) | Analysis |
|---|---|---|
| 1 | existing runs | `logs/phase15/diagnose_existing.py` |
| 2, 8 (c, d), 11 | `uniform-n64-time-count-location-dx-{blind,shufframes,shufstamps,aattn,…}` | `logs/phase15/diagnose_b.py` |
| 3 | unpruned runs | `logs/phase15/label_audit.py` |
| 4, 7, 8 (a, b), 9, 10 | `…-offline-oracle-*`, `…-dx-ret-*`, `…-offline-hermes_exact-*`, `…-dx-{nostamp,offset200,start50,qattn}` | `logs/phase15/diagnose_gpu.py` |
| 5 | `uniform-n64-time-count-location-oracle-window{,-timeline}-v300-native-time` | this report (paired bootstrap as in `logs/phase13/compare.py`) |
| 6 | retention logs above | `logs/phase16/evidence_share.py` |

Specs: `logs/phase15/make_spec.py`, `logs/phase15/make_spec_b.py`, `logs/phase16/make_spec.py`. All
diagnostic options are off by default (`configs/config.yaml`, section "Diagnostics").
