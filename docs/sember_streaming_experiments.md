# Streaming Memory for Egocentric Video QA: Frame Sampling Experiments on S-EMBER

This document describes, in order, every experiment we ran to understand and improve **HERMES
streaming memory** on the **S-EMBER** benchmark: why each experiment was needed, how it was designed,
what it found, and how each result shaped the next step. It is written for readers who have not
followed the project day to day.

All experiments, code and analysis scripts live in this repository (see
[Reproducing the results](#12-reproducing-the-results)).

---

## Contents

1. [Summary](#1-summary)
2. [Background](#2-background)
3. [Experimental setup](#3-experimental-setup)
4. [Chapter 1: Baselines and first ablations](#4-chapter-1-baselines-and-first-ablations)
5. [Chapter 2: Where does streaming lose?](#5-chapter-2-where-does-streaming-lose)
6. [Chapter 3: Changing how frames are sampled (rounds 1-4)](#6-chapter-3-changing-how-frames-are-sampled-rounds-14)
7. [Chapter 4: Explaining the gains and the losses (phase 7)](#7-chapter-4-explaining-the-gains-and-the-losses-phase-7)
8. [Chapter 5: Combining the MCQ wins (round 8)](#8-chapter-5-combining-the-mcq-wins-round-8)
9. [Chapter 6: Temporal grounding under token pruning (phases 9-10)](#9-chapter-6-temporal-grounding-under-token-pruning-phases-910)
10. [All results in one table](#10-all-results-in-one-table)
11. [Conclusions and open questions](#11-conclusions-and-open-questions)
12. [Reproducing the results](#12-reproducing-the-results)

---

## 1. Summary

**The problem.** A streaming video assistant sees frames one at a time and must answer questions at
any moment, from a memory of bounded size. HERMES keeps that memory as a compressed KV cache. On
S-EMBER, streaming HERMES with Qwen3-VL-8B at a fixed 0.2 fps did **no better than simply sampling 32
frames uniformly after the fact**. It lost clearly on grounding (24.8 vs 27.5 mIoU), and lost badly on
questions asked early in a video.

**What we found.**

1. **Early questions are the weak spot.** When a question arrives 60 s into a video, a 0.2 fps stream
   has seen 12 frames while uniform sampling has 32-64. Streaming loses by 10-19 points there and wins
   after about 5 minutes.
2. **Sampling densely early fixes multiple-choice QA.** Early-dense schedules and a new
   *grid-thinned* sampler raise Qwen3 MCQ accuracy from **25.7% to 31.4-33.0%**, 6-8 points above
   both the streaming baseline and uniform sampling. The best configuration (`dup:1.0:60,0.2` at 0.7x
   resolution, KV 10.7k) reaches **33.0%** (+7.3, 95% CI +3.6 to +10.8). Grid 32:8 gets 31.4% for about
   1.3x the baseline's GPU time.
3. **Part of the gain comes from a Qwen-specific effect.** Qwen3 merges every two consecutive frames
   into one set of tokens. At 0.2 fps those two frames are 5 s apart, so every token group blends two
   different moments. Feeding each frame twice (so a group holds one moment) raises MCQ by 2.9 points at
   the same token cost. LLaVA-OV, which does not pair frames, gains nothing from denser sampling.
4. **Grounding does not improve.** No streaming variant beats the 0.2 fps baseline on temporal
   grounding. Grounding gets worse as more frames share the budget, and it needs detailed frames.
5. **Grounding's largest error is interval length, not memory.** Every method, including uniform
   sampling, predicts intervals 3-4x shorter than the annotated evidence. This is a model bias, present
   with or without pruning. Correcting lengths after the fact adds 7-10 mIoU to every method, far more
   than any memory change. Asking for full-length intervals in the prompt does not work.
6. **Under token pruning, grounding suffers from lost temporal coverage, not lost boundaries.**
   Randomly dropping 90% of visual tokens costs only 0.8 mIoU, but attention-based selection (HERMES
   score) concentrates tokens on few frames and collapses at aggressive budgets (14.8 mIoU at 5%).
   Selection that keeps an equal share of every frame ("stratified") is robust.
7. **The model reasons from one 10-second token group, so its interval *and* its answer are too
   narrow.** 20% of streaming intervals are exactly 10 s (one Qwen group at 0.2 fps) and 41% of
   duration questions are answered "10 seconds". Post-processing that widens intervals (fixed x4, or
   retrieval-based refinement "M3") raises mIoU by up to 12 points but leaves the answers wrong, so it
   is reported only as a diagnostic, never as a method (Section 9.5). Grounding must be judged together
   with answer correctness (Section 9.7).
9. **Reasoning over the timeline improves grounded QA.** A prompt that makes the model list timestamped
   moments before answering raises answer accuracy (13.3 to 18.9% with uniform sampling) and GQ@0.5
   (5.3 to 8.8%; streaming `dup:1.0:60,0.2`: 2.9 to 6.7%), by fixing the answer rather than the output.
8. **Under the official S-EMBER protocol, grounded QA is very hard.** Only 9-16% of open-ended answers
   are judged correct (GQ@0.0 = 9-16%) and GQ@0.5 is 1.5-5.5% for every configuration. Open-ended answer accuracy tracks
   MCQ: the best MCQ configuration also gives the best answer accuracy (15.6%, +3.2 over the baseline).

**The resulting picture:** there is a real trade-off. Multiple-choice QA wants *many frames*, sampled
densely where the video is new. Grounding wants *temporal coverage and detailed frames* and, above all,
better interval prediction; improvements must come from what the model sees and reasons over, not
from post-processing its output.

---

## 2. Background

### 2.1 The task: S-EMBER

S-EMBER is a streaming benchmark of long egocentric videos (a person's first-person view while doing
everyday activities, typically 5-10 minutes). Each question is asked **at a specific time** in the
video, and the model may only use video seen up to that moment. Questions come in three categories:

- **Time / duration:** "How long did I spend using the paper cutter?"
- **Counting objects or events:** "How many boxes did I open?"
- **Location trace:** "Where did I first see my phone?"

Every question exists in two formats:

- **MCQ:** choose one of 5 options (chance = 20%). Metric: **accuracy**.
- **Grounding:** answer in free text *and* give the time interval `[start, end]` where the evidence
  appears. Metrics: **mIoU**, the mean temporal overlap (intersection over union) between predicted and
  annotated intervals, and **R@0.5**, the share of questions whose interval overlaps the annotation with
  IoU >= 0.5.

### 2.2 HERMES streaming memory

HERMES turns an offline video-language model into a streaming one:

1. Frames arrive in time order and are encoded in chunks of 16 into the language model's **KV cache**.
2. When the cache exceeds its **budget** (e.g. 6,000 tokens), HERMES **compresses** it: it scores
   visual tokens (attention to predicted future questions, plus a recency bonus that varies by layer)
   and keeps the best ones. Layers deep in the network also fold evicted tokens into a summary token.
3. When a question arrives, the model answers from the compressed cache. Past frames are never
   re-read.

This makes memory and answering latency independent of video length. The question this document asks
is: **which frames should enter that bounded memory, and in what form?**

### 2.3 Two properties of the models that matter

- **Visual tokens are expensive.** At native resolution, Qwen3-VL spends about **330 tokens per
  frame**, so a 6k-token cache holds only about **18 frames' worth**. Compression is heavy from the
  first minute of every video.
- **Qwen3 merges frames in pairs.** Its vision encoder has a temporal patch size of 2: every two
  consecutive input frames become *one* group of tokens. At 0.2 fps the two frames of a group are 5 s
  apart. LLaVA-OV encodes each frame separately. This difference turns out to matter (Chapter 4).
- **Qwen3 sees timestamps.** Each frame group is preceded by a text timestamp such as `<12.0 seconds>`.
  These tokens are what let the model name intervals for grounding.

---

## 3. Experimental setup

| Item | Setting |
|---|---|
| Benchmark | S-EMBER, first **300 videos**: 576 MCQ and 475 grounding questions (time, counting, location). Early exploration used the first 100 videos (183 MCQ / 159 grounding). |
| Models | **Qwen3-VL-8B-Instruct** (main model) and **LLaVA-OneVision-7B** (transfer check) |
| Streaming baseline | HERMES at **0.2 fps**, KV budget **6k** tokens, no per-frame floor (k = 0), `attention_weighted` strategy. For Qwen3, timestamp tokens of frames that still hold visual tokens are always kept ("surviving" mode, see 4.2). |
| Offline reference | **Uniform 32 / 64**: for each question, 32 or 64 frames spread evenly from 0 s to the question time, encoded fresh with no compression. |
| Answer length | 16 tokens (MCQ), 128 tokens (grounding) |
| Statistics | Paired bootstrap over questions (2,000 resamples) gives 95% confidence intervals (CIs) for differences against the baseline. One MCQ question is about **0.17 points**; one grounding question about **0.21 points**. Differences under about 2 points are usually noise. |
| Question-time buckets | 0-80 s (45 MCQ / 50 grounding), 80-160 s (129 / 110), 160-320 s (253 / 196), >= 320 s (149 / 119) |

### Naming used in this document

| Name | Meaning |
|---|---|
| `0.2 fps`, `fps1` | Constant sampling rate (frames per second) |
| `KV 6k`, `10.7k`, `21.4k` | KV-cache budget in tokens |
| `k=0`, `k=1` | Per-frame floor: with k = 1 every frame keeps at least one token (and its timestamp) through compression |
| `1.0:60,0.2` | **Rate schedule**: 1 fps for the first 60 s, then 0.2 fps. `1.0:160,0.2`, `1.0:80,0.5:160,0.2` work the same way. |
| `grid N:S` | **Grid-thinned sampler** (Section 6.2): about N frame pairs kept, pair spacing capped at S seconds |
| `grid N:S:spread` | Grid sampler with the two frames of a pair spread over the grid cell |
| `dup:<schedule>` | Each sampled frame is fed twice in a row, so a Qwen token group holds one moment (Section 7.6) |
| `scale 0.7`, `0.5` | Frames decoded at 0.7x or 0.5x resolution (about 1/2 or 1/4 of the tokens) |
| `full_span` | Grounding prompt that also asks for the interval to cover the whole event |
| `grid N:S:dup` | Grid sampler where each grid frame is fed twice (one moment per Qwen token group) |
| `keep r` (offline) | Offline pruning: after encoding uniform 64 frames, keep a fraction r of visual tokens |
| HERMES / random / recent / stratified score | Which tokens pruning keeps: highest HERMES attention score; random; most recent; or an equal share of every frame, choosing each frame's highest-scoring tokens |
| M3 | Retrieval-based boundary refinement of predicted intervals; post-processing, used only as a diagnostic (Section 9.5) |

Results are written as **MCQ accuracy / grounding mIoU / grounding R@0.5**, all in %.

---

## 4. Chapter 1: Baselines and first ablations

Before changing anything, we needed to know how HERMES streaming compares with a naive offline
approach, and which of its existing knobs matter.

### 4.1 Streaming vs uniform sampling (300 videos)

**Why:** a streaming method is only interesting if it beats (or matches, at lower cost) the obvious
alternative of sampling a fixed number of frames over the video seen so far.

| Model | Method | MCQ | mIoU | R@0.5 |
|---|---|---|---|---|
| LLaVA-OV-7B | Uniform 32 / 64 | 26.0 / 26.0 | 7.1 / 7.8 | 3.2 / 3.6 |
| LLaVA-OV-7B | Stream 0.2 fps, KV 6k | 27.8 | 8.2 | 4.0 |
| Qwen3-VL-8B | Uniform 32 / 64 | 25.0 / 25.3 | **27.5** / 26.9 | **26.7** / 25.3 |
| Qwen3-VL-8B | Stream 0.2 fps, KV 6k | 25.7 | 24.8 | 24.4 |

**Findings:**
- **Streaming roughly ties uniform on MCQ** for both models.
- **Qwen3 streaming loses on grounding** (-2.7 mIoU, CI +0.2 to +5.3 in favour of uniform).
- **LLaVA cannot ground.** Its mIoU of 7-8 is near the floor for every method: it does not produce
  meaningful timestamps. From here on, **grounding results are Qwen3-only**, and LLaVA is used for MCQ.

### 4.2 Timestamp tokens under compression (Qwen3)

**Why:** HERMES compression can evict Qwen3's timestamp tokens like any other token. Without them the
model cannot tell *when* the retained visual tokens happened, which should hurt grounding.

**Design:** three policies. **Default:** timestamps compete for the budget like other tokens.
**Surviving:** a frame group's timestamp is always kept if any of its visual tokens survive.
**All:** every timestamp is always kept.

| KV budget | Default | Surviving | All |
|---|---|---|---|
| 4k | 20.2 | **23.9** | 23.9 |
| 6k | 22.9 | **24.8** | 24.5 |
| 10.7k | 23.5 | 23.6 | 23.9 |
| 21.4k | 23.1 | 23.3 | 23.3 |

(grounding mIoU; MCQ changes by at most 0.3 points across modes)

**Finding:** protecting timestamps adds **2-4 mIoU at small budgets** and costs nothing. All later
Qwen3 streaming runs use **surviving** mode.

### 4.3 A per-frame floor, k = 1 (100 and 300 videos)

**Why:** under heavy compression some frames lose *all* their tokens, leaving holes in the timeline.
Guaranteeing every frame at least one token could help counting and grounding.

**Design:** with k = 1, a frame that would lose everything either keeps its single best token (**patch**
strategies, chosen by HERMES score or by raw attention) or gets one synthetic token pooled over the
frame (**pool** strategies: mean, attention-weighted, softmax over scores). The five strategies were
compared on LLaVA (100 videos, KV 4k and 6k), then the best one (`top_attention_patch`) on 300 videos.

**Finding:** all strategies are within 1-2 questions of each other and of k = 0. On 300 videos, k = 1
adds at most 0.3-0.7 points (e.g. Qwen3 KV 6k: 26.0 / 24.6 / 24.6 vs 25.7 / 24.8 / 24.4). **The floor
is not a lever**, which also told us that missing frames are not the main problem.

### 4.4 A larger memory budget (Qwen3)

**Why:** if compression is what hurts streaming, a bigger cache should help.

| KV budget | MCQ | mIoU | R@0.5 |
|---|---|---|---|
| 4k | 25.0 | 23.9 | 21.9 |
| 6k | 25.7 | 24.8 | 24.4 |
| 10.7k | 26.0 | 23.6 | 22.5 |
| 21.4k | 25.3 | 23.3 | 22.1 |

**Finding:** **more memory does not help.** 6k is as good as 21.4k, and grounding is slightly worse at
larger budgets. Late in long videos the 21.4k model's predicted intervals drift a median 147 s too
early (62 s at 6k). Simply keeping more of the same 0.2 fps stream is not the answer.

### 4.5 Constant higher frame rates (100 videos, MCQ)

**Why:** the simplest way to see more of the video is to sample faster.

| Rate | KV 4k | KV 6k |
|---|---|---|
| 0.2 fps | **30.6** | **32.2** |
| 0.5 fps | 27.3 | 30.6 |
| 1.0 fps | 27.3 | 31.7 |

**Finding:** with a fixed budget, faster sampling just means more compression, and it does not help.
This result was later revisited on 300 videos (Chapter 3), where the picture changed.

### 4.6 Side experiment: counting prompt

Asking the model to state its count before the option letter ("count-first") raises counting accuracy
(Qwen3 13.4 to 18.9, LLaVA 18.4 to 21.9) and overall MCQ by 1-2 points. This is a prompting effect
independent of memory, so it was not pursued further here, and all other runs use the official prompt.

---

## 5. Chapter 2: Where does streaming lose?

The knobs in Chapter 1 did not move the results. Instead of trying more settings, we analysed *where*
streaming loses to uniform sampling, using only existing predictions (no GPU).

### 5.1 By question time: early questions are the problem

At 0.2 fps, a question asked at time *t* has seen *t*/5 frames, while uniform sampling always has 32 or
64 frames, densely packed when *t* is small.

| Question time | Stream 0.2 fps | Uniform 32 | Uniform 64 |
|---|---|---|---|
| 0-80 s | 28.9 / 29.1 / 28.0 | 44.4 / 39.5 / 34.0 | 44.4 / **48.3** / **54.0** |
| 80-160 s | 20.9 / 31.3 / 30.9 | 23.3 / 36.0 / 35.5 | 24.8 / 37.9 / 34.5 |
| 160-320 s | **28.1** / 22.1 / 22.4 | 24.9 / 25.0 / 25.0 | 25.3 / 19.7 / 17.3 |
| >= 320 s | **24.8** / **21.3** / **20.2** | 20.8 / 18.7 / 18.5 | 20.1 / 19.5 / 17.6 |

**Finding:** streaming loses by **15 MCQ points and up to 19 mIoU on questions in the first 80 s**, and
wins after about 5 minutes. The same pattern in both tasks points at **frame count early in the
video**, not at any grounding-specific issue.

### 5.2 By compression: it is not the compression

Using the inference logs we recovered, for each question, whether the cache had already been
compressed. The largest gap to uniform (-9.5 mIoU) is on questions answered **before any
compression** happened. Once compression starts the gap shrinks to -1.8, and after 320 s streaming is
ahead. **Compression is not what makes streaming lose**; seeing too few frames early is.

### 5.3 Predicted intervals are too short

Across all methods, predicted intervals are much shorter than the annotated evidence (median 12.5 s
for streaming and 18 s for uniform, against a median of 43 s). The start is typically right but the
end comes too early. This observation returns in Section 7.2.

**Where this left us:** the most promising lever is **how frames are sampled over time**: sample more
densely when the video is new, as uniform sampling implicitly does.

---

## 6. Chapter 3: Changing how frames are sampled (rounds 1-4)

These rounds add a new option, `run.sample_schedule`, that controls when frames enter the stream. All
runs are Qwen3 on 300 videos, KV 6k and surviving timestamps unless stated.

### 6.1 Round 1: early-dense rate schedules

**Why:** directly target the early-question gap by sampling faster at the start.

**Design:** piecewise-constant rates, e.g. `1.0:60,0.2` (1 fps for 60 s, then 0.2 fps). Real frame times
are passed to the model so timestamps stay correct. Constant 1 fps is a control for "dense everywhere",
and one schedule was repeated at 10.7k to test the budget.

| Config | MCQ | mIoU | R@0.5 |
|---|---|---|---|
| Baseline 0.2 fps | 25.7 | 24.8 | 24.4 |
| `1.0:60,0.2` | 29.0 | 23.1 | 21.7 |
| `1.0:160,0.2` | 29.9 | 21.9 | 19.6 |
| `1.0:80,0.5:160,0.2` | not run | 22.2 | 20.2 |
| Constant 1 fps | 28.3 | 20.8 | 19.6 |
| `1.0:160,0.2`, KV 10.7k | **30.9** | 23.4 | 23.8 |

**Findings:**
- **MCQ improves significantly** (+3.3 to +5.2 points, all CIs above zero), in every question-time
  bucket. Early questions jump from 29% to 44-47%.
- **Grounding gets worse**, especially late: questions after 320 s drop from 21.3 to 11-14 mIoU.
- **Why grounding collapses:** the dense first minute floods the cache (60 early frames and their
  timestamps against one frame every 5 s later). For questions after 320 s, **50% of predicted
  intervals point into the first 60 s**, where only 2% of correct answers lie (29% at baseline). The
  model is pulled toward the over-represented start of the video.

Because the remaining variants of this idea shared the same flaw, the unstarted ones were cancelled
to save GPU time.

### 6.2 Round 2: grid thinning, a streaming version of uniform sampling

**Why:** uniform sampling is dense early *and thins out old frames as the video grows*. Round 1 only
did the first half. The fix is to also evict old frames so memory stays evenly spread over time.

**Design (`grid:N:S`):**
1. Frames come in Qwen pairs `(t, t + 1 s)`.
2. A pair starts only on a grid whose spacing doubles as the stream grows: 2 s, then 4, 8, 16 s. The
   spacing is the smallest value that keeps about **N** groups over [0, now], capped at **S** seconds.
3. When the spacing doubles, pairs that fall off the new grid are **evicted from the KV cache**
   during the next HERMES compression, along with their timestamps. Grids are nested (every 4 s point
   is also a 2 s point), so an evicted pair is never needed again, and new frames are only decoded if
   they land on the current grid.
4. HERMES compression still works as usual within the budget on whatever remains.

For grid 32:8: about 1 fps up to 64 s, then pairs every 4 s, then every 8 s after 128 s (about 0.25 fps,
with memory growing slowly as T/8 groups).

| Config | MCQ | mIoU | R@0.5 |
|---|---|---|---|
| Baseline 0.2 fps | 25.7 | 24.8 | **24.4** |
| grid 16:16 | 27.1 | **25.0** | 21.7 |
| grid 32:16 | 28.1 | 22.7 | 21.7 |
| grid 32:8 | **30.7** | 21.9 | 20.2 |
| grid 64:16 | 30.4 | 21.2 | 20.6 |

**Findings:**
- **The early-drift bias is fixed:** for questions after 320 s, the median error of the predicted
  interval centre is -16 s with grid 32:16, against -187 s for `1.0:60,0.2` (round 1) and -67 s for
  the baseline. Only 26% of late predictions point into the first minute (50% in round 1).
- **MCQ keeps the gain:** grid 32:8 is +5.0 (CI +2.1 to +8.2) and costs about 1.3x the baseline's GPU
  time, less than `1.0:160,0.2`.
- **Grounding still loses**, and gets worse the more frames stay in memory: grid 16:16 (fewest frames)
  ties the baseline on mIoU, while grid 64:16 drops late-question grounding to 8.9 mIoU.
- A pattern emerges: **MCQ improves with more frames, grounding prefers fewer.**

### 6.3 Round 3: spreading the two frames of a pair

**Why:** in round 2 a pair's frames are 1 s apart, so at 8-16 s spacing each group sees 1 s of video
followed by a 7-15 s blind gap. Events inside those gaps could be missed.

**Design (`:spread`):** place the second frame half a grid cell later (e.g. `t` and `t + 4 s` at 8 s
spacing), so coverage is even.

| Config | MCQ | mIoU | R@0.5 |
|---|---|---|---|
| grid 32:16 / spread | 28.1 / 28.8 | 22.7 / 21.3 | 21.7 / 18.9 |
| grid 32:8 / spread | 30.7 / 30.0 | 21.9 / 22.3 | 20.2 / 20.0 |

**Finding:** no improvement; grounding is slightly worse. Blind gaps were not the cause. In hindsight
this was a first hint of the frame-blending effect (Section 7.6): spreading the pair makes each token
group mix two moments further apart.

### 6.4 Round 4: a bigger budget and a per-frame floor

**Why:** if grounding suffers because many frames share a fixed budget (fewer tokens per frame), then
a larger budget, or a floor that keeps every frame's timestamp, should recover it.

| Config | MCQ | mIoU | R@0.5 |
|---|---|---|---|
| Baseline 0.2 fps, 6k / 10.7k | 25.7 / 26.0 | 24.8 / 23.6 | 24.4 / 22.5 |
| grid 32:8, **10.7k** | **31.4** | 23.2 | 21.9 |
| grid 16:16, 10.7k | 26.0 | 24.0 | 21.5 |
| `1.0:60,0.2`, 10.7k | 28.6 | 23.6 | 22.1 |
| grid 32:8, 6k, **k=1** | 30.4 | 22.3 | 20.2 |

**Findings:**
- **Best MCQ of the sampling rounds:** grid 32:8 at 10.7k, **31.4%** (+5.7, CI +2.4 to +9.0).
- **The bigger budget recovers only part of the grounding loss** (+1.3 mIoU for grid 32:8), still below
  the baseline.
- **The floor (k = 1) does nothing**, as in Chapter 1.

---

## 7. Chapter 4: Explaining the gains and the losses (phase 7)

By now two facts needed explaining: *why* does dense sampling help MCQ so much, and *why* does
grounding refuse to improve? Phase 7 combined free analyses with targeted GPU experiments.

### 7.1 What do the gains cost? (no GPU)

**Why:** a gain is only useful if its cost is reasonable. GPU time was recovered from inference logs.

| Method | GPU time (MCQ) | vs baseline | Frames encoded per question |
|---|---|---|---|
| Uniform 32 / 64 | 86 / 121 min | 0.78x / 1.09x | 32 / 64 |
| Stream 0.2 fps | 111 min | 1.00x | 34 |
| grid 16:16 | 100 min | 0.91x | 47 |
| grid 32:8 (6k / 10.7k) | 148 / 145 min | 1.33x / 1.31x | 76 |
| `1.0:160,0.2`, 10.7k | 179 min | 1.61x | 99 |
| Constant 1 fps | 240 min | 2.17x | 171 |

**Findings:** the best MCQ configuration costs about **1.3x** the baseline. Time to first token stays at
60-70 ms for all streaming methods, which is the point of HERMES. (Uniform runs came from an older code
version, so their timings are approximate.)

### 7.2 How much of grounding error is interval length? (no GPU)

**Why:** Section 5.3 showed intervals are too short for every method. If interval length dominates the
error, improving memory will barely move grounding.

**Design:** fit a length scale and a centre shift for each method on half of the videos, apply them to
the other half, then swap (2-fold cross-validation by video, so nothing is fitted on test videos).

| Method | Raw mIoU | Calibrated mIoU |
|---|---|---|
| Uniform 32 | 27.6 | 35.1-35.5 |
| Stream 0.2 fps | 25.0 | 33.5-34.0 |
| grid 16:16 | 25.3 | 34.9-35.5 |
| grid 32:8, 10.7k | 23.6 | 30.9-31.0 |

(Ranges come from two search grids for the scale; the conclusions are the same.)

**Findings:** widening predictions about 4-5x adds **7-10 mIoU to every method**. After calibration the
methods converge: uniform 32, grid 16:16 and the baseline are within 2 points. **Interval sizing is the
largest single grounding error**, and it is shared by all methods, streaming or not.

### 7.3 Does lowering resolution help? (GPU)

**Why:** grounding drops as more frames share the budget. If the cause is too few tokens per frame,
cheaper frames (lower resolution) should let dense sampling keep its frame count without diluting each
frame.

**Design:** new option `run.frame_scale` decodes frames at 0.7x resolution (about 1/2 the tokens,
~170 per frame) or 0.5x (about 1/4, ~85 per frame).

| Config | Scale | MCQ | mIoU | R@0.5 |
|---|---|---|---|---|
| 0.2 fps | 1.0 / 0.7 / 0.5 | 25.7 / 25.7 / 25.7 | **24.8** / 24.6 / 22.2 | **24.4** / 22.5 / 19.2 |
| grid 32:8 | 1.0 / 0.7 / 0.5 | **30.7** / 30.0 / 29.2 | 21.9 / 23.5 / 20.7 | 20.2 / 20.8 / 19.4 |

**Findings:**
- **MCQ does not depend on frame detail**: a quarter of the tokens leaves it unchanged. MCQ depends on
  *how many* moments are seen.
- **Grounding needs detail**: it holds at 0.7x but drops at 0.5x.
- Cheaper frames recover part of grid 32:8's grounding loss (+1.6 mIoU at 0.7x) but do not reach the
  baseline. Token dilution is only part of the explanation.

### 7.4 Does the MCQ gain transfer to LLaVA-OV? (GPU, MCQ)

**Why:** a sampling improvement that only works for one model is much less interesting. Grid eviction
was ported to LLaVA, and the 0.2 fps baseline was re-run with the same code as a control (27.8, which
matches the original run exactly).

| LLaVA-OV-7B | MCQ | 0-80 s | 80-160 s | 160-320 s | >= 320 s |
|---|---|---|---|---|---|
| Stream 0.2 fps | **27.8** | 42.2 | 30.2 | 24.9 | 26.2 |
| grid 32:8 | 26.9 | 35.6 | 26.4 | 23.7 | 30.2 |
| grid 16:16 | 26.0 | 37.8 | 28.7 | 22.5 | 26.2 |
| `1.0:60,0.2` | 28.0 | not broken down | | | |

**Finding:** **it does not transfer.** LLaVA's streaming baseline is already strong on early questions
(42%, against Qwen3's 29%), so there is no early gap to close. This raised the question of what makes
Qwen3 weak early, answered in 7.6.

### 7.5 Can the prompt fix interval length? (GPU, grounding)

**Why:** if intervals are too short (7.2), the cheapest fix is to ask for longer ones.

**Design:** the `full_span` prompt adds: *"The time interval must cover the whole event, from the moment
it begins until it has completely ended, not a single moment within it."*

| Config | Official prompt | `full_span` prompt |
|---|---|---|
| Stream 0.2 fps | 24.8 / 24.4 | 24.6 / 23.2 |
| Uniform 32 | 27.5 / 26.7 | 28.0 / 26.5 |
| grid 16:16 | 25.0 / 21.7 | 24.4 / 21.3 |
| grid 32:8, 10.7k | 23.2 / 21.9 | 23.1 / 21.1 |

(mIoU / R@0.5)

**Finding:** **the model ignores the instruction.** The median predicted length moves by only 1-2 s.
Fixing interval length will need something other than prompting (calibration, decoding constraints or
training).

### 7.6 Why is Qwen3 weak early? The frame-pairing test (GPU)

**Hypothesis:** Qwen3 merges two consecutive frames into one token group. At 0.2 fps these frames are
**5 s apart**, so every group blends two different moments, which could make each group harder to
interpret. Dense sampling (pairs 1 s apart) would reduce this blending, which would explain why it
helps Qwen3 but not LLaVA (no pairing), and why spreading pairs apart (round 3) did not help.

**Design (`dup:`):** feed each sampled frame twice in a row, so every token group contains a single
moment:

```
0.2 fps baseline : [f0 f5] [f10 f15] [f20 f25] ...   each group blends two moments, 5 s apart
dup:0.2          : [f0 f0] [f5 f5]   [f10 f10] ...   each group holds one moment
```

The same frames are seen, but each now costs a full group (about 660 tokens), doubling the token cost.
To isolate blending from token count, **`dup:0.2` at scale 0.7 costs about 336 tokens per frame, the
same as the baseline**, and scale 0.7 alone was shown to be harmless (7.3).

| Config | MCQ | mIoU | R@0.5 |
|---|---|---|---|
| Baseline 0.2 fps | 25.7 | 24.8 | 24.4 |
| `dup:0.2`, scale 0.7 (same token cost) | **28.6** | 23.0 | 20.4 |
| `dup:0.2`, native (2x token cost) | 25.9 | 23.0 | 19.4 |
| `dup:1.0:60,0.2`, scale 0.7 | **31.9** (best overall) | 18.7 | 16.6 |

**Findings:**
- **Removing blending at equal token cost adds +2.9 MCQ** with no extra frames. Frame pairing is a
  real, previously unnoticed source of weakness for Qwen3 streaming at low frame rates.
- At native resolution the doubled token cost cancels the gain, consistent with 7.3.
- **Dense early sampling without blending gives the best MCQ result of all: 31.9%.**
- **Grounding drops again** in both cases. Grounding benefits neither from dense frames nor from
  un-blended ones.

---

## 8. Chapter 5: Combining the MCQ wins (round 8)

**Why:** the two best MCQ ideas were separate: grid thinning (grid 32:8, 31.4% at 10.7k) and
un-blended frame pairs (`dup:1.0:60,0.2`, 31.9% at 6k). If their gains add up, one configuration
should exceed both.

**Design:** a new grid mode `grid:N:S:dup` feeds each grid frame twice, so every Qwen token group holds
one moment. All runs use 0.7x resolution, so a one-frame group costs about 336 tokens. Grid 64:8:dup
doubles the number of groups, to keep as many *distinct* moments as grid 32:8 (which puts two different
frames in each group).

| Config | KV | MCQ | mIoU | R@0.5 |
|---|---|---|---|---|
| Baseline 0.2 fps | 6k | 25.7 | 24.8 | 24.4 |
| grid 32:8 (reference) | 10.7k | 31.4 | 23.2 | 21.9 |
| `dup:1.0:60,0.2`, scale 0.7 (reference) | 6k | 31.9 | 18.7 | 16.6 |
| grid 32:8:dup, scale 0.7 | 6k | 27.8 | 20.9 | 18.7 |
| grid 32:8:dup, scale 0.7 | 10.7k | 29.2 | 21.7 | 20.4 |
| grid 64:8:dup, scale 0.7 | 6k | 29.9 | 20.9 | 20.8 |
| **`dup:1.0:60,0.2`, scale 0.7** | **10.7k** | **33.0** | 18.8 | 17.5 |

**Findings:**
- **The gains do not add up.** Grid thinning with un-blended pairs is worse than either idea alone. With
  one frame per group, 32 groups hold only 32 distinct moments instead of 64. Doubling the groups
  (grid 64:8:dup) recovers most of the loss (29.9%), confirming that **the number of distinct moments
  seen drives MCQ**.
- **Best MCQ overall: `dup:1.0:60,0.2` at 0.7x and KV 10.7k, 33.0%** (+7.3 over the baseline, 95% CI
  +3.6 to +10.8; about 8 points above uniform 64). It is the best configuration in every question-time
  bucket (51.1 / 31.0 / 32.0 / 30.9% from early to late).
- Grounding stays low for all these configurations (18.8-21.7 mIoU): the MCQ/grounding trade-off
  remains.

---

## 9. Chapter 6: Temporal grounding under token pruning (phases 9-10)

The rest of the project targets one goal: **improve temporal grounding when visual tokens are pruned**,
both in streaming memory (HERMES) and in offline pruning (keep a fraction of tokens after encoding
uniformly sampled frames). All methods are training-free. We first diagnosed *why* pruning hurts
grounding (9.1-9.4), then designed a fix for the largest error (9.5).

### 9.1 Offline pruning: does grounding break first? (phase 9, A1)

**Why:** pruning papers usually report QA accuracy only. If grounding degrades much faster than QA,
that is a key motivation; the choice of *which* tokens to keep may also matter more for grounding.

**Design:** Qwen3, uniform 64 frames (about 21.4k visual tokens), one pruning pass right after encoding
(`run.offline_keep_ratio`), keeping 50 / 25 / 10 / 5% of visual tokens. Pruning scores
(`run.prune_score`): HERMES (attention to predicted questions plus recency), random, recent (keep the
latest tokens), and stratified (equal share of every frame, highest-scoring tokens within each frame).
Spatial pooling (lower decode resolution) is included at roughly matched token cost.

| Keep | HERMES score | Random | Stratified | Spatial pooling (matched cost) |
|---|---|---|---|---|
| 100% (no pruning) | 25.3 / 26.9 / 25.3 | | | |
| 50% | 26.9 / 28.4 / 26.7 | 26.0 / 28.6 / 26.9 | – | 26.2 / 25.7 / 24.0 (scale 0.7) |
| 25% | 26.2 / 25.9 / 24.0 | 26.7 / 27.2 / 25.3 | 26.0 / 27.0 / 25.9 | 27.3 / 20.8 / 18.7 (scale 0.5) |
| 10% | **21.5** / **23.3** / 23.2 | 27.4 / 26.1 / 23.6 | 26.7 / 25.9 / 24.2 | 25.0 / 21.8 / 20.8 (scale 0.35) |
| 5% | **20.1** / **14.8** / 13.3 | 26.7 / 25.3 / 22.9 | 25.9 / 25.4 / 24.6 | – |

(MCQ / mIoU / R@0.5, %.) Keeping only the most recent 25% of tokens gives 23.1 / 18.5 / 16.4.

**Findings:**
- **Grounding does not break first.** Randomly dropping 90% of tokens costs only 0.8 mIoU and does not
  change MCQ: video tokens are highly redundant.
- **Attention-based selection is the problem.** At 10%, the HERMES score is worse than random on both
  tasks (-2.8 mIoU, -5.9 MCQ), and at 5% it collapses (MCQ 20.1, mIoU 14.8) while random and stratified
  stay near the unpruned level (25.3-25.4 mIoU with 95% of tokens dropped). Attention concentrates the
  budget on a few frames and loses temporal coverage. Keeping only recent tokens is catastrophic,
  confirming that coverage over time is what matters.
- **Stratified selection is as robust as random**, and principled: it guarantees coverage while still
  choosing the most salient tokens inside each frame.
- **Lower resolution hurts grounding far more than dropping tokens** at the same budget (20.8 vs 27.2
  mIoU at 25%), while MCQ does not care.

### 9.2 Streaming pruning with stratified selection (phases 9-10)

| Config (0.2 fps) | KV | MCQ | mIoU | R@0.5 |
|---|---|---|---|---|
| HERMES score (baseline) | 6k | 25.7 | 24.8 | 24.4 |
| Stratified | 6k | 24.3 | 24.6 | 23.2 |
| HERMES score | 4k | 25.0 | 23.9 | 21.9 |
| Stratified | 4k | 24.8 | **24.6** | **22.7** |
| HERMES score | 2k | 24.5 | 22.5 | 19.8 |
| Stratified | 2k | 24.3 | 22.7 | 19.2 |

**Finding:** in streaming, the choice of score matters little: stratified helps grounding slightly at
4k (+0.7 mIoU, +0.8 R@0.5) and is neutral at 6k and 2k, at a small MCQ cost. The gap is smaller than offline because
streaming compresses incrementally, so every frame is still represented when it arrives.

### 9.3 What survives pruning? (phase 9, A2)

**Why:** a natural hypothesis is that attention-based pruning keeps the salient *peak* of an event and
drops its *boundaries*, which would explain short intervals.

**Design:** a new option `run.retention_snapshot` logs, for every question, how many tokens each streamed
frame still has in the cache when the question is answered, and whether its timestamp survived. Frames
inside the annotated interval are "evidence"; the first and last are "edges".

| Config | Keep rate: background | evidence | edge | interior | Evidence timestamps kept |
|---|---|---|---|---|---|
| 0.2 fps, 6k | 50.3% | 54.0% | 58.8% | 50.8% | 94.9% |
| 0.2 fps, 4k | 37.3% | 38.8% | 44.8% | 35.4% | 94.9% |
| grid 32:8, 6k | 19.0% | 30.3% | 39.4% | 28.1% | 78.9% |

Grounding by how much of the evidence survived (0.2 fps, 6k):

| Evidence retention | Questions | mIoU | R@0.5 | Predicted / gold length |
|---|---|---|---|---|
| Lowest third | 159 | 19.7 | 18.2 | 0.38 |
| Middle third | 158 | 27.5 | 29.7 | 0.52 |
| Top third | 158 | 27.1 | 25.3 | 0.56 |

**Findings:**
- **Boundaries are not lost**: edge frames keep *more* tokens than interior frames, and evidence frames
  keep about as much as background. The "keeps peaks, drops boundaries" hypothesis is rejected.
- **How much evidence survives matters**: questions with the least evidence retained lose 8 mIoU and get
  shorter predictions.
- Timestamps of evidence frames survive 95% of the time with fixed-rate streaming, but only 79% with grid
  thinning.

### 9.4 Too-short intervals are a model bias (phase 9, A3)

| Run | Median predicted length | Median gold length | Median ratio | Shorter than half the gold |
|---|---|---|---|---|
| Uniform 64 (no pruning) | 14.3 s | 43 s | 0.50 | 49% |
| Stream 0.2 fps, 6k | 12.5 s | 43 s | 0.50 | 47% |
| Stream 0.2 fps, 4k | 12.5 s | 43 s | 0.50 | 49% |
| Stream 0.2 fps, 21.4k | 12.5 s | 41 s | 0.48 | 50% |

**Finding:** the length ratio is the same with and without pruning. Short intervals come from the model,
not from pruning. The bias is strongest for time and counting questions (ratio 0.4-0.5), whose evidence
is long (43-77 s); location questions (12 s evidence) are less affected.

### 9.5 Interval-extent corrections: a diagnostic, not a method

> **Important:** the corrections in this section only post-process the predicted interval. They raise
> grounding scores **without changing the model's answer**, which is often wrong. Example (video
> `1064720965773119_start_0.0_end_378.277.mp4`, asked at 350 s): *"How long did I have the faucet on
> before turning it off the first time?"* The model answers **"10 seconds"** with interval [23.5, 34.0];
> the truth is about five minutes, [19, 331]. Refinement widens the interval to [23, 329] (IoU 0.03 to
> 0.98), yet the answer stays wrong and now contradicts the interval. We therefore use these numbers
> only to measure **how much of the grounding error is interval extent**, and do not propose them as
> a method. Real improvements must change what the model perceives, so answers and intervals improve
> together (see 9.7 for answer-aware evaluation).

**Why measure it:** interval length is the largest grounding error for every method (Section 7.2), and
prompting cannot fix it (7.5). Three corrections were evaluated with 2-fold cross-validation by video
(fit on half of the videos, test on the other half, swap).

**Designs:**
1. **Fixed x4:** stretch every predicted interval 4x around its centre (no fitting).
2. **Fitted scale:** the same, with the scale fitted on the training half.
3. **M3, retrieval-based boundary refinement:**
   - A per-frame embedding index is kept *outside* the KV cache, so pruning never touches it: each frame
     is fed alone to Qwen3's vision encoder and its visual tokens are mean-pooled (one 4096-d vector per
     frame; `logs/phase10/dump_frame_embeddings.py`).
   - After the model answers `[s, e]`, only frames up to the question time are used (streaming-legal),
     centred by subtracting their mean embedding.
   - The evidence profile is the mean embedding of frames inside `[s, e]`. Each boundary grows outward
     while neighbouring frames stay similar to it (cosine at least tau times the evidence frames'
     own mean cosine), tolerating gaps of up to `gap` seconds.
   - tau and gap are fitted on the training half.

Grounding (mIoU / R@0.5, all 475 questions). MCQ is unaffected: these corrections only post-process
grounding intervals.

| Setting | Raw | Fixed x4 | Fitted scale | M3, 1 fps index | M3, 0.2 fps index |
|---|---|---|---|---|---|
| Uniform 32 (no pruning) | 27.5 / 26.7 | 35.7 / 33.5 | 35.6 / 32.4 | **38.9 / 36.8** | 37.1 / 35.2 |
| Uniform 64 (no pruning) | 26.9 / 25.3 | 33.3 / 32.0 | 33.4 / 31.6 | **36.7 / 35.2** | 35.9 / 34.1 |
| Offline stratified 25% | 27.0 / 25.9 | 34.0 / 30.3 | 34.1 / 32.2 | **38.1 / 35.8** | 37.1 / 35.6 |
| Offline HERMES 10% | 23.3 / 23.2 | 30.0 / 26.1 | 30.5 / 26.5 | **35.9 / 33.3** | 35.1 / 33.7 |
| Stream 0.2 fps, 6k | 24.8 / 24.4 | 33.8 / 33.1 | 33.8 / 29.9 | **36.9 / 37.5** | 36.5 / 34.3 |
| Stream 0.2 fps, 4k | 23.9 / 21.9 | 34.5 / 33.5 | 34.2 / 30.9 | **36.1 / 35.6** | 35.9 / 35.4 |
| Stream grid 16:16, 6k | 25.0 / 21.7 | 35.7 / 34.5 | 35.7 / 34.5 | **37.7 / 34.9** | 37.2 / 34.7 |

**What the numbers say (as a diagnostic):**
- **Interval extent accounts for 7-12 mIoU** in every setting: widening alone lifts streaming at 6k from
  24.8 to 33.8 (x4) or 36.9 (M3).
- **This headroom does not mean the model understood the event.** On the 87 questions where the
  prediction covers at most 35% of a long event, IoU rises from 0.16 to 0.58 after refinement, but the
  answers are unchanged, and for duration and counting questions they are mostly wrong ("10 seconds",
  "2 times" when the truth is minutes or 10 times).
- **Root cause:** the model points at a single temporal token group. At 0.2 fps a Qwen group spans two
  frames 5 s apart, i.e. 10 s: 20% of streaming intervals are exactly 10 s long (10% with uniform 64),
  and 78 of 191 duration questions are answered "10 seconds". The model reports the granularity it
  sees in memory instead of the event's real extent, and its answer is built from that one moment.
- Detailed cases (video paths, raw outputs, annotator intervals): `logs/phase10/m3_cases.txt`.

### 9.6 Does coverage-preserving pruning transfer to another model? (phase 10, Qwen2.5-VL-7B)

**Design:** Qwen2.5-VL-7B, uniform 64 frames, offline pruning (MCQ only: Qwen2.5-VL has no text
timestamps and its grounding is near the floor, 4.9 mIoU with no pruning, like LLaVA).

| Keep | HERMES score | Random | Stratified |
|---|---|---|---|
| 100% (no pruning) | 25.0 | | |
| 25% | 26.6 | – | 25.0 |
| 10% | **20.8** | **21.0** | **25.9** |

(MCQ accuracy, %.)

**Finding:** on Qwen2.5-VL, at 10% of tokens *both* attention-based and random selection collapse
(about -4 points), while stratified selection keeps the unpruned accuracy. Attention alone loses
temporal coverage; random alone keeps coverage but not the informative tokens within each frame.
**Stratified selection (an equal share of every frame, filled with that frame's most salient tokens) is
the only score robust on both models.**

### 9.7 Evaluation with the official S-EMBER protocol

mIoU and R@0.5 reward an interval even when the answer is wrong (Section 9.5). The S-EMBER paper
([arXiv 2607.02689](https://arxiv.org/abs/2607.02689), [code](https://github.com/facebookresearch/S-EMBER))
therefore also reports answer accuracy and a joint metric:
- **GQ@0.0 (answer accuracy):** an LLM judge marks the free-text answer CORRECT if it is semantically
  equivalent to *any one* of the annotators' answers. No interval requirement.
- **GQ@0.5 (grounded QA):** a question counts only if the answer is judged correct **and** the interval
  has IoU >= 0.5 with the evidence.

**Our implementation** (`logs/phase11/judge_grounding.py`): the official judge prompt, gold list and
verdict parsing, copied unchanged from the released script (`logs/phase11/official/sember_official_judge.py`).
The only difference is the judge model: the official script calls Gemini (`gemini-3.1-flash`), which
needs an API key and sends the data to an external service; we use a local Qwen3.8-27B-FP8 (vLLM,
temperature 0). The judge sees the model's answer text without its "Time:" interval. Absolute numbers
may therefore differ from the paper's; comparisons between our configurations use the same judge.
The paper's clean/overall accuracy and hallucination rate are not reproducible from the released code.

Sanity check: on duration and counting questions the judge agrees with a deterministic rule-based check
(`logs/phase11/rule_answer.py`) on 82% of questions; almost all disagreements are answers the lenient
rule accepts and the judge rejects.

Qwen3-VL-8B, 300 videos (475 grounding / 576 MCQ questions). Differences to the streaming baseline with
95% paired-bootstrap CIs; one grounding question = 0.21 points.

| Configuration | MCQ | mIoU | R@0.5 | **GQ@0.0** (answer acc.) | **GQ@0.5** | GQ@0.0 vs base | GQ@0.5 vs base |
|---|---|---|---|---|---|---|---|
| Uniform 32 (no pruning) | 25.0 | 27.5 | 26.7 | 13.3 | **5.3** | +0.8 [-2.1, +3.6] | +1.7 [-0.6, +4.2] |
| Uniform 64 (no pruning) | 25.3 | 26.9 | 25.3 | 13.5 | 4.4 | +1.1 [-1.9, +4.0] | +0.8 [-1.3, +2.9] |
| **Stream 0.2 fps, 6k (baseline)** | 25.7 | 24.8 | 24.4 | 12.4 | 3.6 | – | – |
| Stream 0.2 fps, 4k | 25.0 | 23.9 | 21.9 | 11.8 | 3.8 | | |
| Stream grid 32:8, 10.7k | 31.4 | 23.2 | 21.9 | 12.4 | 4.4 | +0.0 [-2.5, +2.5] | +0.8 [-1.3, +2.9] |
| Stream `dup:0.2`, scale 0.7, 6k | 28.6 | 23.0 | 20.4 | 14.1 | 3.4 | +1.7 [-0.8, +4.2] | -0.2 [-2.1, +1.5] |
| **Stream `dup:1.0:60,0.2`, scale 0.7, 10.7k** | **33.0** | 18.8 | 17.5 | **15.6** | 2.9 | **+3.2 [+0.2, +6.1]** | -0.6 [-2.7, +1.5] |
| Offline uniform 64, HERMES 5% | 20.1 | 14.8 | 13.3 | 8.8 | 1.5 | -3.6 [-6.3, -1.1] | -2.1 [-4.0, -0.4] |
| Offline uniform 64, random 5% | 26.7 | 25.3 | 22.9 | 10.7 | 3.4 | -1.7 [-4.2, +1.1] | -0.2 [-2.3, +1.7] |
| Offline uniform 64, stratified 5% | 25.9 | 25.4 | 24.6 | 9.7 | 2.9 | -2.7 [-5.5, -0.2] | -0.6 [-2.7, +1.3] |

All 70 Qwen3 grounding runs: `python3 logs/phase11/joint_metrics.py`.

**Findings:**
- **Grounded QA is very hard for every configuration.** Only 9-16% of open-ended answers are judged
  correct, and GQ@0.5 is 1.5-5.5%: a correct answer with a well-placed interval is rare. mIoU and R@0.5
  (20-29%) greatly overstate grounding quality.
- **Answer accuracy (GQ@0.0) follows MCQ, not mIoU.** The configuration with the best MCQ
  (`dup:1.0:60,0.2`, 33.0%) also has the best open-ended accuracy (15.6%, +3.2, CI excludes zero), although its mIoU is the
  lowest of the streaming runs. Memory changes that let the model see more distinct moments improve
  its answers; interval metrics alone would have hidden this.
- **GQ@0.5 cannot yet separate the memory configurations** apart from collapse cases: most differences
  are within +-2 points (about 10 questions) and all CIs include zero, except offline HERMES pruning at
  5%, which clearly hurts both GQ@0.0 and GQ@0.5. Separating configurations on grounded QA needs either
  larger gains or more questions (the full benchmark has 9,448 QA pairs).
- **Coverage-preserving pruning still matters under the official metric:** at 5% of tokens, random and
  stratified keep GQ@0.5 near the baseline (-0.2, -0.6) while HERMES-score pruning collapses (-2.1).
- The weak link is that correct answers rarely come with correct intervals, and wrong answers are the
  majority. Improving grounded QA requires the model to both understand and localize the whole event,
  which is where future memory designs should be judged.

### 9.8 Does the model echo its memory's token-group spacing? (no GPU)

**Why:** most wrong grounding answers name a short slice of a long event (often exactly 10 s, one Qwen
token group at 0.2 fps) and state a wrong duration. If the model simply reports the time step it sees in
memory, changing the group spacing would change both its intervals and its answers, making memory
granularity the lever for grounded QA.

**Design:** Qwen3 stamps each two-frame group with the pair's mean time, so the gap between neighbouring
timestamps depends on sampling: 0.2 fps 10 s, 1 fps 2 s, `dup:0.2` 5 s (one frame per group), uniform N
frames 2 x t / N. Existing runs are compared on predicted interval length, stated durations (duration
questions) and whether interval endpoints land on a group timestamp (`logs/phase11/group_echo.py`).

| Config | Group spacing | Most common predicted lengths | Most common stated durations | Endpoints on a timestamp (chance) |
|---|---|---|---|---|
| 0.2 fps, 6k | 10 s | 10 s (94), 12 s (38), 11 s (20) | 10 s (54), 90 s (31) | 30% (10%) |
| 1 fps, 6k | 2 s | 10 s (45), 4 s (31), 12 s (24) | 90 s (34), 10 s (28) | 32% (52%) |
| `dup:0.2`, 6k | 5 s | 10 s (80), 5 s (52), 15 s (39) | 10 s (35), 90 s (34) | 66% (10%) |
| Uniform 32 / 64 | 3-28 s | 10 s in almost every spacing bin | 10 s, 90 s | – |

**Findings:**
- **Interval position is partly copied from visible timestamps:** with sparse stamps, endpoints land on a
  group timestamp 3-6.5x more often than chance (66% with 5 s integer stamps).
- **Interval length and stated durations are defaults, not read from memory:** 10 s is the most common
  length in every configuration, even with 2 s or 28 s spacing, and stated durations cluster on "10
  seconds" and "90 seconds" everywhere. The model falls back to canned values when it has not tracked
  the event.
- **So memory granularity is not the lever for grounded QA.** Changing the group spacing moves where the
  endpoints land but not the default lengths or the wrong durations. The answers are wrong because the
  model guesses instead of reasoning over the timeline; improving grounded QA needs the model to use the
  timestamps it already sees (e.g. enumerate when the event is visible, then derive duration or count),
  which changes the answer itself rather than post-processing it.

### 9.9 Timeline-reasoning prompt: fixing the answer, not the output (phase 12)

**Why:** Section 9.8 showed the model answers durations and intervals with canned values ("10 s",
"90 s") instead of using the timestamps it sees. Unlike post-processing (9.5), a prompt that makes the
model reason over the timeline changes the answer itself, and the official judge scores it.

**Design:** `dataset.grounding_prompt=timeline` asks the model to (1) list up to 8 moments when the
relevant object or event is visible, using the timestamps shown in the video, one per line starting
with "Seen:"; (2) derive the duration from the first and last moment, or count distinct occurrences;
(3) answer, then give the interval from the first to the last moment. A worked example on an unrelated
question fixes the format (without it the model copied placeholders and produced runaway lists).
Answers stay in the official "Answer: ... / Time: [s, e]" format; up to 384 new tokens. Scored with the
official S-EMBER judge prompt (Section 9.7).

| Configuration | Prompt | Acc. | mIoU | R@0.5 | GQ@0.3 | GQ@0.5 | Acc. by type (time / count / location) |
|---|---|---|---|---|---|---|---|
| Stream 0.2 fps, 6k | official | 12.4 | 24.8 | 24.4 | 4.0 | 3.6 | 10.5 / 13.6 / 14.0 |
| | timeline | 14.3 | 22.6 | 21.1 | 7.6 | 4.8 | 14.1 / 13.6 / 16.0 |
| Uniform 32 | official | 13.3 | 27.5 | 26.7 | 6.5 | 5.3 | 8.9 / 16.3 / 16.0 |
| | timeline | **18.9** | 25.4 | 23.4 | **11.2** | **8.8** | **25.7** / 15.8 / 12.0 |
| Stream `dup:1.0:60,0.2`, scale 0.7, 10.7k | official | 15.6 | 18.8 | 17.5 | 4.2 | 2.9 | 8.9 / 16.8 / 26.0 |
| | timeline | 17.1 | 22.3 | 22.5 | 7.4 | **6.7** | 14.7 / 18.5 / 19.0 |

Timeline minus official (95% paired-bootstrap CI): uniform 32 Acc. +5.7 [+1.9, +9.7], GQ@0.5 +3.6
[+0.4, +6.5]; `dup:1.0:60,0.2` GQ@0.5 +3.8 [+1.3, +6.3], Acc. +1.5 [-2.3, +5.3]; streaming baseline Acc.
+1.9 [-1.9, +5.5], GQ@0.5 +1.3 [-1.1, +3.8].

**Findings:**
- **Reasoning over the timeline fixes answers, not just intervals.** With uniform sampling, answer
  accuracy rises from 13.3 to 18.9% and grounded QA (GQ@0.5) from 5.3 to 8.8%; duration answers almost
  triple (8.9 to 25.7%). The canned "10 seconds" answers disappear (stated durations spread out).
- **mIoU can fall while grounded QA rises:** intervals now come with correct answers more often, which
  is what GQ measures. This is the opposite of post-processing, which raised mIoU with wrong answers.
- **Streaming benefits less.** The model can only reason over timestamps that are still in memory:
  compressed streaming memory keeps fewer timestamped moments than uniform sampling's 32 fresh frames,
  and the gain is smaller and not significant at 0.2 fps. The denser, un-blended streaming memory
  (`dup:1.0:60,0.2`) gets a significant GQ@0.5 gain (+3.8), the best streaming grounded-QA result so far.
- **Counting barely changes**, and location answers drop for two configurations (listing moments may
  distract from naming the place). A category-aware prompt is a natural follow-up.

---

## 10. All results in one table

Qwen3-VL-8B, 300 videos, surviving timestamps for streaming. Format: MCQ / mIoU / R@0.5 (%). The last
column is the MCQ difference from the baseline with its 95% bootstrap CI.

| Configuration | KV | All | 0-80 s | 80-160 s | 160-320 s | >= 320 s | MCQ vs base |
|---|---|---|---|---|---|---|---|
| Uniform 32 | – | 25.0 / **27.5** / **26.7** | 44.4 / 39.5 / 34.0 | 23.3 / 36.0 / 35.5 | 24.9 / 25.0 / 25.0 | 20.8 / 18.7 / 18.5 | -0.7 [-3.6, +2.1] |
| Uniform 64 | – | 25.3 / 26.9 / 25.3 | 44.4 / 48.3 / 54.0 | 24.8 / 37.9 / 34.5 | 25.3 / 19.7 / 17.3 | 20.1 / 19.5 / 17.6 | -0.3 [-3.3, +2.6] |
| **0.2 fps (baseline)** | 6k | 25.7 / 24.8 / 24.4 | 28.9 / 29.1 / 28.0 | 20.9 / 31.3 / 30.9 | 28.1 / 22.1 / 22.4 | 24.8 / 21.3 / 20.2 | – |
| 0.2 fps | 10.7k | 26.0 / 23.6 / 22.5 | 28.9 / 29.1 / 28.0 | 21.7 / 29.0 / 28.2 | 29.2 / 22.2 / 21.4 | 23.5 / 18.5 / 16.8 | +0.3 [-1.0, +1.7] |
| 0.2 fps, scale 0.7 | 6k | 25.7 / 24.6 / 22.5 | 28.9 / 31.6 / 28.0 | 24.8 / 27.7 / 26.4 | 25.3 / 22.5 / 20.4 | 26.2 / 22.2 / 20.2 | 0.0 |
| 0.2 fps, scale 0.5 | 6k | 25.7 / 22.2 / 19.2 | 40.0 / 29.3 / 20.0 | 24.8 / 25.8 / 23.6 | 23.7 / 18.8 / 15.3 | 25.5 / 21.3 / 21.0 | 0.0 |
| Constant 1 fps | 6k | 28.3 / 20.8 / 19.6 | 44.4 / 43.0 / 48.0 | 27.9 / 32.3 / 28.2 | 28.1 / 14.6 / 13.8 | 24.2 / 11.2 / 9.2 | +2.6 [-0.9, +6.1] |
| `1.0:60,0.2` | 6k | 29.0 / 23.1 / 21.7 | 44.4 / 43.3 / 44.0 | 26.4 / 32.7 / 30.0 | 28.1 / 18.5 / 17.9 | 28.2 / 13.5 / 10.9 | +3.3 [+0.7, +5.9] |
| `1.0:60,0.2` | 10.7k | 28.6 / 23.6 / 22.1 | 44.4 / 44.2 / 46.0 | 27.9 / 35.6 / 33.6 | 28.5 / 17.7 / 16.3 | 24.8 / 13.5 / 10.9 | +3.0 [+0.2, +5.7] |
| `1.0:160,0.2` | 6k | 29.9 / 21.9 / 19.6 | 44.4 / 43.0 / 48.0 | 27.9 / 32.3 / 28.2 | 30.4 / 15.5 / 13.3 | 26.2 / 13.9 / 10.1 | +4.2 [+1.4, +7.1] |
| `1.0:160,0.2` | 10.7k | 30.9 / 23.4 / 23.8 | 46.7 / 44.0 / 46.0 | 27.9 / 38.2 / 40.9 | 31.6 / 17.4 / 16.8 | 27.5 / 11.0 / 10.1 | +5.2 [+2.3, +8.2] |
| grid 16:16 | 6k | 27.1 / **25.0** / 21.7 | 44.4 / 38.7 / 38.0 | 24.8 / 31.9 / 28.2 | 26.9 / 21.0 / 17.3 | 24.2 / 19.3 / 16.0 | +1.4 [-1.6, +4.3] |
| grid 16:16 | 10.7k | 26.0 / 24.0 / 21.5 | 42.2 / 38.6 / 36.0 | 26.4 / 31.8 / 29.1 | 24.5 / 19.3 / 15.8 | 23.5 / 18.5 / 17.6 | +0.3 [-2.8, +3.6] |
| grid 32:16 | 6k | 28.1 / 22.7 / 21.7 | 48.9 / 44.3 / 48.0 | 29.5 / 29.4 / 27.3 | 27.7 / 16.9 / 15.3 | 21.5 / 16.7 / 16.0 | +2.4 [-0.5, +5.4] |
| grid 32:16:spread | 6k | 28.8 / 21.3 / 18.9 | 48.9 / 44.1 / 48.0 | 27.1 / 28.5 / 21.8 | 29.2 / 15.5 / 13.3 | 23.5 / 14.4 / 13.4 | +3.1 [+0.0, +6.2] |
| grid 32:8 | 6k | 30.7 / 21.9 / 20.2 | 48.9 / 44.3 / 48.0 | 29.5 / 29.4 / 27.3 | 30.0 / 16.7 / 13.8 | 27.5 / 13.9 / 12.6 | +5.0 [+2.1, +8.2] |
| grid 32:8 | 10.7k | 31.4 / 23.2 / 21.9 | 46.7 / 43.0 / 50.0 | 27.9 / 33.8 / 31.8 | 30.8 / 16.0 / 13.3 | 30.9 / 17.0 / 15.1 | +5.7 [+2.4, +9.0] |
| grid 32:8, k=1 | 6k | 30.4 / 22.3 / 20.2 | 48.9 / 44.4 / 48.0 | 27.9 / 31.0 / 29.1 | 30.0 / 17.0 / 13.3 | 27.5 / 13.5 / 11.8 | +4.7 [+1.7, +7.6] |
| grid 32:8:spread | 6k | 30.0 / 22.3 / 20.0 | 48.9 / 44.1 / 48.0 | 27.1 / 28.5 / 21.8 | 29.6 / 16.7 / 14.3 | 27.5 / 16.3 / 16.0 | +4.3 [+1.4, +7.5] |
| grid 32:8, scale 0.7 | 6k | 30.0 / 23.5 / 20.8 | 44.4 / 41.6 / 44.0 | 28.7 / 33.5 / 30.0 | 29.2 / 18.8 / 15.8 | 28.2 / 14.5 / 10.9 | +4.3 [+1.2, +7.5] |
| grid 32:8, scale 0.5 | 6k | 29.2 / 20.7 / 19.4 | 44.4 / 35.1 / 36.0 | 27.1 / 31.2 / 30.9 | 26.9 / 15.3 / 12.2 | 30.2 / 13.8 / 13.4 | +3.5 [+0.2, +6.6] |
| grid 64:16 | 6k | 30.4 / 21.2 / 20.6 | 46.7 / 43.0 / 48.0 | 28.7 / 32.8 / 30.0 | 30.0 / 16.7 / 16.8 | 27.5 / 8.9 / 6.7 | +4.7 [+1.4, +8.0] |
| `dup:0.2`, scale 0.7 | 6k | 28.6 / 23.0 / 20.4 | 31.1 / 40.1 / 38.0 | 25.6 / 31.6 / 29.1 | 28.5 / 18.3 / 15.8 | 30.9 / 15.5 / 12.6 | – |
| `dup:0.2` | 6k | 25.9 / 23.0 / 19.4 | 35.6 / 41.9 / 34.0 | 24.8 / 32.1 / 27.3 | 25.3 / 17.9 / 14.8 | 24.8 / 14.9 / 13.4 | – |
| `dup:1.0:60,0.2`, scale 0.7 | 6k | 31.9 / 18.7 / 16.6 | 48.9 / 45.9 / 52.0 | 32.6 / 27.5 / 24.5 | 29.6 / 11.8 / 8.2 | 30.2 / 10.6 / 8.4 | – |
| **`dup:1.0:60,0.2`, scale 0.7** | 10.7k | **33.0** / 18.8 / 17.5 | 51.1 / 44.2 / 50.0 | 31.0 / 29.5 / 28.2 | 32.0 / 10.8 / 8.2 | 30.9 / 11.5 / 9.2 | +7.3 [+3.6, +10.8] |
| grid 32:8:dup, scale 0.7 | 6k | 27.8 / 20.9 / 18.7 | 48.9 / 35.5 / 34.0 | 24.8 / 31.8 / 29.1 | 25.3 / 15.2 / 11.7 | 28.2 / 14.0 / 14.3 | – |
| grid 32:8:dup, scale 0.7 | 10.7k | 29.2 / 21.7 / 20.4 | 48.9 / 40.2 / 44.0 | 27.9 / 36.7 / 37.3 | 25.7 / 13.5 / 9.7 | 30.2 / 13.5 / 12.6 | – |
| grid 64:8:dup, scale 0.7 | 6k | 29.9 / 20.9 / 20.8 | 51.1 / 35.4 / 34.0 | 24.0 / 33.4 / 35.5 | 29.2 / 14.3 / 13.8 | 29.5 / 14.0 / 13.4 | – |
| Stratified pruning | 6k | 24.3 / 24.6 / 23.2 | | | | | |
| Stratified pruning | 4k | 24.8 / 24.6 / 22.7 | | | | | |

Grounding with the `full_span` prompt: 0.2 fps 24.6 / 23.2, uniform 32 28.0 / 26.5, grid 16:16
24.4 / 21.3, grid 32:8 at 10.7k 23.1 / 21.1 (mIoU / R@0.5). Offline pruning results are in Section 9.1
and interval-correction results (M3) in Section 9.5.

---

## 11. Conclusions and open questions

### What we learned

1. **The real weakness of HERMES streaming on S-EMBER is early in the video**, where a low frame rate
   has seen too few frames. It is not compression and not a missing per-frame floor.
2. **Sampling densely where the video is new is a strong, cheap lever for multiple-choice QA**: about
   +6 points for Qwen3, beating uniform sampling, for about 1.3x the baseline's compute.
3. **Dense sampling needs old frames to be thinned** (grid thinning). Otherwise the over-represented
   start of the video pulls grounding predictions toward it.
4. **Part of the MCQ gain comes from Qwen3's frame pairing.** At low frame rates each token group blends
   two moments seconds apart. Keeping groups to one moment helps at equal cost. This is a general
   caveat for streaming with any model that uses temporal patches.
5. **Grounding behaves differently from MCQ.** It prefers fewer, detailed frames, and its dominant error
   (intervals 3-4x too short) is a model bias shared by all methods, with or without pruning, and
   resists prompting.
6. **Token pruning hurts through lost temporal coverage.** Attention-based selection concentrates tokens
   on few frames and collapses at aggressive budgets (Qwen3 at 5%, Qwen2.5-VL at 10%). Stratified
   selection (equal share per frame, most salient tokens within each) is the only score robust on both
   models. Event boundaries are not what gets lost.
7. **Interval extent is the largest grounding error, but post-processing is not a fix.** Widening
   intervals (fixed x4 or M3) adds 7-12 mIoU while answers stay wrong. The model reasons from one
   10-second token group; methods must change what it sees and reasons over, and grounding must be
   scored together with answer correctness.

### Open questions and next steps

- **Answer-aware grounding evaluation** (in progress, 9.7): score answer correctness for every
  grounding run and report a joint metric (answer correct and interval overlapping the evidence).
- **Fix the one-group reasoning at its source.** Test whether the model echoes the token-group span
  (vary group spacing with 0.5 fps or `dup:` groups), then change what the model sees (memory
  contents, group granularity, token selection) so answers and intervals cover the whole event.
- **Coverage-aware pruning as a method.** Stratified selection keeps grounding at 5% of tokens; report
  answer-aware grounding and MCQ versus budget against published pruning methods.
- **Separate memories for separate tasks.** Since MCQ and grounding want different memory contents,
  test a hybrid: a dense, un-blended content memory next to a sparse, detailed "timeline" memory.
- **Check generality beyond S-EMBER.** The repository already has StreamingBench scripts; the best
  configurations (grid 32:8, `dup:1.0:60,0.2`) should be tested there.
- **Pairing-aware sampling for other Qwen-style models** (Qwen2.5-VL uses the same temporal patch).
- **Report noise honestly.** Many differences are within 1-2 points; use the bootstrap CIs in this
  document when making claims.

---

## 12. Reproducing the results

### Code

| Component | Location |
|---|---|
| Rate schedules, grid planner, `dup:` schedule | `video_qa/sampling.py` (`parse_sample_schedule`, `schedule_frame_times`, `GridPlan`, `DupSchedule`) |
| Loading scheduled frames, resolution scaling | `video_qa/base.py` (`load_scheduled_video`, `_open_reader`) |
| Streaming loop, grid thinning, keeping pairs whole | `video_qa/hermes_vqa.py` |
| Evicting frames during compression | `inference/abstract_hermes.py` (`_evict_masks`), used in `inference/qwenvl_hermes.py` and `inference/llavaov_hermes.py` |
| Grounding prompt styles | `video_qa/adapters.py` (`sember_grounding_prompt`) |
| Offline pruning, pruning scores (hermes / random / recent / stratified) | `video_qa/hermes_vqa.py` (uniform branch), `inference/qwenvl_hermes.py` (`prune_kv_cache_by_attention`, `_rank_within_frames`) |
| Retention snapshots | `inference/abstract_hermes.py` (`retention_snapshot`), `video_qa/hermes_vqa.py` |
| Frame embedding index and boundary refinement (M3, diagnostic only) | `logs/phase10/dump_frame_embeddings.py`, `logs/phase10/boundary_refine.py` |
| Answer correctness for grounding (rule-based and LLM judge) | `logs/phase11/rule_answer.py`, `logs/phase11/judge_grounding.py` |
| Unit tests for schedules and the grid planner | `tests/test_sampling.py` |

### Configuration options (Hydra)

| Option | Example | Default |
|---|---|---|
| `run.sample_schedule` | `'1.0:60,0.2'`, `'grid:32:8'`, `'grid:32:8:spread'`, `'grid:32:8:dup'`, `'dup:0.2'` | `null` (constant `run.sample_fps`) |
| `run.offline_keep_ratio` | `0.25` (uniform sampling only) | `1.0` (no pruning) |
| `run.prune_score` | `random`, `recent`, `stratified` | `hermes` |
| `run.retention_snapshot` | `true` | `false` |
| `run.frame_scale` | `0.7` | `1.0` |
| `dataset.grounding_prompt` | `full_span` | `official` |
| `run.kv_size`, `run.min_tokens_per_frame`, `run.keep_time_tokens` | `6000`, `0`, `surviving` | see `configs/config.yaml` |

Values containing commas must be quoted for Hydra, e.g. `"run.sample_schedule='1.0:60,0.2'"`.

Example (Qwen3, grid 32:8, KV 6k, MCQ, 300 videos):

```bash
python scripts/run.py experiment=sember_mcq_time_count_location model=qwen3_vl_8b \
  dataset.max_videos=300 run.sample_fps=0.2 run.kv_size=6000 run.min_tokens_per_frame=0 \
  run.frame_summary_strategy=attention_weighted run.keep_time_tokens=surviving \
  "run.sample_schedule='grid:32:8'"
```

### Running many configurations on Slurm

- `logs/sched.py <spec.json>` runs a list of configurations. It splits each run into video chunks that
  fit the 3-hour debug limit, fills both GPU partitions, retries failed chunks, skips nodes with
  hardware errors, and scores each run when all its chunks finish.
- Specs are generated by `logs/phase6/make_spec.py` (rounds 1-4), `logs/phase7/make_spec.py` (phase 7
  and round 8), `logs/phase9/make_spec.py` and `logs/phase10/make_spec.py` (token pruning), e.g.
  `python3 logs/phase7/make_spec.py r7a r7b r7c > logs/phase7/main.json`.

### Analysis scripts (no GPU)

| Script | Output |
|---|---|
| `logs/grounding_breakdown.py` | Chapter 2 analysis: grounding by question time, video length, compression state; interval biases |
| `logs/phase6/analyze.py` | Section 10 table: per-bucket scores and bootstrap CIs against the baseline |
| `logs/phase7/cost.py` | Section 7.1: GPU time, frames encoded, time to first token |
| `logs/phase7/calibrate.py` | Section 7.2: cross-validated interval calibration |
| `logs/phase8/calibration_method.py` | Section 9.5: fixed and fitted interval corrections |
| `logs/phase9/retention_analysis.py` | Section 9.3: token retention of evidence frames |
| `logs/phase9/extent_bias.py` | Section 9.4: interval-length bias with and without pruning |
| `logs/phase10/boundary_refine.py` | Section 9.5: M3 retrieval-based boundary refinement (needs numpy) |
| `tables/*.py` | LaTeX tables for the paper (`tables/*.tex`) |

Results are stored under `results/<model>/<sember_mcq|sember_grounding>/<run name>/`, with the run
name encoding the configuration (e.g. `time-count-location-grid32x8-kv6000-k0-attention_weighted-native-time-keepsurv-v300`).
Each folder contains per-question predictions (`*_scored.jsonl`) and summary metrics
(`*_metrics.json`).
