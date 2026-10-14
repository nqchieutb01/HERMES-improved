# Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget

*Draft v0.1, 14 October 2026. Target: CVPR. Method name **CoCo** (Coverage + Contrast) is a placeholder.*
*Conventions: `[TODO]` = missing experiment or text; `[cite]` = reference to add; `[verify]` = reference taken from our
survey and not yet re-read; numbers in grey boxes are from saved runs listed in Appendix C. All numbers are on
Qwen3-VL-8B with 64 frames.*

---

## Abstract

Video LLMs are increasingly asked not only *what* happened but *when*: grounded questions such as "how long did I use
the paper cutter?" require an answer and the time interval of its evidence. To keep long videos tractable, the visual
memory of these models is pruned to a small fraction of its tokens, and published pruners lose little accuracy on
multiple-choice video QA at 10% of the tokens. We show that temporal grounding behaves differently. Four recent
training-free pruners (FastVID, VidCom², FlashVID, SemVID) fall *below random token selection* on grounding, because
the time a model reports is read from the timestamps that survive in memory, and selection by saliency, uniqueness or
text similarity removes whole stretches of the timeline. Where the memory is silent, the decoder fills the gap with a
prior that does not depend on the video: asked with no video at all, the model still answers, and its intervals start
early and end too soon. We propose **CoCo**, a training-free remedy with two parts. *Coverage* keeps an equal token
quota in every frame and fills it with the tokens the actual question attends to, so that every time anchor survives
pruning. *Contrast* writes each time value by contrasting the model's prediction against its own no-video prediction,
with a strength that fades as the model becomes confident, and reserves room for the final interval so that the
contrast cannot crowd it out. Because grounding benchmarks carry strong positional biases, we evaluate every method
against video-free floors (the best constant interval and length-rescaled predictions). With 10% of the visual tokens,
CoCo improves R@0.5 from 25.0 (random tokens) and 28.6 (all tokens) to 54.8 on QVHighlights, where the best video-free
constant reaches 17.6, and from 24.6 to 40.8 on S-EMBER grounded QA. We also report where it does not help: on
Charades-STA and ActivityNet Captions the uncompressed model itself stays below the video-free floor, and no pruning
method, ours included, rises above it.

---

## 1. Introduction

Ask a video assistant "how long did I use the paper cutter?" and a useful answer has two parts: a duration and the
interval it comes from. Such *grounded* questions are central to egocentric assistants, video search and
step-by-step guidance, and they are hard for the same reason they are useful: the model must know *when* things
happened across the whole window, including the moments just before and after the event. At the same time, video
LLMs pay for every frame. Qwen3-VL spends about 330 visual tokens per pair of frames, so 64 frames already cost about
21,000 tokens, and long or streaming video multiplies this. Pruning the visual tokens to a small budget is therefore
the default, and a fast-growing literature reports that 10–25% of the tokens preserve multiple-choice accuracy
[FastVID, VidCom², FlashVID, SemVID, PruneVid, DyCoke; cite].

This paper asks what pruning does to *temporal grounding*, and finds that the answer is different from QA in three
ways.

**Observation 1: QA pruners lose the timeline.** Video LLMs such as Qwen3-VL report time by reading timestamp tokens
interleaved with the visual tokens. A frame whose visual tokens are all removed leaves its timestamp without content,
or removes it altogether. Pruners designed for QA remove frames that look redundant, unremarkable or unrelated to the
words of the question, which is exactly what makes a timeline usable for grounding. On S-EMBER grounded QA, all four
recent pruners we evaluate fall below *random* token selection at the same budget (Sec. 6.2), and the same holds on
QVHighlights and Charades-STA. Random selection is a surprisingly strong baseline for grounding because it keeps a
few tokens on every frame.

**Observation 2: where the memory is silent, a prior answers.** Asked with no video at all, the model still produces
confident intervals, and 90% of them start in the first 5% of the window. With a pruned memory the same habit
appears: 30–56% of predicted intervals start in the first 5% of the window, against 16% of the gold intervals, and
removing the frames that contain the evidence raises this share from 37% to 51%. The end of the interval suffers
most: the median end error is 2–3× the start error, and without video 87% of predicted ends fall more than half an
interval too early (Sec. 3).

**Observation 3: benchmark floors are high.** Grounding benchmarks have strong positional regularities [Otani et al.
2020; verify]. On S-EMBER, where questions are asked mid-stream and two thirds of the gold intervals end within the
last fifth of the window, the best *constant* interval [0.15q, q] reaches 36.9 mIoU, more than the uncompressed model
(28.0). On ActivityNet Captions and Charades-STA the best constant also beats the uncompressed model. Any claimed
gain in grounding under compression must therefore clear these floors, and we report them for every method.

These observations suggest that a grounding-aware compressed memory needs two things that QA-oriented pruning does
not provide: (i) **coverage**, so that every part of the timeline keeps an anchor in memory, and (ii) a way to stop
the decoder's **prior** from writing the interval where evidence is thin. We propose **CoCo**, a training-free method
with one component for each:

- **Coverage (Sec. 4.1).** After the frames are encoded, the KV cache is pruned layer by layer. Every frame keeps an
  equal share of the layer's budget, and within each frame the tokens that the *actual question* attends to most are
  kept. Because a timestamp survives exactly when one of its frame's visual tokens survives, the equal quota keeps
  every time anchor, and the question decides what each anchor shows.
- **Contrast (Sec. 4.2).** While the answer is written, each time value is chosen by contrasting the model's
  prediction with its prediction from a video-free memory. The contrast is applied only to time values, and its
  strength fades where the model is already confident. A small *answer reserve* (Sec. 4.3) guarantees that the final
  interval is always written, which matters because contrastive decoding otherwise lengthens the list of observed
  moments until the token limit.

Neither operator is new in isolation: per-frame budgets appear in VidCom², KTV and AdaTP, question-attention KV
eviction in SnapKV and PruneVid, and contrast against a degraded input in VCD and M3ID. Our contribution is to show
*why* temporal grounding needs this specific combination, and to make it work:

1. **A diagnosis of grounding under compression** (Sec. 3): QA-oriented pruners lose temporal coverage and fall
   below random selection; the decoder's video-free prior fills the gaps, mostly at the interval end; and benchmark
   floors are high enough to hide both effects.
2. **CoCo** (Sec. 4), a training-free method that couples time-anchor coverage with prior-contrastive decoding of
   time values. It requires no training, no labelled calibration and no change to the model.
3. **A floor-adjusted evaluation** (Sec. 5) that reports, for every method, the best video-free constant interval,
   a length-rescaling baseline, and closed versus open intervals, with paired confidence intervals. All design
   choices were made on S-EMBER; QVHighlights, ActivityNet and Charades-STA were run once with the frozen
   configuration.
4. **Results on four datasets** (Sec. 6). With 10% of the tokens, CoCo exceeds the uncompressed model and every
   pruner on S-EMBER and QVHighlights, and clears the video-free floors by +7.6 mIoU (S-EMBER, closed intervals) and
   +22.1 mIoU (QVHighlights), both significant. We also report the two datasets where the base model itself stays
   below the floor, and the budget below which equal quotas break down.

`[TODO teaser figure: (a) a 64-frame memory pruned by FastVID / SemVID vs CoCo, timestamps shown as ticks, gaps in
red; (b) the predicted interval with greedy decoding vs CoCo against the gold interval; (c) R@0.5 vs budget on
QVHighlights.]`

---

## 2. Related work

**Frame selection.** BOLT [cite] scores frames against the query with CLIP or SigLIP and selects them by inverse
transform sampling; AKS [cite], MDP3 [cite], FOCUS [cite] and Q-Frame [cite] optimise relevance together with
coverage or diversity. All are evaluated on multiple-choice QA. Our optional retrieval front end (Sec. 4.4) is BOLT
adapted to grounding, and we find it contributes little once tokens are chosen by the question.

**Visual token pruning for video LLMs.** Query-agnostic methods compress by redundancy or saliency: FastVID [cite]
segments the video and keeps density peaks, VidCom² [cite] gives each frame a budget by its uniqueness, FlashVID
[cite] selects by [CLS] attention and merges across frames with trees, and DyCoke [cite] and FrameFusion [cite]
merge similar tokens. Query-aware methods use the text: SemVID [cite] scores tokens by word-embedding relevance with
MMR diversity, AdaTP [cite] boosts segments similar to the text, LGTTP [cite] reads temporal words in the question to
reallocate tokens, and PruneVid [cite] keeps the top 40% of tokens by question attention at one LLM layer. LGTTP is
the only one we know that evaluates grounding, at a much milder budget (~35% of compute) and offline. KTV [cite] is
closest in structure (keyframes, then tokens with a CLIP-weighted per-frame budget). CoCo differs in that the
per-frame quota is equal by design, the score is the actual question's attention propagated exactly through all
layers, and the purpose of the quota is to keep every timestamp, which we show matters for grounding and not for QA.

**KV-cache compression and streaming memory.** SnapKV [cite], H2O [cite], PyramidKV [cite] and Ada-KV [cite] evict
KV entries by attention in long-context LLMs; our token stage is SnapKV-style eviction applied to video with a
per-frame constraint. For streaming video, HERMES [cite] compresses with generic pseudo-queries, ReKV [cite]
retrieves KV blocks at question time, and StreamMem [cite], InfiniPot-V [cite], LiveVLM [cite] and StreamKV [cite]
maintain bounded memories. These works report QA only. Our memory is built per question from frames up to the
question time: it is causal, but it is *not* a persistent streaming memory, and we make no efficiency claim beyond
the size of the memory that the decoder reads (Sec. 7).

**Contrastive decoding.** VCD [cite] contrasts against a noised image, ICD [cite] against a disturbed instruction,
M3ID [cite] against the absence of the image, and PAI [cite] amplifies image attention while subtracting the
text-only logits; SEASON [verify] extends this to video with temporally corrupted negatives. These target object
hallucination or QA. We apply the no-input contrast to *time values only*, with a confidence-adaptive strength, and
show that its benefit for grounding comes from correcting interval ends, not from a uniform lengthening (Sec. 6.4).

**Training-free temporal grounding with video LLMs.** NumPro [cite] draws frame numbers on frames, FV-Action
[verify] scans with yes/no questions, REZE [verify] recasts grounding as recognition, and "MLLMs Know When Before
Speaking" [verify] reads intervals from calibrated attention heads, debiased against a blank-video run. All assume
the full set of video tokens. CoCo addresses the complementary question of grounding from a *compressed* memory, and
these readouts are natural candidates to run on top of it `[TODO: FV-Action and attention readout on the same 10%
memory]`.

**Biases in moment retrieval benchmarks.** Otani et al. [verify] showed that blind baselines are competitive on
Charades-STA and ActivityNet Captions because of positional priors in the annotations. We find the same on the
datasets we use and make video-free floors part of every comparison.

---

## 3. Problem setting and diagnosis

### 3.1 Setting

A question $x$ is asked at time $q$ about a video $v_{0:q}$. The model sees $F{=}64$ frames from $[0, q]$, encoded
into a KV cache $M$ of $N$ visual tokens plus interleaved timestamp tokens ("<t seconds>" for each pair of frames in
Qwen3-VL). A budget $\rho$ (default $\rho{=}0.1$) limits the number of tokens kept per layer to $\rho N$, with kept
timestamps counted inside the budget. The model then writes a short timeline prompt answer
`Seen: t_1 seconds, ... / Seen: t_k seconds, ... / Answer: ... / Time: [s, e]`, from which the interval $[s, e]$ is
parsed. Grounding is scored by IoU with the gold interval (mIoU, R@0.3/0.5/0.7); grounded QA additionally requires the
answer to be judged correct (GQ@0.5 = correct and IoU ≥ 0.5).

### 3.2 QA pruners lose temporal coverage

Table 1 (Sec. 6.2) gives the main evidence: at 10% of the tokens, FastVID, VidCom², FlashVID and SemVID all score
below random token selection on S-EMBER, QVHighlights and Charades-STA. Two mechanisms explain this.
First, the timestamp of a frame pair is only useful when visual content of that pair survives with it; a pruner that
drops a frame entirely leaves a hole in the timeline. Second, saliency, uniqueness and word similarity are poor
proxies for *temporal* relevance: the frames just before and after an event, which delimit the interval, are often
neither salient nor similar to the question words. Random selection keeps about 30 tokens on every frame pair and so
never leaves a hole. Selection that keeps holes is harmful even when it is query-aware: ranking all tokens globally
by HERMES's attention score loses 5.3 mIoU [2.9, 7.6] against the same score ranked within each frame.

### 3.3 The prior fills the gaps, mostly at the interval end

We measure the model's video-free behaviour by asking every question with an empty visual memory. The model answers
anyway; 90% of its intervals start in the first 5% of the window, its predicted intervals are 0.45× the gold length
(median), and 87% of its ends fall more than half an interval too early. The same signature appears, attenuated, in
compressed memories: with random 10% pruning, 30% of intervals start in the first 5% of the window (gold: 16%) and 52%
of ends are more than half an interval early. A causal test confirms the direction: removing the frames inside the
gold interval from the memory raises the early-start share from 37% to 51%, while removing the same number of other
frames does not. Across methods, the end error dominates: the median end error is 0.40–0.69 of the gold length
without contrastive decoding, against 0.14–0.30 for the start. The decoder, in short, falls back on a video-free
habit (start early, stop soon) wherever the memory does not show the evidence.

### 3.4 Benchmark floors

Positional regularities make some intervals predictable without the video. On S-EMBER, questions are asked while the
stream is running, and 67% of gold intervals end within the last 20% of the window; the best constant interval
[0.15q, q] scores 36.9 mIoU and 34.7 R@0.5. Snapping a method's predicted end to $q$ helps every S-EMBER method
substantially, because it exploits the same regularity. On ActivityNet Captions, where moments average 38% of the
video, the constant [0, 0.95q] scores 37.9 mIoU; on Charades-STA, [0, 0.45q] scores 29.7. QVHighlights is the least
biased of the four (best constant 26.8 mIoU, 17.6 R@0.5). These floors are tuned on the test questions and are
therefore upper bounds for any video-free rule. We return to them in Sec. 5.

---

## 4. Method: CoCo

CoCo has two components, applied at the two places where Sec. 3 locates the failures: the memory (Coverage) and the
decoder (Contrast). Neither requires training. Figure 2 `[TODO]` gives an overview.

### 4.1 Coverage: equal per-frame quotas filled by the question

After the $F$ frames are encoded, the memory is pruned once, independently in each of the $L$ layers. Let
$\mathcal{T}_f$ be the visual tokens of frame pair $f$ and $B = \rho N$ the per-layer budget (minus the kept
timestamps).

*Question relevance.* The question $x$ is appended to the full, uncompressed memory and propagated exactly through all
layers, so that queries at layer $\ell$ come from the real hidden states. The relevance of visual token $i$ at layer
$\ell$ is its attention from the question, averaged over heads $h$ and question positions $j$:
$$
a_\ell(i) \;=\; \frac{1}{H |x|} \sum_{h=1}^{H} \sum_{j \in x} \mathrm{softmax}_i\!\left(\frac{Q^{(\ell,h)}_j {K^{(\ell,h)}_i}^{\!\top}}{\sqrt{d}}\right).
$$

*Equal quota.* Each frame pair receives the same share of the layer budget, $B_f = \lfloor B / F' \rfloor$ (with
remainders assigned in order), where $F'$ is the number of frame pairs. Layer $\ell$ keeps
$$
\mathcal{K}_\ell \;=\; \bigcup_{f} \operatorname{top}_{B_f}\big\{ a_\ell(i) : i \in \mathcal{T}_f \big\},
$$
and the timestamp tokens of $f$ are kept if and only if $\mathcal{K}_\ell \cap \mathcal{T}_f \neq \emptyset$. With an
equal quota every frame pair keeps at least one token at any practical budget, so **every time anchor survives**, and
the question decides which part of each frame is shown. The pruned cache keeps original rotary positions, so the
relative order of kept tokens is preserved.

*Why equal, and why the question.* An equal quota is a deliberately simple coverage constraint: it guarantees the
timeline, which grounding needs and QA does not, and it prevents the attention mass from collapsing onto a few frames
(attention sinks and recency bias concentrate up to 227 of ~2,140 kept tokens on a single frame when quotas follow
attention mass). We tested relevance-weighted quotas with a floor, from SigLIP similarity or attention mass; none
beat equal quotas (pre-registered test: −0.2 mIoU [−1.2, +0.8]). Using the *actual* question rather than the generic
probe questions used by streaming memories such as HERMES is what makes attention informative: generic probes with
global top-k are the weakest variant we measured (18.3 mIoU).

*Per-layer selection.* Each layer keeps its own set $\mathcal{K}_\ell$. Forcing one shared set (attention averaged
over layers) costs CoCo −1.5 mIoU [−2.9, −0.0]*, whereas it does not affect random selection (+0.7, n.s.); the
benefit of per-layer selection is therefore specific to question-driven scores and does not explain the gap between
random selection and input-level pruners.

*Cost.* One extra forward pass of the question over the uncompressed memory per question. `[TODO: measured latency
and peak memory; the current implementation computes 3–4 scoring passes and can be reduced to one.]`

### 4.2 Contrast: prior-contrastive decoding of time values

Let $p_M(y) = p(y \mid M, x, y_{<t})$ be the model's next-token distribution with the compressed memory, and
$p_\emptyset(y) = p(y \mid \emptyset, x, y_{<t})$ the distribution with the same prompt and generated prefix but *no
visual tokens*. By Bayes' rule,
$$
\log p_M(y) \;=\; \log p_\emptyset(y) \;+\; \underbrace{\log \frac{p_M(y)}{p_\emptyset(y)}}_{\text{evidence from } M},
$$
so when the memory is uninformative about $y$, the evidence term is flat and the video-free prior decides. This is
the failure of Sec. 3.3. Prior-contrastive decoding (PCD) amplifies the evidence term:
$$
s_t(y) \;=\; (1+\alpha_t)\log p_M(y) - \alpha_t \log p_\emptyset(y) \;=\; \log p_\emptyset(y) + (1+\alpha_t)\log \frac{p_M(y)}{p_\emptyset(y)},
\qquad y \in \mathcal{V}_\beta,
$$
restricted to the plausible set $\mathcal{V}_\beta = \{y : p_M(y) \geq \beta \max_{y'} p_M(y')\}$ with $\beta = 0.1$, as
in VCD, so that tokens the memory makes implausible are never promoted. Three choices adapt this rule to grounding.

1. *No-video counterfactual.* The negative memory is empty rather than noised or shuffled. It isolates exactly the
   prior that Sec. 3.3 measures, and it is cheap: the counterfactual cache holds only the prompt.
2. *Time scope.* The contrast is applied only while a time value is being written: inside a `Seen: t` or
   `Event: s – e` value before its description, and inside `Time: [s, e]`. All other tokens, including the answer
   text, are decoded greedily from $p_M$, so the answer is unaffected where the memory already speaks.
3. *Confidence-adaptive strength.* $\alpha_t = \alpha_{\max}\,(1 - \max_y p_M(y))$ with $\alpha_{\max}{=}1$. Where the
   memory makes the model confident (for example, evidence that truly starts at 0 s), the contrast fades and the
   early answer is kept; where the model hesitates between times, the evidence term decides.

The timeline prompt matters: PCD acts on the listed `Seen:` moments and then on the interval, so the interval is
written from contrasted observations. With a direct-answer prompt (only `Time: [s, e]`), PCD does not help in our
experiments `[TODO: include the number]`.

### 4.3 Answer reserve

Contrastive decoding lengthens the list of observed moments: under PCD the model sometimes keeps writing `Seen:` lines
until the generation limit $T$, and the interval is never written. We reserve room for the answer: if $R$ tokens
remain ($R = 64$) and no `Answer:` line has started, the decoder writes `\nAnswer:` and continues normally. This is a
format constraint, not a heuristic on the content: the interval is still chosen by PCD from the memory. It removes
all unparseable outputs (from 26–68 of 500 to 0) and improves every PCD configuration on every dataset
(+1.5 to +6.1 mIoU, all significant; Table 5).

### 4.4 Optional: retrieval-guided frames

When the window is much longer than $F$ frames, which frames are encoded matters as much as which tokens are kept. As
an optional front end we sample frames as in BOLT [cite]: each second of video is embedded once with SigLIP 2 (a
causal 1 fps index), the relevance $z_j$ of second $j \le q$ is the z-scored cosine similarity with the question, and
the $F$ frames are taken at evenly spaced quantiles of $p(j) = \tfrac{1}{2}\cdot\tfrac{1}{q} + \tfrac{1}{2}\,
\mathrm{softmax}(z)_j$, so that half of the frames stay uniform. We skip retrieval when uniform frames are already
less than 1 s apart. Its contribution is small once tokens are chosen by the question (+0.8 to +0.9 mIoU, not
significant on S-EMBER, QVHighlights or ActivityNet), so we report it as an option; the main-table runs include it.

---

## 5. Experimental setup

**Model.** Qwen3-VL-8B-Instruct, 64 frames sampled uniformly from $[0, q]$ (or by Sec. 4.4), native resolution,
greedy decoding (or PCD), at most 384 new tokens, batch size 8. All methods use the same timeline prompt.

**Datasets.** (i) **S-EMBER** grounded QA [cite]: 475 questions (191 duration, 184 counting, 100 location) on the first
300 videos, asked mid-stream; the memory contains only frames up to the question time. (ii) **QVHighlights** val
[cite]: 500 queries with a single relevant window (seed-0 sample of the 1,020 single-window queries), videos of
~150 s. (iii) **ActivityNet Captions** val_1 [cite]: 500 queries, one sentence per video, videos of ~115 s. (iv)
**Charades-STA** test [cite]: 500 queries, videos of ~30 s. For the last three datasets the question is "When does the
following happen in the video: "<query>"?" and the window is the whole video. The QVHighlights and ActivityNet copies
we use are low-resolution re-encodes (153 and ~84 visual tokens per frame pair, against 660 for S-EMBER); budgets are
therefore also reported in absolute tokens.

**Budget.** 10% of the visual tokens of the 64-frame memory (S-EMBER: ~2,140 tokens including timestamps;
QVHighlights: ~520). ActivityNet is reported at 30% (~400–900 tokens) because at 10% the memory holds only 120–300
tokens (Sec. 6.5).

**Baselines.** All tokens (no pruning); random tokens; FastVID, VidCom², FlashVID (vision-side stages) and SemVID,
re-implemented and checked to keep the same tokens as the authors' code on test inputs; each with and without PCD.
`[TODO: SnapKV-style observation-window eviction, PruneVid, global question top-k without quotas, FV-Action and
attention readout on the same memory.]`

**Floor-adjusted evaluation.** For every dataset we report (a) the best constant interval $[aq, bq]$ over a 20×20 grid,
tuned on the test questions (an upper bound for video-free constants); (b) *length rescaling*: the predicted interval
of the no-PCD method stretched about its start by a factor fitted once on S-EMBER (1.76) and applied unchanged to the
other datasets, which tests whether a gain could come from longer intervals alone; and (c) the split into *closed*
questions (gold end < 0.8q) and the rest. On S-EMBER, end-snapping to $q$ is a further floor; we report it in
Appendix B.

**Metrics and statistics.** mIoU and R@{0.3, 0.5, 0.7}; for S-EMBER also answer accuracy and GQ@0.5, judged by the
official S-EMBER judge prompt with a local Qwen3.8-27B-FP8 model, all rows of a table in one judging session. Answers
that state times as minutes:seconds are converted to seconds for every method alike (Appendix A). Paired
bootstrap over questions (2,000 resamples, 95% intervals); * marks intervals that exclude 0. `[TODO: video-clustered
bootstrap.]`

**Protocol.** All design choices (scores, quotas, α, scope, reserve) were made on S-EMBER. The configuration was then
frozen and run once on QVHighlights, ActivityNet and Charades-STA. `[TODO: held-out S-EMBER videos as a test split.]`

---

## 6. Results

### 6.1 Main results

**Table 1. Grounding with 10% of the visual tokens (ActivityNet: 30%). mIoU / R@0.5.** Floors in grey are video-free.

| Method | S-EMBER | S-EMBER closed | QVHighlights | ActivityNet (30%) | Charades-STA |
|---|---|---|---|---|---|
| *Best constant interval (oracle)* | *36.9 / 34.7* | *32.1 / 29.9* | *26.8 / 17.6* | *37.9 / 35.8* | *29.7 / 29.6* |
| All tokens (no pruning) | 28.0 / 26.7 | 30.0 / 27.4 | 29.1 / 28.6 | 22.9 / 18.2 | 16.6 / 14.2 |
| Random tokens | 25.6 / 24.6 | 28.3 / 27.4 | 26.9 / 25.0 | 22.9 / 18.6 | 17.4 / 13.8 |
| FastVID | 17.9 / 15.8 | | 19.2 / 15.4 | | 13.3 / 11.6 |
| VidCom² | 20.4 / 18.7 | | 19.9 / 18.2 | | 12.3 / 8.4 |
| FlashVID | 22.7 / 21.5 | | 22.8 / 20.4 | | 14.1 / 10.8 |
| SemVID | 23.3 / 20.6 | | 24.5 / 21.2 | 22.8 / 18.6 | 14.6 / 11.2 |
| Random + Contrast | 35.0 / 33.9 | 36.8 / 38.2 | 37.8 / 39.2 | 31.4 / 29.4 | **26.2 / 24.0** |
| SemVID + Contrast | 34.8 / 35.8 | 34.3 / 34.4 | 34.8 / 34.8 | 30.2 / 27.8 | 21.7 / 19.2 |
| Coverage only | 32.5 / 34.1 | 33.0 / 35.7 | 34.5 / 35.6 | 25.4 / 21.2 | 15.4 / 12.0 |
| **CoCo (Coverage + Contrast)** | **39.3 / 40.8** | **39.7 / 43.3** | **48.8 / 54.8** | **34.7 / 32.4** | 23.3 / 21.2 |

*Contrast includes the answer reserve. Coverage rows include optional retrieval (Sec. 4.4); on Charades retrieval is
skipped by the gate. Empty cells: not yet computed. `[TODO: fill the closed column for the pruners; add R@0.7.]`*

**Table 2. Paired gains of CoCo (mIoU, 95% CI).**

| Comparison | S-EMBER | S-EMBER closed | QVHighlights | ActivityNet (30%) | Charades-STA |
|---|---|---|---|---|---|
| vs all tokens | +11.3 `[TODO CI]` | | +19.7 [16.7, 22.8]* | +11.8 [9.3, 14.6]* | |
| vs random + Contrast | +4.2 [1.7, 6.7]* | | +11.0 [8.3, 13.5]* | +3.3 [1.1, 5.5]* | −2.9 [−4.7, −1.1]* |
| vs SemVID + Contrast | +4.5 [1.9, 7.2]* | | +14.1 [11.2, 17.1]* | +4.5 [2.3, 6.7]* | |
| vs best constant | +2.4 [−0.7, 5.6] | +7.6 [2.5, 12.6]* | +22.1 [18.7, 25.4]* | −3.1 [−5.9, −0.4]* | −6.4 [−9.5, −3.3]* |
| vs length rescaling | +1.0 [−1.6, 3.7] | +7.5 [3.2, 11.7]* | +8.6 [5.5, 11.8]* | +0.8 [−2.0, 3.5] | −0.7 [−3.1, 1.6] |

Three results stand out. **(i)** On QVHighlights, the least biased benchmark, CoCo with 10% of the tokens nearly
doubles the R@0.5 of the uncompressed model (54.8 vs 28.6) and clears every video-free floor by a wide margin. **(ii)**
On S-EMBER, CoCo beats all tokens, random selection and SemVID under the same decoding; against the video-free floors
its pooled gain is small, but on closed intervals, where the end of the window is not a shortcut, it gains +7.6 mIoU
over the best constant and +7.5 over length rescaling. **(iii)** On ActivityNet and Charades-STA the uncompressed
model is itself far below the constant floor (22.9 vs 37.9; 16.6 vs 29.7). CoCo improves on all tokens and on every
pruner on ActivityNet, but neither it nor any other method reaches the floor, and on Charades-STA (30 s videos) the
coverage stage is worse than random selection (Sec. 6.5). We report these as limits of the base model and of the
method, not as successes.

On S-EMBER, CoCo also improves the answers themselves: accuracy 21.7 vs 19.8 (all tokens) and 17.3 (random), and
GQ@0.5 13.3 vs 8.8 and 7.4 (one judging session). `[TODO: GQ for SemVID + Contrast in the same session.]`

### 6.2 QA pruners on grounding

Table 1 shows that every published pruner falls below random token selection on S-EMBER (−2.3 to −7.6 mIoU, all
significant), QVHighlights (−2.4 to −7.7; SemVID n.s.) and Charades-STA (−2.8 to −5.2, all significant). Random
selection itself varies little across seeds (26.0 ± 0.6 mIoU over three seeds on S-EMBER), so the gaps are not noise.
The ordering is consistent with coverage: the methods that drop frames most aggressively (FastVID's segment peaks,
VidCom²'s uniqueness budgets) lose most, and the methods that spread tokens (FlashVID, SemVID) lose least. Contrast
helps every pruner (+7 to +12 mIoU with the reserve), but it does not close the gap: SemVID + Contrast stays below CoCo on S-EMBER,
QVHighlights and ActivityNet.

### 6.3 Ablation: what each component contributes

**Table 3. Component ladder (mIoU / R@0.5, 10% of tokens; ActivityNet 30%).**

| Frames | Tokens | Decoding | S-EMBER | QVHighlights | ActivityNet |
|---|---|---|---|---|---|
| uniform | random | greedy | 25.6 / 24.6 | 26.9 / 25.0 | 22.9 / 18.6 |
| retrieval | random | greedy | 29.4 / 29.5 | 28.7 / 28.0 | — |
| uniform | Coverage | greedy | 31.7 / 32.4 | 33.7 / 33.0 | 24.6 / 21.6 |
| retrieval | Coverage | greedy | 32.5 / 34.1 | 34.5 / 35.6 | 25.4 / 21.2 |
| retrieval | Coverage | PCD | 37.5 / 39.2 | 42.8 / 48.2 | — |
| retrieval | Coverage | PCD + reserve | **39.3 / 40.8** | **48.8 / 54.8** | **34.7 / 32.4** |

Coverage is the largest gain on the memory side (+6.1 mIoU* on S-EMBER, +6.8* on QVHighlights over random). Retrieval
adds +0.8–0.9 on top (n.s. on all three datasets): once tokens are chosen by the question, which frames are encoded
matters less. Contrast is the largest single gain (+5 to +14 mIoU), and the answer reserve is needed to realise it on
long videos (QVHighlights +6.1*). `[TODO: Coverage + Contrast without retrieval on all datasets, so that the main
method does not depend on the optional stage.]`

**Table 4. Design choices of Coverage (S-EMBER, greedy decoding, mIoU).**

| Variant | mIoU | Δ vs equal quota, question top-k |
|---|---|---|
| Equal quota, question top-k (Coverage) | 32.5 | — |
| Relevance-weighted quota (SigLIP, floor 50%) | 33.0 | +0.5 [−0.5, 1.4] |
| Relevance-weighted quota (attention mass) | 32.1 | n.s.; one frame takes up to 227 tokens |
| Equal quota, sampling instead of top-k (uniform frames) | 30.0 | −1.7 vs top-k |
| Shared token set across layers | 31.9 | −0.6 [−2.3, 1.1] |
| Generic probe questions, global top-k (HERMES) | 18.3 | worst |
| Global top-k, no quota, question attention | `[TODO]` | |

### 6.4 Is Contrast more than a longer interval?

Contrast lengthens predicted intervals (median predicted/gold length 0.57 → 0.67 on S-EMBER), and longer intervals
alone raise mIoU on benchmarks whose gold intervals are long. We test this directly with the length-rescaling floor:
the no-contrast prediction stretched by a factor fitted on S-EMBER (Table 2, last row). On QVHighlights (+8.6*) and on
S-EMBER closed intervals (+7.5*), CoCo clearly beats rescaling: Contrast moves the interval toward the evidence, not
just outward. On pooled S-EMBER, ActivityNet and Charades-STA the two are statistically tied, so on those sets the
benefit of Contrast cannot be separated from a calibrated lengthening. `[TODO: decompose start and end errors on
closed questions; mismatched-video and shuffled-frame negatives as mechanism controls.]`

### 6.5 Budget, and where Coverage breaks

**Table 5. Budget sweep (mIoU).**

| | S-EMBER 5% | 10% | 20% | QVH 5% | 10% | 20% | ANet 10% | 30% | 50% |
|---|---|---|---|---|---|---|---|---|---|
| kept tokens per memory | ~1,070 | ~2,140 | ~4,280 | ~260 | ~520 | ~1,040 | 120–300 | 400–900 | 600–1,500 |
| Random | 23.9 | 25.6 | 27.4 | | 26.9 | | 17.8 | 22.9 | 23.5 |
| SemVID + Contrast | 31.9 | 34.8 | 35.3 | | 34.8 | | 26.8 | 30.2 | 31.4 |
| Random + Contrast | 34.5 | 35.0 | 36.5 | | 37.8 | | 21.5 | 31.4 | 32.2 |
| CoCo | **39.3** | **39.3** | **36.8** | 9.7 | **48.8** | **49.4** | 11.6 | **34.7** | **34.4** |

Two regimes appear. Above roughly 500 kept tokens, CoCo is flat or improves as the budget *shrinks* (S-EMBER: 36.8 at
20%, 39.3 at 10% and 5%), whereas random selection and SemVID degrade; question-driven pruning removes distractors as
well as cost, and its margin over SemVID + Contrast grows from +1.5 (20%, n.s.) to +7.4* (5%). Below roughly 300 kept
tokens (QVHighlights 5%, ActivityNet 10%), an equal quota leaves only 2–4 tokens per frame pair, attention top-k at
that size keeps tokens with little content, and CoCo collapses (9.7 and 11.6 mIoU), below random selection and below
SemVID, whose cross-frame merging degrades more gracefully. The threshold is absolute, not relative: it depends on the
number of tokens per frame, which is low in the re-encoded QVHighlights and ActivityNet copies.

**Short videos.** On Charades-STA (30 s, 64 frames, about two frames per second), every question-aware selector is
at or below random selection, and CoCo is 2.9 mIoU* below Random + Contrast. When frames are this dense, spreading
tokens across all of them is worth more than concentrating them on the most attended ones. The uncompressed model
also predicts near-instantaneous intervals on this dataset (median predicted/gold length 0.00), which no token
selection can fix.

### 6.6 Answer reserve

**Table 6. Effect of the answer reserve (mIoU, paired).**

| Configuration | S-EMBER | QVHighlights | Charades-STA |
|---|---|---|---|
| Random + PCD | +1.9 [1.0, 2.8]* | +1.5 [0.9, 2.2]* | +2.7 [1.7, 3.8]* |
| SemVID + PCD | | | +5.6 [4.2, 7.2]* |
| Retrieval + Coverage + PCD | +1.8 [1.0, 2.8]* | +6.1 [4.5, 7.8]* | +2.4 [1.5, 3.6]* |
| Unparseable answers, PCD → PCD + reserve | 26–39 → 0 | 25–60 → 0 | 28–68 → 0 |

---

## 7. Limitations

- **Base-model floors.** On ActivityNet Captions and Charades-STA, Qwen3-VL-8B with 64 frames is below the best
  constant interval even without pruning, and so is every compressed variant. Our claims on these datasets are
  relative to the model, not absolute.
- **Minimum tokens per frame.** Equal quotas fail below ~300 kept tokens (2–4 per frame pair). A floor-aware variant
  that merges frames before splitting the budget is future work.
- **Short, densely sampled videos.** When uniform frames are under a second apart, random spreading beats
  question-driven selection.
- **Not a streaming memory.** The memory is built per question from frames up to the question time. It is causal,
  but the full memory is prefilled once per question, so peak memory and time-to-first-token are those of the
  uncompressed model; CoCo reduces the memory the decoder reads, not the cost of building it. A write-once memory
  re-ranked at question time is future work `[TODO]`.
- **Contrast doubles decoding** for time-value tokens (a second, prompt-only cache). `[TODO cost table.]`
- **One model, one judge.** All results use Qwen3-VL-8B; answer accuracy uses a local judge with the official prompt.
- **Design on the test split of S-EMBER.** Every hyperparameter was chosen on the 475 S-EMBER questions; the other
  three datasets were run once with the frozen configuration, but a held-out S-EMBER split is still to be run.
- **Subsets.** All datasets are 475–500-question subsets; QVHighlights uses single-window queries and ActivityNet the
  videos of one archive part.

---

## 8. Conclusion

Temporal grounding asks a compressed video memory for something multiple-choice QA does not: a complete timeline.
Pruners designed for QA remove parts of it and fall below random selection, and the decoder's video-free prior fills
the gaps, mostly by ending intervals too early. CoCo keeps every time anchor with an equal, question-filled quota and
writes time values against the model's own no-video prediction. With a tenth of the visual tokens it grounds better
than the uncompressed model on S-EMBER and QVHighlights, and it clears video-free floors where the base model is
above them. The floors themselves are a lesson: on two standard benchmarks, a constant interval beats the
uncompressed model, and grounding results under compression should be read against them.

---

## Appendix A. Parsing

The model sometimes writes times as minutes:seconds (`Time: [1:45, 2:07]`) on videos longer than a minute, which the
S-EMBER parser scores as missing. We convert `h:mm:ss` and `m:ss` to seconds for every method alike. Rankings are the
same with the official parser; the main numbers with the official parser are QVHighlights CoCo 44.6 / 50.2 (vs 48.8 /
54.8) and ActivityNet (30%) 33.3 / 31.0 (vs 34.7 / 32.4). `[TODO: full table.]`

## Appendix B. End-snapping on S-EMBER

Setting the predicted end to $q$ raises every S-EMBER method to 42–46 mIoU, more than CoCo's 39.3, because 67% of
gold intervals end within the last fifth of the window. On closed intervals end-snapping *hurts* every method (to
22–24 mIoU), while CoCo keeps 39.7. On QVHighlights, end-snapping lowers CoCo from 48.8 to 40.5. A method that decides
whether an interval is still open at question time is a natural extension `[TODO]`.

## Appendix C. Reproduction

Configuration (Hydra overrides): `model=qwen3_vl_8b experiment=sember_grounding_uniform run.uniform_num_frames=64
dataset.grounding_prompt=timeline run.keep_time_tokens=surviving run.max_new_tokens=384 run.batch_size=8
run.prune_score=core_topk run.offline_keep_ratio=0.1 run.contrastive_mode=blind run.contrastive_alpha=1.0
run.contrastive_adaptive=true run.contrastive_scope=time run.answer_reserve=64` (+ `run.retrieval=frames_cont
run.retrieval_min_gap=1.0` for the optional retrieval). Code: branch `worktree-p31-gate-pcdfix`. Result tables:
`hermes_pool/reports/report_all_mss.md`; floors: `docs/paper/analysis/floors.py`.

---

## References (to complete)

Frame selection: BOLT (CVPR 2025, arXiv 2503.21483); AKS (CVPR 2025, 2502.21271); MDP3 (2501.02885); FOCUS
(2510.27280); Q-Frame (ICCV 2025) [verify]. Token pruning: FastVID (NeurIPS 2025, 2503.11187); VidCom² (EMNLP 2025,
2505.14454); FlashVID (ICLR 2026, 2602.08024); SemVID (ECCV 2026, 2603.05663); PruneVid (ACL Findings 2025); LGTTP (EMNLP
2025); AdaTP (2505.20100); KTV (2602.03615); DyCoke (CVPR 2025, 2411.15024) [verify]; FrameFusion (2501.01986) [verify].
KV memory: SnapKV (NeurIPS 2024) [verify]; H2O [verify]; PyramidKV [verify]; Ada-KV [verify]; HERMES (ACL 2026); ReKV
(ICLR 2025); StreamMem (2508.15717); InfiniPot-V (2506.15745); LiveVLM (2505.15269) [verify]; StreamKV (AAAI 2026)
[verify]. Contrastive decoding: VCD (CVPR 2024) [verify]; ICD (ACL Findings 2024) [verify]; M3ID (CVPR 2024) [verify];
PAI (ECCV 2024) [verify]; SEASON (2512.04643). Grounding: NumPro (CVPR 2025) [verify]; FV-Action (2608.08315);
REZE (2608.04480) [verify]; MLLMs Know When Before Speaking (2605.21954); Otani et al., Uncovering Hidden Challenges
in Query-Based Video Moment Retrieval (BMVC 2020) [verify]. Datasets: S-EMBER [cite]; QVHighlights (Moment-DETR,
NeurIPS 2021) [verify]; ActivityNet Captions (ICCV 2017) [verify]; Charades-STA (TALL, ICCV 2017) [verify]. Model:
Qwen3-VL [cite]; SigLIP 2 [cite].
