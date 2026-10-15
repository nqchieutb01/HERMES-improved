# Peer Review Report — Perspective Review (Peer Reviewer 3)

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (CoCo)
- **Manuscript ID**: CVPR 2026 draft, commit 2c84a78 (main.tex, sec/*.tex incl. sec/6b_transfer.tex, fig/*.tex)
- **Supporting material read**: docs/paper/analysis/results_2026-10-15.md
- **Review Date**: 2026-10-15
- **Review Round**: Round 2 (spotlight bar)

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Researcher in LLM decoding and calibration (contrastive decoding, classifier-free guidance, PMI scoring, contextual calibration) who has also shipped streaming egocentric assistants. I am an outsider to temporal-grounding benchmarks. I take the benchmark numbers as reported and do not audit dataset specifics or statistics; those belong to other seats.

### Review Focus
(1) Is PCD principled, and is it placed correctly in the decoding lineage? (2) What does the new evidence mean for the thesis? It shows that PCD helps reasoning-style answers but not one-line answers, and that Coverage transfers under the standard prompt. (3) In deployment, when does pruning at query time, after a full prefill, pay off? (4) Does the method generalise beyond time tokens and beyond the Qwen3-VL family?

### Calibration Status
`NOT_CALIBRATED`

criteria_binding_unavailable

(No criteria manifest was supplied. Venue comments are field-general and judged against the shared "Spotlight bar" in field_analysis.md.)

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**. Everything that bears on the decision can be repaired in one cycle. For the spotlight bar, see the separate section near the end.

### Confidence Score
4. Decoding/calibration and deployment are core expertise. Grounding-benchmark conventions are outside my competence.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment
The paper diagnoses why temporal grounding breaks under visual-token pruning. QA pruners empty frames, so timestamps disappear, and the decoder's video-free prior fills the gaps. It proposes two fixes: Coverage (an equal per-frame quota filled by the question's attention) and PCD (contrast against a no-video pass, applied to time values only, with a strength that fades as the model grows confident).

From a decoding standpoint, the experiments around PCD are unusually careful. They compare counterfactuals, replay decision rules offline against a gold-tuned threshold, and add a length-rescaling floor. The PCD rule itself, however, is the classifier-free-guidance / context-aware-decoding rule with VCD's plausibility constraint. Positioned only against hallucination-mitigation work, it misses the lineage that would make it principled: PMI scoring, calibration against a content-free input, CFG and CAD.

The new evidence changes the thesis more than the draft admits:
- Coverage is the robust, general contribution. It transfers under the standard prompt and to a second model.
- PCD is conditional on the answer format. Its "prior" may be partly produced by the timeline format itself, a confound the paper has not separated.

Deployment is unmeasured: Coverage needs a full prefill and a question pass over the uncompressed memory. The regime where it pays (decode-bound, many questions per video, offloaded caches) is neither defined nor shown. Generality is shown on one architecture family with textual timestamps.

All are repairable; reframed around a predictive account of when contrast helps, the paper could reach the spotlight bar.

---

## Strengths

### S1: The counterfactual is chosen by experiment, not by analogy
Most contrastive-decoding papers inherit VCD's noised input. This one compares no-video, noised-frame (VCD), permuted-timestamp, attention-gain (PAI-style) and instruction-based alternatives on the same memory. The empty input wins clearly (+6.2* vs +0.5). The explanation is mechanistic: noised frames still produce the prior, so contrasting against them cancels content, not prior. This is the kind of evidence the CD literature usually lacks.
**Evidence Anchor**: `table: tab:cf — No video (ours) 32.1 (+6.2*) vs Noised frames (VCD) 26.3 (+0.5)`

### S2: Offline replay of decision rules against a gold-tuned oracle
Logging both distributions at the first time value and replaying alternative rules offline is excellent practice. The rules are constant-α, adaptive, evidence-against, skewed or tempered prior, and the best threshold using gold labels. It shows PCD is near the Pareto front of what a decoding rule can do, and it supports the claim that the remaining errors need evidence, not decoding.
**Evidence Anchor**: `text: §6.4 "Even a threshold tuned on the gold labels cannot beat PCD's trade-off between keeping true early starts and removing leaked ones."`

### S3: Scoping to time values plus adaptive strength protects true early starts
The scope ablation documents a known failure of contrastive decoding: it disturbs content tokens (answer text changed in 85% of questions, over-counting doubled). Restricting PCD to time slots is shown to fix this. The confidence-adaptive α rescues the 78 early-evidence questions that constant-α CD hurts (16.7 → 23.1 accuracy). Selective, slot-scoped contrast is a transferable design idea; see W7.
**Evidence Anchor**: `table: tab:scope — early-evidence questions, PCD constant α 16.7 vs adaptive α 23.1 accuracy`

### S4: Coverage transfers under the standard protocol and to a second model
Under the one-line TimeLens-Bench prompt, Coverage beats random selection, SemVID and a matched-budget native low-res input on both Qwen3-VL-8B and TimeLens-8B. It also matches the uncompressed model at about a quarter of the tokens. A practitioner would adopt this result, because it needs no format change.
**Evidence Anchor**: `table: tab:standard — Coverage (31%) 61.7 / 67.3 mIoU vs Random 58.2 / 65.3 vs All tokens 60.5 / 67.1`

### S5: Candid limitations on cost
The paper states plainly that the method reduces decoder reads, not the cost of building the memory. Few efficiency papers say this, and it makes the deployment discussion below possible.
**Evidence Anchor**: `text: §7 Limitations "reduces the memory the decoder reads, not the cost of building it"`

### S6: Floor-adjusted evaluation
Making video-free constant intervals and a length-rescaling baseline part of every comparison is the grounding analogue of the "blind baseline" discipline in VQA. This is a methodological contribution in its own right.
**Evidence Anchor**: `table: tab:floors — best constant beats the uncompressed model on S-EMBER, ActivityNet and Charades-STA`

---

## Weaknesses

### W1: PCD is CFG / context-aware decoding, and the paper does not say so
**Problem**: Eq. (eq:pcd) is `s = (1+α) log p(y|M,x) − α log p(y|∅,x)`. This is the classifier-free-guidance rule applied to LMs: Sanchez et al., 2023, "Stay on topic with Classifier-Free Guidance" [UNVERIFIED venue]. It is also context-aware decoding with the video as context: Shi et al., NAACL 2024, "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding". On top it adds the adaptive plausibility set of Contrastive Decoding (Li et al., ACL 2023) / VCD. M3ID's contrast against an absent image is already cited.

In the PMI / calibration view, the no-video pass is a content-free input in the sense of Zhao et al., ICML 2021 ("Calibrate Before Use"). Amplifying `log p_M/p_∅` is a tempered form of domain-conditional PMI scoring (Holtzman et al., EMNLP 2021, "Surface Form Competition"). None of these five appear in main.bib. §2 positions PCD only against hallucination-mitigation methods.
**Evidence Anchor**: `absence: §2 Contrastive decoding and §4.2 — expected Contrastive Decoding (Li et al. 2023), CFG for LMs (Sanchez et al. 2023), CAD (Shi et al. 2024), contextual calibration (Zhao et al. 2021), DC-PMI (Holtzman et al. 2021); checked main.bib (41 entries), sec/2_related.tex, sec/4_method.tex`
**Why it matters**: An informed reader will see "PCD" as a renamed CAD/CFG. Without the lineage, the novelty claim reads as overreach. With it, the method becomes principled: it is guidance toward the input-conditional PMI. It also inherits known theory and failure modes, which explains the paper's own observations:
- Amplified noise when the evidence term is small. This is why adaptive α helps.
- Degenerate continuations. This is why the answer reserve was needed (W9).
**Suggestion**:
- Rewrite the CD paragraph of §2 and the opening of §4.2. State that Eq. (eq:pcd) is CFG/CAD with guidance weight 1+α, and that α→∞ (under the plausibility set) approaches PMI ranking.
- Rename the contribution as what is actually new: (a) *slot scoping* to time values, (b) the *confidence-adaptive* weight, (c) the *diagnosis* that the no-video prior, not the content, is the right negative for grounding.
- Cite adaptive-weight CAD variants as related or concurrent. Example: "AdaCAD" [UNVERIFIED — search lead: adaptive context-aware decoding, 2024/25].
**Severity**: Major
**Confidence**: 5 — core expertise: contrastive decoding / CFG / calibration

### W2: Is the "temporal prior" a decoder property or a format property? The paper has not separated the two
**Problem**: The diagnosis says the prior lives in the decoder. "Start early, stop soon" is measured with the timeline prompt, whose first line is a chronological `Seen:` list. A chronological narration naturally starts at the beginning: 89% of first `Seen:` moments are at 0 s. The new evidence shows that the timeline format alone costs about 30 mIoU against the one-line prompt on QVHighlights, uncompressed. It also shows that PCD has no effect under the one-line prompt.

An alternative reading fits all of this. Much of the "prior" is format-induced (a convention of narrating in time order), and PCD mainly repairs a leak the timeline prompt created. The paper's evidence that the prior "acts in the output distribution" (the instruction test) was also run only with the timeline prompt. I found no no-video run under the one-line prompt in the draft or in results_2026-10-15.md.
**Evidence Anchor**: `table: tab:prompt — All tokens 60.5 (one-line) vs 30.4 (timeline); Coverage one-line greedy 61.7 vs PCD° 61.6`
**Why it matters**: This decides whether the paper's second half ("contrast the prior") is a statement about video LLMs under compression, or about one answer format. The title and abstract currently claim the former.
**Suggestion**: Run a cheap, decisive check. Use the no-video memory under the one-line prompt on QVH-513 and S-EMBER, and report:
- the early-start share and the predicted/gold length ratio;
- the per-slot KL(p_M ‖ p_∅) at the first time token under both prompts;
- the leak rate of pruned memories under the one-line prompt.

Then rescope the claim to fit what is found. If the one-line no-video prior is far less skewed than 90% / 0.45×, state that the leak is largely format-induced and that PCD repairs reasoning-style answers. That is still a useful finding, because reasoning-style answers are where the field is heading (W3). If it is as skewed, explain why the one-line answer resists it; the likely reason is that a single interval is decoded with a high-confidence first token, which α_t then fades.
**Severity**: Major
**Confidence**: 4 — core expertise: priors in LM decoding; I cannot rule out dataset-specific effects

### W3: There is no predictive account of when contrast helps, yet "PCD is a plug-in for any memory" is stated without qualification
**Problem**: §6.4 asserts that PCD is a plug-in for any memory, based on eight memories that all use the same timeline prompt and the same dataset. §6.2b then shows a gain of zero under the one-line prompt, and §4.2 adds a paragraph limiting PCD to reasoning-style answers. As written, the reader gets a list of conditions under which PCD helps or does not help, with no unifying variable.
**Evidence Anchor**: `text: §6.4 "PCD is a plug-in for any memory."`
**Why it matters**: The contrastive-decoding literature lacks a predictive account of when CD helps. VCD-style gains are known to vary widely across settings. This paper already holds the data to supply one, and that would make it more general than a time-token method.
**Suggestion**: Define a per-slot *prior dominance* or *evidence share*, for example:
- `KL(p_M ‖ p_∅)` at time slots,
- the agreement rate of argmax p_M with argmax p_∅, or
- the mass p_M assigns to the prior's mode.

Plot PCD's paired gain against it across every condition already run: 8 memories × 3 budgets in tab:pcd8, both prompts in tab:prompt, two models in tab:tl8b, and four datasets. tab:replay says both distributions were logged at the first time value, so part of this is offline analysis. If the points line up, the paper can state a law: "contrast against the input-free prior helps in proportion to how much the prior already decides the slot". Pre-register one prediction before running it. For example, PCD ≈ 0 on NExT-GQA's one-line interval query, and a positive gain when answer and interval are asked jointly with a rationale. Replace "plug-in for any memory" with the measured scope.
**Severity**: Major
**Confidence**: 4 — core expertise: CD/calibration analysis

### W4: The cost claims are unmeasured
**Problem**: Coverage requires the full prefill plus a question pass over the uncompressed memory. The paper notes that the current implementation runs 3–4 scoring passes. PCD adds a second, prompt-only decoding stream for time tokens. No latency, peak memory, throughput or energy numbers are reported. There is a further engineering question. The relevance score in Eq. (eq:relevance) needs explicit attention probabilities from every layer and head over N ≈ 21k visual tokens, and fused attention kernels (FlashAttention-style) do not return these. How this is computed, and at what memory cost, is not described.
**Evidence Anchor**: `text: §4.1 Cost "Coverage adds one forward pass of the question over the uncompressed memory per question."`
**Why it matters**: The paper's motivation is cost ("video LLMs pay for every frame"). Without a cost table, a reviewer cannot tell whether CoCo is cheaper than simply not pruning, which under the standard prompt is equally accurate (tab:standard).
**Suggestion**: Add a cost table on one GPU type for: unpruned; random 10%; SemVID; Coverage (1 scoring pass); CoCo (Coverage + PCD). Columns:
- prefill time, scoring-pass time, TTFT;
- decode tokens/s, plus PCD overhead as a fraction of decode time;
- peak GPU memory;
- KV bytes resident during decode.

Report the fraction of generated tokens that are time slots, which is what PCD's overhead scales with. Note that the no-video stream can be batched with the main stream as in CFG implementations. Describe the attention-probability computation, for example a separate eager pass over the question rows only.
**Severity**: Major
**Confidence**: 5 — core expertise: LLM serving

### W5: The regime in which query-time pruning after a full prefill pays off is not defined or shown
**Problem**: The memory is question-specific, so it cannot be shared across questions. Peak memory and TTFT equal those of the uncompressed model. Pruning therefore saves only during decode: KV reads per generated token, and the KV resident while decoding. That pays off under one of four conditions, none of which is shown:
- (a) Long answers, where decode dominates. The timeline answers here are up to 384 tokens, which is a synergy with PCD worth stating.
- (b) Many concurrent sessions at fixed GPU memory, if the full cache is released or offloaded after scoring.
- (c) Many questions per video against a full cache held in CPU or disk, with per-question selection gathered into GPU.
- (d) Streaming, with a bounded write-time memory and question-time re-selection.

The intro cites streaming ("long or streaming video multiplies this"), and §2 positions against streaming memories, but the method is not a streaming memory.
**Evidence Anchor**: `text: §7 Limitations "the full memory is prefilled once per question, so peak memory and time-to-first-token are those of the uncompressed model"`
**Why it matters**: A deployment reader (an egocentric assistant that answers many questions over hours of video) cannot tell whether to adopt Coverage. The paper's own TODO ("write-once memory re-ranked at question time") is the experiment that would turn a benchmark trick into a systems contribution.
**Suggestion**: Two experiments.
1. **Amortisation curve.** On S-EMBER, about 475 questions over about 300 videos, so some videos carry several questions. Keep one full cache per video offloaded to CPU, and per question gather the Coverage subset to GPU. Plot GPU-resident memory, end-to-end latency and mIoU against questions per video. Compare against re-prefilling per question and against the unpruned cache.
2. **Two-tier streaming memory.** Write each frame into a bounded memory with a query-agnostic equal quota, the "stratified" selector already in tab:retention, at for example 30%. At question time, re-select within it by the question's attention down to 10%. Compare at equal resident memory with the HERMES streaming 4k/6k memories that already exist in tab:pcd8. Report accuracy against question delay (how long after the event the question is asked).

If (2) holds up, the paper gets a deployable streaming variant, and the "Not a streaming memory" limitation becomes a result.
**Severity**: Major
**Confidence**: 5 — core expertise: streaming assistant deployment

### W6: Generality beyond the Qwen3-VL timestamp mechanism is untested
**Problem**: Both models are Qwen3-VL-8B; TimeLens-8B is a fine-tune of it. The mechanism behind Coverage is specific to the textual timestamp tokens interleaved in Qwen3-VL ("a timestamp survives exactly when one of its frame's visual tokens survives"). Other designs encode time differently:
- Absolute time in rotary position ids (my understanding of Qwen2.5-VL's MRoPE; to verify). Every kept token then carries its time, so an emptied frame loses content but no separate anchor.
- No time encoding at all (many LLaVA-style video models). There, time must come from frame index, prompt text or overlays such as NumPro.

Whether "keep every anchor" or "keep every frame non-empty" is the operative principle cannot be told apart on one family.
**Evidence Anchor**: `text: §5 Models "a Qwen3-VL-8B model fine-tuned for grounding"`
**Why it matters**: The spotlight bar asks for gains on more than one model. Two checkpoints of one family with one time-encoding scheme is closer to one model than two.
**Suggestion**: Add one model from a different family and time-encoding design. The cheapest informative choice is a positional-time model (e.g., Qwen2.5-VL-7B; verify its time encoding) or a non-Qwen video LLM (e.g., InternVL or LLaVA-Video family [UNVERIFIED for time handling]). Run, under the one-line prompt on QVH-513: Coverage vs random vs SemVID at 31%. Also run the no-video prior measurement from W2. State which principle the result supports.
**Severity**: Major
**Confidence**: 4 — core expertise in decoding; architecture details of other families to verify

### W7: Generality beyond time tokens is the paper's most transferable idea and is left implicit
**Problem**: The paper's own scope ablation shows that contrasting every token damages content (over-counting doubled), while contrasting only the slots whose value must come from the input is safe. That idea, selective contrast on evidence-bearing slots, applies to any structured output read from a compressed input: box coordinates in referring-expression grounding, counts, page or line numbers in document QA, speaker turns or timestamps in audio LLMs. Coverage likewise generalises to "anchor coverage" for any positional marker (pages, spatial tiles, audio segments).
**Evidence Anchor**: `text: §6.4 "over-counting rose from 14% to 34% of counting questions"`
**Why it matters**: A CVPR spotlight is remembered for an idea that travels. Here that idea is "under compression, keep the anchors and contrast only the anchored slots", not "a better time decoder".
**Suggestion**: Add one small table outside time, for example spatial grounding (box coordinates) under 10% visual-token pruning with Qwen3-VL. Apply PCD scoped to the coordinate tokens, and Coverage with an equal quota per spatial tile. If space is short, at least state the generalisation explicitly in the discussion, as a hypothesis.
**Severity**: Minor
**Confidence**: 4 — core expertise: decoding; spatial grounding outside my core

### W8: The right "length" comparator is a calibration baseline, not only rescaling
**Problem**: On pooled S-EMBER, ActivityNet and Charades-STA, CoCo is statistically tied with length rescaling. PCD's effect there cannot be separated from a calibrated shift of the time distribution. From a calibration standpoint, the natural cheap competitor is *contextual calibration* of the time-token distribution: divide by a precomputed, question-independent no-video distribution, or apply an affine correction fitted without labels. This needs no second decoding stream.
**Evidence Anchor**: `table: tab:paired — Length rescaling row: +1.0 (S-EMBER), +0.8 (ActivityNet), −0.7 (Charades-STA), none significant`
**Why it matters**: If a one-pass calibration matches PCD, that is a better deployment result (W4) and a sharper theory result: the prior is static. If it does not match, it shows the prior is prefix-dependent and that per-step contrast is needed. tab:replay's "skewed or tempered prior" row hints at this but is not a calibration baseline.
**Suggestion**: Add the following, replayable offline from the logged distributions where possible: "calibrated time tokens", meaning `log p_M(y) − λ log p̄_∅(y)`, where p̄_∅ is averaged over questions per slot position and λ is set without labels.
**Severity**: Minor
**Confidence**: 4 — core expertise: calibration

### W9: The answer reserve is a symptom of a known CD degeneracy and should be framed as constrained decoding
**Problem**: Contrast rewards continuations the prior finds unlikely, which produced run-on `Seen:` lists (25–68 unparseable of 500). The paper frames the fix as part of PCD and as "not a heuristic on the content". It is a forced-token intervention after a step count. It is reasonable, but it is a band-aid on a degeneracy the CD literature already documents.
**Evidence Anchor**: `text: §4.2 Answer reserve "This is a format constraint, not a heuristic on the content"`
**Why it matters**: Reviewers from decoding will ask why the plausibility set (β = 0.1) does not prevent this. The likely reason is that PCD is off on the newline / "Answer:" decision, so the list length is governed by greedy decoding over contrasted contents. A principled fix generalises better.
**Suggestion**: Explain the mechanism (which tokens decide list continuation, and whether PCD touches them). Compare the reserve with grammar-constrained decoding (cap on `Seen:` lines, or an FSM over the timeline format), or with excluding list-continuation tokens from contrast. Present the reserve as a decoding constraint in the CD tradition rather than as a PCD component.
**Severity**: Minor
**Confidence**: 4 — core expertise: CD failure modes

### W10: The no-video branch is conditioned on video-derived text, so it is not a pure prior
**Problem**: p_∅ conditions on the prefix generated so far, which was written with the video (earlier `Seen:` times and descriptions). Late in the answer, p_∅ therefore already contains evidence copied from M through the text. This weakens the "evidence from M" reading of Eq. (eq:decomp) and may explain why PCD's effect concentrates at the first time values. Context-aware decoding has the same property. That is not a flaw, but it should be stated.
**Evidence Anchor**: `equation: Eq. (eq:decomp) — "evidence from M" term when y_<t already carries video-derived content`
**Why it matters**: It changes how the decomposition should be read, and it bears on W3, because evidence share falls along the answer.
**Suggestion**: Add one sentence. Report the per-slot KL along the answer (first, middle, final interval tokens).
**Severity**: Minor
**Confidence**: 4 — core expertise: CD theory

### W11: There is no abstention or confidence output, although the method computes the needed signal
**Problem**: The error analysis shows that with sparse evidence (fewer than 4 of 64 frames inside the gold interval), every method fails. A deployed assistant answering "how long did I use the paper cutter?" would then return a confident, wrong interval. PCD already computes `log p_M/p_∅` at the time slots, which is a natural "was this in memory?" score for selective prediction.
**Evidence Anchor**: `text: §6.10 "no decoding rule recovers evidence that is not in the input"`
**Why it matters**: For egocentric assistants, a wrong interval presented with confidence is worse than "I did not see it". It erodes trust and can mislead time-tracking uses such as health or safety logs.
**Suggestion**: Report a risk–coverage curve: GQ@0.5 when abstaining on the lowest-evidence-share fraction of questions. One figure in the appendix would do.
**Severity**: Minor
**Confidence**: 3 — deployment experience; metric conventions for grounding abstention to check

---

## Detailed Comments

### Assumption Audit
- **Explicit assumptions**:
  - "The prior acts in the output distribution and has to be removed there" (§3.3). This rests on the instruction test with a single prompt; see W2.
  - The equal quota keeps every time anchor. This is true for textual timestamps; see W6.
- **Implicit assumptions**:
  - (i) The video-free distribution is the model's prior rather than a property of the answer template (W2).
  - (ii) Pruning saves cost even though it happens after a full prefill (W4/W5).
  - (iii) A single no-video counterfactual suffices for every slot. It is prefix-conditioned (W10).
  - (iv) The reasoning-style answer is a fair host format. It is natural for grounded QA (S-EMBER), but on single-moment benchmarks it costs about 30 mIoU.
- **Paradigmatic assumptions**: Benchmark-centric evaluation, best number per dataset. A decoding-research reader would rather see one *mechanistic variable* (evidence share) that predicts gains across conditions (W3). A deployment reader would rather see cost per answered question.

### Cross-Disciplinary Connections
- **Parallel research (NLP decoding)**: CFG for LMs, CAD, Contrastive Decoding, contextual calibration, DC-PMI (W1). These give PCD a theory: guidance toward PMI with respect to a content-free input. They also predict its failure modes: noise amplification at low evidence share, and degenerate continuations.
- **Parallel research (VQA language bias)**: Subtracting a "blind" branch at inference is the core of RUBi (Cadene et al., NeurIPS 2019) and Counterfactual VQA (Niu et al., CVPR 2021). In CF-VQA's terms, PCD estimates something like the effect of the video beyond the language-only effect. VQA-CP (Agrawal et al., CVPR 2018) is the precedent for making blind floors part of evaluation. Citing this line connects the floor-adjusted evaluation and PCD to a well-known CVPR debate and strengthens the novelty case for the *diagnosis*.
- **Borrowing opportunities**:
  - Evidence share as a selective-prediction score (W11).
  - Grammar-constrained decoding for the timeline format (W9).
  - KV-offload or prefix-caching infrastructure for the amortisation experiment (W5). Paged KV storage as in vLLM (Kwon et al., SOSP 2023) makes per-question gather of a subset of a stored cache natural.
- **Methodological borrowing**: Pre-registered predictions (the authors already pre-registered one test; extend this to the PCD-gain law in W3). Cost reporting per answered question, as in serving papers.

### Practical Impact
- **Real-world application**:
  - Coverage under the standard prompt is immediately useful: it equals unpruned accuracy at about 24–31% of the tokens on two checkpoints.
  - PCD is useful where the assistant explains before it grounds, the natural UX for grounded QA ("I saw you at the cutter at 12 s, 40 s… so about 2 min"). S-EMBER is exactly this, and the +10.7 on TimeLens-8B is the strongest practical PCD result.
- **Implementation feasibility**:
  - Coverage needs attention probabilities from a full question pass (not available from fused kernels) and a full prefill. With no offload or amortisation, it saves only decode-time KV.
  - PCD needs a second, small KV stream and per-token slot detection (a format parser in the loop). Both are feasible, but they need the cost table (W4).
- **Stakeholders**: End users of egocentric assistants, who care about wrong-but-confident intervals (W11). Serving operators, who care about cost per answered question (W5).

### Broader Implications
- **Ethical dimensions**: An always-on egocentric memory that guarantees every moment keeps an anchor also retains more of the user's timeline. Retention policy and on-device processing deserve a sentence. Confident wrong durations can matter in health, work-time or safety logs (W11).
- **Social impact**: Low; the method is training-free and reduces compute at decode time.
- **Future directions**:
  - (1) Evidence-share law for contrastive decoding across modalities.
  - (2) Two-tier streaming memory (write-time quota, read-time question re-selection).
  - (3) Slot-scoped contrast for spatial or count outputs.
  - (4) Reasoning ("thinking") video LLMs. If they increasingly write reasoning before the interval, PCD's niche grows. Test on a thinking variant (e.g., a Qwen3-VL "Thinking" checkpoint [UNVERIFIED]) and on RL-tuned grounding reasoners [UNVERIFIED — search lead: R1-style video temporal grounding, 2025].

---

## What the new evidence means for the thesis, and how to make it a stronger, more general insight

The draft's title gives equal weight to "Keep the Clock" and "Contrast the Prior". The new results are asymmetric:
- **Coverage** is format-independent and model-independent within the family tested (tab:standard, NExT-GQA at 30%).
- **PCD** helps only when time values are written before the interval. On one-line answers its gain is about 0 (tab:prompt).

I recommend making that asymmetry the thesis rather than hedging it:

> *Under compression, a video LLM loses anchors before it loses content. Where anchors are missing, the output is decided by an input-free prior, and how much the prior decides a slot is measurable (evidence share). Keeping anchors (Coverage) helps every answer format. Contrasting the prior helps exactly in proportion to prior dominance, which is high in reasoning-style answers and low in single-interval answers.*

This turns a format limitation into a prediction, makes PCD's lineage (W1) an asset, and is testable with data the authors mostly already have (W3). It also gives a clear answer to the question a decoding audience will ask: "why does your contrastive method work here when VCD-style methods are fragile elsewhere?"

---

## What would make it a spotlight (separate from the acceptance bar)

Acceptance requires W1–W5 to be addressed: lineage rewrite, format-prior check, scoped claims, cost table, and a defined deployment regime. For spotlight, in priority order:
1. **One insight, one figure.** The evidence-share vs PCD-gain plot across all existing conditions (W3), with one pre-registered prediction confirmed on a new condition such as NExT-GQA joint vs one-line.
2. **Headline under the standard protocol.** Lead with Coverage on the full TimeLens-Bench (all three sets, both checkpoints, matched-budget native inputs). Keep the timeline-prompt QVH 54.8 as a secondary result about reasoning-style answers, not the teaser number.
3. **A second model family** with a different time encoding (W6).
4. **A cost and amortisation figure** (W4/W5), ideally with the two-tier streaming variant. This is what a deployment-minded AC would remember.
5. **One non-time slot** (W7) to show the idea travels.

Items 1, 2 and 4 are mostly analysis plus queued compute. Items 3 and 5 are one table each.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #4 focus (principled grounding of PCD); spotlight bar (1) | PARTLY_MEETS | `table: tab:cf` (S1); W1 absence anchor | The diagnosis and slot-scoped adaptive contrast are new. The PCD rule itself is CFG/CAD and is not positioned as such | Novelty of Coverage judged by other seats | yes — the lineage rewrite is required |
| Methodological Rigor | — | NOT_ASSESSED | — | Outside this seat's remit (R1) | — | no |
| Evidence Sufficiency | Card #4 (deployment; generality) | PARTLY_MEETS | `table: tab:prompt` (W2); W4 and W6 anchors | Decoding ablations are strong. Cost, deployment regime, second family and the format-prior check are missing | Benchmark numbers taken as reported | yes — W2, W4, W5 |
| Argument Coherence | Card #4 (thesis vs new evidence) | PARTLY_MEETS | `table: tab:pcd8` (W3 context); W3 anchor | Unqualified PCD claims coexist with format-dependence results. The thesis should be re-centred on an asymmetry with a predictive variable | Internal consistency auditing is DA's role | yes — rescoping is required |
| Writing Quality | — | NOT_ASSESSED | — | Outside remit | — | no |
| Literature Integration | — | NOT_ASSESSED | — | Systematic coverage is R2's role. Lineage gaps relevant to positioning are raised under W1 only | — | no |
| Significance & Impact | Card #4 (deployment; generality); spotlight bar (1)–(3) | PARTLY_MEETS | `table: tab:standard` (S4); W5 anchor | Coverage under the standard prompt is practically useful. The general insight and the deployment value are not yet demonstrated | Single model family | yes — determines spotlight vs poster |

The recommendation (Major Revision) follows from three repairable, decision-bearing gaps: lineage and novelty framing (W1), the format-vs-decoder prior confound with the resulting scope of the PCD claims (W2/W3), and unmeasured cost and deployment regime (W4/W5). None is fatal. Strengths on Originality do not offset the Evidence Sufficiency gaps.

---

## Cross-Disciplinary Reading Recommendations
- Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization", ACL 2023. The plausibility constraint PCD inherits via VCD.
- Shi et al., "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding", NAACL 2024. The same rule as Eq. (eq:pcd), with context = document.
- Sanchez et al., "Stay on topic with Classifier-Free Guidance", 2023 [UNVERIFIED venue; arXiv 2023]. CFG for LMs; guidance weight = 1+α.
- Ho & Salimans, "Classifier-Free Diffusion Guidance", 2021/22 [UNVERIFIED venue]. Origin of the guidance form.
- Zhao et al., "Calibrate Before Use: Improving Few-Shot Performance of Language Models", ICML 2021. Content-free input as the calibrator; motivates the one-pass calibration baseline (W8).
- Holtzman et al., "Surface Form Competition: Why the Highest Probability Answer Isn't Always Right", EMNLP 2021. Domain-conditional PMI; the limit of strong contrast.
- O'Brien & Lewis, "Contrastive Decoding Improves Reasoning in Large Language Models", 2023 [UNVERIFIED venue; arXiv]. CD helps reasoning-style generation, which parallels the finding that PCD helps timeline answers.
- Cadene et al., "RUBi: Reducing Unimodal Biases for Visual Question Answering", NeurIPS 2019; Niu et al., "Counterfactual VQA: A Cause-Effect Look at Language Bias", CVPR 2021. Subtracting a blind branch; causal framing for PCD.
- Agrawal et al., "Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering", CVPR 2018. Precedent for prior-aware evaluation.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. Infrastructure for the per-question KV gather (W5).
- [UNVERIFIED] Search lead: adaptive context-aware decoding (e.g., "AdaCAD", 2024/25), for the confidence-adaptive α.
- [UNVERIFIED] Search lead: "chain-of-thought can hurt perception-heavy tasks" (e.g., Liu et al., 2024, "Mind Your Step (by Step)"). Context for the 30-mIoU cost of the timeline format.

---

## Questions for Authors
1. Under the one-line prompt with **no video**, what are the early-start share and the predicted/gold length ratio on QVH-513 and S-EMBER? Do they match the 90% / 0.45× measured with the timeline prompt? (W2)
2. Across the conditions you have already run, does PCD's paired gain correlate with a per-slot prior-dominance measure such as KL(p_M ‖ p_∅) or argmax agreement at the first time token? (W3)
3. At what questions-per-video, answer length or concurrency does Coverage beat simply keeping the uncompressed cache on end-to-end cost per answered question? How are attention probabilities over about 21k tokens obtained in practice? (W4/W5)
4. On a model whose time is encoded positionally rather than as text, does the equal quota still beat random selection under the one-line prompt? (W6)

---

## Minor Issues

### Language / Grammar
- §4.2 heading text "Answer reserve" carries a `\label` inside a paragraph; the reference resolves to the subsection number. Check `cref{sec:reserve}` renders as intended.

### Citation Format
- §2 "Contrastive decoding" paragraph: once the CFG/CAD lineage is added (W1), cite M3ID as the closest vision instance of the no-input contrast, rather than listing it alongside noise-based contrast.

### Figures and Tables
- tab:prompt mixes PCD without the reserve (one-line) with PCD with it (timeline). The matched runs are still missing (results doc §2e); mark the asymmetry in the table body, not only in the caption.
- The teaser headline (×1.9 the uncompressed model, QVH timeline prompt) is the number most exposed to W2. Consider leading with the S-EMBER grounded-QA gain or the standard-prompt Coverage result.

### Layout
- None beyond the above.
