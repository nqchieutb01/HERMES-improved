# Peer Review Report

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (CoCo)
- **Manuscript ID**: CVPR 2026 draft (commit 2c84a78; main.tex + sec/*.tex incl. sec/6b_transfer.tex)
- **Review Date**: 2026-10-15
- **Review Round**: Round 2 (spotlight bar)

---

## Domain Review Report (Peer Reviewer 2)

### Reviewer Identity
Senior researcher in video temporal grounding and grounded VideoQA (Charades-STA, ActivityNet Captions, QVHighlights, NExT-GQA, TimeLens-Bench; grounding-tuned Video-LLMs; benchmark-bias literature). Blind spot: decoding theory (left to Reviewer 3). Statistics, batch-size noise and cost measurement are left to Reviewer 1.

### Review Focus
Are the results state-of-the-art-relevant on standard grounding benchmarks under standard protocols? Is the literature on grounding-tuned Video-LLMs, benchmark bias, video token pruning / KV compression and contrastive decoding covered? Does the insight change how the community builds or evaluates grounding under compression, and what would make it spotlight-level?

### Overall Recommendation
**Major Revision** (for acceptance). Not spotlight-level in its current form. One result is missing: a 10%-budget comparison under the standard one-line protocol on the full TimeLens-Bench. That result decides between "accept after reframing" and "spotlight".

### Confidence Score
4: the task is within my expertise. Scope limit: I cannot verify the exact published numbers of every 2025–2026 grounding model from memory, so I place the paper's numbers mainly against the TimeLens-Bench anchors that the paper itself reproduces.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

No criteria manifest was supplied (`criteria_binding_unavailable`), so I make no venue-alignment claim. The judgements below are field-general.

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #3 + spotlight bar (1) | PARTLY_MEETS | `text: §1 "Neither operator is new in isolation"` | The diagnosis is new for grounding: pruners fall below random selection, and a no-video prior causes early starts. The operators are known. | The novelty of "random selection is strong" relative to the QA-pruning literature is still to verify (W5) | yes: the insight is the main contribution |
| Methodological Rigor | (R1's remit) | NOT_ASSESSED | — | Left to Reviewer 1 | — | no |
| Evidence Sufficiency (SOTA relevance) | Card #3; spotlight bar (2) | DOES_NOT_MEET | `table: Table tab:standard — TimeLens-8B Coverage 31% vs All tokens +0.2` | No 10% result under the standard protocol. The headline numbers use a prompt that costs the uncompressed model about 30 mIoU. | The full sweep is running and may change this | yes |
| Argument Coherence | Card #3 | PARTLY_MEETS | `figure: Fig. 1 right panel — R@0.5 28.6 vs 54.8` | The body states the prompt caveat honestly. The teaser and title do not. | none identified | yes (framing) |
| Literature Integration | Card #3 | PARTLY_MEETS | `absence: §2 Related Work — expected grounding-tuned Video-LLMs beyond TimeLens; checked §2, §5, §6b, main.bib` | Pruning and contrastive-decoding coverage is good. Grounding-model and bias coverage is thin. | none identified | yes, repairable |
| Significance & Impact | Spotlight bar (1)(2) | PARTLY_MEETS | `table: Table tab:tl8b — CoCo vs all tokens +10.7 (TimeLens-8B)` | The S-EMBER result is strong. The general grounding claim reduces to Coverage at parity under the standard protocol. | depends on the pending sweep | yes |
| Writing Quality | (R1/EIC) | NOT_ASSESSED | — | — | — | no |

### Summary Assessment
The paper asks what visual-token pruning does to temporal grounding in Video-LLMs. It diagnoses three effects. QA-oriented pruners empty frames and fall below random token selection. A video-free decoder prior writes early, short intervals. Benchmark floors are high. It then proposes CoCo: equal per-frame quotas filled by question attention (Coverage), plus contrastive decoding of time values against a no-video run (PCD).

From a grounding perspective, the diagnosis is the most valuable part. It is careful: the evidence-removal causal test, per-dataset video-free floors, a length-rescaling control, and a negative-results table. The paper also reproduces published TimeLens-Bench numbers within 0.3 mIoU. That is better practice than most compression papers.

The SOTA-relevance story is weaker than the abstract and teaser suggest:
- **Prompt handicap.** The large gains on QVHighlights, ActivityNet and Charades-STA (Tab. main) are measured under a reasoning-style timeline prompt. That prompt costs the uncompressed model about 30 mIoU. Under the standard one-line protocol, the unpruned Qwen3-VL-8B scores 64.9 R@0.5 on a 513-query TimeLens subset of QVHighlights, above CoCo's headline (54.8).
- **What survives under the standard protocol.** Coverage reaches parity with the unpruned model at 24–31% of the tokens and beats random selection by 2–3.5 mIoU. PCD has no effect.
- **The strongest result is on a dev subset.** The strongest result, +10.7 mIoU over an unpruned grounding-tuned model, is on a 475-question, three-type subset of S-EMBER, which also served as the development set.

Coverage of grounding-tuned Video-LLMs and benchmark-bias work is thin. I recommend Major Revision. A full-TimeLens-Bench 10% result plus one long-video grounding benchmark would move the paper toward spotlight.

---

### Strengths

#### S1: Published grounding numbers are reproduced before anything is compared
The authors run the official TimeLens script on all three full TimeLens-Bench sets for two models and match the published mIoU within 0.3. They then show that their own pipeline matches the script on the same queries. Few token-compression papers anchor their unpruned baseline this way, and it makes the standard-protocol rows of Tab. standard credible.
**Evidence Anchor**: `table: Table tab:anchors — Qwen3-VL-8B 48.5/59.6/47.1 vs published 48.3/59.4/46.8`

#### S2: Video-free floors are reported for every method and dataset
The paper builds on Otani et al. (BMVC 2020). For every method and dataset it reports a tuned best-constant floor, a length-rescaling floor and a closed/open split. This is a concrete, reusable evaluation contribution for grounding under compression. It also exposes that a constant beats the uncompressed model on three of four datasets under the paper's prompt.
**Evidence Anchor**: `table: Table tab:floors — best constant 36.9/37.9/29.7 mIoU vs uncompressed 28.0/22.9/16.6`

#### S3: A causal test for the temporal prior
Dropping the frames inside the evidence, against a control that drops the same number of frames elsewhere, raises the leak by 14 points. The answers move to the start of the window rather than spreading out. This is an informative diagnostic for the grounding community, which has mostly documented positional bias in annotations rather than in model decoding.
**Evidence Anchor**: `table: Table tab:causal — leak 31% → 46% (random 10%), 37% → 51% (all tokens)`

#### S4: The gain transfers to a grounding-tuned model on grounded QA
On S-EMBER, CoCo improves the unpruned TimeLens-8B by +10.7 mIoU, and the method ordering is the same for both models. TimeLens-8B is a grounding-specialised fine-tune, so this partly rules out "it only fixes a weak generalist".
**Evidence Anchor**: `table: Table tab:tl8b — CoCo vs all tokens +11.2* (Qwen3) / +10.7* (TimeLens-8B)`

#### S5: Limits are reported honestly
The paper reports that no method reaches the floor on ActivityNet and Charades-STA, that Coverage loses to random selection on short, densely sampled videos, and that PCD needs reasoning-style answers. Reviewers in this area see this kind of honesty rarely.
**Evidence Anchor**: `text: §6.1 "We report these as limits, not successes."`

---

### Weaknesses

### W1: Teaser and title overstate the result, because the headline uses a non-standard prompt
**Problem**: Fig. 1 and the abstract lead with QVHighlights R@0.5 54.8 against 28.6 for all tokens. Under the TimeLens-Bench one-line prompt, the unpruned model reaches 60.5 mIoU / 64.9 R@0.5 on a 513-query re-annotated subset of the same benchmark (Tab. standard), higher than CoCo's headline. On that same subset, switching to the timeline prompt alone drops the unpruned model to 30.4 mIoU (Tab. prompt). The headline comparison (500-query original val sample, low-res, 64 frames) and the standard rows differ in sample and inputs, but the prompt effect is isolated. Section 6b and the Limitations section state this. The teaser caption, which most readers see first, does not.
**Evidence Anchor**: `text: Fig. 1 caption "over 500 queries, \method nearly doubles the R@0.5 of the"`
**Why it matters**: A grounding reader who sees only Fig. 1 will conclude that a 10% memory roughly doubles Qwen3-VL's QVHighlights grounding. Under the community's standard protocol that is false. This is the first thing an area-chair-level grounding expert will check.
**Suggestion**: Rebuild the teaser around a claim that holds under the standard protocol. Two options: (a) S-EMBER grounded QA with both models, or (b) Coverage against random selection and the matched-budget TimeLens input on TimeLens-Bench. Move the timeline-prompt QVHighlights number into the PCD analysis, labelled "reasoning-style answers". Tie the title's "Contrast the Prior" explicitly to reasoning-style or grounded-QA outputs.
**Severity**: Major
**Confidence**: 5 (core expertise; the numbers are in the manuscript's own tables)

### W2: No result shows the method is SOTA-relevant at the claimed 10% budget under standard protocols
**Problem**: The standard-protocol evidence exists only at 24–31% of the tokens. There, Coverage gains +1.2 mIoU over the unpruned Qwen3-VL-8B (lower bound at about 0) and +0.2 over TimeLens-8B. PCD changes mIoU by −0.1. On NExT-GQA at 10%, Coverage grounds no better than random selection (−0.5 / +0.1 mIoU) and loses about 3 Acc@GQA against the unpruned model.

The Charades-STA and ActivityNet columns of Tab. main (uncompressed 16.6 and 22.9 mIoU, against 48.5 and 47.1 under the standard prompt) cannot be compared with any published grounding number. On Charades-STA, CoCo also loses to Random + PCD (−2.9\*).

In the full TimeLens-Bench sweep (results file §2d), no Random cell is complete. The partial Cover chunks are −0.2 (QVHighlights) and −1.1 (ActivityNet) against the unpruned model.
**Evidence Anchor**: `table: Table tab:standard — Coverage 31% vs All tokens +1.2`
**Why it matters**: A grounding paper at CVPR is judged by where its numbers sit on Charades-STA, ActivityNet Captions, QVHighlights and TimeLens-Bench under the protocol everyone uses. The current evidence supports "Coverage is a good near-lossless pruner at about 30%". It does not support "a 10% memory grounds as well as or better than the full model".
**Suggestion**: Finish the full TimeLens-Bench sweep for both models at 10% and 31%. Include Random, SemVID, a SnapKV-style observation-window eviction and PruneVid, plus the matched-budget TimeLens input. Report R@{0.3,0.5,0.7} and mIoU in one table, next to the published TimeLens-Bench rows for other Video-LLMs that the TimeLens paper reports.
**Severity**: Major
**Confidence**: 4 (core expertise; the sweep may still change the picture)

### W3: The main grounding columns use non-standard inputs and subsets
**Problem**: QVHighlights, ActivityNet and Charades-STA in Tab. main are 500-query samples. The QVHighlights and ActivityNet videos are low-resolution re-encodes (153 and about 84 visual tokens per frame pair). The budget is defined on a 64-frame memory, so "10%" on QVHighlights is about 520 tokens. ActivityNet has to be reported at 30%. The paper cites and reproduces the TimeLens-Bench protocol (full re-annotated sets, 2 fps, 14,336-token cap; Tab. anchors), so the field norm is set by the paper's own chosen reference. These columns follow neither that protocol nor the original benchmarks' full val/test splits.
**Evidence Anchor**: `text: §5 "Our QVHighlights and ActivityNet copies are low-resolution re-encodes (153 and about 84 visual tokens per frame pair, against 660 on S-EMBER)"`
**Why it matters**: The budget-collapse finding (CoCo falls to 9.7 and 11.6 mIoU below about 300 tokens) is partly an artefact of these copies. Readers cannot place any of these numbers next to the literature.
**Suggestion**: Make the TimeLens-Bench / original-resolution full sets the primary grounding table. Move the 500-query timeline-prompt runs to the analysis or supplement as "reasoning-style prompt" studies. Report budgets in absolute tokens per frame alongside percentages everywhere.
**Severity**: Major
**Confidence**: 4 (core expertise; the norm is grounded in the TimeLens-Bench protocol the paper adopts)

### W4: The strongest results come from an under-specified S-EMBER subset that was also the development set
**Problem**: S-EMBER is a published benchmark: S-EMBER, "A Large-Scale Benchmark for Streaming Egocentric Memory Retrieval", arXiv:2607.02689, released by facebookresearch. It has 3,141 videos, 9,448 QA pairs and eight question types. The paper's bib entry is a placeholder (`[authors to verify]`). The paper uses 475 questions of three types (duration, counting, location) from the first 300 videos. Every design choice was tuned on these questions, and the held-out split is a TODO. The headline transfer result (+10.7 mIoU on TimeLens-8B) is on this same set.
**Evidence Anchor**: `text: §5 "475 questions (191 duration, 184 counting, 100 location) on the first 300 videos, asked mid-stream"`
**Why it matters**: The paper's most distinctive result, grounded streaming QA under compression, sits on a non-standard slice of a new benchmark, chosen and tuned without a held-out split. Reviewers will treat it as development-set evidence.
**Suggestion**:
- Cite S-EMBER properly.
- State why these three types were chosen.
- Evaluate the frozen configuration on held-out S-EMBER videos, ideally the full test set with all eight question types, using the official GSER metrics.
- Compare with the baselines reported in the S-EMBER paper.
**Severity**: Major
**Confidence**: 4 (core expertise; S-EMBER details from its arXiv abstract and Hugging Face listing)

### W5: The literature on grounding-tuned Video-LLMs and on benchmark bias is thin
**Problem**:
- **Grounding-tuned Video-LLMs.** TimeLens is the only one cited. TimeChat, VTimeLLM, Momentor, LITA, TRACE, VTG-LLM and the RL-tuned grounding models of 2025 are absent.
- **Benchmark bias.** The section cites only Otani et al. (2020). The distribution-shift re-splits (Charades-CD / ActivityNet-CD) and the compositional splits (Charades-CG / ActivityNet-CG) are not discussed, although they are the community's standard answer to the positional bias the paper re-discovers.
- **Training-free grounding.** The list omits Zheng et al.'s training-free VTG (ECCV 2024).
- **Pruning baselines.** The pruning section does not engage with recent work showing that random or pooling baselines are competitive for multimodal token pruning. That work bears directly on the "below random" claim.
**Evidence Anchor**: `absence: §2 Related Work — expected grounding-tuned Video-LLMs (TimeChat, VTimeLLM, TRACE) and OOD-split bias work (Charades-CD); checked §2, §1, §6b, main.bib`
**Why it matters**: The paper claims a diagnosis of how Video-LLMs ground under compression, so it should say whether the temporal prior and early-start bias appear in grounding-tuned models generally, and how the floors relate to existing debiased splits. Without this, the "benchmark floors" contribution looks like a rediscovery.
**Suggestion**: Add a "Grounding with Video-LLMs" paragraph and expand "Benchmark biases" (see Missing Key References). Ideally, add one result on Charades-CD (OOD test), where positional floors are low by construction. That would be the cleanest test that Coverage/PCD gains are not floor-exploiting.
**Severity**: Major
**Confidence**: 4 (core expertise; references tiered by my certainty below)

### W6: The claimed lesson ("coverage, not saliency") holds only under the reasoning-style prompt and on longer clips
**Problem**: Under the timeline prompt on the QVHighlights-513 subset, Coverage is worse than random selection without PCD (−2.8\*) and tied with it with PCD (+0.1). On Charades-STA and NExT-GQA, which are short and densely sampled, Coverage does not ground better than random selection. With identical inputs (QVHighlights-513, 300 protocol-size frames, 31%), changing only the prompt flips Coverage minus Random from −2.8\* (timeline) to +3.5\* (one-line). The direction of the effect therefore depends on the prompt, and on sampling density across datasets, and the paper does not yet explain this.
**Evidence Anchor**: `table: Table tab:prompt — timeline greedy Random 33.4 vs Coverage 30.7`
**Why it matters**: A spotlight-level insight should tell practitioners when to use equal quotas. Right now the rule is empirical, the frames-per-token threshold was derived on the test results (TODO), and the reasons for the sign flip are not analysed.
**Suggestion**: Report Coverage minus Random as a function of seconds per sampled frame and tokens per frame pair, across all datasets, under the standard prompt. Show whether one curve explains Charades, NExT-GQA, QVHighlights and ActivityNet. If it does, it becomes the paper's general, memorable finding.
**Severity**: Major
**Confidence**: 3 (domain expertise; the mechanism analysis partly overlaps R1's remit)

### W7: The premise of long and streaming video is not tested on a long-video grounding benchmark
**Problem**: The introduction motivates pruning by long and streaming video. The grounding evaluation uses videos of about 30–150 s (Charades-STA, QVHighlights, ActivityNet, NExT-GQA) plus the S-EMBER subset. No hour-scale grounding benchmark is used, so it is unclear whether pruning is forced there and whether equal quotas still keep every anchor when F is far below the video length.
**Evidence Anchor**: `text: §1 "64 frames already cost about 21{,}000 tokens, and long or streaming video multiplies this"`
**Why it matters**: The core claims about compressed grounding survive without this, so it is Minor for acceptance, but it is a spotlight requirement. The setting where compression is unavoidable is where the community most needs guidance, and where reviewers will expect the method to shine.
**Suggestion**: Add one long-video grounding set, either MAD or Ego4D-NLQ (both egocentric or long-form and widely used). Report the uncompressed model at the maximum feasible frames against CoCo at a fixed token budget.
**Severity**: Minor
**Confidence**: 4 (core expertise)

---

### Detailed Comments

#### Literature Review
- **Coverage**: Good on video token pruning (FastVID, VidCom², FlashVID, DyCoke, FrameFusion, PruneVid, LGTTP, KTV, AdaTP), on KV eviction (SnapKV, H2O, PyramidKV, Ada-KV), and on streaming memories (ReKV, StreamMem, InfiniPot-V). Good on visual contrastive decoding (VCD, ICD, M3ID, PAI). Weak on grounding-tuned Video-LLMs and on benchmark bias (W5).
- **Integration quality**: Critical rather than enumerative. The paper says where each line differs from CoCo, and LGTTP is correctly identified as the only pruner evaluated on grounding.
- **Research gap argument**: Convincing for "pruning has not been evaluated for grounding". Less convincing for "the prior is a newly identified failure", because positional bias in model outputs is close to the known annotation bias. The causal test (S3) is what makes it new, and it should be foregrounded.

#### Theoretical Framework
- **Appropriateness**: The two-lever framing (memory coverage vs decoder prior) fits the problem and is supported by the oracle table: an oracle selector still leaks 33%.
- **Application depth**: The framing guides both the design and the error analysis (Tab. errors: "entirely before gold" drops from 33% to 19%).
- **Alternative frameworks**: Debiased evaluation via OOD splits (Charades-CD) is an alternative or complement to tuned constant floors and should be discussed.

#### Academic Argument Quality
- **Factual accuracy**: The reproduction numbers check out against the supporting results file. "Random selection keeps a few tokens on every frame" holds by construction.
- **Argument logic**: The Conclusion's "under the standard grounding prompt, Coverage alone keeps the accuracy of the uncompressed model with a quarter of the tokens" rests on one 513-query subset and a partial Charades run. Phrase it as provisional until the full sweep finishes.
- **Terminology precision**: "Grounded QA" (S-EMBER, NExT-GQA) and "moment retrieval / VTG" (QVHighlights, Charades-STA) are sometimes pooled as "grounding". Keep them separate in the claims. Defining "leak" as an early start with later gold evidence is useful. Define it once in the main text before Fig. 2.

#### Contribution to the Field
- **Incremental contribution**: (a) The first systematic evidence that QA-oriented video pruners damage grounding more than random selection does, at least on longer clips. (b) A floor-adjusted evaluation protocol. (c) A training-free method that clearly helps grounded QA under compression on S-EMBER for two models.
- **Positioning**: Correctly modest about the operators. Needs to be more modest about standard-benchmark gains (W1, W2).
- **Overclaiming**: The teaser and abstract sentence "raises R@0.5 on QVHighlights from 25.0 … and 28.6 … to 54.8" (W1). Contribution 4, "Results on six benchmarks", should say that TimeLens-Bench re-annotates three of the other datasets.

#### Does the insight change how the community builds or evaluates grounding under compression?
Potentially yes, on evaluation:
- Reporting video-free floors and random-token baselines for every compressed-grounding result is a concrete norm the community could adopt.
- "Random selection beats published pruners on grounding" is a memorable warning.

On building, the take-away is not yet stable. Equal quotas help in some regimes and not others (W6), and PCD helps only reasoning-style outputs. To change practice, the paper needs a rule that predicts when coverage matters, validated under the standard protocol.

#### Where the numbers sit
- **Standard protocol, about 30% budget.** Coverage at 31% on QVHighlights-513 gives 61.7 mIoU / 66.7 R@0.5 (Qwen3-VL-8B) and 67.3 / 73.9 (TimeLens-8B). This is on par with the unpruned models (60.5 / 64.9 and 67.1 / 73.1), so near-lossless compression to about 3,300 tokens. That is competitive with the TimeLens-8B operating point and above the matched-budget TimeLens input (+3.0 to +3.7). It is a solid efficiency result, but not "SOTA beating".
- **Timeline-prompt columns.** Tab. main's QVHighlights, ActivityNet and Charades-STA columns are not comparable to published Video-LLM grounding numbers at all.
- **Against token-pruning methods.** On grounding, CoCo beats every re-implemented input-level pruner. It has not been compared with KV-level eviction (SnapKV, PruneVid), which are the closest baselines (TODO in §5). This last point is noted for R1.

#### Missing Key References
Confident (I can attest that these exist):
- TimeChat: Ren et al., "TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding", CVPR 2024. Canonical grounding-tuned Video-LLM.
- VTimeLLM: Huang et al., CVPR 2024. Boundary-aware grounding Video-LLM.
- Momentor: Qian et al., ICML 2024. Segment-level temporal reasoning.
- LITA: Huang et al., "Language Instructed Temporal-Localization Assistant", ECCV 2024. Time tokens for localisation, relevant to the "reads time from timestamps" argument.
- E.T. Bench: Liu et al., NeurIPS 2024 Datasets & Benchmarks. Event-level time-sensitive benchmark.
- Yuan et al., "A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric" (Charades-CD / ActivityNet-CD), 2021. Directly about the positional bias the floors measure.
- Li et al., "Compositional Temporal Grounding with Structured Variational Cross-Graph Correspondence Learning" (Charades-CG / ActivityNet-CG), CVPR 2022.
- MAD: Soldan et al., CVPR 2022. Long-form grounding (W7).
- Ego4D (NLQ episodic-memory task): Grauman et al., CVPR 2022. Egocentric long-video grounding (W7), and the natural sibling of S-EMBER.
- Li et al., "Contrastive Decoding", ACL 2023, and Zhao et al., "Calibrate Before Use" (contextual calibration), ICML 2021. Text-only antecedents of the no-input contrast. Details left to R3.
- S-EMBER: arXiv:2607.02689 (facebookresearch). Replace the placeholder bib entry.

Fairly confident (verify the venue or details before citing):
- TRACE (Guo et al., ICLR 2025), TimeSuite (Zeng et al., ICLR 2025), VTG-LLM (Guo et al., AAAI 2025): grounding Video-LLMs with time tokens. [verify]
- ReXTime (Chen et al., NeurIPS 2024 D&B) and CG-Bench (ICLR 2025): grounded reasoning and long-video clue grounding. [verify]
- Wen et al. 2025, "Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?". Reports that random or pooling baselines are competitive. Needed to position the "below random" claim. [verify venue]
- Zheng et al., "Training-free Video Temporal Grounding using Large-scale Pre-trained Models", ECCV 2024: training-free VTG baseline. [verify]

Search leads only [UNVERIFIED]:
- RL-tuned grounding Video-LLMs of 2025 (e.g. Time-R1, VideoChat-R1).
- Streaming benchmarks with temporal-grounding tasks (e.g. OVO-Bench, StreamingBench).
- Long-video grounding sets such as LongVALE.
- Visual-token pruning methods VisionZip and FastV as image-side pruning references.

### Questions for Authors
1. On the full TimeLens-Bench under the one-line prompt, what are Coverage, Random, SnapKV-style eviction and the matched-budget TimeLens input at **10%** of the tokens for both models? Does Coverage stay above Random there, and how far below the unpruned model is it?
2. Why were only the duration, counting and location types of S-EMBER used, and only the first 300 videos? What does the frozen configuration give on held-out S-EMBER videos and on the other five types?
3. Does the early-start prior also appear in TimeLens-8B with no video, and in other grounding-tuned models? If the prior is weaker after grounding fine-tuning, PCD's role should be scoped accordingly.
4. On a debiased split (e.g. Charades-CD test-OOD), does Coverage still beat random selection under the standard prompt?

### Minor Issues
- Bib placeholders: `sember`, `hermes`, `semvid`, `season`, `reze`, `knowwhen`, `fvaction` carry "[verify title]" or "[authors to verify]".
- "Six benchmarks" in contribution 4: say explicitly that TimeLens-Bench re-annotates three of the other datasets (Charades-STA, QVHighlights, ActivityNet).
- Tab. paired: the S-EMBER "All tokens" cell still shows `\TODO{sig.}`. The results file gives +11.2\* (Qwen3), so fill it in.
- The NExT-GQA two-query protocol differs from many published zero-shot reports. Add the joint-protocol row to the table so readers can compare.
- Charades-STA appears both in its original form (Tab. main) and re-annotated (TimeLens-Bench). Name the version consistently in every table.

---

### What would make it spotlight-level (domain view)
1. **The gating result.** Full TimeLens-Bench (all three sets), one-line prompt, Qwen3-VL-8B and TimeLens-8B, at 10% and 31%. Compare with Random, SemVID, SnapKV-style eviction, PruneVid and matched-budget native inputs. A spotlight needs CoCo or Coverage at about 10% within about 1 mIoU of the unpruned model and clearly above every pruner. Parity at 31% supports acceptance, not a spotlight.
2. **One benchmark where compression is forced.** MAD or Ego4D-NLQ, with ReXTime or CG-Bench as alternatives to verify first. Show that CoCo at a fixed token budget beats the uncompressed model at its maximum feasible frame count. This is where the "keep the clock" argument should be strongest.
3. **Full S-EMBER with a held-out split.** All question types and the official GSER metrics, next to the S-EMBER paper's baselines. Grounded streaming egocentric QA is a distinctive niche, and owning it would be memorable.
4. **One predictive rule.** Coverage minus Random as a function of seconds per frame and tokens per frame, across datasets, under the standard prompt. That turns W6 into the paper's general insight.
5. **Reframing.** Base the title and teaser on claims that hold under the standard protocol. Scope PCD to reasoning-style and grounded-QA outputs. Make floors plus a random-token baseline the recommended reporting norm, with Charades-CD as an extra check.
6. **Literature.** Add grounding-tuned Video-LLMs and debiased-split work (W5), and include at least one grounding-tuned baseline beyond TimeLens.

Items noted for other seats: batch-size noise, bootstrap clustering, cost and latency tables, the reserve-vs-PCD confound (R1); the relation of PCD to CFG and PMI (R3).
