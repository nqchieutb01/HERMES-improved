# Peer Review Report

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (CoCo)
- **Manuscript ID**: CVPR 2026 draft, worktree `paper-draft` at commit 2c84a78
- **Review Date**: 2026-10-15
- **Review Round**: Round 2 (spotlight-bar review)

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Researcher in evaluation methodology for efficient video-LLM inference (visual-token pruning, KV-cache compression benchmarks). Blind spot: novelty positioning (left to R2 and the EIC).

### Review Focus
Would the evidence convince a skeptical expert at CVPR? I checked: standard protocols and full splits; whether budgets are matched in tokens and in compute; whether the strongest input-level and KV-level baselines are present and run in the same pipeline; noise control (batch size, multiple tests, clustering); held-out validation of design choices; reproducibility. Where the paper and the code disagree I read the inference code read-only (`video_qa/hermes_vqa.py`, `inference/qwen3vl_hermes.py`, `inference/qwenvl_hermes.py`, `inference/batched_decoding.py`).

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**: acceptable at CVPR once items W1 to W6 are fixed. **Not yet at spotlight level** (see "Path to spotlight" at the end).

### Confidence Score
4. Core expertise in pruning and KV-compression evaluation; temporal-grounding benchmark specifics are adjacent.

Confidence describes my uncertainty and scope only.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper claims three things. Visual-token pruners built for QA lose the timeline and fall below random selection on temporal grounding. A video-free decoder prior fills the gaps. A training-free fix (equal per-frame quotas filled by question attention, plus prior-contrastive decoding of time values) gives large gains, measured against video-free floors.

The diagnostic half is unusually careful: a causal frame-removal test, video-free floors for every method, a reproduction of published TimeLens-Bench numbers, paired bootstrap throughout, and an honest table of negative results.

The method half is not yet identified well enough:
- The headline CoCo rows include a retrieval front end that the main baselines lack. On the one benchmark where Coverage and random are compared under PCD without retrieval, they tie (+0.1).
- The "below random" result compares KV eviction after a full prefill (random, Coverage) with input-level pruning (FastVID, VidCom², FlashVID, SemVID). These are different regimes in both information and compute.
- The largest numbers (QVHighlights 54.8 vs 28.6 R@0.5) are relative to a prompt that costs the uncompressed model about 30 mIoU.
- Several design choices were fixed after seeing results on the evaluation sets.
- Batch-size noise (±2 points) sits under the "matches the uncompressed model" claims.
- No cost numbers exist, although Coverage needs a full prefill and PCD doubles every decode step.

All of these can be fixed in one revision cycle with roughly 150 to 500 GPU-hours, so I recommend Major Revision. For a spotlight, the paper additionally needs a full-split, two-family, strong-baseline main table under the standard protocol.

---

## Strengths

### S1: A causal test of the prior, not just a correlation
Removing the frames inside the gold interval, against a matched control, raises the leak by 14 points on both the unpruned and the random-10% memory. Answers move toward the start rather than spreading out. This is the right design for the claim that missing evidence causes the prior to take over.
**Evidence Anchor**: `table: Tab. causal — leak 31% control vs 46% evidence (Random 10%), 37% vs 51% (All tokens)`

### S2: Video-free floors are part of every comparison
Each dataset gets a best constant interval tuned on the test questions (an upper bound on constant rules), a length-rescaling baseline, and an open/closed split. The paper reports where CoCo does not clear them (ActivityNet −3.1*, Charades −6.4*, pooled S-EMBER +2.4 n.s.). Few pruning papers do this, and it protects against positional-bias artefacts.
**Evidence Anchor**: `table: Tab. paired — rows "Best constant" and "Length rescaling" across four datasets`

### S3: The pipeline is checked against published numbers before comparisons are made
With the TimeLens script, all six model×dataset anchors reproduce the published numbers within 0.3 mIoU. The authors' own pipeline matches the script on the same queries (paired Δ 0.0 to +1.4, n.s.).
**Evidence Anchor**: `table: Tab. anchors — e.g. TimeLens-8B ActivityNet 53.2 vs 53.2 published`

### S4: Paired statistics and honest negative results
Differences are reported as paired bootstrap intervals over the same questions. A dedicated table lists ideas that did not work. The offline replay of decision rules at the first time value, including a gold-tuned threshold, is a good way to show that PCD sits near the decoding frontier.
**Evidence Anchor**: `table: Tab. replay — "Best threshold (uses gold)" 46% / 6% vs PCD adaptive 51% / 7%`

### S5: Mechanism-level ablations of PCD
Swapping the counterfactual (no video vs noised frames vs permuted timestamps), the scope (all tokens vs time values) and the strength (constant vs adaptive) one at a time supports each PCD design choice. The adaptive rule is tested on the subgroup where it should matter: evidence that truly starts at 0 s.
**Evidence Anchor**: `table: Tab. scope — early-evidence questions, Acc 16.7 (constant α) vs 23.1 (adaptive α)`

---

## Weaknesses

### W1: Coverage's contribution under PCD is not identified; the headline rows bundle retrieval
**Problem**:
- Every "Coverage" and CoCo row in Tab. main includes the optional SigLIP-2 retrieval front end. The strongest baselines (Random + PCD, SemVID + PCD) use uniform frames.
- Retrieval alone lifts random tokens by +3.8* mIoU on S-EMBER (Tab. ladder: 25.6 → 29.4). The headline gain of CoCo over Random + PCD is +4.2* (Qwen3) and +5.5* (TimeLens-8B), so it could come mostly from retrieval.
- The one comparison without retrieval is QVH-513 with the timeline prompt (results file §3). There, (Coverage + PCD) − (Random + PCD) = +0.1 [−1.9, +2.0], and Coverage − Random without PCD = −2.8*.
- The missing cell is a TODO in the paper ("Coverage + PCD without retrieval on all datasets").
- Fig. pruners labels its rightmost bar "Coverage (ours)" under "same 10% budget, greedy decoding". The bar values (+6.9, +7.6) equal R + C − random (32.5 − 25.6, 34.5 − 26.9), so the figure also bundles retrieval. The retrieval-free values are +6.1 and +6.8.

**Evidence Anchor**: `table: Tab. ladder — "R, random tokens" 29.4 vs "Uniform, random tokens" 25.6 on S-EMBER`

**Why it matters**: The paper's central claim is that grounding needs *this combination*. If Coverage adds nothing once PCD is applied, the method reduces to retrieval + PCD and contribution 2 changes. This is the single experiment most likely to change the story, so it goes first.

**Suggestion**:
- Add rows (a) C + PCD on uniform frames and (b) R + random + PCD on all four datasets and both models. Then define CoCo once, with or without retrieval, and use that definition everywhere.
- Relabel Fig. pruners, or plot C on uniform frames.
- Cost: about 2–3 GPU-h per S-EMBER run with PCD and about 1 GPU-h per 500-query low-res run, so about 15–20 GPU-h for both models. **Priority 1.**

**Severity**: Major
**Confidence**: 5 (numbers read directly from Tab. ladder, Tab. main and the re-scored results file)

### W2: "QA pruners fall below random selection" compares two different pruning regimes
**Problem**: I checked the code (`video_qa/hermes_vqa.py` l.369–449).
- **Random and Coverage**: every frame is first prefilled through the full LLM (`encode_video_chunk` on all chunks). The KV cache is then evicted per layer (`predict_and_compress`). Each kept KV entry was therefore computed with attention over all ~21k visual tokens, and each layer keeps a different set.
- **FastVID / VidCom² / FlashVID / SemVID**: these run through `encode_video_fastvid`, which feeds *only* the kept tokens to the LLM. They keep one set for all layers, and their kept tokens never see the dropped ones.

So "random" is a post-prefill per-layer KV-eviction baseline, and the four pruners are input-level methods. The two differ in what information survives and in cost: input pruning saves prefill compute, KV eviction does not. The per-layer vs shared ablation (Tab. shared: random +0.7 n.s.) controls for per-layer diversity but not for full-context prefill.

The within-regime evidence (Tab. retention: HERMES global 18.3 < stratified 23.7 < random 25.8, all KV-level) does support the coverage story. That is why I rate this Major rather than Critical. But the abstract's headline sentence about four pruners is cross-regime.

**Evidence Anchor**: `text: §5 Baselines "FlashVID (vision-side stages) and SemVID, re-implemented and checked to keep the same tokens as the authors' code"`

**Why it matters**: A skeptical reviewer will ask whether random wins because it keeps tokens on every frame (the paper's claim) or because its kept KV entries encode the dropped tokens through full-context prefill. The mechanism claim "a model reads time from the timestamps that survive" is not isolated.

**Suggestion**: Two cheap controls, both supported by the existing code paths:
- (a) **Random at input level**: random token selection through the `encode_video_fastvid` path, same budget, same timestamp rule.
- (b) **Timestamps restored**: run FastVID and SemVID with `keep_time_tokens=all`, so every frame keeps its timestamp. If this recovers most of the gap to random, the timestamp mechanism is confirmed directly.

Optionally, (c) apply each input pruner's kept set as post-prefill KV eviction. Report all three on S-EMBER and QVHighlights.

Cost: about 1–2 GPU-h per run, about 12 runs, so about 20 GPU-h. **Priority 1.**

**Severity**: Major
**Confidence**: 5 (verified in code)

### W3: The strongest baselines for a question-attention KV method are missing, including the ablation that defines the method
**Problem**: Coverage is question-attention KV eviction with a per-frame quota. Its closest competitors are not run:
- SnapKV-style observation-window eviction (top-k by the question window's attention, global, with pooling);
- PyramidKV / Ada-KV layer- and head-adaptive budgets;
- PruneVid;
- the "Question, global top-k, no quota" row, which is an empty TODO cell in Tab. design.

Without that last row, "the quota should be equal / must exist" is shown only for HERMES's generic-probe score (23.7 vs 18.3), not for the question score CoCo actually uses. The other missing baseline is a cheap one: **matched-token low-resolution input** on S-EMBER (64 frames resized so that the visual tokens total about 2,140). It is input-level and saves prefill compute, so any KV method must beat it. Only the QVH-513 TimeLens low-res rows play this role, and only under the one-line prompt.

**Evidence Anchor**: `table: Tab. design — row "Question, global top-k, no quota" is an empty TODO cell`

**Why it matters**: At CVPR, a KV-eviction method that is not compared with SnapKV and with its own no-quota variant will be read as an un-ablated SnapKV with a frame constraint. Contribution 2 depends on these rows.

**Suggestion**:
- Run the question global top-k (no quota), SnapKV (window = question tokens, pooling kernel 7, global top-k), PyramidKV, and low-res matched-token input. Run each with and without PCD on S-EMBER, QVHighlights and Charades, both models.
- Add PruneVid if code permits.
- Cost: about 4 methods × 2 decodings × 3 datasets × 2 models × ~1.5 GPU-h ≈ 70 GPU-h. **Priority 1.**

**Severity**: Major
**Confidence**: 5

### W4: The headline numbers compare against a self-handicapped uncompressed model
**Problem**:
- The abstract, teaser and contribution 4 lead with QVHighlights R@0.5 54.8 vs 28.6 ("nearly doubles the uncompressed model").
- §6b shows that the same uncompressed model scores 60.5 mIoU / 64.9 R@0.5 on QVH-513 with the standard one-line prompt. The timeline prompt costs it −30.0* mIoU.
- Under the standard prompt, PCD has no effect (−0.1 n.s.), and CoCo with the timeline prompt is −17.8* below Coverage with the one-line prompt.
- §6b states the caveat ("the gain over 'all tokens' there does not hold against the one-line prompt"), but the abstract and teaser still lead with the handicapped comparison.

**Evidence Anchor**: `text: Abstract "raises R@0.5 on QVHighlights from 25.0 (random tokens) and 28.6 (all tokens) to 54.8 with our reasoning-style prompt"`

**Why it matters**: Any reader who knows that Qwen3-VL-8B reaches about 60 mIoU on QVHighlights will discount the paper on sight. The large PCD gains are conditional on an answer format the community does not use for single-moment grounding.

**Suggestion**:
- Make the standard-prompt comparison the headline for single-moment benchmarks.
- Present the timeline format only where it is natural (S-EMBER grounded QA, where answers explain before grounding) or as a diagnostic.
- Keep the 54.8 number only next to the 60.5 one-line reference, in the same table.
- Cost: none (rewriting only). **Priority 1.**

**Severity**: Major
**Confidence**: 5

### W5: Design choices were tuned on the evaluation sets, against the stated frozen-configuration protocol
**Problem**: §5 states that all choices were made on S-EMBER and then run once on the other datasets. Four things contradict this:
- ActivityNet is reported at 30% "because at 10% the memory holds only 120–300 tokens", i.e. after the 10% run collapsed.
- The "≥8 tokens per frame pair, otherwise halve F" rule was derived from the QVHighlights and ActivityNet results. A TODO admits this.
- The closed-interval subset (gold end < 0.8q, with floors re-tuned on the subset) supports a contribution-level claim (+7.6*). No sensitivity to the 0.8 threshold is shown.
- Every S-EMBER number, including the main-table one, is a development-set number, because all scores, quotas, α, scope and the reserve were chosen on those 475 questions. The held-out split is a TODO.

**Evidence Anchor**: `text: §5 Protocol "The configuration was then frozen and run once on QVHighlights, ActivityNet and Charades-STA."`

**Why it matters**: These choices make the reported numbers optimistic, and the paper's own power (W6) is too low to absorb that optimism on the smaller effects. Contribution 3 promises a floor-adjusted evaluation, and that promise requires a clean test split.

**Suggestion**:
- **(a)** Run the frozen CoCo and three baselines (Random + PCD, R + random + PCD, unpruned) on held-out S-EMBER videos beyond the first 300, and report them as the S-EMBER test result. About 4 runs × 2–3 GPU-h ≈ 10 GPU-h.
- **(b)** Validate the 8-token rule on a dataset and budget not used to derive it, e.g. full Charades-STA or TimeLens-QVH at 5% with F chosen by the rule before the run. About 10 GPU-h.
- **(c)** Report the closed-subset claim for thresholds 0.6 / 0.7 / 0.8 / 0.9. CPU only.
- **(d)** Report ActivityNet at 10% in the main table with the rule-selected F, rather than switching the budget.

**Priority 1** for (a) and (c), Priority 2 for (b) and (d).

**Severity**: Major
**Confidence**: 4 (protocol inferred from the paper's own statements and TODOs)

### W6: Noise control: batch-size confound, no clustering, no multiplicity control, and low power for the null ablations
**Problem**:
1. **Batch size.** Unpruned runs use batch size 1–2 and pruned runs use batch size 8. The paper itself says this changes about half of the outputs and up to about two points of a score. That is the size of the claims it is meant to support:
   - "Coverage matches the uncompressed model": +1.2 [+0.005, +2.55], where the results file warns the lower bound depends on the seed.
   - NExT-GQA Coverage-10% vs unpruned: −2.8* Acc@GQA.
   - TimeLens-8B Retrieve + Coverage vs unpruned on QVH-513: −2.8*.
2. **No clustering.** The bootstrap resamples questions. NExT-GQA has 1,005 questions on 191 videos (about 5 per video) and S-EMBER has 475 on 300, so the intervals are too narrow. Clustered bootstrap is a TODO.
3. **No multiplicity control.** The manuscript carries 106 starred differences with no correction and no designated primary endpoints. The text says "our pre-registered primary test" for the SigLIP quota, but no registration artefact is cited.
4. **Low power.** From CoCo − (Random + PCD) on S-EMBER, +4.2 [1.7, 6.8], the paired SE is about 1.3 mIoU, so the minimum detectable effect at 80% power is about 3.6 mIoU. The design-choice ablations (±0.5, n.s.) therefore cannot support "the quota should be equal" or "retrieval adds little". They show only that no large effect exists.
5. **Matching claims.** "Matches" and "lossless" are claimed from non-significant differences, without an equivalence test.

**Evidence Anchor**: `text: §5 Models "comparisons with the uncompressed model, which runs at batch size 1–2, carry this noise"`

**Why it matters**: The paper's most deployment-relevant claim is lossless Coverage at a quarter of the tokens under the standard prompt. It currently sits inside a known ±2-point artefact.

**Suggestion**:
- **(a)** Rerun every unpruned reference at batch size 8, or the key pruned rows at batch size 1. About 2 GPU-h per S-EMBER run and about 1 GPU-h per QVH-513 run, about 10 GPU-h in total.
- **(b)** Measure the batch-noise distribution directly: the same config at batch size 1 vs 8, as a paired difference. Report it as a noise floor.
- **(c)** Use a video-clustered bootstrap everywhere. CPU only, minutes.
- **(d)** Declare about 6 primary endpoints: CoCo vs Random + PCD and vs unpruned (mIoU) on each test dataset. Apply Holm correction to those and label everything else exploratory.
- **(e)** Use TOST with a pre-declared margin (e.g. ±1.5 mIoU) for every "matches" or "lossless" claim.
- **(f)** Rephrase null ablations as "no effect larger than about 3.6 mIoU detected".
- Provide the pre-registration artefact (a timestamped commit or log), or drop the word "pre-registered".

**Priority 1** (cheap).

**Severity**: Major
**Confidence**: 5

### W7: No cost accounting; the Limitations text understates PCD's decode cost
**Problem**: The paper motivates itself by token cost but reports no latency, memory or FLOPs. From the code:
- **Coverage** needs the full uncompressed prefill and then 3–4 question-scoring passes over the full memory. The paper's TODO says this is reducible to one.
- **PCD** runs the counterfactual row at *every* decode step. This holds in both `contrastive_answering` and `batched_decoding.batched_answering`, which builds a second group of rows for every job. It is not limited to time-value tokens, so decode compute and KV reads roughly double for up to 384 timeline tokens.
- Limitations says "PCD also adds a second, prompt-only decoding pass for time-value tokens", which describes neither the cost nor the implementation correctly.

As a result, CoCo is likely *more* expensive end-to-end than the uncompressed model with greedy decoding. Its only saving is the size of the KV cache that the decoder reads.

The budget labels are also inconsistent:
- "31%" in Tab. standard keeps 3,271 of 13,650 visual tokens (24%), because timestamps count inside the budget.
- SemVID "31%" keeps 4,232.
- Random "8.2%" keeps about 0 visual tokens because timestamps fill the budget.

**Evidence Anchor**: `text: §4.1 Cost "Coverage adds one forward pass of the question over the uncompressed memory per question."`

**Why it matters**: Reviewers from the efficiency community will ask for compute-matched comparisons. Input-level pruners and low-res input save prefill. If CoCo cannot show a regime where it wins at equal or lower cost, the paper is about grounding *accuracy* under a KV-size constraint and must say so.

**Suggestion**:
- Add a cost table for unpruned, random-KV, random-input, FastVID, SemVID, low-res input, Coverage, Random + PCD and CoCo, at batch size 1 and 8 on S-EMBER. Columns: TTFT, total latency per question, peak GPU memory, decode KV size, prefill and decode FLOPs, generated tokens.
- Fix the Limitations sentence.
- Report absolute visual tokens (excluding timestamps) in every table, with the budget definition stated once.
- Optionally restrict the negative pass to time slots by caching the negative KV lazily.
- Cost: about 3–4 GPU-h. **Priority 1.**

**Severity**: Major
**Confidence**: 5 (verified in code)

### W8: Scope: non-standard subsets, low-resolution copies, and one model family
**Problem**:
- The main table uses 500-query samples of QVHighlights, ActivityNet and Charades-STA. QVHighlights and ActivityNet are low-resolution re-encodes (153 and about 84 tokens per frame pair, against 660 on S-EMBER), so absolute numbers are not comparable with the literature. The collapse at small budgets (Tab. budget, Fig. budget) is partly an artefact of these copies.
- The full TimeLens-Bench sweep has no complete cell. It is 10 of 144 chunks done, and no full-set Cover vs Random comparison exists (results file §2d).
- "Two models" is one family: TimeLens-8B is a Qwen3-VL-8B fine-tune. The Coverage rationale (time is read from textual timestamp tokens that survive only with their frame) is specific to Qwen3-VL's interleaved timestamps. A model that encodes time only through positions (e.g. Qwen2.5-VL's absolute-time MRoPE) or uses no timestamps (LLaVA-OneVision / LLaVA-Video, for which the codebase already has `llavaov_hermes.py`) is the real generality test.
- S-EMBER accuracy and GQ@0.5 use a local "Qwen3.8-27B-FP8" judge, which is not validated against the official judge. TimeLens-8B answers are unjudged.

**Evidence Anchor**: `text: §5 Datasets "Our QVHighlights and ActivityNet copies are low-resolution re-encodes (153 and about 84 visual tokens per frame pair"`

**Why it matters**: The spotlight bar requires large, consistent gains on standard benchmarks under standard protocols, on more than one model. The current evidence is a Qwen3-VL-specific result on subsets.

**Suggestion**:
- **(a)** Complete the full TimeLens-Bench sweep (all three sets, native protocol). Rows: Coverage, Random, SnapKV and matched-budget TL low-res input, on both models. At about 1.2 GPU-h per 250 protocol-size queries, that is about 45 GPU-h per method per model, so about 360 GPU-h, or about 2 days on the 8-GPU cap.
- **(b)** Add a second model family at 8B: InternVL3.5-8B, or LLaVA-Video-7B / Qwen2.5-VL-7B. Run S-EMBER and TimeLens-QVH with Random, Coverage, Random + PCD, CoCo and SnapKV. About 40 GPU-h plus engineering.
- **(c)** Check judge agreement with the official S-EMBER judge on about 200 answers (κ and accuracy delta), and judge the TimeLens-8B answers. Under 5 GPU-h.
- **(d)** Rerun QVHighlights and ActivityNet at native resolution, or drop the low-res copies in favour of TimeLens-Bench.

**Priority**: 1 for (a), 2 for (b) (essential for spotlight), 1 for (c).

**Severity**: Major
**Confidence**: 4 (cost estimates inferred from the pool ETA in the results file)

### W9: The mixed pipeline versions across tables are hard to audit
**Problem**:
- Tab. pcd8, cf, scope, replay, oracle, errors and causal predate the answer reserve, and use the official parser and batch size 1. The main tables use the reserve, the m:ss parser and batch size 8.
- The reserve itself changes mIoU by +1.5* to +6.1*, so effect sizes are not comparable across tables.
- The text says "on S-EMBER the two parsers differ by at most 0.7 mIoU and batch size changes no paired result". The second half is not shown, and it conflicts with the ±2-point note in §5.

**Evidence Anchor**: `text: §6 Reading the tables "predate the answer reserve, so PCD there runs without it, and use the official parser and batch size 1"`

**Why it matters**: Readers cannot tell which effects survive the current configuration.

**Suggestion**: Re-run at least Tab. pcd8 (8 memories) and Tab. cf with the final configuration (about 10 runs × 2–3 GPU-h ≈ 25 GPU-h), or move the older tables to the appendix with a version banner. **Priority 2.**

**Severity**: Minor
**Confidence**: 5

### W10: Length rescaling is in-sample on S-EMBER, and PCD's benefit is not separable from lengthening on three of four sets
**Problem**:
- The rescaling factor (1.76) is fitted on S-EMBER and applied there too.
- On pooled S-EMBER, ActivityNet and Charades, CoCo ties length rescaling (+1.0, +0.8, −0.7). So on those sets PCD's gain cannot be distinguished from calibrated lengthening, as the paper concedes.
- The proposed start/end error decomposition and mismatched-video negatives are TODOs.

**Evidence Anchor**: `table: Tab. paired — "Length rescaling" row: +1.0 / +8.6* / +0.8 / −0.7`

**Why it matters**: "PCD moves the interval toward the evidence" currently rests on QVHighlights (timeline prompt) and the post-hoc closed subset.

**Suggestion**:
- Fit the factor with cross-validation (fit on half the videos, test on the other half).
- Add the start/end error decomposition and a mismatched-video counterfactual (contrast against another video's memory). CPU plus about 5 GPU-h.

**Priority 2.**

**Severity**: Minor
**Confidence**: 4

---

## Detailed Comments

### Title & Abstract
- The abstract's lead number has the W4 problem. "Four recent training-free pruners fall below random token selection" has the W2 problem: the claim is cross-regime.
- "On S-EMBER the gain over the uncompressed model transfers to TimeLens-8B (+10.7 mIoU)" is well supported (paired CI [8.0, 13.5]). The paired comparison carries the batch-size caveat, but the effect is far larger than the noise.

### Introduction
- Contribution 3 promises "paired confidence intervals", and contribution 4 says "Results on six benchmarks and two models". Both need the scoping in W6 and W8.

### Methodology / Research Design
- The design is a set of paired, within-question comparisons, which is appropriate. The problems are confounded factors in the comparisons (retrieval in W1, pruning stage in W2, prompt in W4), not the paired design.
- I checked Eq. 3 (relevance) and Eq. 5 (PCD) against `qwenvl_hermes.py` (`core_topk`: per-frame top-k by within-frame z-scored question attention, then `_quota_scores`) and `contrastive_answering`. The equations match the code, except that within-frame scores are z-scored and passed through a sigmoid before the quota (rank-preserving, so harmless). One addition is not described: the repetition penalty is applied to the contrasted score.
- Random selection is seeded per question (`2024 + total_processed_frames`). Three seeds give 26.0 ± 0.6. Report the seed spread for the CoCo vs Random + PCD gap as well.

### Results / Findings
- Results are reported completely, including failures (ActivityNet, Charades, NExT-GQA mIoU). That is good practice.
- Selective-reporting risk is low. The risk is in *framing*: abstract and teaser (W4), and the figure label in Fig. pruners (W1).

### Discussion / Limitations
- The Limitations section is candid about prompt dependence and non-streaming memory. It should add the cost facts from W7 and the cross-regime caveat from W2.

### Conclusion
- "With a tenth of the visual tokens it grounds better than the uncompressed model on S-EMBER grounded QA, for a general and a grounding-tuned model" holds as stated, subject to the S-EMBER held-out split (W5).

### Statistical reporting
| Criterion | Judgement | Note |
|---|---|---|
| Effect sizes | MEETS | Paired Δ in metric units throughout |
| Confidence intervals | PARTLY_MEETS | Computed in the results file; the paper shows only * marks; intervals are not clustered |
| Power | DOES_NOT_MEET | No MDE; null ablations read as equivalence |
| Multiple comparisons | DOES_NOT_MEET | 106 starred tests, no correction, no primary endpoints |
| Missing data | MEETS | Unparsed answers scored IoU 0 for every method alike; counts reported |
| Red flags | PARTLY_MEETS | Post-hoc budget switch for ActivityNet, post-hoc closed subset and frames rule (W5); no evidence of p-hacking otherwise |
| Arithmetic recompute | n/a | The paper reports no *t*, *F*, χ² or *p* statistics, only bootstrap intervals, so no bounded recompute procedure applies. I spot-checked internal consistency: Tab. types' per-type means reweight exactly to the pooled Tab. main values (CoCo 39.27, Random 25.55, All tokens 28.02, R+C 32.54). Tab. paired deltas match Tab. main differences to within ±0.1 rounding. |

### Methodological fallacies detected
- **Confounding** in W1 (retrieval), W2 (pruning stage and compute) and W4 (prompt).
- **Overfitting to the evaluation set / HARKing risk** in W5.
- **Absence of evidence read as evidence of absence** in W6.4.

---

## Questions for Authors
1. With retrieval removed, what is (Coverage + PCD) − (Random + PCD) on S-EMBER and with TimeLens-8B? If it is near zero, as on QVH-513, how does the paper's claim of a necessary combination change?
2. If random tokens are dropped *before* the LLM (input level), and if FastVID or SemVID keep every timestamp, does random still beat the input-level pruners?
3. What artefact documents the "pre-registered" SigLIP-quota test, and which other comparisons were pre-specified?
4. Was the Charades-STA retrieval gate ("skipped when uniform frames would be less than 1 s apart") fixed before the Charades run, or after it?

---

## Minor Issues
- Tab. paired: the S-EMBER vs All tokens cell says "+11.3 \TODO{sig.}". The re-scored value is +11.2* [8.5, 14.1], and Tab. tl8b already prints +11.2*.
- Tab. reserve: "Unparseable (of 500)" for S-EMBER, which has 475 questions.
- The body still contains 19 `\TODO` macros, several in tables (Tab. design, Tab. paired) and in the Related Work body.
- §6b: "Coverage at 24% of the visual tokens", while Tab. standard says "31%". Use one budget definition.
- §5: the judge model name "Qwen3.8-27B-FP8" looks wrong. Give the exact model identifier and revision.
- Teaser: say how the example query was chosen (random draw vs selection).
- The reproduction appendix should list the commit hash, and the scripts for Tab. paired and the closed subset (`floors.py` covers only floors; `collect_2026-10-15.py` lives in a tmp directory).

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #2 blind spot | NOT_ASSESSED | — | Outside my remit | — | no |
| Methodological Rigor | Card #2: matched budgets, strongest baselines, same pipeline | PARTLY_MEETS | Tab. ladder; `hermes_vqa.py` l.369–449; Tab. design TODO row | Diagnosis is rigorous; method comparisons are confounded by retrieval, pruning stage and prompt (W1–W4) | Severity of W1 depends on an unrun cell | yes: W1–W3 must be fixed |
| Evidence Sufficiency | Spotlight bar (2)–(3) | PARTLY_MEETS | Tab. main; results §2d | Subsets, low-res copies, one family, no cost table, full sweep incomplete (W7, W8) | GPU-hour estimates inferred | yes for spotlight; repairable |
| Statistical Reporting | references/statistical_reporting_standards.md §1.5, §4.5 | PARTLY_MEETS | §5 Metrics; results §3 | Paired bootstrap is good; no clustering, multiplicity control or power; batch confound (W6) | — | yes; cheap to repair |
| Reproducibility | Card #2 | MEETS | Tab. anchors; Suppl. Reproduction | Published numbers reproduced; config overrides listed; scorers documented | Mixed pipeline versions (W9) | no |
| Argument Coherence | Lens 1: does evidence support claims | PARTLY_MEETS | Abstract; §6b | Abstract and teaser over-reach relative to §6b (W4) | — | yes; text-only fix |
| Writing Quality, Literature, Significance | Other seats | NOT_ASSESSED | — | Other seats | — | no |

**Recommendation rationale.** All the unresolved decision-bearing items (W1–W7) can be repaired within one revision. The two that could change the story are W1 (whether Coverage adds anything under PCD) and W2 (whether "below random" is a pruning-stage artefact). Both are cheap (about 40 GPU-h together) and should be run before anything else. Strengths on diagnosis and reproducibility do not offset these.

---

## Path to spotlight (methodology view), with costs and priorities

GPU-hour basis: inferred from the pool ETA in the results file (about 1.5 days on 6 GPUs for 178 chunks of about 250 protocol-size queries, so about 1.2 GPU-h per 250 queries). An S-EMBER run at 64 native frames is about 2 GPU-h, or about 3 with PCD. A 500-query low-res run is about 1 GPU-h.

| # | Experiment / change | Fixes | Cost | Priority |
|---|---|---|---|---|
| 1 | C + PCD (uniform) and R + random + PCD on 4 datasets × 2 models | W1 | ~15–20 GPU-h | P1, run first |
| 2 | Random at input level; FastVID/SemVID with all timestamps; (opt.) input pruners' sets as KV eviction | W2 | ~20 GPU-h | P1 |
| 3 | Question global top-k (no quota), SnapKV, PyramidKV, matched-token low-res input, ± PCD | W3 | ~70 GPU-h | P1 |
| 4 | Rewrite abstract/teaser/contributions around standard-prompt results; timeline only for S-EMBER | W4 | 0 | P1 |
| 5 | Held-out S-EMBER videos with frozen config; closed-threshold sensitivity; ActivityNet at 10% with rule-chosen F | W5 | ~10–20 GPU-h + CPU | P1/P2 |
| 6 | Batch-matched unpruned reruns; batch-noise floor; clustered bootstrap; Holm on primary endpoints; TOST for "matches" | W6 | ~10 GPU-h + CPU | P1 |
| 7 | Cost table (TTFT, latency, peak memory, FLOPs, KV size) at bs 1/8 | W7 | ~4 GPU-h | P1 |
| 8 | Full TimeLens-Bench: Coverage, Random, SnapKV, TL low-res, both models | W8 | ~360 GPU-h (~2 days on 8 GPUs) | P1 for spotlight |
| 9 | Second model family (no textual timestamps) on S-EMBER + TimeLens-QVH | W8 | ~40 GPU-h + engineering | P2 (needed for spotlight) |
| 10 | Judge validation (~200 answers) + judge TimeLens-8B answers | W8 | <5 GPU-h | P1 |
| 11 | Re-run Tab. pcd8 / cf with final config; length rescaling cross-validated | W9, W10 | ~30 GPU-h | P2 |

**What would make it spotlight-level.** The diagnostic insight (coverage loss plus a video-free prior, measured against floors) is the paper's most general and memorable element. It is already near the spotlight bar in rigor. The method evidence is not. A spotlight version would have:
1. One main table on the **full standard splits** (TimeLens-Bench ×3 under its protocol, NExT-GQA full, S-EMBER held-out), on **two model families**. Coverage or CoCo would beat random-KV, random-input, SnapKV, question global top-k, PruneVid and matched low-res input, at **matched visual tokens with the cost columns shown**.
2. A clean decomposition showing that each of Coverage and PCD contributes on its own, with retrieval held fixed (experiment 1).
3. The cross-regime claim either confirmed by the input-level random control and the timestamp-restoration control, or restated as a KV-level result (experiment 2).
4. Statistics that a skeptic cannot dismiss: clustered intervals, Holm-corrected primary endpoints, batch-matched references, equivalence tests for "lossless".

Experiments 1–7 and 10 total roughly 150 GPU-hours, about one week on the 8-GPU cap. With experiment 8 added, the total is about 500 GPU-hours, about 2.5–3 weeks. If experiment 1 shows that Coverage adds nothing under PCD, the right spotlight story is "diagnosis + PCD + floors", with Coverage presented as the standard-prompt result. That story is still strong, but the paper should be reframed before the large sweep is spent.
