# Peer Review Report

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (CoCo)
- **Manuscript ID**: anonymous CVPR 2026 submission (draft at commit f839c4d, branch worktree-paper-draft)
- **Review Date**: 2026-10-15
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
I work on ML evaluation methodology and efficient inference, and specialise in experimental design for training-free methods: how faithfully baselines are re-implemented, ablations, multiple-comparison control, paired bootstrap, and seed and decoding noise. (Configuration Card #2.)

### Review Focus
1. Are the comparisons like for like in prompt, parser, frames, budget accounting and compute?
2. Does the "design on S-EMBER, then run once frozen" protocol hold, and is the significance reporting adequate now that the CIs have been removed?
3. Does the missing cost analysis undermine the efficiency framing?

I also judge the results obtained after the draft (recent_results.md) for their bearing on validity.

### Calibration Status
`NOT_CALIBRATED`

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**. Substantial new experiments are required, and a re-review is needed. In CVPR's single-round terms this is a reject as submitted, but every blocking item can be fixed with experiments the authors have already shown they can run.

### Confidence Score
4. Experimental design, statistics and baseline fidelity are my core expertise. I am less placed to judge the grounding-specialist literature.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment
The paper diagnoses why visual-token pruning hurts temporal grounding in Qwen3-VL-8B: pruners lose timeline coverage, and a video-free decoder prior fills the gaps. It then proposes CoCo, which combines per-frame equal-quota, question-attention KV selection (Coverage) with prior-contrastive decoding of time values (PCD). Results are reported against video-free floors on S-EMBER, QVHighlights, ActivityNet Captions and Charades-STA, using 475–500-question subsets.

The evaluation design is above the norm for this subfield. It includes test-tuned constant floors, a length-rescaling floor, a causal frame-removal test with a matched control, gold-informed upper bounds, offline decision-rule replay and a candid negative-results table. However, the central comparisons are not yet like for like.

The most serious problem is the evaluation prompt. Every main-table number uses a "timeline" prompt that the model handles badly. The authors' own later runs on the same QVHighlights subset and inputs show that Coverage scores 30.7 mIoU under the timeline prompt and 61.7 under the standard one-line prompt. Under the standard prompt PCD has no effect. Under the standard TimeLens protocol the uncompressed model also clears the "floors" that the paper says it cannot. The headline "nearly doubles the uncompressed model" and the "base-model floor" limitation are therefore properties of the protocol, not of the model.

Further problems:
- The pruner comparison mixes input-level pruning with post-prefill KV pruning.
- There is no cost or latency analysis.
- The "frozen, run once" claim is contradicted by visible iteration on the transfer datasets.
- Batch-size nondeterminism is uncontrolled.
- Removing the CIs hides boundary cases among roughly 60 uncorrected tests.

I recommend Major Revision: rerun under the standard protocol, add matched-level baselines and a cost table, and restore the interval reporting.

---

## Strengths

### S1: Floor-adjusted evaluation is a real methodological contribution
For every method and dataset the paper reports three references: a best constant interval tuned on the test questions (an upper bound on constant rules), a length-rescaling floor, and a closed/open split. Gains are then reported against those references as paired differences. This turns "beats the baseline" into "beats a video-free rule", and the paper reports honestly where CoCo loses (ActivityNet −3.1*, Charades −6.4* against the constant). I recomputed every cell of tab:paired from tab:main (for example 48.8 − 29.1 = 19.7 and 23.3 − 29.7 = −6.4). All agree within rounding.
**Evidence Anchor**: `table: tab:floors and tab:paired — Best constant / Length rescaling rows`

### S2: Causal intervention with a matched control
The leak hypothesis is tested by removing the frames inside the gold interval, against removing the same number of frames elsewhere. Every other frame keeps its true timestamp, and the test is run on two memories. This is a proper interventional design rather than a correlation, and the leak denominator is stated (397 questions; 397 + 78 early-evidence = 475, which is consistent).
**Evidence Anchor**: `table: tab:causal — control vs evidence rows, leak 31% → 46% (random 10%)`

### S3: Gold-informed upper bounds and offline replay bound the method
The oracle selector, gold-window zoom and oracle window (tab:oracle) separate what "which tokens" can deliver from what decoding can deliver. The replay of decision rules at the first time value, including a threshold tuned on gold labels (tab:replay), shows PCD near the trade-off frontier. These are informative ceiling analyses that few pruning papers provide.
**Evidence Anchor**: `table: tab:replay — "Best threshold (uses gold)" 46% / 6% vs PCD adaptive 51% / 7%`

### S4: Negative results and seed variance are reported
The paper lists 11 ideas that did not work, with paired deltas. It reports three random-selection seeds (26.0 ± 0.6) and states the datasets where the method fails (Charades below Random + PCD, −2.9*). This reduces the risk of selective reporting.
**Evidence Anchor**: `table: tab:negative — 11 rows incl. "128 frames at the same budget" −3.6*`

### S5: Parser change is applied uniformly and disclosed
The minutes:seconds conversion is applied to every method, and the official-parser numbers for the headline cells are given (QVHighlights 44.6 / 50.2 vs 48.8 / 54.8).
**Evidence Anchor**: `text: X_suppl §A "We convert h:mm:ss and m:ss to seconds for every method alike."`

### S6: Configuration is reproducible from stated overrides
The supplement lists the exact configuration overrides for the main method.
**Evidence Anchor**: `text: X_suppl §Reproduction "corresponds to the following overrides of the HERMES code base"`

---

## Weaknesses

### W1: The headline gains and the "base-model floor" limitation are artefacts of the evaluation prompt
**Problem**: All main results use a bespoke "timeline" prompt (Seen: lines, Answer, Time) for every method (§5 "All methods share the timeline prompt"). The results obtained after the draft (recent_results.md §3c, phase 38) show:
- On the same 513-query QVHighlights subset with identical inputs, Coverage scores 30.7 mIoU under the timeline prompt and 61.7 under the one-line TimeLens prompt. Timeline + PCD + reserve against the TimeLens prompt is a paired −17.8*.
- Under the TimeLens prompt, PCD on Coverage changes mIoU by −0.1 / −0.3 (n.s.).
- Under the standard TimeLens script, the unpruned Qwen3-VL-8B reaches 48.5 mIoU on Charades-STA and 47.1 on ActivityNet (§3a, matching the published 48.3 / 46.8). These are far above the paper's floors (29.7 / 37.9) and far above the paper's "all tokens" rows (16.6 / 22.9).

So the paper's +10–13 PCD gain largely repairs a format the model mishandles. "Nearly doubles the uncompressed model" (29.1 → 48.8) compares against a crippled reference. The limitation that the base model "stays below the video-free floor" on ActivityNet and Charades is a property of the protocol (prompt, 64 frames, low-resolution re-encodes), not of the model. There is one caveat. The §3a/§3c full-set anchors also differ in fps and resolution (13,650 vs ~5,200 tokens per video), so only the phase-38 paired row isolates the prompt. That row alone is large enough to change the reading of tab:main.

**Evidence Anchor**: `text: sec/6_results.tex §6.1 "On the least biased benchmark, \method nearly doubles the uncompressed model."`

**Why it matters**: The paper's two lead claims both depend on the prompt: CoCo beats the uncompressed model, and it clears the floors where the base model does. A reader cannot tell how much of the 19.7-point QVHighlights gain survives under the protocol the community uses. The diagnosis in §3.2 is also prompt-dependent: 89% of first Seen: moments at 0 s describes the timeline format specifically.

**Suggestion**:
1. Make the standard one-line grounding prompt and the TimeLens-Bench protocol (full splits, published resolution and fps) the primary evaluation. Anchor the unpruned rows to published numbers; the authors already reproduce them to within 0.3 mIoU.
2. Report Coverage, the pruners and random selection at matched token counts under that protocol.
3. Report PCD under both prompts, and state plainly that PCD requires a reasoning-style answer. §6.4 "Prompt format" gives +0.4 under the direct-answer prompt; this belongs in the abstract and method, not one paragraph.
4. Retract or re-scope the "base-model floor" limitation.

**Severity**: Critical
**Confidence**: 4 (core expertise: evaluation protocol confounds). The evidence comes from the authors' own runs that postdate the draft; the full-set sweep is about 1.5% complete.

### W2: Pruner comparison mixes input-level pruning with post-prefill KV pruning
**Problem**: The baselines are implemented at different levels of the model:
- In the code (video_qa/hermes_vqa.py, lines 370–413), FastVID, FlashVID, VidCom² and SemVID prune at the input: "the memory holds only the kept / merged tokens".
- Random, HERMES, Stratified and Coverage prune the KV cache after the full 64-frame memory has been prefilled, with an independent set per layer.

KV entries kept after a full prefill have already attended to every dropped token in lower layers, and per-layer sets make the union of tokens seen across layers far larger than 10%. "Pruners fall below random selection" (Observation 1, fig:pruners) therefore conflates coverage with pruning level. The shared-set ablation (tab:shared) addresses per-layer diversity but not input vs post-prefill. Further gaps in the baselines:
- No question-aware KV baseline (SnapKV, PruneVid, global question top-k) is included; §5 still lists these as a TODO, and the "Question, global top-k, no quota" cell in tab:design is empty.
- FlashVID is reduced to its "vision-side stages".
- FastVID uses the authors' Qwen2.5-VL hyperparameters unchanged on Qwen3-VL (inference/fastvid.py header).
- No re-implementation is sanity-checked against published numbers. Equivalence of kept tokens on test inputs is checked, but not the reported MCQ accuracy on the original benchmarks.

**Evidence Anchor**: `text: sec/5_setup.tex Baselines "re-implemented and checked to keep the same tokens as the authors' code on test inputs"`

**Why it matters**: The diagnosis that QA pruners lose the timeline is a core contribution, and so is CoCo's margin over SemVID + PCD (+4.5*, +14.1*). Both could be partly a comparison between a method that sees the full contextualised memory and one that never does.

**Suggestion**:
1. Run random selection at the input level (drop tokens before the LLM, same token count, timestamps handled identically) and compare the four pruners against it.
2. Run the selectors that do not merge tokens at the KV level too, giving a 2 × 2 of selector × level.
3. Add SnapKV-style observation-window eviction and question top-k without quotas at the same post-prefill level as Coverage. This is the direct test of whether the equal quota matters.
4. Reproduce one published MCQ number per pruner (for example VideoMME at the paper's retention ratio) to show fidelity.
5. State how timestamps are counted for the input-level pruners.

**Severity**: Major
**Confidence**: 4 (core expertise: baseline fidelity). Verified in the code path.

### W3: No cost, latency or memory analysis for a method framed by efficiency
**Problem**: The paper motivates pruning by cost ("video LLMs pay for every frame", §1) and titles the method "under a Visual-Token Budget". Yet the cost analysis is a TODO, and the paper concedes that CoCo has the uncompressed model's peak memory and time-to-first-token:
- Coverage needs a full prefill plus "3–4 scoring passes" (§4.1 TODO).
- PCD adds a second, prompt-only decoding pass for time tokens.
- Retrieval adds a SigLIP-2 1 fps index.

The input-level baselines, by contrast, reduce prefill cost. CoCo therefore compares at matched decoder-visible tokens but unmatched compute.

**Evidence Anchor**: `text: sec/7_conclusion.tex Limitations "the full memory is prefilled once per question, so peak memory and time-to-first-token are those of the uncompressed model"`

**Why it matters**: Without measured costs, a reader cannot place CoCo on a quality–cost frontier against input-level pruners or against simply using all tokens with PCD. tab:oracle shows all tokens + PCD at 32.7, comparable to many compressed rows. Whether "compression" buys anything depends on numbers the paper does not give.

**Suggestion**: Add a table with prefill FLOPs, time-to-first-token, decode latency per answer, and peak and steady-state KV memory for each row of tab:main, measured on one GPU type with the batch size stated. Include the one-pass Coverage implementation the authors say is possible. Plot mIoU against latency and against memory. If the intended benefit is the decoder-read memory (long multi-turn sessions), state that scenario and measure the amortised cost over k questions per video.

**Severity**: Major
**Confidence**: 5 (core expertise: efficient inference evaluation).

### W4: The "design on S-EMBER, run once frozen" protocol is contradicted by iteration on the transfer datasets
**Problem**: The paper claims that "All design choices ... were made on S-EMBER. The configuration was then frozen and run once on QVHighlights, ActivityNet and Charades-STA." The manuscript itself shows otherwise:
- tab:reserve reports QVHighlights and Charades runs with and without the answer reserve.
- The m:ss parser was introduced after seeing long-video outputs (X_suppl §A).
- ActivityNet is reported at 30% "because at 10% the memory holds only 120–300 tokens", a choice made after the 10% collapse.
- A retrieval "coverage gate" is applied only on Charades.
- The frames rule (halve F below about 8 tokens per pair) "was derived from the results of" tab:frames, which are QVHighlights and ActivityNet results (§4 TODO, §7).

S-EMBER, the dataset all choices were tuned on, also supplies half the headline claims, and no held-out split exists.

**Evidence Anchor**: `text: sec/5_setup.tex Protocol "The configuration was then frozen and run once on QVHighlights, ActivityNet and Charades-STA."`

**Why it matters**: Test-set iteration inflates the transfer results, which are the paper's main evidence of generality. The S-EMBER numbers are in-sample by construction.

**Suggestion**:
1. Evaluate on a held-out set of S-EMBER videos (the TODO).
2. Report the first-run numbers on each transfer dataset alongside the revised ones, and log every configuration change with its date and trigger.
3. Pre-specify the budget and frame rule and validate them on a dataset not used to derive them (for example the TimeLens-Bench full sets).
4. Rewrite the protocol paragraph to describe the actual sequence.

**Severity**: Major
**Confidence**: 4 (core expertise: experimental protocol).

### W5: Batch-size and decoding nondeterminism are not controlled
**Problem**: The paper states that "batch size changes no paired result". However:
- Table runs mix batch size 1 (early phases, the Charades unpruned run "unpruned-bs1" in analysis/floors.py) with batch size 8.
- The recent runs mix bs 1/2 (unpruned) with bs 8 (pruned).
- The authors' own measurements elsewhere put the effect at about half of outputs changed and roughly ±2 points of accuracy.

The manuscript shows the noise directly. The same configuration (random 10% + PCD, no reserve) scores 32.6 mIoU in tab:pcd8 and 32.1 in tab:cf and tab:scope. The same ΔGQ@0.5 of +2.9 is starred in tab:cf but not in tab:pcd8. The bootstrap (report_all.py `delta`: iid over questions, fixed seed, percentile) captures question-sampling variance only, not run-to-run variance.

**Evidence Anchor**: `table: tab:pcd8 vs tab:cf — Random 10% PCD mIoU 32.6 vs 32.1; ΔGQ +2.9 unstarred vs +2.9*`

**Why it matters**: Many design claims rest on deltas of 0.5 to 1.5 mIoU: unequal quotas within ±0.5, retrieval +0.8 to 0.9, shared sets −1.5*. These are at or below the run-to-run noise, and their significance stars are not reliable.

**Suggestion**: Rerun every compared row at one batch size (bs 1, or a fixed deterministic kernel setting). Alternatively, estimate replicate variance from three reruns of two reference configurations, and use a test that includes it, for example a bootstrap over questions combined with a random effect for run. Report which table rows share a batch size.

**Severity**: Major
**Confidence**: 4 (core expertise: decoding noise in LLM evaluation).

### W6: Significance reporting is weakened by removing the CIs, by many uncorrected tests and by ignoring video clustering
**Problem**: Commit f839c4d removed all bootstrap intervals and kept only stars. Some of the hidden intervals matter:
- The shared-set ablation was −1.5 [−2.9, −0.0]*, a boundary case that now reads as a clean result.
- The closed-interval headline was +7.6 [2.5, 12.6], on about 157 questions.

There are about 60 paired tests across the tables, with no multiplicity control. S-EMBER has 475 questions on 300 videos, and QVHighlights queries share videos, but the bootstrap resamples questions iid; a video-clustered bootstrap is a TODO. The introduction still promises "paired confidence intervals", which the text no longer gives. Small subgroups (78 early-evidence questions, 100 location questions, the ~157 closed questions) carry strong claims without power considerations. The S-EMBER "All tokens" cell in tab:paired carries "\TODO{sig.}".

**Evidence Anchor**: `text: sec/1_intro.tex contribution 3 "closed versus open intervals, with paired confidence intervals."`

**Why it matters**: Without the intervals, readers cannot see how close several claims are to zero, and the family-wise error over about 60 tests makes some stars likely false positives. Video clustering will widen every interval.

**Suggestion**: Restore the 95% intervals, at least in a supplementary table with one row per starred claim. Apply Holm correction within each table family. Switch to a video-clustered bootstrap. For subgroup claims (tab:scope bottom, tab:types, tab:closed), report the interval and n next to the estimate. Fill the missing S-EMBER significance cell.

**Severity**: Major
**Confidence**: 5 (core expertise: paired bootstrap and multiple comparisons).

### W7: External validity is narrower than claimed (one base model, non-standard subsets, NExT-GQA null)
**Problem**: The draft evaluates one model on 475–500-question subsets with low-resolution re-encodes (153 and about 84 tokens per frame pair). The collapse thresholds in tab:budget and tab:frames are partly products of these re-encodes. The recent results cut both ways:
- In support: TimeLens-8B on S-EMBER reproduces the ordering (CoCo 43.9 vs unpruned 33.1; §2), and Coverage beats Random by +2.0* to +3.5* on the QVHighlights TimeLens subset (§3c).
- The TimeLens-8B S-EMBER answers are unjudged, and the unpruned run used bs 1 against bs 8 for the pruned runs.
- On NExT-GQA (~40 s videos, 1,005 questions; §1), Coverage vs Random at 10% is a wash on grounding (mIoU −0.5 and +0.1, n.s.). Both 10% methods regress about 3 points against unpruned on Acc&GQA, which is near the batch-size noise level.
- PCD has not been run on NExT-GQA. Its two-query grounding answer is one-line, where W1 predicts no effect.

The paper's limitation "short, densely sampled videos" points at this, but the boundary is broader than stated.

**Evidence Anchor**: `text: sec/7_conclusion.tex Scope "All results use one model, 475--500-question subsets"`

**Why it matters**: The generality of Coverage, and especially of PCD, is not established beyond one model, one prompt family and long videos.

**Suggestion**: Add TimeLens-8B (with judged answers) and NExT-GQA (two-query protocol, stated) to the paper. Run PCD there and report the null if it is null. Use full splits at native resolution for at least QVHighlights and Charades-STA. State the video-length and tokens-per-frame regime in which each component helps.

**Severity**: Major
**Confidence**: 4 (adjacent expertise: grounding benchmarks; core: external-validity design).

### W8: The length-rescaling floor is asymmetric with the constant floor
**Problem**: The constant floor is re-tuned on each dataset's test questions. The rescaling factor (1.76) is fitted on S-EMBER only and applied unchanged elsewhere. "PCD moves the interval toward the evidence, not just outward" (QVHighlights +8.6*) is therefore tested against a floor that is not tuned on QVHighlights.
**Evidence Anchor**: `text: sec/5_setup.tex Floor-adjusted evaluation "a factor fitted once on S-EMBER (1.76) and applied unchanged to the other datasets"`
**Why it matters**: A per-dataset oracle factor (and a rescale-plus-shift variant) is the fair upper bound for "longer intervals alone". It may narrow the +8.6 margin that supports the mechanism claim.
**Suggestion**: Report per-dataset oracle-tuned rescaling, and rescale plus shift, alongside the fixed factor. This is a CPU-only analysis on saved outputs.
**Severity**: Minor
**Confidence**: 4 (core expertise: baseline design).

### W9: The main CoCo rows include retrieval and the PCD baselines do not
**Problem**: CoCo rows include SigLIP-2 retrieval frames. Random + PCD and SemVID + PCD use uniform frames. Retrieval gives random tokens +3.8*, so the paired gains in tab:paired (for example +4.2* over Random + PCD on S-EMBER) partly measure retrieval. The paper acknowledges this in a TODO ("Coverage + PCD without retrieval on all datasets").
**Evidence Anchor**: `table: tab:ladder — "R, random tokens" 29.4 vs "Uniform, random tokens" 25.6`
**Why it matters**: The attribution to Coverage is weaker than the table suggests.
**Suggestion**: Report Coverage + PCD on uniform frames as the primary row, or give the baselines the same retrieval frames.
**Severity**: Minor
**Confidence**: 5 (direct reading of the tables).

### W10: Analysis code does not match the reproducibility statement
**Problem**: X_suppl says "Floors and paired intervals are computed by docs/paper/analysis/floors.py". That script computes the constant floors, the closed split and end-snapping only. It contains no bootstrap and no length rescaling. The bootstrap lives outside the repository (hermes_pool/reports/report_all.py, nextgqa/paired.py) with hard-coded absolute paths. The judge is a local model named "Qwen3.8-27B-FP8", which I could not identify as a public checkpoint.
**Evidence Anchor**: `text: X_suppl §Reproduction "Floors and paired intervals are computed by \texttt{docs/paper/analysis/floors.py}."`
**Why it matters**: The paired claims cannot be reproduced from the released artefacts, and the accuracy and GQ@0.5 numbers depend on an unidentified judge.
**Suggestion**: Move the bootstrap and rescaling code into docs/paper/analysis with relative paths. Release per-question predictions for every table row. Name the judge checkpoint exactly, or use a public judge and report agreement with the official one on a subset.
**Severity**: Minor
**Confidence**: 5 (verified by reading the scripts).

---

## Detailed Comments

### Title & Abstract
- "Under a Visual-Token Budget" invites an efficiency reading that the paper does not measure (W3).
- The abstract's "from 24.6 to 40.8 on S-EMBER grounded QA" uses R@0.5, not a QA metric; label it.
- The abstract should state that PCD needs a reasoning-style (timeline) answer (W1).

### Introduction
- Observation 1 depends on the input vs KV confound (W2).
- Contribution 3 promises CIs that are no longer reported (W6).
- "Same holds on QVHighlights" omits that SemVID is not significant there (the figure caption says so).

### Methodology / Research Design
- **Design type.** A paired, within-question comparison of inference-time methods on fixed question sets. The design is appropriate, but internal validity depends on holding the prompt, batch size, frames and pruning level fixed across rows. W1, W2, W5 and W9 each break one of these.
- **Hyperparameter selection.** α_max = 1, β = 0.1, R = 64, the 8-token rule and the 1 s retrieval gate were selected on test data (W4). "Pre-registered" (§4 and §6.3) is asserted without a registration record.
- **Budget accounting.** Timestamps are counted inside the budget for KV methods. It is not stated how they are handled for the input-level pruners.

### Results / Findings
- **Internal arithmetic.** I checked all 20 cells of tab:paired against tab:main and the means in tab:pcd8 (greedy 24.0, PCD 30.7, Δ 6.65 → 6.7). All are consistent within ±0.1. I found two unexplained discrepancies between runs of the same configuration (W5).
- **Incomplete main table.** FastVID and VidCom² are missing on ActivityNet, and FlashVID's ActivityNet cell is at 10% inside a 30% column.
- **"PCD helps every pruner (+7 to +12)".** This is shown only for FastVID and SemVID.

### Discussion / Limitations
The limitations section is unusually candid (no streaming, one model, held-out split pending). Its "Base-model floors" paragraph, however, should be revised in light of W1.

### Conclusion
"clears video-free floors wherever the base model does" needs re-evaluation under the standard protocol, where the base model clears all of them.

---

## Methodological Fallacies Detected
- **Confounding (protocol).** Prompt format confounded with method effect (W1). Pruning level confounded with selection rule (W2).
- **Overfitting / test-set reuse.** Design and budget choices made on evaluation data (W4).
- **Uncorrected multiple comparisons.** About 60 starred tests (W6).
- **Non-independence.** Questions clustered within videos (W6).
- **Selective reporting risk.** Low: negative results are reported. The NExT-GQA null and the PCD null under the TimeLens prompt must be added to keep it low.

---

## Statistical Reporting Judgement (Step 4a)
- **Effect sizes.** Paired mIoU and R@0.5 differences are natural effect sizes and are reported: MEETS.
- **Confidence intervals.** Removed from the text: DOES_NOT_MEET.
- **Power and sample size.** No consideration for subgroups of 78–157 questions: PARTLY_MEETS.
- **Assumptions.** The iid-question bootstrap ignores video clustering and run-to-run variance: PARTLY_MEETS.
- **Missing data.** Unparseable answers are scored IoU 0 and counted in tab:reserve and tab:errors: MEETS.
- **Red flags.** Uncorrected multiplicity and test-set tuning.

**Arithmetic recompute.** In standard mode I record no_recomputable_statistics: the manuscript reports no t, F, χ² or z statistics, no degrees of freedom, and no Likert-scale means or SDs. All inference is by percentile paired bootstrap, which the bounded procedures (p_from_test_statistic, grim, grimmer, n_from_df) do not cover. Internal consistency checks were done in prose above, and one arithmetic slip is listed under Minor Issues.

---

## Questions for Authors
1. Under the standard one-line grounding prompt on the full QVHighlights, Charades-STA and ActivityNet splits, what are the unpruned, Random, best pruner, Coverage and CoCo scores at matched kept tokens? Does any PCD gain remain?
2. Are Random, HERMES and Stratified pruned after a full prefill (KV-level) while FastVID, VidCom², FlashVID and SemVID prune at the input? If so, what does input-level random selection score?
3. Which configuration changes were made after the first runs on QVHighlights, ActivityNet and Charades-STA (reserve, parser, gate, budget, frame rule), and what were the first-run numbers?
4. What is the measured time-to-first-token, decode latency and peak memory of CoCo against all tokens and against SemVID at 10%? Where is the "pre-registered" test recorded?

---

## Minor Issues

### Language / Grammar
- §1: "about 330 visual tokens per pair of frames, so 64 frames already cost about 21,000 tokens". 32 pairs × 330 ≈ 10,600; §5 states 660 per frame pair on S-EMBER. Fix one of the two.
- §6.1 (ii): "two thirds of S-EMBER intervals end at the question time" overstates §3.4, which says "within the last 20% of the window".
- §5: the judge model name "Qwen3.8-27B-FP8" looks like a typo.

### Figures and Tables
- tab:main: mark the ActivityNet column as incomplete, or fill FastVID and VidCom².
- tab:paired: the S-EMBER "All tokens" significance is still a TODO.
- tab:design: the empty "Question, global top-k, no quota" cell is exactly the ablation that isolates the quota (W2).
- tab:retention vs tab:main: Random is 25.8 vs 25.6 because of different runs and batch sizes. State this in the caption.

### Layout
- Several TODO macros remain in the compiled text (§4.1 twice, §4.2, §5 three times, §6.1, §6.3, §6.4 twice, X_suppl twice).

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`. There are no bound venue criteria (criteria_binding_unavailable), so the criterion source is the methodology-reviewer rubric (role file Steps 1–6) applied to CVPR field-general expectations.

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Outside methodology remit | NOT_ASSESSED | — | Reviewed by the EIC and R2 | — | no |
| Methodological Rigor | Role file Steps 2, 4, 5 | PARTLY_MEETS | tab:floors; tab:causal; sec/5_setup.tex Baselines; hermes_vqa.py lines 370–413 | Strong floors and causal and oracle designs, undermined by prompt and pruning-level confounds and test-set iteration | The full-set standard-protocol sweep is about 1.5% done | yes: W1, W2 and W4 must be resolved |
| Evidence Sufficiency | Role file Steps 3, 4a | PARTLY_MEETS | tab:paired; recent_results §1–3 | One model, 500-question subsets, CIs removed, no cost data, NExT-GQA null not reported | TimeLens-8B results are partial and unjudged | yes: W3, W6 and W7 |
| Argument Coherence | Role file Step 5 | PARTLY_MEETS | sec/6_results.tex §6.1; sec/7_conclusion.tex Limitations | Claims are internally consistent but rest on a protocol the authors' later data contradict | — | yes, through W1 |
| Writing Quality | Outside main remit | NOT_ASSESSED | — | Only methodology-relevant wording is noted under Minor Issues | — | no |
| Literature Integration | Outside methodology remit | NOT_ASSESSED | — | Reviewed by R2 | — | no |
| Significance & Impact | Role file Step 5 (claims vs data) | PARTLY_MEETS | tab:main; tab:oracle | The diagnosis (coverage, prior, floors) is useful. The size of the method's gain under standard protocols is unknown | Depends on the reruns in W1 | yes |

**Recommendation rationale**: Major Revision. The unresolved decision-bearing items are:
- W1 (Critical, repairable): rerun under the standard prompt and protocol on full splits.
- W2 (Major): matched-level baselines, including input-level random and a SnapKV-style KV baseline.
- W3 (Major): a cost table.
- W4 (Major): held-out validation and an honest protocol history.
- W5 and W6 (Major): controlled batch size, restored and corrected intervals, clustered bootstrap.

The strengths in floor design and diagnostics are real, but they do not offset the confounds in the headline comparisons.
