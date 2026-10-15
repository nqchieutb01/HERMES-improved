# Peer Review Report — Journal-Fit Reviewer (EIC seat), round 2 (spotlight bar)

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (CoCo)
- **Manuscript ID**: CVPR 2026 draft, worktree `paper-draft`, commit 2c84a78
- **Review Date**: 2026-10-15
- **Review Round**: Round 2 (author question: what is needed to reach CVPR spotlight level)

criteria_binding_unavailable

No #684 criteria manifest was supplied. Venue comments below use the shared "Spotlight bar" in field_analysis.md and general CVPR norms (8-page body limit, double-blind). I make no claim of alignment with formal venue criteria.

---

## Reviewer Information

### Reviewer Role
Journal-Fit Reviewer (EIC seat)

### Reviewer Identity
CVPR area chair for video understanding and multimodal LLMs who has served on award and oral-selection committees (Card #1). Focus: the one insight the paper will be remembered for, whether the story is tight enough for an oral or spotlight, what to cut to reach 8 pages, and the framing of the title, teaser and contributions. Blind spot: statistical details. Bootstrap, batch-size noise and parser issues are left to the methodology seat unless they change the framing.

### Review Focus
Is there one crisp, general and surprising insight, and do the title, abstract, teaser and main table deliver it under standard protocols? What would an oral or spotlight committee see in the first two pages and the main table, and what has to change for that to land in the top 5–10% of accepted papers?

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**. For CVPR this means: not acceptable as it stands (desk-reject risk on length and anonymity, many TODOs). After the cuts and reframing below it would be a borderline-to-accept poster. It is not yet at spotlight level.

### Confidence Score
4. Mostly within my area (video LLMs, grounding, efficient inference). I did not check the statistics in detail.

Confidence is an uncertainty and scope disclosure only. It does not change consensus counts, severity, decision bearing or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper asks what visual-token pruning does to temporal grounding in video LLMs. It answers with three observations: QA pruners fall below random token selection because they empty frames and so lose timestamps; where the memory is silent, a video-free "start early, stop soon" prior writes the interval; and positional floors on grounding benchmarks are high. It then proposes CoCo, which has two parts. Coverage gives every frame an equal quota and fills it by the actual question's per-layer attention. PCD contrasts time-value tokens against a no-video pass and adds an answer reserve. The diagnosis is the strongest part of the paper. It is surprising, it is shown mechanistically (Tab. retention, Tab. causal), and it is useful beyond this method. Coverage is simple and transfers to a grounding-tuned model and to the standard one-line prompt.

The headline is the main problem. The abstract, teaser and main table use the authors' reasoning-style "timeline" prompt. Their own §6b shows that this prompt costs the uncompressed model about 30 mIoU. So the "nearly doubles the uncompressed model" message does not hold under the standard protocol, and PCD, half of the method's name, has no effect there. Three further problems block acceptance: the body runs to roughly 14 pages (about 20 tables), the paper is unfinished (19 `\TODO`s, empty cells, 15 placeholder references including the development benchmark), and there is no cost table for what is an efficiency paper. My recommendation is Major Revision. The core finding is worth publishing, but the story has to be rebuilt around the claims that survive the standard prompt.

---

## Strengths

### S1: A surprising, general and mechanistically explained diagnosis
All four recent training-free pruners score below random token selection on grounding at 10%, and the paper explains why: they empty frames, and a frame with no visual tokens loses its timestamp. Tab. retention shows the mechanism directly. HERMES keeps 27.6% of tokens in the last quarter of the window and 2.5–4.7% elsewhere, and FastVID keeps tokens from only 25% of frames. Ranking the same HERMES scores within each frame recovers +5.3* mIoU. A result like "published methods lose to random, and here is the one-line reason" is the kind of finding a spotlight can rest on.
**Evidence Anchor**: `table: Tab. retention (sec/3_diagnosis.tex) — HERMES 27.6% in 75–100% vs 2.5–4.7% elsewhere; stratified 23.7 vs global 18.3 mIoU`

### S2: A causal intervention, not only a correlation
Removing the frames inside the gold interval, against a control that removes the same number of frames elsewhere, raises the leak by 14 points for both the unpruned and the random-10% memory. Answers move to the start of the video rather than spreading out. This is the right experiment to tie the prior to missing evidence, and few pruning papers do anything like it.
**Evidence Anchor**: `table: Tab. causal — leak 31%→46% (random 10%) and 37%→51% (all tokens), evidence vs control`

### S3: Coverage transfers to the standard prompt and to a grounding-tuned model
Under the TimeLens-Bench one-line prompt, Coverage at about 24% of the visual tokens beats random selection on both Qwen3-VL-8B (+3.5*) and TimeLens-8B (+2.0*), beats SemVID (+2.9*), and matches the uncompressed model (61.7 vs 60.5; 67.3 vs 67.1). This is the most protocol-robust positive result in the paper.
**Evidence Anchor**: `table: Tab. standard — Coverage 61.7/66.7 (Qwen3-VL-8B) and 67.3/73.9 (TimeLens-8B) vs All tokens 60.5/64.9 and 67.1/73.1`

### S4: Unusual honesty, with floors and negative results reported
Every comparison is set against tuned video-free constants and a length-rescaling floor, the authors state where the method fails, and they include a full negative-results table. Area chairs reward this, and it will help in rebuttal.
**Evidence Anchor**: `text: §6.1 "We report these as limits, not successes."`

### S5: The pipeline reproduces published numbers
With the TimeLens script, both models reproduce published TimeLens-Bench numbers within 0.3 mIoU on all three full sets, and the authors' own pipeline matches the script on the same queries. This makes the standard-prompt results credible.
**Evidence Anchor**: `table: Tab. anchors (Appendix) — e.g. Qwen3-VL-8B QVHighlights 59.6 vs 59.4 published; TimeLens-8B ActivityNet 53.2 vs 53.2`

### S6: The full method holds up on grounded QA across two models
On S-EMBER grounded QA, CoCo at 10% exceeds the uncompressed model by +11.2* (Qwen3-VL-8B) and +10.7* (TimeLens-8B) mIoU, with the same ordering of methods on both models. This is the setting where the reasoning-style answer is natural, so it is the strongest case for the full CoCo.
**Evidence Anchor**: `table: Tab. tl8b — CoCo 43.9/46.9 vs All tokens 33.1/34.1 (TimeLens-8B)`

### S7: Training-free and drop-in
The method needs no training, no calibration data and no change to the model, which keeps adoption cheap.
**Evidence Anchor**: `text: §1 contribution 2 "It needs no training, no labelled calibration and no change to the model."`

---

## Weaknesses

### W1: The headline result is measured against a prompt-handicapped baseline
**Problem**: The teaser, the abstract's lead numbers (QVHighlights R@0.5 54.8 vs 28.6) and Tab. main all use the timeline prompt. §6b shows this prompt costs the uncompressed model about 30 mIoU on QVHighlights (60.5 → 30.4 on the 513-query subset). On that same subset, Coverage + PCD with the timeline prompt reaches 43.9 mIoU (Tab. prompt), about 17 points below the uncompressed model under the standard prompt. The authors acknowledge this in §6b and in the Limitations, but the teaser and the first lines of the abstract still lead with it.
**Evidence Anchor**: `text: Fig. 1 caption "nearly doubles the R@0.5 of the uncompressed model and is far above the best video-free constant"`
**Why it matters**: The first thing an AC or oral committee checks is whether the headline survives the standard protocol. Here it does not. A reader who finds §6b will discount the whole paper, and the honest caveats will not undo that first impression. The paper would mislead readers about the size of the gain.
**Suggestion**: Remove "×1.9" and the QVHighlights 54.8 vs 28.6 lead from the teaser and abstract. Make the main table the standard-prompt TimeLens-Bench comparison at a matched budget (both models, random, the four pruners, SemVID, Coverage). Present S-EMBER grounded QA, where a reasoning-style answer is natural, as the setting for the full CoCo. Keep the timeline-prompt QVHighlights/ActivityNet/Charades table only in the supplement, labelled as a prompt study.
**Severity**: Major
**Confidence**: 4 (core expertise: grounding benchmarks and protocols)

### W2: The body is about 14 pages against an 8-page limit
**Problem**: field_analysis.md reports that the compiled PDF is 18 pages including the appendix. I could not render it, so the body length is an estimate. In the 14 Oct build (16 pages), the bibliography is loaded at pages 13–14. The current PDF has 18 pages and adds §6b, so the body is about 14 pages, with about 20 tables and 5 figures. §6 alone has eleven subsections.
**Evidence Anchor**: `text: §6 opening "\cref{sec:negative} lists what did not work, and \cref{sec:errors} analyses the remaining errors."`
**Why it matters**: A CVPR submission longer than 8 pages is desk-rejected, so as it stands acceptance is impossible. This is a structural problem rather than a formatting one: what is in the main paper decides what the paper is about, and at present the strongest message is buried among about 20 tables of similar weight.
**Suggestion**: Use the 8-page outline in (d) below. Move the oracles, the retrieval analysis, the negative results, the error taxonomy, the decision-rule replay, the scope and adaptive-strength tables, the answer-reserve table, the question types, the closed intervals, end-snapping, parsing, the anchors, the prompt study and the reproduction details to the supplement.
**Severity**: Critical (repairable, by cutting and restructuring)
**Confidence**: 3 (page count estimated from the build log and appendix length, not measured on a rendered page)

### W3: Half of the method's name works only with a non-standard answer format
**Problem**: PCD helps only when the model writes several times before the interval. Under the one-line prompt it changes mIoU by −0.1 (n.s.). With the direct-answer prompt on S-EMBER it gives +0.4. On pooled S-EMBER, ActivityNet and Charades it cannot be separated from a calibrated lengthening. The title, the method name and the contribution list nevertheless present Coverage + PCD as one method for grounding.
**Evidence Anchor**: `text: §6b "under the one-line prompt it has no effect"`
**Why it matters**: A spotlight needs one method with a clear scope. As written, a reviewer can fairly say that PCD repairs a problem the timeline prompt creates. That makes the second half of the title ("Contrast the Prior") look prompt-specific and weakens the paper's identity.
**Suggestion**: Choose one of two framings and make the title, abstract and contributions consistent with it. (a) Coverage is the general method, and PCD is a decoding fix for reasoning-style grounded QA (S-EMBER, and possibly NExT-GQA in a single-reply format), shown as a plug-in on eight memories. (b) If the authors want to keep CoCo as a single method, they need to show PCD helping on a second grounded-QA benchmark where the reasoning-style answer is the natural format (see experiment E5).
**Severity**: Major
**Confidence**: 4 (core expertise: framing of method contributions)

### W4: The development benchmark is uncited, and all choices were tuned on it
**Problem**: Every design choice was made on S-EMBER. The held-out S-EMBER split is a TODO. The S-EMBER bibliography entry is a placeholder. The largest gains of the full method (+10.7*/+11.2* over the uncompressed model) are on S-EMBER only.
**Evidence Anchor**: `dataset: S-EMBER — main.bib entry has author "[authors to verify]" and note "[cite: S-EMBER benchmark paper]"; §5 states all design choices were made on S-EMBER`
**Why it matters**: The full-method story depends on a benchmark that readers cannot yet look up, on which the configuration was also selected. If S-EMBER is unpublished or is the authors' own, the community cannot calibrate the result, and that alone keeps a paper out of oral or spotlight consideration.
**Suggestion**: Cite S-EMBER properly and describe it in a short paragraph (source, annotation, licence, and why questions are asked mid-stream). Evaluate the held-out S-EMBER split with the frozen configuration. Add one public grounded-QA benchmark on which the full method is evaluated in the same way (NExT-GQA, or an Ego4D-NLQ-style egocentric set if one fits).
**Severity**: Major
**Confidence**: 3 (I cannot tell from the manuscript whether S-EMBER is public)

### W5: An efficiency paper with no cost numbers
**Problem**: Coverage scores tokens by running the question over the full uncompressed memory, and the current implementation uses 3–4 scoring passes. PCD adds a second decoding pass. The Limitations state that peak memory and time-to-first-token are those of the uncompressed model. No latency, FLOPs or memory table is reported.
**Evidence Anchor**: `text: §7 Limitations "peak memory and time-to-first-token are those of the uncompressed model"`
**Why it matters**: The obvious AC question is: if prefill cost is unchanged, why not use the uncompressed model with the standard prompt, which grounds better? Without a cost table that shows what is saved (decode-time KV reads, cache footprint across several questions, per-token decode latency), the motivation in the introduction ("video LLMs pay for every frame") is unsupported.
**Suggestion**: Add one compact table: prefill time, decode time per output token, peak memory and KV size for all tokens, random, SemVID, Coverage and Coverage + PCD at 10% and 25%, on one GPU. State plainly that savings are on the decoder side. If possible, show one setting where the saving compounds, such as several questions over one cached memory or long-horizon decoding.
**Severity**: Major
**Confidence**: 4 (core expertise: efficient video-LLM inference)

### W6: The general version of Observation 1 is untested under the standard prompt
**Problem**: The claim that four pruners fall below random is shown on S-EMBER, QVHighlights and Charades only with the timeline prompt. Under the standard one-line prompt, the only pruner tested is SemVID, at 31%, and there it is slightly above random (58.8 vs 58.2).
**Evidence Anchor**: `table: Tab. standard — SemVID (31%) 58.8/62.4 vs Random tokens (31%) 58.2/60.8`
**Why it matters**: Observation 1 is the paper's most memorable claim, and the spotlight case rests on it being general. If it holds only under a reasoning-style prompt, it becomes a curiosity rather than an insight.
**Suggestion**: Run FastVID, VidCom², FlashVID and SemVID at 10% (and 25%) against random and Coverage under the one-line TimeLens-Bench prompt on both models (experiment E2). Then report the figure in Fig. pruners under the standard protocol.
**Severity**: Major
**Confidence**: 4 (core expertise: pruning baselines)

### W7: The ablation that would isolate the named mechanism is an empty cell
**Problem**: The paper's explanation is that the equal quota keeps every timestamp. The cell that separates "the quota" from "the actual question's attention", namely question attention with global top-k and no quota, is a `\TODO{}` in Tab. design. The paper does have related evidence (HERMES scores stratified vs global, +5.3*), but that uses a different score.
**Evidence Anchor**: `table: Tab. design — row "Question, global top-k, no quota", mIoU cell is \TODO{}`
**Why it matters**: Without this cell, a reviewer can say that the gain comes from SnapKV-style question attention and that coverage is incidental. That would remove the link between the diagnosis and the method, which is the paper's main narrative.
**Suggestion**: Run the cell on S-EMBER and on the standard-prompt QVHighlights subset. Report it next to the frame-coverage statistic (the share of frames that keep at least one token) so the mechanism is visible.
**Severity**: Major
**Confidence**: 4 (core expertise: ablation design for token selection)

### W8: Observation 3 (high floors) is largely a product of the prompt
**Problem**: The introduction presents "the best constant beats the uncompressed model on ActivityNet and Charades" as a property of the benchmarks. Under the standard one-line prompt, the uncompressed model scores 47.1 and 48.5 mIoU, well above the constants (37.9 and 29.7). The floors therefore beat a handicapped prompt, not the model. S-EMBER's open-interval shortcut is a genuine benchmark property.
**Evidence Anchor**: `text: §1 Observation 3 "On ActivityNet Captions and Charades-STA the best constant also beats the uncompressed model."`
**Why it matters**: One of the three headline observations would be partly an artefact, and grounding specialists will notice immediately. That costs credibility for the two observations that are robust.
**Suggestion**: Scope Observation 3 to what holds. S-EMBER's open intervals make the constant [0.15q, q] beat the model, and floors should be reported for any grounding evaluation under compression. Drop the ActivityNet and Charades claim from the introduction, or qualify it explicitly as specific to the prompt. Keep the floors as an evaluation practice (one row in the main table) rather than as a headline observation.
**Severity**: Major
**Confidence**: 4 (core expertise: grounding benchmark bias)

### W9: The paper is not finished
**Problem**: There are 19 `\TODO` markers in the text, empty cells in the main table (FastVID and VidCom² on ActivityNet) and in the paired table (Charades rows; "+11.3 \TODO{sig.}" against "+11.2*" in Tab. tl8b), 15 placeholder bibliography entries, and several tables run before the answer reserve existed, mixed with tables run after it.
**Evidence Anchor**: `table: Tab. paired — "All tokens" row, S-EMBER cell "+11.3 \TODO{sig.}" and empty Charades cells`
**Why it matters**: A draft in this state would be scored down on presentation, and the inconsistencies (+11.3 vs +11.2*, PCD with and without the reserve) invite doubts about whether the numbers have been checked.
**Suggestion**: Freeze one configuration (with the reserve) and regenerate every main-paper table from it. Move pre-reserve tables to the supplement, labelled as such, or rerun them. Resolve every bibliography placeholder.
**Severity**: Major
**Confidence**: 5 (directly observed)

### W10: The teaser shows the result, not the insight
**Problem**: The left panel shows one example in which several methods miss and CoCo gets IoU 0.97. The right panel is a bar chart against the prompt-handicapped baseline. Neither panel shows the mechanism the paper should be remembered for: holes in the timeline left by QA pruners, and the prior filling them. Fig. overview (a) and (b) shows that idea far better than the teaser does.
**Evidence Anchor**: `figure: Fig. 1 — left: per-method interval bars on one QVHighlights query; right: R@0.5 bars incl. "×1.9 the uncompressed model"`
**Why it matters**: Spotlight selection depends heavily on whether a reader understands the idea from page 1. A single cherry-picked example plus a contested bar chart does not achieve that.
**Suggestion**: Redesign the teaser as three aligned strips over one timeline. (i) The tokens a QA pruner keeps per frame, with empty frames greyed out and their timestamps crossed out. (ii) Coverage's equal quota, with every timestamp kept. (iii) The no-video prior's start distribution (90% in the first 5%) against the predictions from the pruned memory. Add a small inset with the standard-protocol result: published pruners < random < Coverage ≈ uncompressed at 24% of tokens.
**Severity**: Minor
**Confidence**: 4 (experience on oral-selection committees)

### W11: Positioning undersells the paper's novelty
**Problem**: The introduction concedes that neither operator is new (per-frame budgets in VidCom²/KTV/AdaTP, question-attention eviction in SnapKV/PruneVid, no-input contrast in VCD/M3ID), and the contribution list then gives four items of similar weight. Read quickly, it looks like a recombination paper.
**Evidence Anchor**: `text: §1 "Neither operator is new in isolation"`
**Why it matters**: The novelty lies in the diagnosis: grounding needs coverage that QA does not, and published pruners lose to random for a reason one can name. The current ordering puts the method first in the reader's mind.
**Suggestion**: State the contribution as "a finding and its fix". Contribution 1 is the finding, with the mechanism and the causal test. Contribution 2 is the minimal fix, Coverage, which works under the standard protocol on two models. Contribution 3 is PCD, a scoped decoding fix for reasoning-style grounded QA. Fold the floor-adjusted evaluation into the experimental protocol rather than listing it as a contribution.
**Severity**: Minor
**Confidence**: 4 (core expertise: CVPR framing)

### W12: The appendix may break double-blind review
**Problem**: The reproduction section says that CoCo is a set of overrides of "the HERMES code base", and HERMES is also a cited baseline (and appears in Tab. retention and Tab. pcd8). If HERMES is the authors' prior work, this links the submission to them.
**Evidence Anchor**: `text: Appendix Reproduction "corresponds to the following overrides of the HERMES code base"`
**Why it matters**: Breaking anonymity is grounds for desk rejection at CVPR.
**Suggestion**: Remove the code-base name and the repository paths (`docs/paper/analysis/floors.py`) from the submission, and refer to HERMES in the third person throughout.
**Severity**: Minor (it becomes a desk-reject issue only if HERMES is the authors' own work)
**Confidence**: 3 (I do not know whether HERMES is the authors' prior work)

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Spotlight bar (1), field_analysis.md | MEETS | `table: Tab. retention`; `table: Tab. causal` | The "QA pruners lose to random on grounding because they empty frames" finding is new and general in form. The operators are not new, as the authors say. | Generality under the standard prompt is untested for pruners (W6) | yes. It carries the spotlight case if E2 confirms it |
| Methodological Rigor | Spotlight bar (3) | PARTLY_MEETS | `table: Tab. standard`; `table: Tab. design` | Strong floors and a causal test, but the main table uses a handicapped prompt, the key mechanism cell is empty, and there is no cost table | Statistical detail is left to the R1 seat | yes (W1, W5, W7) |
| Evidence Sufficiency | Spotlight bar (2) | PARTLY_MEETS | `table: Tab. tl8b`; `table: Tab. standard` | Two models. Under the standard prompt the gains are modest (+2–3.5 mIoU over random, parity with uncompressed). The large gains are on S-EMBER only | Full TimeLens-Bench sweep is in progress | yes |
| Argument Coherence | Card #1 | PARTLY_MEETS | `text: §6b "under the one-line prompt it has no effect"` | Title and abstract promise a two-part method whose second part depends on the prompt, and Observation 3 is partly a prompt artefact | — | yes (W3, W8) |
| Writing Quality | CVPR 8-page norm | DOES_NOT_MEET | `text: §6 opening "\cref{sec:negative} lists what did not work, and \cref{sec:errors} analyses the remaining errors."` | Roughly 14-page body, about 20 tables, 19 TODOs. The prose itself is clear and plain | Page count estimated | yes (W2, W9) |
| Literature Integration | Card #1 | MEETS | `text: §1 "Neither operator is new in isolation"` | Related work is accurate and candid, but 15 references are placeholders | Coverage of the grounding literature is left to the R2 seat | no (repairable) |
| Significance & Impact | Spotlight bar (1, 2) | PARTLY_MEETS | `table: Tab. standard`; `figure: Fig. 1` | Potentially high if the finding about pruners is general. As framed, the impact claim rests on a prompt-relative headline | Depends on E1/E2 outcomes | yes |

The recommendation rests on unresolved, decision-bearing items: W2 (length; repairable), W1/W3/W8 (framing; repairable by rewriting plus E1–E2), W5/W7 (missing evidence; repairable within one cycle), and W4 (benchmark provenance; repairability depends on S-EMBER's status). The strong originality judgement does not offset these.

---

## Detailed Comments

### Journal Fit
This is squarely a CVPR topic, sitting between video LLMs, efficient inference and temporal grounding, and it is timely now that token pruning for video LLMs is a fast-growing area. The paper's tone (floors, negative results, reproduction) suits a CVPR readership. In form it currently reads like a journal or TPAMI paper: exhaustive, with many tables of similar weight. CVPR rewards the opposite: one message, one main table, one figure that makes the point.

### Originality
The originality is in the diagnosis, not the operators. "Random beats every published pruner on grounding, because grounding needs a timestamp on every frame" is crisp and counter-intuitive, and it should change how pruners are evaluated. The paper should claim this as its main contribution.

### Significance
If E2 shows that published pruners also fall below random under the standard protocol on both models, the significance is broad: every pruning paper that reports only MCQ accuracy would need to add a grounding check. If E2 shows that they do not, the significance shrinks to "a fix for reasoning-style grounded QA under compression", which is a solid poster.

### Structural Coherence
The introduction promises three observations and a two-part method. The results then show that the method's second part, and one of the three observations, depend on the prompt. The Limitations section is candid about this, but the title, abstract, teaser and conclusion have not been updated to match. That gap between the honest body and the over-promising front matter is the biggest coherence problem.

### Title & Abstract
The title delivers both halves of the method, but "Contrast the Prior" is the half that does not survive the standard prompt. Consider "Keep the Clock: Why Token Pruning Breaks Temporal Grounding in Video LLMs, and a Training-Free Fix", or keep the current title only under framing (b) of W3. The abstract is long (about 340 words) and states two different benchmark stories (timeline prompt, then one-line prompt). It should lead with the finding about pruners, then give the standard-protocol Coverage result and the S-EMBER grounded-QA result for the full method, and drop the 54.8 vs 28.6 comparison.

### Conclusion
The conclusion is accurate and appropriately scoped. Its last sentence, on floors, should be scoped as described in W8.

---

## (a) Current standing

**Borderline, below acceptance as submitted; not spotlight.**
- *As it stands*: it would be desk-rejected for length (W2), and possibly for anonymity (W12). If it were reviewed anyway, the TODOs (W9) and the prompt-relative headline (W1) would put it at weak reject or borderline.
- *After the cuts and the reframing in (d), with the experiments already run*: a borderline-accept poster. The diagnosis, Coverage under the standard protocol on two models, and the S-EMBER full-method result add up to a coherent and honest paper.
- *Spotlight*: requires the general version of Observation 1 under the standard protocol (E2), a mechanism ablation (E3), a cost story (E4), and a teaser that conveys the insight. Experiment results cannot be predicted, so I cannot say whether spotlight is reachable. The finding is of the kind that gets spotlights, but at present it is only confirmed under a handicapped prompt.

## (b) Gap to spotlight, ranked by impact

1. **Headline under the standard protocol** (W1, W6). Replace the main table and teaser with a standard-prompt TimeLens-Bench comparison at a matched budget on both models, including all four pruners. This is the single biggest change, and without it nothing else matters.
2. **One insight, one method identity** (W3, W11, W8). Lead with the finding. Make Coverage the general fix and scope PCD to reasoning-style grounded QA, or show that it generalises beyond S-EMBER. Scope Observation 3.
3. **8 pages and a finished draft** (W2, W9, W12). These are mechanical changes, but they decide acceptance.
4. **A cost table** (W5). An efficiency paper without cost numbers will be asked about cost first.
5. **Mechanism ablation** (W7). This one cheap cell connects the diagnosis to the method.
6. **Benchmark provenance and a held-out split** (W4). The full-method result needs a citable benchmark and a held-out split, and ideally a second grounded-QA benchmark.
7. **Teaser redesign** (W10). This is needed for spotlight, though not for acceptance.

## (c) The one-sentence insight

As I read it: **"Pruning for QA keeps what is salient, but grounding needs what is everywhere: a QA pruner empties frames, deletes their timestamps and loses to random selection, and the model then fills the gaps from its own video-free prior."**

The first clause (coverage, losing to random) is the remembered part, and the clause on the prior explains *why the errors look the way they do* (early starts, short intervals). As far as the evidence goes, the first clause survives the standard prompt for Coverage, but for the pruners it has been checked under the standard prompt only for SemVID at 31% (E2 is needed). The clause on the prior is shown by the no-video and causal tests, but the fix for it (PCD) only helps reasoning-style answers. I would therefore keep the prior in the explanation and treat PCD as a scoped tool.

Do the front matter deliver it?
- **Title**: partly. "Keep the Clock" is memorable and exactly right. "Contrast the Prior" promises a component that does not generalise to the standard prompt.
- **Abstract**: partly. The first half states the insight well ("fall below random token selection ... removes whole stretches of the timeline"). The second half buries it under prompt-relative numbers and two benchmark stories.
- **Teaser**: no. It shows a winning example and a contested bar chart, not the mechanism (W10).

## (d) A concrete 8-page outline

| § | Content | Pages | Main-paper floats |
|---|---|---|---|
| Teaser | Redesigned: three timeline strips (QA pruner with empty frames and lost timestamps / Coverage / no-video prior vs pruned predictions) and a small inset with the standard-protocol result | (top of p.1) | Fig. 1 |
| 1 Intro | Paper-cutter hook. Finding 1 (pruners < random, because they empty frames). Finding 2 (prior fills gaps, causal). Fix. Three contributions as in W11 | 0.9 | — |
| 2 Related | Three paragraphs: token/KV pruning; contrastive decoding; training-free grounding and benchmark bias | 0.5 | — |
| 3 Diagnosis | Setting (0.2). Coverage loss (Tab. retention). Prior: no-video stats plus causal test (Tab. causal), with Fig. leak if space allows. Floors as a short evaluation-practice paragraph with numbers only | 1.3 | Tab. retention, Tab. causal, (Fig. leak) |
| 4 Method | Coverage (Eq. 1–2, why equal, per-layer, minimum tokens per frame rule). PCD (Eq. 3–4, no-video negative, time scope, adaptive α, reserve in two lines). Cost paragraph. Retrieval reduced to one sentence pointing to the supplement | 1.3 | Fig. overview |
| 5 Setup | Models, datasets (with a proper S-EMBER description), standard vs reasoning-style protocol, floors, statistics | 0.5 | — |
| 6.1 | **Main table A, standard protocol**: TimeLens-Bench (Charades, QVH, ANet) × {Qwen3-VL-8B, TimeLens-8B} at 10% and ~25%: all tokens, random, 4 pruners, Coverage, matched-budget low-res input | 0.8 | Tab. A |
| 6.2 | **Main table B, grounded QA**: S-EMBER (+ held-out split) and NExT-GQA × two models: all tokens, random, best pruner, Coverage, PCD on random, CoCo, floors row; Acc/GQ | 0.6 | Tab. B |
| 6.3 | Why it works: one merged ablation table (equal vs weighted quota, question vs generic score, global top-k without quota, per-layer vs shared; counterfactual for PCD: no-video vs noised vs permuted) | 0.6 | Tab. C |
| 6.4 | Budget and frames-vs-tokens (Fig. budget only) and cost (Tab. D, compact) | 0.6 | Fig. budget, Tab. D |
| 7 | Limitations (prompt scope of PCD, short videos, not streaming) + conclusion | 0.4 | — |

About 7.8 pages, with 4–5 tables and 3–4 figures. **Move to the supplement**: Tab. qa details, question types, Tab. pcd8 (keep a one-line summary in 6.3), Tab. scope, Tab. replay, Tab. reserve, the prompt-format study (Tab. prompt), the timeline-prompt QVH/ANet/Charades table (current Tab. main), Tab. paired (fold the stars into A/B), oracles, retrieval analysis, negative results, error taxonomy, closed intervals, end-snapping, parsing, anchors and reproduction. Fig. pruners can become a column of Tab. A, or stay as the figure if Fig. leak is cut.

## (e) Minimal experiments to move it toward spotlight, with outcome risks

| # | Experiment | Why it is needed | Cost | Outcome risk |
|---|---|---|---|---|
| E1 | S-EMBER, uncompressed and random-10%, with the one-line and direct-answer prompts (both models), against CoCo | Decides whether the S-EMBER headline (+10.7*/+11.2*) also depends on the prompt. The paper already notes a direct-answer run (+0.4 for PCD) but does not report the uncompressed score | A few GPU-hours | **High**. If the uncompressed model under a standard prompt approaches CoCo, the full-method claim shrinks to "under a reasoning-style answer format". Either way, reporting it removes the reviewer's strongest objection |
| E2 | FastVID, VidCom², FlashVID, SemVID vs random vs Coverage at 10% and 25%, one-line prompt, full TimeLens-Bench, both models (extends the queued p40 sweep) | Makes Observation 1 general. This is the spotlight finding | Largest: about 1.5–3 days on the pool | **Moderate**. SemVID at 31% is already slightly above random under this prompt, so at milder budgets pruners may tie random. At 10% the frame-emptying argument predicts they still lose. If they do not, the paper becomes "a fix for grounded QA" (a poster) |
| E3 | Question attention with global top-k and no quota (Tab. design TODO), on S-EMBER and standard-prompt QVH-513, plus a frame-coverage statistic | Isolates the named mechanism (coverage) from SnapKV-style question attention | Under 1 GPU-hour per setting | **Low cost, high information**. If it matches the equal quota, the "every timestamp survives" story is not the explanation, and the paper has to say so |
| E4 | Cost table: prefill, per-token decode latency, peak memory and KV size for all tokens / random / SemVID / Coverage / CoCo | Answers "why prune if prefill is unchanged?" | Hours, no new accuracy runs | **Low**. It will confirm that savings are decoder-side only. The framing must then claim that honestly (multi-question reuse, decode-bound regimes) |
| E5 | PCD (with reserve) on NExT-GQA in the single-reply, reasoning-style format, and the held-out S-EMBER split | Shows the full method on a second grounded-QA setting and a held-out split | About a day | **Moderate**. The joint format already costs the uncompressed model 16.5 Acc@GQA, so PCD may recover part of that without beating the two-query protocol. Report it either way |
| E6 | Rerun the pre-reserve tables that remain in the main paper (the PCD counterfactual rows) with the reserve | Coherence of the main paper | Hours | **Low** |

E1–E4 are the minimum. E2 decides whether the paper is a spotlight candidate or a solid poster. E1 decides how the S-EMBER result has to be worded.

---

## Questions for Authors
1. What does the uncompressed model score on S-EMBER with the one-line or direct-answer prompt, and does CoCo still exceed it? (E1)
2. Is S-EMBER public, and can it be cited? Is it the authors' own benchmark? This affects both W4 and anonymity.
3. Is HERMES (the cited baseline) the authors' prior work? If so, how will the appendix be anonymised? (W12)
4. In the intended application (egocentric assistants), is the memory reused across several questions? If so, can Coverage's question-dependent pruning be reconciled with reuse, for example by a write-once memory re-ranked at question time? This is the strongest deployment argument the paper could make.

---

## Minor Issues

### Consistency
- Tab. paired gives S-EMBER CoCo vs all tokens as "+11.3 \TODO{sig.}", while Tab. tl8b gives +11.2*. Use one number.
- The Fig. budget caption says CoCo "leads at every budget", but Tab. budget shows it below SemVID + PCD at ActivityNet 10% (11.6 vs 26.8) and very low at QVHighlights 5% (9.7). Scope the caption to S-EMBER, or to budgets of at least about 500 tokens.
- §6b text says "Coverage at 24% of the visual tokens", while Tab. standard's row label reads "Coverage (31%)". Explain the keep-ratio vs token-share difference once (timestamps and minimum per frame).
- The judge is named "Qwen3.8-27B-FP8" in §5. Check the model name.

### Citation Format
- 15 bibliography entries carry "[authors to verify]" or "[cite: ...]" placeholders (sember, hermes, semvid, ktv, adatp, lgttp, mdp3, focus, qframe, streammem, infinipotv, season, fvaction, reze, knowwhen).

### Figures and Tables
- Several tables mark best and second best per row and others per column. In the cut-down main paper, use one convention.
- Floors appear as grey rows. Keep that convention in the new Tab. A and Tab. B.

### Layout
- Remove the `\TODO` macros and the "Supplementary material (remove from the submission PDF)" appendix from the submission PDF, and submit the supplement separately.
