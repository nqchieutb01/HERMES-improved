# Peer Review Report

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (CoCo)
- **Manuscript ID**: CVPR 2026 draft (anonymous, review mode)
- **Review Date**: 2026-10-15
- **Review Round**: Round 1 (simulated panel)

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Senior researcher in video temporal grounding and moment retrieval (Charades-STA, ActivityNet Captions, QVHighlights, NExT-GQA). Familiar with grounding-tuned Video-LLMs (TimeChat, VTimeLLM, TRACE, Time-R1, TimeLens-style models) and with the benchmark-bias literature (Otani et al. 2020; Yuan et al., Charades-CD / ANet-CD).

### Review Focus
(1) Whether the related-work section covers grounding-specialised Video-LLMs, the benchmark-bias and debiasing literature, and nearby token-pruning and contrastive-decoding work. (2) Whether the results use standard protocols and can be compared with the state of the art, and what the post-draft TimeLens-Bench, TimeLens-8B and NExT-GQA results (`recent_results.md`) do to the paper's positioning. (3) Whether the method helps grounding-tuned models or only a base model under one particular prompt.

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**: substantial revisions needed, and the paper must be re-reviewed after them.

### Confidence Score
4. Grounding protocols, benchmarks and the Video-LLM literature are my core area. I am less sure about the exact published numbers of individual grounding-tuned models, so those are marked "to verify" below.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment
The paper studies temporal grounding with Qwen3-VL-8B when the visual KV memory is pruned to 10%. It makes three claims. QA-oriented pruners lose the timeline and fall below random selection. A video-free decoder prior writes intervals that start early and end too soon. Positional floors on standard benchmarks are high. The proposed CoCo combines equal per-frame quotas filled by question attention (Coverage) with no-video contrastive decoding of time tokens (PCD), and it reports large gains: QVHighlights R@0.5 rises from 28.6 to 54.8.

The diagnosis is careful and the negative results are reported openly. The positioning, however, rests on a baseline that the post-draft results show to be an artifact of the prompt. The paper's "timeline" prompt halves Qwen3-VL-8B's grounding. On the same 513 QVHighlights queries with the same Coverage memory, changing only the prompt moves mIoU from 30.7 to 61.7. Under the standard prompt, the uncompressed model clears every video-free floor, and PCD has no effect. So the headline claims ("nearly doubles the uncompressed model"; "the base model stays below the floor on Charades/ActivityNet") do not hold as stated. In addition, none of the public benchmarks is evaluated on its standard split, and the related work omits the whole grounding-tuned Video-LLM literature.

What survives is Coverage. Under the standard prompt it beats random selection on both Qwen3-VL-8B and TimeLens-8B, and it matches unpruned grounding with about a quarter of the visual tokens (3,271 of ~13,650). That is a publishable contribution, but it is a different paper from the one written. I recommend Major Revision, because the repair (re-baseline under the standard protocol, reframe PCD, add the missing literature) fits within one cycle.

---

## Strengths

### S1: The diagnosis that QA-oriented pruners lose the timeline is new and well evidenced for grounding
The token-retention analysis by window position (`tab:retention`) and the paired gaps of all four pruners against random selection on three datasets (`fig:pruners`) are concrete and mechanistic. The paper reads time from interleaved timestamp tokens, and a frame emptied by pruning loses its timestamp. That is the right explanation for why grounding differs from multiple-choice QA, and the efficient-video-LLM literature reports QA almost exclusively. The post-draft results partly support it: on NExT-GQA, SemVID 30% is below Cover 30% on mIoU (+1.8*).
**Evidence Anchor**: `table: tab:retention (caption), FastVID keeps tokens from only 25% of frames; HERMES global top-k mIoU 18.3 vs stratified 23.7`

### S2: Video-free floors are reported next to every method
Putting the best constant interval and a length-rescaling baseline beside every row of the main table, plus a closed-interval split on S-EMBER, follows Otani et al. 2020 and is better practice than most Video-LLM grounding papers. The S-EMBER end-snapping analysis (Appendix C) is a useful warning for streaming grounded-QA benchmarks.
**Evidence Anchor**: `table: tab:floors, best constant [.15, 1] scores 36.9 mIoU on S-EMBER vs 28.0 for the uncompressed model`

### S3: Coverage transfers to a grounding-tuned model and to the standard prompt
In the post-draft results, Coverage beats random selection under the TimeLens prompt on both Qwen3-VL-8B (+3.5*) and TimeLens-8B (+2.0*). At 3,271 tokens it matches the unpruned model (TimeLens-8B 67.3 vs 67.1). The 3,271 tokens are out of ~13,650 visual tokens, labelled "31%" in the run names. On S-EMBER with TimeLens-8B, Retrieve + Coverage beats random by +9.0 mIoU. This answers the community's question of whether the method helps models that are already good at grounding: for Coverage, the answer is yes.
**Evidence Anchor**: `table: recent_results.md §3c, TimeLens-8B Random 31% / Cover 31% = 65.2 / 67.3 mIoU; unpruned 67.1`

### S4: Negative results and limits are reported openly
The paper keeps the ActivityNet and Charades failures, the collapse below about 300 tokens, the PCD failure under the direct-answer prompt and a table of ideas that did not work. This makes the paper easy to audit, and that audit is how the prompt issue in W1 surfaced.
**Evidence Anchor**: `text: §6.4 "with the official direct-answer prompt it does not help (+0.4 mIoU)"`

---

## Weaknesses

### W1: The uncompressed baseline is a prompt artifact, and the headline gains and "below the floor" claims do not hold under the standard grounding prompt
**Problem**: All main tables use a reasoning-style timeline prompt (`Seen: … / Answer: … / Time: [s, e]`). Under it, unpruned Qwen3-VL-8B scores 29.1 mIoU on QVHighlights, 22.9 on ActivityNet and 16.6 on Charades-STA. With the standard one-line TimeLens grounding prompt, the authors' own pipeline reproduces the published Qwen3-VL-8B numbers: 48.5 vs published 48.3 on Charades, 59.6 vs 59.4 on QVH, 47.1 vs 46.8 on ANet. The decisive test changes only the prompt. On the same 513 QVH queries with the same Cover-31% memory, the timeline prompt gives 30.7 mIoU and the TimeLens prompt gives 61.7, and timeline + PCD is −17.8* below the TimeLens prompt. Under the standard prompt, PCD is null on both models (−0.1 and −0.3, n.s.).
**Evidence Anchor**: `table: recent_results.md §3c / observation 2b, QVH-TL subset, Qwen3 Cover under timeline prompt 30.7 mIoU vs one-line TimeLens prompt 61.7; paired −17.8*`
**Why it matters**: Three claims change.
(a) The abstract and Limitations say "on Charades-STA and ActivityNet Captions the uncompressed model itself stays below the video-free floor". Under the standard prompt, Qwen3-VL-8B scores 48.5 on Charades (floor 29.7) and 47.1 on ANet (floor 37.9). These are approximate comparisons: the floors were tuned on the paper's 500-query subsets, while the model numbers come from TimeLens-Bench's refined annotations. The gap is still far larger than that mismatch could explain. The model is above the floor; the prompt is below it.
(b) §6.1(i), "nearly doubles the uncompressed model", compares CoCo's 48.8 mIoU against a handicapped baseline. In the controlled comparison (same 513 queries, same Cover memory of 3,271 of ~13,650 visual tokens, labelled "31%" in the runs), timeline prompt + PCD + reserve gives 43.9 mIoU, against 61.7 for Coverage alone under the standard prompt and 60.5 for the unpruned model. No 10%-budget CoCo run under the standard prompt exists yet (Question 1), so the size of the 10% gain under the standard protocol is unknown.
(c) Most of PCD's +7 to +20 point gains repair failure modes of the timeline format (run-on `Seen:` lists, unparseable outputs, early starts). These are not shown to exist under the format the field uses. The paper already concedes that PCD does nothing under the direct-answer prompt; the new data show this is structural.
A CVPR reader comparing these numbers with the Qwen3-VL or TimeLens reports would conclude that the paper's model is broken, or that the gains are inflated.
**Suggestion**:
(1) Make the standard one-line grounding prompt the primary protocol for QVH, Charades and ANet, and report the timeline prompt as a secondary setting.
(2) Present Coverage under the standard prompt as the headline: it beats random, it beats SemVID on NExT-GQA, and it matches unpruned with 3,271 of ~13,650 tokens. Extend this to the 10% budget.
(3) Reposition PCD as a decoding fix for reasoning-style or timeline outputs. These matter for grounded QA (S-EMBER, where the answer needs listed moments), not for single-moment retrieval. Say explicitly that PCD is null under the one-line format.
(4) Rewrite the abstract, Obs. 3, §6.1(iii) and the "Base-model floors" limitation.
(5) Report the unpruned standard-prompt number in every main-table column, so readers can see the cost of compression under the standard protocol.
**Severity**: Major. Coverage survives the prompt swap, so the core contribution remains. The headline numbers, the "doubles the uncompressed model" claim and the "base model below the floor" framing must all be redone.
**Confidence**: 5. Core expertise: grounding evaluation protocols. The evidence is the authors' own controlled prompt swap.

### W2: No public benchmark is evaluated on its standard split or metric, so no number can be compared with the literature
**Problem**: QVHighlights uses 500 of the 1,020 single-window val queries. The standard QVH moment-retrieval protocol reports R1@0.5, R1@0.7 and mAP on the full val or test set, including multi-window queries (Lei et al. 2021). Charades-STA uses 500 of the 3,720 test queries (Gao et al. 2017). ActivityNet Captions uses 500 queries, one sentence per video, from low-resolution re-encodes (84 visual tokens per frame pair, against 660 on S-EMBER). All of this is disclosed in §5. The consequence is never stated: the paper contains no number that can be put next to a published one. Its only external anchor (W1) shows the base model at about twice the paper's figure.
**Evidence Anchor**: `text: §5 "Our QVHighlights and ActivityNet copies are low-resolution re-encodes (153 and about 84 visual tokens per frame pair"`
**Why it matters**: Grounding papers at CVPR are expected to report on the full standard test splits so results can be compared, and both dataset papers define them. The low-resolution re-encodes also drive the "collapse below about 300 tokens" result (§6.5), which the paper itself attributes to the copies ("low in our re-encoded … copies"). That result may not hold on the original videos.
**Suggestion**: Re-run on full splits with the original videos. The TimeLens-Bench sweep already under way (full Charades 3,363, QVH 1,541, ANet 4,500) is the natural vehicle: it is a recognised, refined benchmark with published Qwen3-VL-8B and TimeLens-8B anchors that the authors reproduce to within 0.3 mIoU. Report the original-annotation splits as well, or state why TimeLens-Bench replaces them. Re-derive the token-floor threshold on full-resolution video.
**Severity**: Major. Without standard splits, the claim of comparability to the state of the art cannot be assessed.
**Confidence**: 4. Core expertise; norm grounded in the dataset papers' defined splits and the published TimeLens-Bench numbers in `recent_results.md` §3a.

### W3: The related work omits grounding-tuned Video-LLMs and has no comparison against them
**Problem**: The paper frames itself as "temporal grounding for Video LLMs". Its related work covers frame selection, token pruning, KV compression, contrastive decoding, training-free grounding and benchmark bias. It never mentions the grounding-specialised Video-LLMs that define the state of the art: TimeChat, VTimeLLM, LITA, Momentor, VTG-LLM, TimeSuite, TRACE, Time-R1, TimeLens. The bib has none of them, yet the post-draft experiments run TimeLens-8B and use the TimeLens evaluation scripts.
**Evidence Anchor**: `absence: §2 Related Work and main.bib — expected grounding-tuned Video-LLMs (TimeChat, VTimeLLM, TRACE, Time-R1, TimeLens) and their zero-shot Charades/QVH/ANet numbers; checked sec/2_related.tex, sec/1_intro.tex, sec/5_setup.tex, main.bib`
**Why it matters**: A reader cannot tell where 54.8 R@0.5 on a 500-query QVH subset sits relative to these models. Nor can they tell whether a training-free method on a general model is competitive with, complementary to, or far below grounding fine-tuning. The paper also needs to show that the method still helps once a model is grounding-tuned. The TimeLens-8B results answer this positively for Coverage.
**Suggestion**: Add a paragraph on grounding-tuned Video-LLMs. Add a reference table of published zero-shot numbers on the standard splits (Qwen3-VL-8B, TimeLens-8B, and earlier timestamp-aware models), with this paper's unpruned and 10% or 31% rows under the same protocol. Include the TimeLens-8B results (S-EMBER: CoCo 43.9 vs unpruned 33.1; QVH-TL: Cover 67.3 vs unpruned 67.1) as a second model.
**Severity**: Major. Positioning against the field's main line of work is missing, and it affects how the contribution is judged.
**Confidence**: 5. Core expertise.

### W4: The benchmark-bias literature is thin, and "floor-adjusted evaluation" is listed as a contribution although it is largely a blind baseline
**Problem**: Only Otani et al. 2020 is cited. The paper does not cite the follow-up line that built out-of-distribution splits for the same biases: Yuan et al.'s Charades-CD / ActivityNet-CD re-splits with a debiased metric, and work on deconfounded or debiased moment retrieval. Contribution 3 presents the floor-adjusted evaluation as a contribution of this paper. The best-constant floor is a test-tuned version of Otani's blind baseline, and the paper's own related-work paragraph says it "find[s] the same".
**Evidence Anchor**: `text: §2 "We find the same on the datasets we use, and make video-free floors part of every comparison."`
**Why it matters**: The paper overstates its novelty, and it misses an alternative evaluation the community already uses for exactly this concern: OOD re-splits, where the positional prior is broken by construction.
**Suggestion**: Cite the Charades-CD / ANet-CD line (verify the exact reference; see Missing References). Describe floors as an adopted practice rather than a contribution. Ideally, report Coverage on Charades-CD test-ood, which would test whether Coverage, unlike PCD's partial lengthening, helps where the prior is wrong by construction. Keep the length-rescaling baseline and the S-EMBER closed and end-snapping analyses, which are new.
**Severity**: Minor. This concerns literature completeness and the framing of a contribution; the core claims are unaffected.
**Confidence**: 4. Core expertise; the Charades-CD metadata needs verification.

### W5: "All four pruners fall below random" is untested under the standard prompt and at a 30% budget, and one post-draft data point is mixed
**Problem**: The pruner gaps (`fig:pruners`) are all measured under the timeline prompt at 10%. Post-draft, on the QVH-TL subset with the standard prompt, SemVID 31% (58.8) is slightly above Random 31% (58.2). It also keeps more tokens (4,232 vs 3,271), so the comparison is not matched. On NExT-GQA, SemVID 30% is below Random 30% (35.8 vs 37.3 mIoU).
**Evidence Anchor**: `table: recent_results.md §3c, Qwen3 Random 31% / SemVID 31% / Cover 31% = 58.2 / 58.8 / 61.7 mIoU`
**Why it matters**: Observation 1 is the paper's most transferable finding. It should be shown under the protocol other researchers will use.
**Suggestion**: Re-run the four pruners against random at matched tokens under the standard prompt on the full TimeLens-Bench splits. Report the result whichever way it goes.
**Severity**: Minor. Only one data point conflicts, it is not budget-matched, and it may be noise.
**Confidence**: 3. The direction is plausible, but the evidence is a single unmatched cell.

### W6: NExT-GQA (post-draft) uses a non-standard two-query protocol, and at 10% Coverage does not help grounding
**Problem**: NExT-GQA metrics (Acc@QA, Acc@GQA, mIoP, IoP@0.5, mIoU) are standard (Xiao et al., CVPR 2024). The two-query protocol is not: answering and grounding are asked separately. It gives 38.9 Acc@GQA for unpruned Qwen3-VL-8B, while the joint prompt reproduces the published zero-shot row (22.4). At 10%, Cover vs Random on grounding is a wash for both models (mIoU −0.5 and +0.1, n.s.), and both regress against unpruned. PCD has not been run on NExT-GQA.
**Evidence Anchor**: `table: recent_results.md §1, Qwen3 Unpruned two-query Acc&GQA 38.9 vs joint LOVER prompt 22.4; Cover10 − Random10 mIoU −0.5 (n.s.)`
**Why it matters**: If NExT-GQA is added, the protocol difference has to be explicit, or readers will compare 38.9 against published joint-prompt numbers. The null result at 10% on short (~40 s) videos also bounds the claim that Coverage helps grounding: it helps on long windows, not on short densely sampled clips. This matches the paper's own "short videos" limitation and should be stated together with it.
**Suggestion**: If NExT-GQA goes into the paper, report both protocols, compare methods only within one protocol, cite NExT-GQA and the source of the joint-prompt baseline, and present the 10% grounding result as a boundary condition.
**Severity**: Minor. These results are not in the draft yet. The concern is how they are integrated.
**Confidence**: 4. Core expertise in GQA protocols.

### W7: The primary benchmark (S-EMBER) and the judge are not identifiable
**Problem**: All design choices were made on S-EMBER. Its bib entry is a placeholder. The judge is named "Qwen3.8-27B-FP8", which does not match any model release I know of.
**Evidence Anchor**: `text: main.bib sember "author = {{[authors to verify]}}, note = {[cite: S-EMBER benchmark paper]}"`
**Why it matters**: Reviewers cannot check the S-EMBER protocol, its question types or its official metrics. The accuracy and GQ@0.5 numbers depend on a judge whose identity is unclear.
**Suggestion**: Give the full S-EMBER citation and state whether it is public and peer-reviewed. Correct the judge's model name and version, and state its agreement with the official judge if a different model is used.
**Severity**: Minor. This is fixable by citation, but it is a prerequisite for review.
**Confidence**: 3. I cannot verify S-EMBER's status from the materials.

---

## Detailed Comments

### Title & Abstract
- The title promises "Temporal Grounding for Video LLMs". The evaluation covers one general-purpose model under a non-standard prompt. If TimeLens-8B is added and the standard prompt becomes primary, the title fits.
- The final abstract sentence ("the uncompressed model itself stays below the video-free floor") must be removed or qualified as specific to the timeline prompt (W1).
- "Raises R@0.5 on QVHighlights from … 28.6 (all tokens) to 54.8" should name the prompt and the subset.

### Introduction
- Obs. 1 and 2 are well motivated. Obs. 3 ("benchmark floors are high … the best constant also beats the uncompressed model") is true only under the timeline prompt.
- The motivating example, "how long did I use the paper cutter", is grounded QA, where a reasoning-style answer is natural. That supports positioning PCD for grounded QA rather than for single-moment retrieval.

### Literature Review / Theoretical Framework
- **Coverage**: The token-pruning coverage is broad and current: FastVID, VidCom², FlashVID, SemVID, PruneVid, DyCoke, FrameFusion, KTV, AdaTP, LGTTP. FastV, the canonical attention-based visual-token pruner that most of these compare against, is missing. Grounding-tuned Video-LLMs are entirely missing (W3), and so is the OOD / debiasing grounding line (W4). The contrastive-decoding coverage is VLM-hallucination-centric. The text-only origin, contrastive decoding (Li et al., ACL 2023), and the calibration / PMI line are absent; I leave depth on that to Reviewer 3.
- **Integration quality**: The paragraphs are synthetic and state differences clearly; the KTV and LGTTP contrast is good. The claim that "LGTTP is the only method we know that evaluates grounding" under pruning should be verified: token-efficient grounding models may exist [UNVERIFIED].
- **Gap argument**: That compression breaks the timeline is convincing. That a decoder prior is the bottleneck is convincing for the timeline format and unproven for the standard format.

### Methodology / Research Design
- (Domain view only.) The causal evidence-removal test (`tab:causal`) and the gold-informed oracles (`tab:oracle`) are good diagnostic designs that the grounding community would value.
- The choice of the timeline prompt is the single most consequential design decision, and the paper justifies it only by PCD's needs: "The timeline prompt matters".

### Results / Findings
- The main table has no row for the standard-prompt unpruned model and no published reference numbers.
- QVHighlights results should include mAP if they are labelled "QVHighlights", or else be labelled single-window R1.
- The ActivityNet result at 30% while the other datasets are at 10% complicates comparison; budgets in absolute tokens on full-resolution video would remove the need for this exception.
- The post-draft TimeLens-8B S-EMBER result (CoCo 43.9 vs unpruned 33.1, same ordering as Qwen3) is strong evidence for the grounded-QA claim and should be in the paper. Its answers still need judging.

### Discussion
- The "Base-model floors" limitation needs rewriting under the standard prompt. The real limitation is that PCD's gains depend on the format.
- The label-ambiguity analysis for S-EMBER location questions (62 of 100 with annotator IoU < 0.6) is useful for the benchmark community.

### Conclusion
- "It clears video-free floors wherever the base model does" and "on two standard benchmarks a constant interval beats the uncompressed model" both need revision (W1).

### References
- Many entries carry `[verify]` or `[authors to verify]` markers (S-EMBER, HERMES, SEASON, REZE, FV-Action, knowwhen, LGTTP, AdaTP, StreamMem, InfiniPot-V). Under the paper's own rules, every one must be resolved before submission.

---

## Missing Key References
Confident (I can attest these exist; check the bibliographic details before use):
- Ren et al., **TimeChat**: a time-sensitive multimodal LLM for long video understanding, CVPR 2024. Canonical timestamp-aware Video-LLM with zero-shot Charades-STA and QVH results.
- Huang et al., **VTimeLLM**: empower LLM to grasp video moments, CVPR 2024. Boundary-aware grounding Video-LLM.
- Huang et al., **LITA**: Language Instructed Temporal-Localization Assistant, ECCV 2024. Time tokens for localization.
- Guo et al., **TRACE**: temporal grounding Video LLM via causal event modeling, ICLR 2025.
- Xiao et al., **Can I Trust Your Answer? Visually Grounded Video Question Answering** (NExT-GQA), CVPR 2024. Required if NExT-GQA is added.
- Chen et al., **FastV** ("An Image is Worth 1/2 Tokens After Layer 2"), ECCV 2024. Canonical attention-based visual-token pruning.
- Li et al., **Contrastive Decoding: Open-ended Text Generation as Optimization**, ACL 2023. Origin of the expert-minus-amateur contrast.
- Zhao et al., **Calibrate Before Use**, ICML 2021. Content-free-input prior correction, conceptually the closest precedent to a "no-video" prior (depth left to R3).

To verify (real lines of work; confirm the exact metadata before citing):
- Yuan et al., "A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric" (Charades-CD / ActivityNet-CD OOD re-splits and a debiased metric), around 2021 [UNVERIFIED venue].
- Nan et al., "Interventional Video Grounding with Dual Contrastive Learning", CVPR 2021, deconfounding temporal location bias [UNVERIFIED details].
- Wang et al., **Time-R1** (RL post-training of an LVLM for temporal video grounding), 2025 [UNVERIFIED authors and venue].
- **TimeLens** (rethinking video temporal grounding with MLLMs; TimeLens-Bench refined annotations; TimeLens-8B), 2025 [UNVERIFIED authors and venue]. Must be cited if TimeLens-Bench or TimeLens-8B results are used.
- Qian et al., **Momentor**, ICML 2024; Guo et al., **VTG-LLM**; Zeng et al., **TimeSuite**, ICLR 2025 [UNVERIFIED venues]. Optional, for completeness of the grounding-LLM paragraph.
- **VideoMind** and **LOVER** as the sources of the NExT-GQA MCQ prompt and joint-prompt baseline [UNVERIFIED metadata].

---

## Questions for Authors
1. Why is the timeline prompt the primary protocol for single-moment benchmarks (QVH, Charades, ANet) when the standard prompt doubles unpruned performance (30.7 → 61.7 on the same QVH queries and memory)? What do CoCo and each pruner score under the standard prompt at 10% on the full TimeLens-Bench splits?
2. Is there any configuration with the standard (one-line) answer format in which PCD gives a significant gain? If not, will you restrict the PCD claim to reasoning-style or grounded-QA answers?
3. What is S-EMBER (citation, public availability, peer-review status, official metrics), and which model and version is the judge ("Qwen3.8-27B-FP8")?
4. Does the collapse below about 300 tokens, and the "about 8 tokens per frame pair" rule, still hold on the original-resolution QVHighlights and ActivityNet videos?

---

## Minor Issues

### Language / Grammar
- "Qwen3.8-27B-FP8" (§5) looks like a typo.
- "We call an answer an early start … and a leak …": "leak" is defined well, but the abstract and introduction use "leak" informally before the definition.

### Citation Format
- Placeholder entries (`sember`, `hermes`, `season`, `reze`, `fvaction`, `knowwhen`) have empty or invented titles and must be completed.
- `otani2020uncovering` is correctly cited to BMVC 2020.

### Figures and Tables
- `tab:main` should add a standard-prompt unpruned row and a note on the evaluation subset.
- The `\TODO{sig.}` cell in `tab:paired` and the empty cells for FastVID / VidCom² on ActivityNet in `tab:main` (and Charades / S-EMBER cells in `tab:paired`) must be filled or removed.

### Layout
- The 16-page draft (main text plus appendix) will need heavy cuts. The negative-results and replay tables are good candidates for the supplement. (Deferring to the Journal-Fit Reviewer.)

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #3 focus 1; quality_rubrics | PARTLY_MEETS | `table: tab:retention`; `text: §2 "We find the same on the datasets we use"` | The timeline-loss diagnosis for grounding under pruning is new. Floors and no-input contrast are adopted, and the paper says so for PCD but not for floors | Novelty against very recent grounding-under-compression work [UNVERIFIED] | No. Repairable by reframing |
| Methodological Rigor | Card #3 focus 2 (protocol standardness) | DOES_NOT_MEET | `table: recent_results.md §3c, 30.7 vs 61.7`; `text: §5 "low-resolution re-encodes"` | Non-standard prompt and subsets make the baseline unrepresentative (domain view only; statistics are left to R1) | The full TimeLens-Bench sweep is ~98% outstanding | Yes. Must re-baseline |
| Evidence Sufficiency | Card #3 focus 2–3 | PARTLY_MEETS | `table: recent_results.md §2 TimeLens-8B S-EMBER CoCo 43.9 vs 33.1`; `table: §3c Cover +3.5*/+2.0*` | Coverage evidence is strong across two models. PCD evidence holds only under the timeline format | Post-draft numbers are partly from logs, not re-scored | Yes. Defines what can be claimed |
| Argument Coherence | quality_rubrics | PARTLY_MEETS | `text: Abstract "the uncompressed model itself stays below the video-free floor"` | Diagnosis → method is coherent. "Below the floor" and "doubles uncompressed" follow from a prompt choice, not from the model | none identified | Yes |
| Writing Quality | quality_rubrics | MEETS | `text: §6 "We report these as limits, not successes."` | Clear and direct; honest negative results | Length is out of scope | No |
| Literature Integration | Card #3 focus 1 | DOES_NOT_MEET | `absence: §2 and main.bib — expected grounding-tuned Video-LLMs; checked sec/2_related.tex, main.bib` | The field's main SOTA line and the OOD-split line are both absent | Some suggested refs are marked to verify | Yes. Needed for positioning |
| Significance & Impact | Card #3 focus 2 | PARTLY_MEETS | `table: recent_results.md §3c TimeLens-8B Cover 67.3 vs unpruned 67.1 with 3,271 of ~13,650 tokens (run label "31%")` | Lossless grounding at ~1/4 of the tokens on a SOTA grounding model is significant if confirmed on full splits. A 10%-budget gain under the standard prompt is not yet shown | The 10% standard-prompt result is unknown; NExT-GQA at 10% is null | Yes |

The decision-bearing failures are protocol standardness (W1, W2) and positioning against grounding-tuned Video-LLMs (W3). All three can be repaired in one revision. The pipeline already reproduces published TimeLens-Bench numbers to within 0.3 mIoU, so a standard-protocol re-run is feasible. Coverage's positive evidence on two models gives the revised paper a defensible core. Strong diagnostics do not offset the protocol issue: as written, the headline claims would mislead readers about Qwen3-VL-8B's grounding ability and about the size of the gain.
