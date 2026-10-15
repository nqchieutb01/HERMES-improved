# Peer Review Report

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget (method name: CoCo)
- **Manuscript ID**: n/a (anonymous CVPR 2026 review-mode draft; `docs/paper/latex/main.tex` + `sec/*.tex` + `fig/*.tex`)
- **Review Date**: 2026-10-15
- **Review Round**: Round 1 (simulated pre-submission panel), including results produced after the draft (`recent_results.md`)

---

## Reviewer Information

### Reviewer Role
Journal-Fit Reviewer (internal role EIC)

### Reviewer Identity
CVPR area chair for video understanding and multimodal LLMs; has chaired sessions on long-video LLMs and efficient video inference.

### Review Focus
Whether the contribution is new compared with token pruning (FastVID, VidCom2, FlashVID, SemVID, SnapKV, PruneVid) and contrastive decoding (VCD, M3ID, ICD), or is a combination of known parts. Whether the story (diagnosis → method → floors) holds together and its claims match the evidence. Whether the headline numbers use standard protocols and can be compared with published numbers. How the new results (NExT-GQA, TimeLens-8B, TimeLens-Bench) change the paper.

### Venue criteria
`criteria_binding_unavailable`

CVPR 2026 is the venue the authors name, but no formal criteria manifest was supplied. Venue-fit comments below use general expectations for a top computer-vision conference. They are not bound criteria and make no claim of alignment with the venue.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score
4. Video LLMs, temporal grounding and efficient inference are central to this seat. Statistical details and how faithfully the baselines were re-implemented are outside my remit and are left to R1.

Confidence only discloses uncertainty and scope. It does not change severity or decision bearing.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper studies temporal grounding when a video LLM's visual memory is pruned to about 10% of its tokens. It makes three claims. (1) QA-oriented pruners fall below random token selection on grounding because they empty frames and so lose timestamps. (2) Where the memory is silent, a video-free "start early, stop soon" prior writes the interval. (3) Benchmark floors are high enough to hide both effects. The method, CoCo, combines Coverage (an equal per-frame quota filled by question attention, chosen per layer) with PCD (contrastive decoding of time values against a no-video pass, plus an answer reserve). The diagnosis is the strongest part. The evaluation discipline is unusually honest: video-free floors, length-rescaling floors, and a section of negative results. These would interest CVPR readers.

The headline framing does not survive the results the authors collected after drafting. The teaser and abstract claim that CoCo "nearly doubles" the uncompressed model on QVHighlights (54.8 vs 28.6 R@0.5). Under the standard one-line TimeLens prompt, on standard data, the same uncompressed Qwen3-VL-8B scores 64.5–64.9 R@0.5, and Coverage alone at 31% scores 66.7. The draft's baseline is held back by its timeline prompt. The same protocol effect makes the abstract's claim that the base model "stays below the video-free floor" on Charades and ActivityNet false as a statement about the model. PCD helps only under the timeline format. Coverage, by contrast, transfers to a second model, to the standard prompt and to NExT-GQA answering.

The paper is worth publishing if it is re-centred on the diagnosis, Coverage and the floors, with PCD scoped to grounded-QA output formats. That needs a substantial rewrite and a re-run of the main table, which is why I recommend Major Revision.

---

## Strengths

### S1: A clear causal diagnosis that is new for this literature
The paper does more than observe that pruners hurt grounding. It isolates the mechanism. The share of kept tokens by position (Tab. retention) shows that pruners empty frames. Removing the evidence frames raises the early-start share by 14 points for both the unpruned and the random memory, while a control removal does not (Tab. causal). Stating the bias in the instruction does nothing. I have not seen this chain ("pruning removes timestamps → the prior fills the gap") made in the token-pruning literature, which reports QA only (Sec. 2). None of the new results contradict it. SemVID is again the worst on grounding on NExT-GQA (recent_results §1, Cover30 − SemVID30 mIoU +1.8*).
**Evidence Anchor**: `table: Tab. causal (sec/3_diagnosis.tex) — leak 31%→46% (random 10%) and 37%→51% (all tokens) when evidence frames are removed`

### S2: Floor-adjusted evaluation as a methodological contribution
Every main comparison reports a best constant interval tuned on test, and that constant is labelled as an upper bound on video-free constants. Every comparison also reports a length-rescaling floor that tests whether gains are just longer intervals. The paper says where it loses: ActivityNet −3.1*, Charades −6.4* against the constant. This is the evaluation discipline the grounding community has asked for since Otani et al., and it shields the paper against the most common reviewer objection.
**Evidence Anchor**: `table: Tab. paired (sec/6_results.tex) — rows "Best constant" and "Length rescaling" reported for all four datasets, including the negative cells`

### S3: Coverage is a robust, simple component, and the new results strengthen it
Within the draft, Coverage gives +6.1* (S-EMBER) and +6.8* (QVHighlights) over random tokens (Tab. ladder). The new results show it transfers beyond the draft's protocol:
- QVH-TimeLens subset, standard prompt: Cover 31% scores 61.7 vs 60.5 unpruned for Qwen3-VL-8B, and 67.3 vs 67.1 for TimeLens-8B. Cover − Random is +3.5* and +2.0* (recent_results §3c).
- NExT-GQA Acc@QA at 10%: +3.1* over random for both models (§1).
- S-EMBER with TimeLens-8B: Retrieve + Coverage scores 37.2 vs 28.2 random (§2).
This is the claim most likely to survive review.
**Evidence Anchor**: `table: Tab. ladder (sec/6_results.tex) — "C (uniform frames)" 31.7 / 33.7 vs random 25.6 / 26.9 mIoU`

### S4: Negative results and replayed decision rules are reported
Tab. negative and Tab. replay record what did not work: VCD-style contrast, PAI-style attention gain, unequal quotas, prior-subtracted retrieval. They also show that a threshold tuned on the gold labels does not beat PCD at the first time value. This raises confidence that the method was not cherry-picked, and it is useful to readers.
**Evidence Anchor**: `table: Tab. negative (sec/6_results.tex) — 11 ideas with ΔmIoU against stated references`

### S5 (new evidence): the S-EMBER ordering holds on a grounding-finetuned model
This answers the "one weak model" objection for S-EMBER specifically. With TimeLens-8B the ordering random < unpruned < Coverage < Random+PCD < CoCo is the same as with Qwen3-VL-8B. CoCo scores 43.9 / 46.9 (mIoU / R@0.5) against 33.1 / 34.1 for the unpruned model (recent_results §2), and TimeLens-8B has no m:ss parsing issue. It is not yet in the paper, which states the single-model limitation.
**Evidence Anchor**: `text: sec/7_conclusion.tex Limitations "All results use one model, 475--500-question subsets"`

---

## Weaknesses

### W1: The headline gain is measured against a baseline held back by its prompt
**Problem**: The teaser (right panel and its caption "×1.9 the uncompressed model"), the abstract (25.0/28.6 → 54.8 R@0.5) and conclusion (i) in §6.1 present CoCo as nearly doubling the uncompressed model. All of these numbers use the timeline prompt ("Seen: … / Answer: … / Time: [s, e]") on low-resolution re-encoded copies. The new results show that the uncompressed baseline is badly handicapped by this prompt:
- On the 513-query QVH-TimeLens subset, Qwen3-VL-8B unpruned with the standard one-line TimeLens prompt scores **60.5 mIoU / 64.9 R@0.5** (recent_results §3c). The draft's all-tokens row is 29.1 / 28.6.
- The cleanest evidence is the paired comparison on the same subset, model and memory (§2b). Cover with the timeline prompt scores 30.7 mIoU; Cover + PCD + reserve, 43.9; Cover with the TimeLens prompt, 61.7. Timeline + PCD vs the TimeLens prompt is −17.8*.
- Under the standard prompt, Coverage alone at 31% (66.7 R@0.5) is far above the draft's CoCo headline (54.8).

There are confounds (31% vs 10% budget, 2 fps vs 64 frames, full-resolution vs re-encoded copies), but §2b holds them fixed and the gap remains. Much of the "×1.9" is PCD repairing a format the model handles badly, not CoCo beating the model.

**Evidence Anchor**: `figure: Fig. teaser (fig/teaser.tex) right panel — "×1.9 the uncompressed model", all tokens 28.6 vs CoCo 54.8 R@0.5`

**Why it matters**: The first thing a CVPR reader or AC will do is compare 28.6 R@0.5 with the published Qwen3-VL-8B QVHighlights numbers (≈64 R@0.5 on TimeLens-Bench). The authors' own anchor run reproduces those numbers within 0.3 mIoU (§3a). Shipped as is, the headline would mislead readers. Once a reviewer finds this, it is likely to sink the paper.

**Suggestion**:
1. Rebuild the main QVHighlights / Charades / ActivityNet comparison on the standard one-line prompt and standard data (TimeLens-Bench), with both models. The §3c table is ready to become the new main QVH table: unpruned, TL-script 2 fps low-res at matched tokens, Random/SemVID/Cover 31%, and Retrieve + Cover at 897 tokens vs TL 1 fps at 1,125, for both models.
2. Keep the timeline prompt only where the task needs both a free-form answer and an interval (S-EMBER grounded QA). On S-EMBER, also report the unpruned model under a one-line or direct prompt, so readers can see the cost of the timeline format there.
3. Remove "×1.9" and "nearly doubles" from the teaser, abstract and §6.1. A defensible replacement headline is "Coverage at 10–31% matches or exceeds the uncompressed model under the standard protocol, on two models; QA pruners fall below random". Its full-set version must wait for the §3b sweep (see W6).

**Severity**: Critical (repairable). The core headline claim is invalid as worded; the underlying contribution survives if the paper is re-framed.
**Confidence**: 4. Core expertise in video-LLM grounding protocols; relies on recent_results §2b/§3a/§3c as reported.

### W2: "The base model stays below the video-free floor" describes the protocol, not the model
**Problem**: The abstract closes with "on Charades-STA and ActivityNet Captions the uncompressed model itself stays below the video-free floor". §7 lists this under "Base-model floors" and the conclusion calls it "a lesson of their own". Under the standard TimeLens protocol, Qwen3-VL-8B scores 48.5 mIoU on full Charades-STA and 47.1 on full ActivityNet (recent_results §3a). The authors' own pipeline gives 48.0 and 47.1 on the 400-query bridge subsets. Those are far above the draft's floors (29.7 and 37.9) and the draft's unpruned numbers (16.6 and 22.9). The model is not below the floor. The timeline prompt plus the re-encoded copies put it there.

**Evidence Anchor**: `text: sec/0_abstract.tex "the uncompressed model itself stays below the video-free floor, and no pruning method, ours included, rises above it"`

**Why it matters**: The paper generalises this as a property of the model and of the benchmarks ("on two standard benchmarks a constant interval beats the uncompressed model"). A reviewer who knows published Qwen3-VL numbers will read it as a sign of a broken pipeline, and that undermines trust in every other table.

**Suggestion**: Drop the claim from the abstract and conclusion. Keep the floors as an evaluation tool (S2), and re-compute them on the TimeLens-Bench splits you adopt. State plainly that, under the standard protocol, Qwen3-VL-8B clears the Charades and ActivityNet floors by about 9–19 mIoU. If you keep the timeline-prompt runs in the appendix, label the sub-floor result as specific to that prompt.

**Severity**: Major. A secondary headline claim that is false as stated; the core method claim survives.
**Confidence**: 4. Published numbers are known to this seat and the authors' anchor runs reproduce them.

### W3: PCD is presented as general but works only with a timeline-style output format
**Problem**: The abstract and §1 describe PCD as part of a general method, and §6.4 calls it "a plug-in for any memory". The draft itself reports in §6.4 "Prompt format" that PCD gives +0.4 with the official direct-answer prompt. The method section carries a TODO to report this. The new results add a second null: under the TimeLens one-line prompt, PCD on Cover is −0.1 / −0.3 (n.s.) for Qwen3-VL-8B (recent_results §3c). The likely mechanism is that PCD only changes the decimal of times written as a list of observations, while TimeLens-style answers round to 1 s. Where PCD does work (S-EMBER on both models: +10.2 on random and +6.7 on R + C with TimeLens-8B, §2), it works inside a format that the plain model handles worse than the one-line format (W1).

**Evidence Anchor**: `text: sec/6_results.tex §6.4 "PCD needs a reasoning-style answer: with the official direct-answer prompt it does not help"`

**Why it matters**: PCD is half of the method name ("Contrast the Prior"). If its gains depend on a format that is itself sub-optimal for single-moment grounding, the contribution is narrower than the title suggests. Reviewers will find the TODO and the +0.4.

**Suggestion**: Scope PCD explicitly, in the abstract and in a short paragraph in §4.2. It corrects the time prior when the model must write intermediate observations or an answer together with an interval (grounded QA, counting and duration questions). It does not help one-line interval answers. Move the direct-prompt and TimeLens-prompt null results from TODO and footnote into a visible table. Optionally, test whether the S-EMBER gain survives a stronger structured format (the event timeline, already +7.4* greedy, with CoCo, which is the open TODO) so that PCD is not credited for repairing the timeline prompt alone. The planned NExT-GQA PCD run is a cheap prediction test and should be reported whatever the outcome.

**Severity**: Major.
**Confidence**: 4. Decoding mechanics are adjacent expertise; the evidence is the paper's own §6.4 plus recent_results §3c.

### W4: The ablation that would separate the claimed novelty from SnapKV/PruneVid is missing
**Problem**: The paper says that "neither operator is new in isolation" and that the contribution is the combination for grounding. The key question is whether gains come from the equal quota (new) or from question-attention selection (SnapKV, PruneVid). The row that answers this, "Question, global top-k, no quota", is an empty `\TODO{}` in Tab. design. SnapKV-style eviction and PruneVid are also TODO baselines in §5. The equal-quota claim currently rests on the stratified-vs-global comparison of HERMES scores (23.7 vs 18.3), not on question attention.

**Evidence Anchor**: `table: Tab. design (sec/6_results.tex) — row "Question, global top-$k$, no quota" with empty \TODO{} cell`

**Why it matters**: Without this row, a reviewer can argue that Coverage is SnapKV applied to video with per-layer selection, and that the "keep the clock" mechanism is not shown for the selector the paper actually uses. This is the main originality risk for a CVPR AC.

**Suggestion**: Run question global top-k (no quota) at 10% and 5% on S-EMBER and the QVH-TimeLens subset. Report its frame-emptying statistics in the same form as Tab. retention. Add SnapKV (observation-window eviction) and PruneVid as baselines in the main table. If the global top-k variant also empties frames and loses, the novelty claim becomes strong. If it does not, re-centre the contribution on per-layer question selection and the diagnosis.

**Severity**: Major.
**Confidence**: 4. Core expertise in the token-pruning literature.

### W5: The efficiency framing is not supported by a cost analysis
**Problem**: The title ("under a Visual-Token Budget") and the introduction justify pruning by cost ("video LLMs pay for every frame"). But CoCo has several sources of cost:
- It prefills the full uncompressed memory per question.
- It runs the question through the full memory for scoring; the method section notes "the current implementation runs 3–4 scoring passes".
- PCD adds a second decoding pass.
§7 states that peak memory and time-to-first-token are those of the uncompressed model. The cost table is a TODO.

**Evidence Anchor**: `text: sec/4_method.tex §4.1 Cost "measured latency and peak memory; the current implementation runs 3--4 scoring passes, reducible to one."`

**Why it matters**: CVPR reviewers in efficient video inference will ask what is saved. Without numbers, "budget" reads as an accuracy-only constraint. Combined with W1, a reader could conclude that the method costs more than the uncompressed model and grounds worse than it under a standard prompt.

**Suggestion**: Add a cost table: prefill time, scoring-pass time, decode time with and without PCD, peak memory, and the size of the KV cache that the decoder reads. Compare it with unpruned, random and SemVID on the same hardware. Then either (a) reframe honestly as "decoder-side memory compression: the KV cache read during answering, which matters for multi-question or long-answer sessions", and show a multi-question-per-memory setting where the saving is real, or (b) implement the single scoring pass and chunked prefill. The TimeLens-Bench comparison at matched tokens (Retrieve + Cover at 897 tokens vs TL 1 fps at 1,125: 64.3 vs 57.5 mIoU on TimeLens-8B, §3c) is the best existing efficiency argument and belongs in the main text.

**Severity**: Major.
**Confidence**: 4. Core expertise in efficient video inference.

### W6: Headline data are non-standard copies and subsets; standard full-set results are not yet available
**Problem**: The QVHighlights and ActivityNet copies are low-resolution re-encodes (153 and about 84 tokens per frame pair, against 660 on S-EMBER). Every dataset is a 500-question subset. S-EMBER is both the design set and a headline set, and the held-out split is a TODO. Several of the draft's own findings are artefacts of the low token count: the collapse below about 300 tokens, the 32-frame rule, the 30% ActivityNet budget. The authors now have a validated standard pipeline: TimeLens-Bench anchors reproduce published numbers within 0.2–0.3 mIoU (recent_results §3a). But the full-set Cover-vs-Random sweep is about 1.5% done, and every TimeLens-8B and random31 directory is empty (§3b).

**Evidence Anchor**: `dataset: QVHighlights / ActivityNet copies (sec/5_setup.tex) — "low-resolution re-encodes (153 and about 84 visual tokens per frame pair, against 660 on S-EMBER)"`

**Why it matters**: My card asks whether the headline numbers can be compared with published work. Currently they cannot. The standard-protocol evidence that would replace them exists only for a 513-query subset.

**Suggestion**: Make the full TimeLens-Bench sweep (both models, Cover vs Random vs SemVID at matched tokens, plus the matched TL-script baselines) the main grounding table, and do not submit before it finishes. Keep the low-resolution runs only for the budget and frames analysis, labelled as a low-token-per-frame regime. Run the held-out S-EMBER split, or state that S-EMBER is a development set and weight claims towards the frozen-configuration datasets.

**Severity**: Major.
**Confidence**: 4. Core expertise in grounding benchmarks; sweep status taken from recent_results §3b.

### W7: NExT-GQA bounds the claim "coverage helps grounding" to long videos, and the paper does not say so
**Problem**: On NExT-GQA (1,005 questions, two-query protocol; recent_results §1):
- At 10%, grounding regresses relative to unpruned on both models. Qwen3-VL-8B: Acc&GQA 38.9 → 36.1, mIoU 37.7 → 34.8. TimeLens-8B: 41.5 → 38.5.
- Cover vs Random on mIoU and IoU@0.5 is a wash (−0.5 / +0.1, n.s.); Cover's gain is in answering (+3.1* Acc@QA).
- At 30%, both Cover and Random are lossless, so that budget does not separate the methods.
This fits the draft's existing limitation on short, densely sampled videos (≈40 s clips covered by 64 frames). But the abstract and §1 state the timeline-loss mechanism without a duration condition.

**Evidence Anchor**: `text: sec/1_intro.tex Observation 1 "Random selection is a strong baseline for grounding because it keeps a few tokens on every frame."`

**Why it matters**: NExT-GQA is the standard grounded-QA benchmark that CVPR reviewers will expect, given that the paper motivates itself with grounded questions. A reviewer who runs it will find no grounding gain. Reporting it first, framed correctly, turns a liability into a boundary condition.

**Suggestion**: Add NExT-GQA (both models; unpruned, Random, SemVID, Cover at 10% and 30%; also 5% to find where grounding separates) as a grounded-QA table. Frame it as "Coverage preserves answering at 10%; on 40 s clips a random spread already keeps the timeline". State the two-query protocol and its relation to LOVER's joint protocol (22.4 joint vs 38.9 two-query Acc&GQA, which reproduces LOVER), and compare methods only within one protocol. Add the duration condition (number of frames per second of video, or tokens per frame) to the Observation 1 claim.

**Severity**: Major. It changes the claimed scope of a core claim and requires new tables.
**Confidence**: 4. Core expertise in grounded-QA benchmarks.

### W8: The draft is visibly unfinished, and the length is far over the limit
**Problem**:
- There are `\TODO` markers on 16 lines across the sections, and several sit in headline places. Tab. paired has "+11.3 \TODO{sig.}" and empty Charades/SemVID cells. Tab. main has empty ActivityNet cells for FastVID and VidCom2. Others cover the event timeline with CoCo, the video-clustered bootstrap and the held-out split.
- The compiled PDF is 16 pages (main.log). The main text visibly exceeds the 8-page CVPR limit. I did not extract the exact split.
- The results section alone has about 15 tables and figures, many of them S-EMBER design tables.

**Evidence Anchor**: `table: Tab. main (sec/6_results.tex) — empty ActivityNet cells for FastVID and VidCom2 ("Empty cells: not yet computed")`

**Why it matters**: Desk-reject risk on length, and reviewers will read unfilled cells as incomplete work.

**Suggestion**: After the re-framing in W1–W3, keep in the main paper:
- teaser
- diagnosis (Tab. retention, Fig. leak, Tab. causal, Tab. floors)
- method figure
- main table: TimeLens-Bench, both models
- S-EMBER grounded QA: both models, with Acc and GQ once the TimeLens-8B answers are judged
- NExT-GQA
- pruners figure
- one Coverage ablation (including W4's row)
- one PCD table (Tab. cf plus the prompt-format nulls)
- cost table

Move Tab. scope, Tab. replay, Tab. oracle, Tab. negative, Tab. errors, the budget and frames analysis on re-encoded copies, retrieval analysis, and Tab. closed to the supplement. Fill or delete every TODO cell.

**Severity**: Minor. Presentation and completeness; it does not by itself change a core claim.
**Confidence**: 5. Venue-format expectations.

### W9: Optional retrieval is included in the headline rows although it is not significant
**Problem**: §4.3 calls retrieval optional, and it adds +0.8 to +0.9 n.s. once tokens are chosen by the question. Yet "the main-table runs include it". The "Coverage $+$ PCD without retrieval on all datasets" ablation is a TODO. On the QVH-TimeLens subset, Retrieve + Cover at 64 frames (58.3) is below uniform Cover 31% (61.7), though at fewer tokens (897 vs 3,271).

**Evidence Anchor**: `text: sec/4_method.tex §4.3 "so we treat it as optional; the main-table runs include it."`

**Why it matters**: The headline method then includes a SigLIP index that the paper's own ablation says is not needed. This makes the method look more complex than it is and blurs which component carries the result.

**Suggestion**: Report the main rows without retrieval (pure Coverage, or Coverage + PCD), and show retrieval as a separate row. This also simplifies the method box.

**Severity**: Minor.
**Confidence**: 4. Adjacent; follows the paper's own ablation.

---

## How the new results should change the paper (integration plan)

| New result (recent_results.md) | Changes which claim | Action |
|---|---|---|
| §3a: TimeLens-Bench anchors reproduce published numbers (Qwen3 Charades 48.5, QVH 59.6, ANet 47.1; TimeLens-8B Charades 55.4, QVH 65.7) | Invalidates "base model below floor" (W2) and exposes the prompt handicap (W1) | Put the anchors in the setup section as a protocol check. Re-base the main table on this pipeline |
| §3c: QVH-TL subset, both models, Cover 31% ≥ unpruned; Cover − Random +3.5* / +2.0*; PCD −0.1 / −0.3 n.s. | Supports Coverage (S3); limits PCD (W3) | New main QVH table until the §3b full sweep finishes. Report the PCD null visibly |
| §2b: timeline vs TimeLens prompt, −17.8* | Explains the draft's low baselines; reframes PCD | One paragraph plus a small table in the PCD analysis: "PCD repairs the timeline format; it is needed where an answer must accompany the interval" |
| §2: TimeLens-8B on S-EMBER, CoCo 43.9 vs 33.1 unpruned, same ordering | Removes the single-model limitation for S-EMBER (S5) | Add as a second column block in the S-EMBER table. Judge the answers (Acc, GQ@0.5) first. Location questions show no gain (20.3 vs 20.6); keep that in the question-type table |
| §1: NExT-GQA, Cover helps answering (+3.1*), not grounding; 10% regresses vs unpruned | Bounds Observation 1 to long or sparsely sampled videos (W7) | New grounded-QA table. Two-query protocol stated; LOVER reproduction in the supplement |
| §1 / §3b: SemVID worst again (Cover30 − SemVID30 mIoU +1.8*) | Reinforces "QA pruners lose time" (S1) | One sentence plus a cell in the pruners figure or table |
| §3c: Retrieve + Cover at 897 tokens vs TL 1 fps at 1,125 (TimeLens-8B 64.3 vs 57.5) | The best efficiency evidence (W5) | Put it in the main text as a matched-token comparison against frame-rate reduction |

Caveats the authors must handle when integrating (these are not findings against the current draft):
- Batch size differs between unpruned runs (bs 1–2) and pruned runs (bs 8). By the authors' own note this moves ±2 points, so NExT-GQA "regressions" of about 3 points sit near noise. Re-run the unpruned baselines at bs 8, or report all runs at one batch size.
- `frame_max_pixels` differs between models (TimeLens-8B 1,572,864, Qwen3 unset), so cross-model comparisons are not yet clean. Within-model comparisons are fine.
- The S-EMBER TimeLens-8B answers are not yet judged.
- The full TimeLens-Bench sweep is about 98% outstanding. No claim should cite it until it is complete.
- The paired CIs and every *(log)* value in recent_results.md are as reported by the overnight scripts. This panel did not re-score them; re-verify them before they enter the paper.

Suggested narrative after revision: "Pruning for QA silently breaks the timeline. We diagnose why (coverage loss, then a video-free prior), give an evaluation discipline (floors) that exposes it, and show that a simple question-filled equal quota keeps grounding at or above the uncompressed model on two models under the standard protocol. When the task needs an answer with its interval (grounded QA), a time-scoped prior-contrastive decoder removes the remaining prior." Retitling should be considered so that the title does not promise more for PCD than the evidence supports. For example, keep "Keep the Clock" and qualify "Contrast the Prior" as applying to grounded QA.

---

## Detailed Comments

### Journal Fit
The topic fits CVPR well: video LLMs, efficient video inference and temporal grounding are all active CVPR areas, and the paper uses CVPR-standard benchmarks (QVHighlights, Charades-STA, ActivityNet Captions, and now NExT-GQA). The style is empirical and analysis-heavy, which suits a CVPR audience if the analysis is compressed (W8). Length and the TODO count are the fit risks, not the scope.

### Originality
Medium. The authors admit that the operators come from earlier work (per-frame budgets in VidCom2/KTV/AdaTP, question-attention eviction in SnapKV/PruneVid, no-input contrast in M3ID). The original parts are:
- the diagnosis linking timestamp loss and the decoder prior;
- scoping PCD to time values with a confidence-adaptive strength and an answer reserve;
- the floor-adjusted evaluation.
Video-free floors are not new in themselves (Otani et al. 2020, cited). Making them a per-method reference under compression is new. The novelty argument for Coverage depends on the missing global top-k row (W4).

### Significance
If re-framed, significance is moderate to high for the efficient-video-LLM sub-community. "QA pruners fall below random on grounding" is a useful warning that will change how pruning papers evaluate. As currently framed, the significance rests on a ×1.9 gain that comes largely from the protocol (W1), so it is overstated.

### Structural Coherence
Title → abstract → intro → method → results is internally consistent, and the observations map cleanly to the components. The conclusion overreaches in two places, on floors (W2) and on PCD's generality (W3). The Limitations section is candid, but it frames the sub-floor result as a model limit when it is a protocol limit.

### Title & Abstract
The title is memorable, and its two halves map onto the two components. The abstract is well written but contains the two claims that the new data contradict (the 28.6 → 54.8 framing and the floor sentence). Its final sentence should become a statement of scope (short videos; one-line answers for PCD).

### Conclusion
The conclusion that grounding "better than the uncompressed model on S-EMBER and QVHighlights" with a tenth of the tokens holds only under the timeline prompt. For QVHighlights, rewrite it against the standard-protocol results. "Clears video-free floors wherever the base model does" should be restated once the floors are re-computed on the standard data.

---

## Questions for Authors
1. On S-EMBER, what do the unpruned model and Coverage score with a one-line or direct-answer grounding prompt, and with a separate answer query as in your NExT-GQA two-query protocol? If they approach CoCo's 39.3, is PCD's gain on S-EMBER also a repair of the timeline format?
2. What does question global top-k without a quota (Tab. design, TODO row) score, and how many frames does it empty? This decides whether the novelty lies in the quota or in the per-layer question attention.
3. What is the wall-clock and peak-memory cost of CoCo against unpruned inference on the same 64 frames? Is there any deployment setting (several questions per memory, long answers) where the decoder-side saving is the dominant cost?
4. Will the full TimeLens-Bench sweep (§3b) and the judged TimeLens-8B S-EMBER answers be finished before submission? If not, which subset results do you intend to present as main results?

---

## Minor Issues

### Figures and Tables
- In Fig. budget (b), CoCo at 64 frames (9.7) comes from the low-token re-encode. Label the regime ("~2–4 tokens/frame pair, re-encoded copy") on the panel itself.
- Several tables mix runs with and without the answer reserve and with different parsers and batch sizes (explained in "Reading the tables"). After re-running on the standard pipeline, regenerate the S-EMBER design tables (Tab. pcd8, cf, scope, replay, oracle, errors) with one configuration, or move them to the supplement with that caveat.
- Teaser bar chart: if kept, add the standard-prompt unpruned bar, or replace the chart.

### Layout
- 16 compiled pages. The supplement is input after `\clearpage` in main.tex; produce a separate supplementary PDF for submission.

### Citation Format
- NExT-GQA, LOVER, TimeLens and grounding-finetuned Video-LLMs (TimeChat, VTimeLLM, TRACE) are not cited. They need to be once the new results are integrated.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`. Criteria source: field-general CVPR expectations from Card #1 (`criteria_binding_unavailable`).

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #1 focus 1 | PARTLY_MEETS | text: sec/1_intro.tex "Neither operator is new in isolation"; table: Tab. design TODO row | Diagnosis and floors are new; Coverage novelty versus SnapKV is untested | Depends on the W4 ablation | yes: AC-level novelty question |
| Methodological Rigor | general | PARTLY_MEETS | table: Tab. paired; table: Tab. causal | Floors, paired tests and negative results are strong; the protocol choice handicaps the baseline | Statistical details deferred to R1 | yes, through W1 |
| Evidence Sufficiency | Card #1 "standard protocols" | DOES_NOT_MEET | figure: Fig. teaser; dataset: re-encoded copies (sec/5_setup.tex) | Headline is not on a standard protocol; full standard sweep about 1.5% done | New results partly fill the gap (subset only) | yes |
| Argument Coherence | general | PARTLY_MEETS | text: sec/0_abstract.tex floor sentence | Story is coherent but two abstract-level claims are contradicted | none identified | yes |
| Writing Quality | general | MEETS | sec/1_intro.tex; sec/3_diagnosis.tex | Clear and precise prose; over length; TODOs | Page split not extracted | no, if fixed |
| Literature Integration | Card #1 focus 1 | PARTLY_MEETS | sec/2_related.tex | Pruning and contrastive decoding are well covered; grounding-tuned Video-LLMs, NExT-GQA and LOVER are missing | R2 owns the depth | no |
| Significance & Impact | Card #1 focus 2 | PARTLY_MEETS | table: Tab. ladder; recent_results §3c | The "pruners break the timeline" warning plus Coverage is significant; the ×1.9 headline overstates it | Depends on the outcome of the full sweep | yes |

**Recommendation rationale**: Major Revision. The decision-bearing defects (W1, W2, W3, W6) are all repairable within one cycle using data the authors already have or have queued: the TimeLens-Bench pipeline is validated and the QVH-TL subset results exist. The core contributions survive and are strengthened by the new data: the diagnosis, the floors, and Coverage transferring across models and prompts. What does not survive is the headline framing. Submitting the current draft would very likely be rejected once a reviewer compares the 28.6 R@0.5 baseline with published Qwen3-VL numbers. Strengths do not offset W1: until the main comparison is on a standard protocol, the paper's central quantitative claim is not established.
