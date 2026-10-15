# Editorial Decision Letter, round 2 (spotlight bar): "Keep the Clock, Contrast the Prior" (CoCo)

**Decision: Major Revision.** All five seats agree the paper is not at spotlight level today.
- **As submitted:** below the acceptance bar (Journal-Fit Reviewer).
- **After the reframing and cuts below:** a borderline-to-accept poster.
- **For a spotlight:** one decisive experiment must come out well (§5, E2), and the paper must be rebuilt around the result that survives the standard protocol.

**Disclosures**
- **Venue:** CVPR 2026 is stated by the author, but there is no confirmed criteria manifest (`criteria_binding_unavailable`). The "spotlight bar" is a field-general rubric (field_analysis.md), not CVPR criteria.
- **Calibration:** `NOT_CALIBRATED`.
- **Panel:** five role-separated seats, each in a fresh context. None could see the others' reports or the round-1 reports. All five run on the same model family and provider (Claude, Anthropic), so their errors may be correlated.
- **Protocol:** lightweight. Each seat made a single call; this was not the two-call paper-blind sprint contract, and no checker replay was run.
- **Evidence:** manuscript at commit 2c84a78, plus `docs/paper/analysis/results_2026-10-15.md`, which was re-scored from raw outputs.
- **Claims checked by the editor (in the manuscript or code):**
  - The PCD no-video row is computed at every decode step: in `inference/batched_decoding.py`, the time scope only selects which score is used.
  - The appendix says "HERMES code base".
  - The budget-figure caption says CoCo "leads at every budget", but `tab:budget` shows 9.7 and 11.6 at QVH 5% and ANet 10%.
  - There are 19 `\TODO`s in sec/*.tex.
  - The judge is named "Qwen3.8-27B-FP8".
  - TimeLens-8B is fine-tuned from Qwen3-VL-8B-Instruct (TimeLens README).
- **Not verified by the editor:** S-EMBER = arXiv:2607.02689 (R2's finding) and the GPU-hour estimates (R1). Several literature suggestions are marked "to verify" by the reviewers.

## 1. What changed since round 1 (morning panel)
- **Resolved:** the new results are in the paper (§6.2, with 4 tables), together with a reproduction table showing the pipeline matches published TimeLens-Bench numbers to within 0.3. The "nearly doubles" and "below the floor" claims are scoped to the timeline prompt in §6.1 and in the Limitations. The intro arithmetic is fixed.
- **Partly resolved:** the claims were scoped in the body, but the title, the teaser (×1.9), contribution 4, Observation 3 and parts of the conclusion still sell the timeline-prompt result. All five seats flag this (DA-C3).
- **Unresolved, and now sharper:** whether the prior is a format artefact, missing baselines, cost, lineage, statistics, page length.

## 2. Consensus strengths
- **Diagnosis rigor** (all 5): QA pruners empty frames and fall below random. The causal frame-removal test and the video-free floors for every method are spotlight-grade in rigor (R1).
- **Reproduction and honesty** (EIC, R1, R2): published numbers reproduced within 0.3 mIoU, a candid negative-results table, and limits stated.
- **Coverage transfers under the standard prompt** (EIC, R2, R3, DA-observations): it matches the uncompressed model at about 24% of tokens on both Qwen3-VL-8B and TimeLens-8B, and beats random (+3.5\*, +2.0\*) and SemVID.
- **The full method on grounded QA** (EIC, R2): +11.2\* and +10.7\* mIoU over the uncompressed model on S-EMBER, for a general and a grounding-tuned model.
- **PCD's decoding analysis** (R3): the counterfactual choice, the replay against a threshold tuned on gold labels, and the time-only scope with adaptive strength.

## 3. Consensus issues

### S1. The front matter contradicts §6.2 (CRITICAL by DA-C3; Major by EIC, R1, R2, R3)
The title ("Contrast the Prior"), the teaser (×1.9, 54.8 vs 28.6), contribution 4, Observation 3 and the conclusion's "floors" lesson rest on the timeline prompt. Under the one-line prompt, the uncompressed model scores 60.5 on QVH-513 and 47.1 / 48.5 on ActivityNet / Charades, above the floors, and PCD gives −0.1.
**Adjudication: validated.** Writing only, no GPU needed, but it blocks acceptance.

### S2. "The temporal prior" is measured only under the timeline prompt (CRITICAL by DA-C1; Major by R3-W2, EIC)
Every measurement of the prior uses the timeline format: 89% of the first `Seen:` times at 0 s, the leak rates and the causal test. No no-video run exists under the one-line or direct-answer prompt.
**Adjudication: validated as unresolved.** It is cheap to settle (E0a), and the answer decides the title.

### S3. The S-EMBER headline has no standard-prompt baseline (CRITICAL by DA-C2; EIC-E1)
CoCo beats the uncompressed model by about +11 only when both use the timeline prompt. The paper mentions an official direct-answer S-EMBER prompt but never reports the uncompressed model under it.
**Adjudication: validated as unresolved.** This is the highest-risk check, because it guards the last "beats uncompressed" claim.

### S4. No decisive result under the standard protocol at the paper's budget (Major; EIC-E2, R2, R1-W8)
The standard-prompt evidence is at about 24–31% of tokens on a 513-query subset. The full TimeLens-Bench sweep has no complete Coverage-vs-Random cell, and no run exists at 10%. EIC and R2 independently call the full TimeLens-Bench at 10%, under the one-line prompt, with both models and the published pruners, the experiment that decides spotlight vs poster.

### S5. Coverage's contribution is not identified (Major; R1-W1, DA-M4, R2-W6)
- CoCo rows include SigLIP retrieval and Random + PCD does not; retrieval alone gives +3.8\*.
- Without retrieval (QVH-513, timeline prompt), Coverage + PCD ties Random + PCD (+0.1).
- The sign of Coverage − Random changes with the setup: +6.8\*, −2.8\* and +3.5\* across QVH setups, and level on Charades and NExT-GQA.
- Fig. pruners' "Coverage" bar is actually Retrieve + Coverage (R1).

### S6. Baselines compared across regimes, strongest ones missing (Major; R1-W2/W3, DA-M1/M2, EIC-E3)
- Random and Coverage evict per layer after a full prefill; the four pruners feed fewer tokens to the LLM (`hermes_vqa.py`).
- Missing: question global top-k without the quota (an empty cell in Tab. design), SnapKV / PyramidKV-style eviction, PruneVid, and a matched-token low-resolution input on S-EMBER.

### S7. Cost and deployment (Major; all seats)
- Coverage needs the full prefill plus 3–4 scoring passes, so TTFT and peak memory equal the uncompressed model's.
- PCD's no-video row runs at every decode step (editor-verified in code); the Limitations text ("for time-value tokens") understates this.
- "31%" in labels is 24% of visual tokens (R1, EIC).
- R3 asks for amortisation over several questions per video and a two-tier streaming memory.

### S8. PCD lineage (Major; R3-W1, R2)
Eq. (pcd) is the CFG / context-aware decoding rule with VCD's plausibility set. Contrastive Decoding, CFG for LMs, CAD, contextual calibration and DC-PMI are not cited. Claim only what is new: the time-slot scope, the adaptive strength and the diagnosis.

### S9. Generality (Major for spotlight; R1-W8, R3-W6, R2-W7)
- TimeLens-8B is a Qwen3-VL fine-tune, so both models are one family.
- The timestamp-loss mechanism depends on Qwen3-VL's text timestamps.
- There is no long-video grounding benchmark (MAD, Ego4D-NLQ), though the paper's premise is long and streaming video.
- S-EMBER is used as a 475-question development set, with no held-out split.

### S10. Statistics hygiene (Major; R1-W6)
- Batch size 1–2 vs 8 sits under the "matches uncompressed" claims (+1.2, lower bound ≈ 0).
- The bootstrap is not clustered by video.
- About 106 tests are starred with no correction.
- There is no equivalence test (TOST) for "matches".
- "Pre-registered" is claimed without an artefact.

### S11. Presentation and compliance (Critical-but-repairable by EIC)
- **Length:** the body is about 14 pages; CVPR allows 8.
- **Unfinished content:** 19 `\TODO`s and about 15 placeholder references, including S-EMBER.
- **Possible anonymity leak:** the appendix says "the HERMES code base", and HERMES is also a cited baseline.
- **Inconsistencies:**
  - the budget caption's "leads at every budget";
  - +11.3 (TODO) vs +11.2\*;
  - the judge name typo;
  - TimeLens-8B Retrieve + Coverage at 897 tokens is −2.8\* vs uncompressed, and this is not reported (DA-M6);
  - the one-line PCD runs lack the reserve (DA-M7).

## 4. Disagreements and arbitration
- **Is anything spotlight-level left?** DA says no: every positive result depends on the authors' own choices. EIC, R2 and R3 say the diagnosis plus Coverage's standard-prompt transfer could carry a spotlight if E2 succeeds and the story is rebuilt. *Arbitration:* DA's verdict holds for the current evidence. The spotlight route goes through S4/E2 and one general, predictive insight.
- **What is the thesis?** Three framings were proposed:
  - EIC: "Pruning for QA keeps what is salient; grounding needs what is everywhere."
  - R3: "Compression loses anchors before content; keeping anchors helps every answer format; contrasting the prior helps in proportion to how much the prior decides the slot."
  - DA: the paper reads as two weaker papers.

  *Arbitration:* R3's framing turns the Coverage/PCD asymmetry into a testable prediction (E5) and keeps both components in one story. EIC's sentence is the better title-level hook. Both are compatible, so use EIC's sentence as the headline and R3's prediction as the analysis spine.
- **Keep PCD in the main paper?** EIC suggests scoping it to grounded QA or showing a second benchmark. R3 suggests the prediction figure. *Arbitration:* keep PCD, scoped to reasoning-style grounded QA, as the second contribution. Its general value is shown by E5, not by "plug-in for any memory".

## 5. Roadmap to spotlight (ordered; GPU estimates are R1's, unverified)

**Phase 0: cheap checks that decide the framing (≈1 day).**
| # | Experiment | Seats | Decides |
|---|---|---|---|
| E0a | No-video answers plus early-start and leak rates under the one-line and direct prompts (S-EMBER, QVH-513); causal test under one prompt | DA, R3, EIC | Whether "the temporal prior" stays in the title (S2) |
| E0b | Uncompressed S-EMBER under the direct-answer (official) and one-line prompts; CoCo must beat the best of them | DA, EIC | Whether the S-EMBER headline survives (S3) |
| E0c | Coverage + PCD on uniform frames; Retrieve + random + PCD | R1 | Whether Coverage adds anything under PCD (S5). If not, reframe as diagnosis + PCD + floors before the big sweep |
| E0d | Question global top-k without the quota; random at input level; input pruners with all timestamps kept | R1, EIC, DA | Coverage's novelty over SnapKV-style eviction; the cross-regime fairness of "below random" (S6) |
| E0e | Cost table: TTFT, total latency, peak memory, FLOPs, KV size at bs 1/8; amortisation over questions per video | all | S7 |

**Phase 1: the decisive result (≈1–3 weeks on 8 GPUs).**
- **E2:** full TimeLens-Bench at 10% (and 30%) under the one-line prompt, both models, against Random, SemVID, FastVID, SnapKV-style eviction and PruneVid, with matched-budget TimeLens inputs. Report the uncompressed model within about 1 mIoU. **This is the spotlight-vs-poster experiment** (EIC, R2).
- **E3:** a held-out S-EMBER split (or the full S-EMBER test set) with official metrics; NExT-GQA with PCD.
- **E4:** a second model family. Ideally it should encode time differently (positions rather than text timestamps), to test the anchor mechanism (R3, R1).

**Phase 2: what makes it memorable.**
- **E5, prediction figure:** PCD gain against the per-slot prior strength (e.g. KL(p_M ‖ p_∅)) across all existing conditions, plus one prediction stated before the run (R3). Optionally, plot Coverage − Random against seconds per frame and tokens per frame, to explain when Coverage matters (R2).
- **E6:** one result outside time, such as counts or box coordinates, using slot-scoped PCD (R3).
- **E7:** a long-video grounding benchmark (MAD or Ego4D-NLQ), where compression is forced (R2).

**Writing (no GPU, do now):**
1. Rebuild the title, teaser, abstract and contributions around what survives: the diagnosis, Coverage under the standard protocol, and PCD for grounded QA. Remove ×1.9 from the teaser.
2. Cut to 8 pages. EIC's plan:
   - two main tables: full TimeLens-Bench under the standard prompt, and grounded QA (S-EMBER held-out + NExT-GQA);
   - one merged ablation table;
   - the budget figure and a cost table;
   - about 15 tables moved to the supplement.
3. Add the PCD lineage and the grounding-tuned Video-LLM and debiasing literature; replace the S-EMBER placeholder with the real citation (verify arXiv:2607.02689).
4. Fix the verified defects:
   - "leads at every budget";
   - the judge name;
   - the PCD cost wording;
   - "31%" vs 24%;
   - +11.3 TODO;
   - report the TimeLens-8B −2.8\* at 897 tokens;
   - remove "HERMES code base" from the anonymous version;
   - clear the TODOs.
5. Statistics: batch-matched reruns of the uncompressed rows, a video-clustered bootstrap, Holm correction over the primary endpoints, and TOST for "matches".

**Decision rule after Phase 0:**
- If E0a shows the prior under standard prompts and E0b keeps the S-EMBER gain, keep the current two-part story.
- If E0a shows no prior outside the timeline format, retitle around coverage and anchors, and scope PCD to reasoning-style answers.
- If E0c shows Coverage adds nothing under PCD, lead with the diagnosis + PCD + floors.

## 6. Questions to the authors
1. Under the official S-EMBER prompt, what does the uncompressed model score?
2. Does the no-video model still start at 0 s under a one-line prompt?
3. Which deployment benefits from pruning after a full prefill? Show the amortisation curve.
4. How are attention probabilities obtained with fused attention kernels, and at what cost?
5. What does Coverage add over question top-k without the quota?

## Attachment: Acronym Check (advisory, #849)
Run on the concatenated LaTeX source of sec/*.tex (as plain text, so LaTeX markup may add noise).

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 12: PCD (the initials before its parentheses do not spell it)
- Body, line 195: mIoU (the initials before its parentheses do not spell it)
- Body, line 1096: IoU (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 2 | Not defined | LLM | 6 |
| Body | 17 | Not defined | EMBER | 62 |
| Body | 17 | Not defined | QA | 22 |
| Body | 21 | Not defined | GQA | 11 |
| Body | 21 | Not defined | NExT | 7 |
| Body | 21 | Not defined | STA | 28 |
| Body | 31 | Not defined | VL | 20 |
| Body | 67 | Not defined | KV | 7 |
| Body | 77 | Not defined | KTV | 2 |
| Body | 78 | Not defined | AdaTP | 2 |
| Body | 78 | Not defined | SnapKV | 4 |
| Body | 79 | Not defined | M3ID | 2 |
| Body | 79 | Not defined | VCD | 6 |
| Body | 103 | Not defined | BOLT | 3 |
| Body | 104 | Not defined | AKS | 1 |
| Body | 104 | Not defined | FOCUS | 1 |
| Body | 104 | Not defined | MDP3 | 1 |
| Body | 112 | Not defined | CLS | 1 |
| Body | 114 | Not defined | MMR | 1 |
| Body | 114 | Not defined | SemVID | 26 |
| Body | 115 | Not defined | LGTTP | 2 |
| Body | 125 | Not defined | HERMES | 18 |
| Body | 126 | Not defined | ReKV | 1 |
| Body | 132 | Not defined | ICD | 1 |
| Body | 133 | Not defined | PAI | 2 |
| Body | 134 | Not defined | SEASON | 1 |
| Body | 140 | Not defined | FV | 3 |
| Body | 141 | Not defined | REZE | 1 |
| Body | 144 | Not defined | TODO | 19 |
| Body | 173 | Not defined | GQ | 18 |
| Body | 338 | Not defined | SigLIP | 11 |
| Body | 1036 | Not defined | AUROC | 3 |
| Body | 1241 | Not defined | mIoP | 1 |
