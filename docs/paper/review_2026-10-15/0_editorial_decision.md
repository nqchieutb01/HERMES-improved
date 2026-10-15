# Editorial Decision Letter — "Keep the Clock, Contrast the Prior" (CoCo)

**Decision: Major Revision** (unanimous: Journal-Fit Reviewer, R1 Methodology, R2 Domain, R3 Cross-disciplinary all recommend Major Revision; Devil's Advocate raises 2 CRITICAL, 9 MAJOR, 7 MINOR).

**Review status disclosures**
- Venue: CVPR 2026 is author-stated; no confirmed criteria manifest → `criteria_binding_unavailable`. Venue comments are field-general.
- Calibration: `NOT_CALIBRATED`.
- Panel provenance: five role-separated seats, each a fresh subagent context with no access to the others' reports before committing; **all five on the same model family and provider** (Claude, Anthropic), no cross-model seat → errors may be correlated. Persona separation is not independence.
- Protocol: lightweight run — single-call reports, not the two-call paper-blind sprint contract; no `check_phase_conformance` / `check_panel_synthesis` replay.
- Evidence base: manuscript at commit f839c4d + `recent_results.md` (post-draft runs, 2026-10-14/15). Several post-draft numbers are copied from run logs and pool reports, not re-scored by the panel.

## 1. Summary of the submission
CoCo targets temporal grounding when a Video-LLM's visual memory is pruned to ~10% of its tokens. It diagnoses (i) QA-oriented pruners losing temporal coverage and (ii) a video-free "temporal prior" filling the gaps, then proposes Coverage (equal per-frame quota filled by question attention, per layer) and PCD (contrast against a no-video run on time values, adaptive α, answer reserve), evaluated against video-free floors on S-EMBER, QVHighlights, ActivityNet and Charades-STA with Qwen3-VL-8B.

## 2. Consensus strengths (≥3 seats)
- **Video-free floors and honest limits** (EIC, R1, R2, R3, DA-observations): best-constant and length-rescaling floors, closed/open split, negative-results table, reporting where the method fails.
- **Careful diagnostics** (R1, R3, DA-observations): causal evidence-removal test with matched control, gold-informed oracles, offline replay of decision rules, no-video counterfactual chosen by ablation against VCD-style noise.
- **Coverage is the robust component** (EIC, R1, R2, DA-observations): beats random under the standard one-line prompt on both Qwen3-VL-8B (+3.5\*) and TimeLens-8B (+2.0\*) on QVH-TimeLens; +3.1\* Acc@QA on NExT-GQA for both models; matches unpruned TimeLens-8B with ~24% of tokens (67.3 vs 67.1).
- **Transfer to a grounding-tuned model on S-EMBER** (EIC, R2, DA-observations): TimeLens-8B reproduces the ordering, CoCo 43.9 vs 33.1 unpruned (answers not yet judged).

## 3. Consensus issues

### C1 — The headline gain is measured against a prompt-handicapped baseline (CRITICAL; all 5 seats)
All main-table rows use the reasoning-style timeline prompt. On the same 513 QVH-TimeLens queries with the same Coverage memory, the timeline prompt gives 30.7 mIoU, timeline + PCD + reserve 43.9, and the one-line TimeLens prompt 61.7 (paired −17.8\*). Unpruned Qwen3 under the standard prompt reaches 59.6 (QVH), 48.5 (Charades), 47.1 (ActivityNet), matching published numbers within 0.3. Consequences:
- "Nearly doubles the uncompressed model" (abstract, §1, §6.1(i)) does not survive a standard-prompt baseline.
- "The base model stays below the video-free floor on ActivityNet and Charades" (abstract, §6.1(iii), §7) is a property of the protocol (prompt + low-res re-encodes + subsets), not of the model.
*Caveat (DA, R2):* the 29.1-vs-~60 cross-table comparison is also confounded by subset (500 vs TimeLens-Bench refined annotations), resolution and fps; the clean evidence is the 513-query paired comparison.
**Editorial adjudication of DA-CRITICAL-1: validated.** It blocks acceptance until resolved.

### C2 — PCD's effect is format-dependent; "the temporal prior" may be a prompt artefact (CRITICAL by DA; Major by EIC, R2, R3)
PCD gives +0.4 under the direct-answer prompt (draft §6.4), −0.1 / −0.3 (n.s.) under the one-line prompt, and a format change alone gives +7.4\*. The "instruction does not remove it" test (§3.2) was run inside the timeline format. The no-video prior has not been measured under other formats.
**Editorial adjudication of DA-CRITICAL-2: validated as unresolved** (the evidence is suggestive, not conclusive: the prior may exist under every format but only bite where the format exposes it). It blocks acceptance until the format study is run.
Partial mitigation: on S-EMBER grounded QA, where a reasoning answer is natural, PCD transfers to TimeLens-8B (+10.2 to random, +6.7 to Retrieve + Coverage).

### C3 — Efficiency claims without a cost table (Major; EIC, R1, R3, DA)
Coverage scores after a full prefill (3–4 scoring passes in the current code), PCD adds a second decoding pass, and the timeline answer allows up to 384 tokens against ~15 for a one-line answer. No latency, FLOPs, peak memory or time-to-first-token is reported. R3: nothing is saved in prefill or TTFT; the method fits "reduce the cached memory read by the decoder" or "answer many questions from one memory", which needs to be the stated use case.

### C4 — Baseline comparisons are not like for like (Major; R1, EIC, DA)
FastVID, VidCom², FlashVID and SemVID prune at the input (`video_qa/hermes_vqa.py` l.370–413); Random and Coverage prune the KV cache after a full prefill with per-layer sets. That confounds "every pruner falls below random". No pruner is checked against its published numbers. The KV-eviction baselines (SnapKV, PruneVid) and the "question global top-k, no quota" ablation, which would isolate what the equal quota adds, are still TODO. Under the standard prompt, one unmatched cell already reverses the ordering (SemVID 58.8 vs Random 58.2, QVH-TimeLens).

### C5 — Non-standard evaluation data and test-set adaptivity (Major; R1, R2, EIC, DA)
- 500-query subsets and low-resolution re-encodes (QVH, ActivityNet), so no number is comparable to published work.
- "Design on S-EMBER, run once frozen" is contradicted by later choices: the m:ss parser, ActivityNet's 30% budget after the collapse at 10%, the Charades-only retrieval gate, the frames rule derived from QVH and ActivityNet, and reserve on/off variants.
- The token-floor collapse may come from the re-encodes (R2).

### C6 — Statistics (Major; R1, DA)
- Batch sizes 1, 2 and 8 are mixed within comparisons; the same configuration scores 32.6 and 32.1 in two tables; the same +2.9 is starred in one table and not in the other.
- About 60 uncorrected tests; question-level bootstrap only (no video clustering).
- With intervals removed, boundary cases are hidden (−1.5 with an interval ending at −0.0; +7.6 on ~157 questions with [2.5, 12.6]), and the intro still promises confidence intervals.

### C7 — Related-work gaps and novelty framing (Major; R2, R3, EIC)
- No grounding-tuned Video-LLMs are cited: TimeChat, VTimeLLM, LITA, TRACE, Time-R1, TimeLens.
- Missing decoding lineage, although PCD's formula matches it:
  - Contrastive Decoding (Li et al., ACL 2023), the source of the plausibility cut;
  - context-aware decoding (Shi et al., NAACL 2024) and classifier-free guidance for LMs (Sanchez et al.);
  - contextual calibration (Zhao et al. 2021) and DC-PMI (Holtzman et al. 2021).
- FastV and the NExT-GQA paper are missing.
- The "floor-adjusted evaluation" contribution is largely Otani et al.'s blind baseline. Present it as adopted practice, not a contribution.
- Several suggested references are marked "to verify" by the reviewers.

### C8 — Short videos and NExT-GQA (Major; EIC, R2, DA)
On NExT-GQA (~40 s clips), 10% pruning loses ~3 Acc&GQA against unpruned on both models (bs 2 vs bs 8 confound), and Coverage ≈ Random on grounding. Together with Charades, this bounds where Coverage helps; it should be presented as a scope limit, not omitted.

## 4. Disagreements and arbitration
- **Severity of C1:** EIC/R1/R3 rate it fixable within a revision; DA calls it CRITICAL. *Arbitration:* both are right — it is fatal to the current headline, but the reframed paper (Coverage + PCD for grounded QA, standard-prompt grounding as a second track) is viable. Hence Major Revision, not Reject.
- **Is the S-EMBER headline affected?** R3: only the QVH headline is directly refuted; S-EMBER has no one-line baseline yet. *Arbitration:* S-EMBER claims stand provisionally, but they need an all-tokens baseline under a short-answer prompt and with PCD, so the comparison is against the best unpruned configuration.
- **Value of PCD:** DA and R2 suggest repositioning PCD as a repair for reasoning-style answers; R3 sees principled value but asks for lineage and an α/β sweep. *Arbitration:* keep PCD, scope it to grounded-QA / reasoning formats, and report the null results under one-line formats in a visible table.

## 5. Minor issues (single-seat or editorial)
- Intro arithmetic: "330 tokens per pair → 21,000 tokens" is wrong; 32 × 330 ≈ 10,600, and §5 says 660 per pair (R1).
- Judge model "Qwen3.8-27B-FP8" looks like a typo (R1, R2).
- S-EMBER citation is a placeholder (R2).
- `floors.py` does not compute the paired intervals the supplement attributes to it (R1).
- Length-rescaling factor fitted on S-EMBER only, while the constant floor is re-tuned per dataset (R1).
- CoCo main rows include retrieval but the PCD baselines do not (R1).
- PCD scores digit tokens rather than whole time values; consider a value-level variant (R3).
- "Near the limit of decoding" rests on replaying only the first time value (R3).
- The answer reserve partly repairs run-ons that PCD itself causes (+6.1\* on QVH); say so (DA, R3).
- The paper is 16 pages; the CVPR main text must fit in 8 (EIC).

## 6. Revision Roadmap (ordered by what unblocks acceptance)

| # | Item | Seats | Type | Concrete action |
|---|---|---|---|---|
| 1 | Re-baseline grounding under the standard prompt | all | experiment | Finish the full TimeLens-Bench sweep (Charades, QVH, ANet; both models): unpruned, Random, SemVID, Coverage at 10% and ~30%; add CoCo with and without PCD. Make this the main grounding table. |
| 2 | Format study of the prior | DA, R3, EIC | experiment | On the same S-EMBER and QVH questions, measure no-video early-start rate, leak, and PCD gain under timeline, direct and one-line prompts. Decide from this whether "temporal prior" is a model or a format property and rewrite §3.2 accordingly. |
| 3 | Rewrite claims | all | text | Drop "nearly doubles the uncompressed model" and "below the floor" as model claims; scope PCD to reasoning / grounded-QA formats; put PCD's one-line null results in a visible table. |
| 4 | Cost table | EIC, R1, R3, DA | experiment | Prefill, scoring passes, decode tokens, latency, TTFT and peak memory for unpruned, input pruners, Coverage and CoCo; state the deployment setting (decoder memory / multi-question reuse). Reduce scoring to one pass if possible. |
| 5 | Matched baselines | R1, EIC, DA | experiment | Add SnapKV/PruneVid-style KV eviction and the "question global top-k, no quota" ablation; run input pruners in a KV-cache variant or Coverage at input level; check each pruner against its published numbers. |
| 6 | Standard data and held-out protocol | R1, R2, DA | experiment + text | Full splits at native resolution; re-check the token floor there; declare which choices were made after seeing test results and validate the frames rule on a held-out split. |
| 7 | Statistics hygiene | R1, DA | analysis | One batch size within each comparison (or report its noise); video-clustered bootstrap; Holm or FDR correction for families of tests; restore intervals for boundary cases in the supplement; fix the intro's promise of CIs. |
| 8 | Related work | R2, R3, EIC | text | Add grounding-tuned Video-LLMs, decoding lineage (CD, CAD, CFG, contextual calibration, DC-PMI), FastV, NExT-GQA, and the debiasing line; reframe the floors as adopted practice. Verify every newly added reference. |
| 9 | New results into the paper | EIC, R2 | text | NExT-GQA (two-query protocol, stated) as a short-video scope result; TimeLens-8B S-EMBER after judging; Coverage under the standard prompt as the transferable finding. |
| 10 | PCD analysis | R3 | experiment | Report mean adaptive α; sweep α_max and β; try value-level scoring of times. |
| 11 | Minor fixes | various | text | Items in §5; cut to 8 pages and move diagnostics to the supplement. |

**Most efficient path:** items 1–3 decide the paper's story; 4–5 are what a CVPR reviewer will ask first; 6–11 can proceed in parallel.

## 7. Questions to the authors
1. What is the best unpruned configuration (any prompt, with or without PCD) on each dataset, and does CoCo beat it?
2. Does the no-video prior survive under a one-line prompt?
3. Which deployment benefits from pruning after a full prefill?
4. Who is the judge model, and what is S-EMBER's provenance?
5. Does the token-floor rule hold on full-resolution video?

## Attachment: Acronym Check (advisory, #849)
Run on the concatenated LaTeX source of sec/*.tex (as plain text, so LaTeX markup may add noise).

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 12: PCD (the initials before its parentheses do not spell it)
- Body, line 191: mIoU (the initials before its parentheses do not spell it)
- Body, line 1079: IoU (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 2 | Not defined | LLM | 6 |
| Body | 16 | Not defined | EMBER | 59 |
| Body | 16 | Not defined | QA | 18 |
| Body | 17 | Not defined | STA | 19 |
| Body | 28 | Not defined | VL | 5 |
| Body | 64 | Not defined | KV | 7 |
| Body | 74 | Not defined | KTV | 2 |
| Body | 75 | Not defined | AdaTP | 2 |
| Body | 75 | Not defined | SnapKV | 4 |
| Body | 76 | Not defined | M3ID | 2 |
| Body | 76 | Not defined | VCD | 6 |
| Body | 99 | Not defined | BOLT | 3 |
| Body | 100 | Not defined | AKS | 1 |
| Body | 100 | Not defined | FOCUS | 1 |
| Body | 100 | Not defined | MDP3 | 1 |
| Body | 108 | Not defined | CLS | 1 |
| Body | 110 | Not defined | MMR | 1 |
| Body | 110 | Not defined | SemVID | 22 |
| Body | 111 | Not defined | LGTTP | 2 |
| Body | 121 | Not defined | HERMES | 18 |
| Body | 122 | Not defined | ReKV | 1 |
| Body | 128 | Not defined | ICD | 1 |
| Body | 129 | Not defined | PAI | 2 |
| Body | 130 | Not defined | SEASON | 1 |
| Body | 136 | Not defined | FV | 3 |
| Body | 137 | Not defined | REZE | 1 |
| Body | 140 | Not defined | TODO | 16 |
| Body | 169 | Not defined | GQ | 17 |
| Body | 334 | Not defined | SigLIP | 11 |
| Body | 1019 | Not defined | AUROC | 3 |
