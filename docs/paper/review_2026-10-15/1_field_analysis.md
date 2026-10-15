# Field Analysis Report

## Paper Basic Information
- **Title**: Keep the Clock, Contrast the Prior: (CoCo) temporal grounding with compressed visual memory
- **Source**: docs/paper/latex/main.tex + sec/*.tex (CVPR 2026 author kit, review mode); compiled main.pdf, 16 pages incl. appendix
- **Target venue**: CVPR 2026 (author-stated in the draft header and by the user; no formal Review Target Context / criteria manifest → `criteria_binding_unavailable`; venue-fit comments are field-general CVPR expectations, not bound criteria)

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Computer vision — video understanding with multimodal LLMs (temporal grounding, efficient inference) |
| Secondary Disciplines | Efficient ML (KV-cache / visual-token pruning); decoding methods for LLMs (contrastive decoding, hallucination mitigation); evaluation methodology / benchmark bias |
| Research Paradigm | Quantitative, empirical |
| Methodology Type | Experimental (training-free method; ablations; paired bootstrap; diagnostic/causal interventions) |
| Target Journal Tier | Top-tier CV conference (CVPR); field-general observation only |
| Paper Maturity | Revised draft / near pre-submission: complete structure, many results, but visible TODOs (cost table, CIs on some cells, event timeline with CoCo, open-interval decoding), one model only, 500-question subsets |

## Reviewer Configuration Cards

### Reviewer Configuration Card #1
**Role**: EIC
**Display role**: Journal-Fit Reviewer
**Identity Description**: CVPR area chair for video understanding & multimodal LLMs; has chaired sessions on long-video LLMs and efficient video inference.
**Review Focus**:
  1. Is the contribution novel relative to token pruning (FastVID, VidCom², FlashVID, SemVID, SnapKV, PruneVid) and contrastive decoding (VCD, M3ID, ICD), or a combination of known parts?
  2. Is the story (diagnosis → method → floors) compelling and well scoped for CVPR readers; are claims proportionate to evidence (one model, 4 + new benchmarks)?
  3. Presentation: figure/table density, 16-page length vs 8-page limit, what goes to the supplement.
**Will particularly care about**: whether headline numbers (QVHighlights 54.8 R@0.5) are on standard protocols and comparable to published numbers.
**Possible blind spots**: statistical details of paired tests; implementation correctness of baselines.

### Reviewer Configuration Card #2
**Role**: Peer Reviewer 1
**Display role**: Peer Reviewer 1 — Methodology
**Identity Description**: Researcher in ML evaluation methodology and efficient inference, specialising in experimental design for training-free methods (ablations, baselines re-implementation fidelity, multiple-comparison control, paired bootstrap, seed/decoding noise).
**Review Focus**:
  1. Baseline fidelity: are FastVID/VidCom²/FlashVID/SemVID re-implementations faithful (same frames, same budget accounting, published-number sanity checks)? Are budgets matched in tokens and in FLOPs/latency?
  2. Design-on-S-EMBER then "run once frozen" claim; test-set tuning of floors; 500-question subsets; decoding/batch-size noise (Acc ±2); significance reporting after the CIs were removed.
  3. Cost: PCD doubles decoding passes and Coverage needs a full-prefill scoring pass — is compression meaningful if the full memory must be built first?
**Will particularly care about**: whether the comparisons are apples-to-apples (prompt, parser, frames, budget) and whether the missing cost table undermines the efficiency framing.
**Possible blind spots**: positioning relative to grounding-specialist literature.

### Reviewer Configuration Card #3
**Role**: Peer Reviewer 2
**Display role**: Peer Reviewer 2 — Domain (video temporal grounding)
**Identity Description**: Senior researcher in video temporal grounding / moment retrieval (Charades-STA, ActivityNet Captions, QVHighlights, NExT-GQA), familiar with grounding-tuned Video-LLMs (TimeChat, VTimeLLM, TRACE, TimeLens-style models, Time-R1) and benchmark-bias literature (Otani et al. "Uncovering hidden challenges", Yuan et al. distribution bias, Charades-CD/ANet-CD).
**Review Focus**:
  1. Literature coverage: grounding-specialised Video-LLMs and benchmark-bias/debiasing work; whether "video-free floors" are novel or known (blind baselines).
  2. Are results competitive with the state of the art on standard splits/metrics, and does the method help grounding-tuned models (e.g. TimeLens-8B) or only a weak base model below the constant floor?
  3. Grounded QA (NExT-GQA, S-EMBER) — are metrics and protocols standard?
**Will particularly care about**: the base model being below a constant floor on ANet/Charades — does the paper study a regime the community cares about?
**Possible blind spots**: decoding theory; efficiency engineering.

### Reviewer Configuration Card #4
**Role**: Peer Reviewer 3
**Display role**: Peer Reviewer 3 — Cross-disciplinary (LLM decoding / calibration & streaming systems)
**Identity Description**: Researcher in LLM decoding and calibration (contrastive/classifier-free guidance decoding, prior correction, label-bias calibration such as contextual calibration / domain-conditional PMI) with experience deploying streaming assistants (egocentric/AR).
**Review Focus**:
  1. Relation of PCD to PMI decoding, classifier-free guidance, contextual calibration (Zhao et al. 2021), DC-PMI (Holtzman et al.), and to VCD/M3ID; is the adaptive α principled?
  2. Practicality for streaming: per-question full prefill, no write-once memory; latency of two decoding passes; when would a system use this?
  3. Generality beyond time tokens (counts, locations) and to other models.
**Will particularly care about**: whether "the prior" is a model property or a prompt/format artefact (timeline prompt, m:ss formats).
**Possible blind spots**: benchmark specifics.

(Fifth seat: fixed Devil's Advocate — no card.)

## Review Strategy Recommendations
- Recent results (NExT-GQA, TimeLens-8B, TimeLens-Bench) are not yet in the paper; reviewers should read `recent_results.md` and judge how they change the claims.
- Tension to watch: R1 (cost/fairness) vs EIC (novelty) on whether a training-free method that needs a full prefill is "compression".
