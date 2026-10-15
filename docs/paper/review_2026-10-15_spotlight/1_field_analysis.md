# Field Analysis Report — round 2 (spotlight bar)

## Paper
- **Title**: Keep the Clock, Contrast the Prior (CoCo): temporal grounding with compressed visual memory
- **Source**: docs/paper/latex/main.tex + sec/*.tex (incl. new sec/6b_transfer.tex) at commit 2c84a78; compiled main.pdf (18 pages incl. appendix)
- **Supporting data**: docs/paper/analysis/results_2026-10-15.md (re-scored post-draft results with paired bootstrap)
- **Target**: CVPR 2026, and the author asks specifically what is needed to reach **spotlight / highlight level** (roughly the top 5–10% of accepted papers). No formal criteria manifest → `criteria_binding_unavailable`; venue comments are field-general.

## Field analysis
| Dimension | Result |
|---|---|
| Primary discipline | Computer vision: video understanding with multimodal LLMs (temporal grounding, efficient inference) |
| Secondary | Efficient ML (visual-token / KV-cache pruning); LLM decoding (contrastive decoding, calibration); evaluation methodology / benchmark bias |
| Paradigm / method | Quantitative, experimental; training-free method; ablations; paired bootstrap; diagnostics |
| Maturity | Revised draft: complete structure, many results, TODOs remain (full TimeLens-Bench sweep, cost table, KV-eviction baselines, judging) |

## Spotlight bar (shared by all seats)
A CVPR spotlight typically has: (1) one crisp, surprising and general insight; (2) a simple method with large, consistent gains on standard benchmarks under standard protocols, on more than one model; (3) clear, convincing evidence (strong baselines, efficiency numbers, ablations); (4) a memorable figure/teaser and a tight 8-page story. Judge the paper against this bar, not just against acceptance.

## Reviewer Configuration Cards

### Card #1 — EIC / Journal-Fit Reviewer
CVPR area chair for video understanding and multimodal LLMs who has served on award / oral-selection committees. Focus: the single insight the paper would be remembered for; whether the story is tight enough for an oral/spotlight; what to cut to reach 8 pages; title, teaser and contributions framing. Blind spot: statistical details.

### Card #2 — Peer Reviewer 1, Methodology
Researcher in evaluation methodology for efficient video-LLM inference (token pruning, KV-cache compression benchmarks). Focus: whether the evidence would convince a skeptical expert: standard protocols, matched budgets (tokens, FLOPs, latency), strongest baselines (input-level and KV-level), statistics and noise (batch size, multiple tests), reproducibility. Blind spot: novelty positioning.

### Card #3 — Peer Reviewer 2, Domain (temporal grounding)
Senior researcher in video temporal grounding and grounded VideoQA (Charades-STA, ActivityNet Captions, QVHighlights, NExT-GQA, TimeLens-Bench; grounding-tuned Video-LLMs; benchmark bias). Focus: whether results are state-of-the-art-relevant on standard benchmarks; literature coverage; whether the insight changes how the community evaluates or builds grounding under compression. Blind spot: decoding theory.

### Card #4 — Peer Reviewer 3, Cross-disciplinary (LLM decoding + deployment)
Researcher in LLM decoding/calibration (contrastive decoding, CFG, PMI, contextual calibration) with experience shipping streaming egocentric assistants. Focus: principled grounding of PCD; when the method matters in deployment (latency, memory, multi-question reuse, streaming); generality beyond time tokens and beyond one model family. Blind spot: benchmark specifics.

(Fifth seat: fixed Devil's Advocate.)
