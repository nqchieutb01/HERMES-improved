# Prior leakage in pruned video memory, and prior-contrastive decoding

Qwen3-VL-8B, S-EMBER grounded QA (475 questions on 300 videos), training-free. Full, auto-generated result tables:
`logs/phase18/report.md` (`python3 logs/phase18/report.py`). Diagnosis behind the method: `docs/phase15_diagnosis.md`.

## 1. Problem and finding

Grounded video QA asks for an answer and the time interval of its evidence. When the visual memory is pruned to fit
a token budget, both degrade. The diagnosis (phases 15–16) shows that the main loss is not which tokens are kept:
a perfect selector at 10% of the tokens only matches the unpruned model. Instead, the model fills the gaps left by
pruning with its **temporal language prior**:

- Asked without any video, the model places evidence at the start of the video: 89% of its first `Seen` moments are
  at 0 s and 90% of its intervals start in the first 5% of the window.
- That prior leaks into answers with video. With HERMES pruning at 5%, 41% of answers open with a "0.0 s" moment
  that was never shown (18% unpruned); 38–41% of intervals start in the first 5% of the window even when frames
  start halfway through the window; with timestamps shifted the model keeps writing early times.
- The leak grows as evidence shrinks, which is what a mixture p(y | memory) ∝ evidence · prior predicts.
- The leak is strongest when the model reasons in timestamped steps (timeline prompt): 38% of its intervals start
  in the first 5% of the video at 10% of the tokens, 56% with HERMES pruning.

## 2. Method: prior-contrastive decoding (PCD)

At each decoding step, the next token maximises

    (1 + α) · log p(y_t | M, q, y_<t)  −  α · log p(y_t | ∅, q, y_<t),

restricted to tokens with p(y_t | M, q, y_<t) ≥ β · max p(· | M, q, y_<t) (adaptive plausibility, as in Visual
Contrastive Decoding, Leng et al., CVPR 2024). M is the pruned visual memory; ∅ is the same prompt with no visual
tokens (the prior). The contrast removes what the model would say regardless of the video, and keeps what the
memory supports. Defaults: α = 0.5, β = 0.1; both ablated.

- **No change to the memory or the selector**: PCD is applied on top of any pruning (HERMES, random, stratified,
  streaming).
- **Cost**: the counterfactual branch holds only the prompt (no visual tokens), so its KV cache is a few hundred
  tokens; decoding costs one extra small forward per generated token.
- **Not post-processing**: the model's own token choices change during generation; the answer and the interval are
  produced together.

Implementation: `contrastive_answering` in `inference/qwen3vl_hermes.py`; the counterfactual memory in
`HermesVQA._counterfactual_memory` (`video_qa/hermes_vqa.py`); option `run.contrastive_mode=blind`.

## 3. Predictions and how they are tested

| Prediction | Test |
|---|---|
| P1. PCD reduces prior leakage | rate of 0 s first moments and early interval starts, with vs without PCD |
| P2. PCD improves grounding without hurting answers | Acc., mIoU, R@0.5, GQ@0.5, paired against the same memory |
| P3. The gain is larger when the memory is smaller | budgets 5 / 10 / 25 / 100% |
| P4. The gain holds across selectors and memory types | HERMES / random / stratified pruning; HERMES streaming at 4k / 6k |
| P5. The prior (no video) is the right counterfactual | contrast with a temporally corrupted memory (permuted timestamps) instead |

## 4. Results

Tables: `tables/pcd_main.tex`, `tables/pcd_ablation.tex` (`python3 tables/make_pcd_tables.py`); every paired comparison
with intervals: `logs/phase18/report.md`. Timeline prompt; Δ is paired against the same memory; * = 95% interval excludes 0.

| Memory | Acc. | mIoU | R@0.5 | GQ@0.5 | Early start (%) |
|---|---|---|---|---|---|
| Random 5% → +PCD | 15.8 → 15.4 (−0.4) | 23.6 → 29.2 (+5.6*) | 21.1 → 27.4 (+6.3*) | 5.9 → 9.1 (+3.2*) | 39 → 20 |
| Random 10% → +PCD | 15.2 → 16.2 (+1.1) | 25.8 → 32.1 (+6.2*) | 24.2 → 31.2 (+6.9*) | 7.2 → 10.1 (+2.9*) | 38 → 19 |
| Random 25% → +PCD | 17.7 → 18.9 (+1.3) | 28.2 → 33.0 (+4.8*) | 27.4 → 33.5 (+6.1*) | 8.2 → 11.8 (+3.6*) | 37 → 21 |
| Unpruned → +PCD | 18.5 → 19.8 (+1.3) | 28.0 → 31.5 (+3.5*) | 26.7 → 31.8 (+5.1*) | 8.6 → 11.2 (+2.5) | 41 → 23 |
| HERMES 10% → +PCD | 13.1 → 15.4 (+2.3) | 18.3 → 22.6 (+4.3*) | 16.2 → 21.9 (+5.7*) | 4.0 → 7.2 (+3.2*) | 56 → 31 |
| Stratified 10% → +PCD | 14.1 → 14.5 (+0.4) | 23.7 → 29.9 (+6.3*) | 21.7 → 29.9 (+8.2*) | 6.1 → 8.8 (+2.7*) | 36 → 17 |
| HERMES streaming 6k → +PCD | 14.3 → 14.7 (+0.4) | 22.6 → 28.1 (+5.6*) | 21.1 → 28.8 (+7.8*) | 4.8 → 7.6 (+2.7*) | 46 → 24 |
| HERMES streaming 4k → +PCD | 13.5 → 17.1 (+3.6) | 21.6 → 27.4 (+5.8*) | 20.6 → 28.4 (+7.8*) | 4.0 → 8.2 (+4.2*) | 45 → 24 |

- **P1 (leakage) holds everywhere**: PCD roughly halves early-start intervals (e.g. 38% → 19%), to below the
  unpruned model (41%). The no-video prior itself is at 90%.
- **P2 holds**: grounding improves significantly in every pruned setting (mIoU +4.3 to +6.3, R@0.5 +5.7 to +8.2,
  GQ@0.5 +2.7 to +4.2) and answer accuracy is never significantly hurt (−0.4 to +3.6).
- **P3 holds**: the gain on pruned memory (+4.3 to +6.3 mIoU) exceeds the gain on unpruned memory (+3.5).
- **P4 holds**: HERMES, random and stratified selectors and HERMES streaming at 6k / 4k all improve. Random 10% + PCD
  (mIoU 32.1, GQ@0.5 10.1) and random 25% + PCD (Acc. 18.9, mIoU 33.0, GQ@0.5 11.8) exceed the unpruned model
  (18.5 / 28.0 / 8.6) with 10–25% of its visual tokens.
- **P5 holds**: contrasting with a temporally corrupted memory (permuted timestamps) does not help (mIoU +0.9 n.s.);
  the useful counterfactual is the prior.
- **Robust to its hyperparameters**: α ∈ {0.25, 0.5, 1.0} gives mIoU +5.0 / +6.2 / +6.1, β = 0.2 gives +5.7.
- **Boundary: the official (direct-answer) prompt does not benefit** (mIoU +0.4, GQ@0.5 −0.2 n.s.). With the timeline
  prompt the model writes timestamped moments before answering, and that listing is where the early-time prior shows
  (38% early starts vs 22% with the official prompt). Timeline reasoning improves answers (phase 13) but brings the
  prior in; PCD removes it. The two are complementary, and the contribution is their combination.

## 5. What did not work (reported as negative results)

| Method (10%, random) | Idea | Result |
|---|---|---|
| Visual-attention gain (γ = 4, PAI-style) | rebalance attention toward visual tokens while decoding | mIoU +1.3, GQ +0.8 (n.s.); does not reduce the early-start prior (38 → 42%) |
| Relevance-guided temporal importance sampling | allocate frames and tokens by grounding-head relevance | mIoU −1.5, Acc. −1.7 (n.s.); the heads localise weakly (median lift 1.07) |
| Coverage-then-zoom, gold window | dense frames in the gold interval at the same budget | mIoU +2.3, Acc. −0.6 (n.s.): re-allocation at a fixed budget is near its ceiling |
| Coverage-then-zoom, self window | dense frames around the model's own first answer | Acc. +4.6* at share 0.6 and 0.8, but not robust (margin 0.5×: +0.4; gold window with the same margin: +0.8; official prompt: +1.1); no grounding gain; combined with PCD it is worse than PCD alone (mIoU 28.3 vs 32.1) |

## 6. Limitations

- One model (Qwen3-VL-8B) and one benchmark subset (300 videos, three question types), by design of this study.
- 3% of PCD answers keep listing moments until the token limit (the prior expects lists to stop early).
- PCD needs a reasoning-style answer (timeline prompt); it does not help direct answers.
- The judge is a local Qwen3.8-27B with the official S-EMBER judge prompt, not the official Gemini judge.
