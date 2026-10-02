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

## 4. Results so far (random pruning to 10% of the tokens, timeline prompt)

| | Acc. | mIoU | R@0.5 | GQ@0.5 | first moment at 0 s | interval in first 5% |
|---|---|---|---|---|---|---|
| Random 10% | 15.2 | 25.8 | 24.2 | 7.2 | 22.5% | 38.0% |
| **+ PCD (α = 0.5)** | 16.2 | **32.1** | **31.2** | **10.1** | **11.8%** | **19.4%** |
| Unpruned 64 frames | 18.5 | 28.0 | 26.7 | 8.6 | 26.7% | 41.4% |

PCD vs its baseline: mIoU +6.2 [+3.9, +8.5], GQ@0.5 +2.9 [+0.2, +5.7] (both significant), Acc. +1.1 (n.s.).
PCD at 10% of the tokens exceeds the unpruned model on mIoU (+4.0, significant). P1 and P2 hold. Budgets, selectors,
streaming, the official prompt and hyperparameters (P3–P4) are in `logs/phase18/report.md` as they complete.

**P5 (counterfactual choice).** Contrasting with the same frames but permuted timestamps gives no grounding gain
(mIoU +0.9 / +0.5 at α = 0.5 / 1) and makes 17% of answers run on without an answer line: the useful counterfactual is
the prior, not a corrupted memory.

## 5. What did not work (reported as negative results)

| Method | Idea | Result at 10% |
|---|---|---|
| Visual-attention gain (γ = 4, PAI-style) | rebalance attention toward visual tokens while decoding | mIoU +1.3, GQ +0.8 (n.s.) |
| Relevance-guided temporal importance sampling | allocate frames and tokens by grounding-head relevance | mIoU −1.5, Acc. −1.7 (n.s.); head relevance localises weakly (median lift 1.07) |
| Coverage-then-zoom, gold window | dense frames in the gold interval at the same budget | mIoU +2.4, Acc. −0.7: re-allocation at a fixed budget is near its ceiling |
| Coverage-then-zoom, self window | dense frames around the model's own first answer | Acc. +4.6 (significant), mIoU +0.4; ablations pending, mechanism unclear (gain appears even when the window misses the evidence) |

## 6. Limitations

- One model (Qwen3-VL-8B) and one benchmark subset (300 videos, three question types), by design of this study.
- 3% of PCD answers keep listing moments until the token limit (the prior expects lists to stop early); the
  plausibility cut-off β is the lever, ablated in the report.
- The judge is a local Qwen3.8-27B with the official S-EMBER judge prompt, not the official Gemini judge.
