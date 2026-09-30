# Why grounded QA fails under pruning: phase 15 diagnosis

Qwen3-VL-8B on S-EMBER grounding (475 questions, 300 videos). Uniform 64 frames unless noted; answers judged
with the official S-EMBER judge prompt; ± values are 95% paired bootstrap intervals (`*` = significant).
Scripts: `logs/phase15/diagnose_existing.py` (no GPU, existing runs), `logs/phase15/diagnose_gpu.py`
(phase 15 runs), `logs/phase15/diagnose_b.py` (phase 15b runs), `logs/phase15/label_audit.py`.
Techniques borrowed from the literature: grounding heads and the perception-generation gap
(MLLMs Know When Before Speaking, arXiv 2605.21954), positional/recency bias of attention pruning
(Attention Debiasing for Token Pruning, arXiv 2508.17807), blind and shuffled-frame tests (VBenchComp,
arXiv 2505.14321), evidence-position analysis (needle-in-a-haystack; S-EMBER recency), label audit (TimeLens).

## 1. What limits performance most: the model, not the memory

| Evidence | Result |
|---|---|
| Oracle pruning (keep gold-interval frames first; ≥75% of other frames get no token) | Equals unpruned: 10% timeline GQ@0.5 8.4 vs 8.6, mIoU 27.9 vs 28.0; official 4.6 vs 4.4 |
| Oracle vs the pruners at 10%, timeline prompt | vs HERMES: GQ@0.5 +4.4*, mIoU +9.6*; vs stratified: mIoU +4.2*, GQ +2.3 (edge); vs random: +1.3 GQ, +2.1 mIoU (n.s.) |
| Questions no run answers correctly (100 judged runs) | 36.6%; never GQ@0.5: 61.7% |
| Blind (no video) | Accuracy 6.9% official / 10.5% timeline, vs 13.5 / 18.5 with video: about half of correct answers come from the language prior; grounding needs video (blind mIoU 3.4) |
| Counting | Undercounts 69% of the time even unpruned (median 3 vs gold 5); count = number of "Seen" lines in 79% of answers |
| Answer right vs interval right | ~20% of questions have IoU ≥ 0.5 but a wrong answer |

Pruning only costs what selection throws away: a perfect selector at 5–10% of tokens loses nothing. Everything
above that ceiling is the model's perception and reasoning.

## 2. Why HERMES pruning collapses: recency concentration

| Pruner | Share of kept tokens in the last 10% of frames | Top 6 frames hold | Median predicted start / question time |
|---|---|---|---|
| HERMES 10% | 68% | 68% | 0.10 |
| HERMES 5% | 75% | 77% | 0.04 |
| hermes_exact 10% (probes propagated through every layer) | 74% | 74% | 0.01 |
| stratified 10% | 9% (even across deciles) | 9% | 0.36 |

HERMES's layer-dependent recency term (pure recency in early layers) plus attention's recency bias put about
70% of the budget on the final ~6 frames. The rest of the timeline keeps a sliver, and the model's intervals
fall back to the video start. Scoring with properly propagated queries (`hermes_exact`) makes it worse
(mIoU −11.3* at 10%), so the embedding-level query approximation is not the cause; the recency bias is. Gold
frames are kept no more than other frames by any pruner (HERMES 13.4% vs 11.0%): no selector is evidence-aware.

## 3. How the model tells time: it reads the timestamp text, with a strong early prior

| Test | Result |
|---|---|
| Remove the timestamp text | mIoU 26.9 → 5.0, GQ@0.5 4.4 → 0.6: without text stamps the model cannot ground |
| Shift every stamp by +200 s | 56% of predictions follow the shift; mIoU after undoing it 20.1 (vs 26.9) |
| Permute stamps only (frames in order) | 27% of predictions land at the wrong stamps shown on gold frames, 7% at the true time |
| Permute frames together with their stamps | Accuracy unchanged; mIoU −3.9* (official), −7.6* (timeline): order/position is also used |
| Early layers | 45% of video attention goes to timestamp text (≈1% of video tokens); 8% in late layers |
| "Seen" timestamps | Only 12–23% copy a shown stamp exactly (median 1.4 s off); a "0.0 s" moment that was never shown opens 18% of answers unpruned, 41% at HERMES 5% |

## 4. Where it fails: primacy and "lost in the middle"

| Evidence | Result |
|---|---|
| Question attention by frame position (late layers) | U-shaped: first decile 14.4%, last 16.1%, middle deciles ~7% |
| Frames start at half the question time | 59% of predicted intervals start in the first 10% of the shown window; 38% start before it (times never shown) |
| Accuracy/grounding by evidence position | mIoU 40 (evidence in first quarter) → 15 (last quarter) unpruned; 27 → 4 at HERMES 5% |
| Interval errors | Too short (median 0.5× gold length) and early; "disjoint, before gold" grows from 25% (unpruned) to 49–60% (HERMES 5%) |
| Official prompt stock answers | 26% of stated durations are "10 s", 18% "90 s" |

## 5. Perception vs generation: grounding heads exist but do not decide the output

| Evidence | Result |
|---|---|
| Real-question attention (all heads) | Barely targets evidence: median lift 1.0–1.16 over the gold frame share |
| Grounding heads (chosen on half the videos, tested on the other half) | Layers 20–21; lift 2.4 while writing timestamps vs 1.2 for all heads |
| Same heads when the output interval is right vs wrong | 2.38 vs 2.13 (timestamp rows), 1.41 vs 1.42 (prompt rows); 28% of wrong outputs still have heads favouring gold |
| Attention to video while writing timestamps | 5.7% of attention (vs 9% while reading the prompt): timestamps are written mostly from text context |
| Under pruning | Grounding-head lift drops to 1.55 (HERMES 10%) and 1.38 (stratified 10%) from 2.42 |

## 6. Labels

Inter-annotator interval IoU 0.76 (location 0.50, duration 0.86, counting 0.81); duration answers differ by a
median 7%. Scoring against the best-matching annotator raises uniform-64 mIoU 26.7 → 30.8, R@0.5 23.6 → 28.6.
Labels are reliable except for location, where they are ambiguous.

## Implications for methods

1. Coverage-preserving, evidence-aware selection closes the pruning gap (oracle = unpruned); the gain over
   random is small (selection is mostly about not starving the timeline), so pruning alone cannot beat
   unpruned.
2. Debias recency in HERMES scoring (the literature's positional debiasing) — the main cause of the collapse.
3. Timestamps are the model's only clock: protect them, and fix the early/0 s prior (primacy) in decoding or
   prompting.
4. Grounding heads (layers 20–21) carry a usable localisation signal that the output does not fully use:
   read-then-regenerate (crop to what the heads see) is a direct, training-free follow-up.
5. The largest headroom is reasoning (counting, stock durations, language prior), not memory.
