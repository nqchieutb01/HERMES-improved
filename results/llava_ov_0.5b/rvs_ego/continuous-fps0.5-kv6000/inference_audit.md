# HERMES RVS-Ego inference audit

Audited 2026-09-14. Existing predictions and evaluations are from the original,
uncorrected inference run; the code corrections below have not been benchmarked
on the full dataset.

## Findings

The most actionable findings are an incorrect pseudo-query attention forward
and a middle-layer recency-weight mismatch. The headline FPS/cache settings are
correct. Neither the magnitude of these defects' effect on accuracy nor the
effect of changing the judge can be established without controlled reruns.

### 1. Incorrect compression attention under SDPA/FlashAttention — fixed

`inference/llavaov_hermes.py::_compute_attention_scores_manually` previously
initialized `hidden_states` from the pseudo-query embeddings, then reused those
embeddings for every decoder layer. It computed Q/K/V projections but never
applied the attention output projection, residual connections, or MLP to produce
the next layer's hidden states. It also applied softmax without a causal mask,
allowing earlier pseudo-query tokens to attend to future pseudo-query tokens.

Consequently, the attention scores used to select visual memory were not the
backbone's actual attention scores. The error affects local, global, and mixed
guidance queries. Deeper layers are particularly exposed because their scores
depend on attention rather than recency alone.

Evidence tying this to the saved run:

- Before editing, the inference file's SHA256 was
  `45fee483fbc37fe65433c84146eba42fc18dae40f6878c7a2f50fd8e55a387f0`,
  exactly matching `execution_manifest.json`.
- The installed Transformers source is commit
  `66bc4def9505fa7c7fe4aa7a248c34a026bb552b`, matching `requirements_llava.txt`.
- Running the installed library's automatic attention selection against the
  actual checkpoint configuration selects `sdpa`. A small Qwen2 instantiation
  likewise selects `Qwen2SdpaAttention`. The loader does not override this.
  The historical log did not explicitly record the backend, so this is a
  reconstruction from the matching code/configuration and current environment.

The fix reconstructs causal attention and advances hidden states through each
decoder layer. It preserves the persistent visual KV cache and uses physical
cache indices for causal masking, separately from logical RoPE positions.
Future loads now log the attention backend and hierarchy settings.

### 2. Middle-layer recency weighting differs from the paper — fixed

[Paper Eq. (3)](https://arxiv.org/pdf/2601.14724) specifies
`omega = 0.75 - 0.6 * progress`; the saved run used
`omega = 1.0 - 0.9 * progress` in `prune_kv_cache_by_attention`.
This makes the first middle layer entirely recency-driven before smoothing,
instead of assigning 25% weight to attention. The implementation now follows
the equation. This is a discrepancy in the repository baseline, not a change
introduced by the continuous-run script.

For the 24-layer 0.5B checkpoint, the active loader assigns layers 0–1 to shallow
memory, 2–15 to middle memory, and 16–23 to deep memory. Before the fix, middle
recency weights ran from 1.0 to about 0.1643; they now run from 0.75 to about
0.1929. The constructor's separate `short_term_ratio = 0.3` is misleading but
inactive in this loader: `load_model` bypasses that constructor and sets 0.1.

### 3. Evaluation protocol remains different

The matching [paper Table 2](https://arxiv.org/pdf/2601.14724) row is
**LLaVA-OV-0.5B + HERMES (6K): 53.0% accuracy, 3.8 score**.

| Evaluation of the saved predictions | Accuracy | Mean score |
| --- | ---: | ---: |
| Local Qwen judge, previous rubric, all 1,465 questions | 42.53% | 2.406 |
| Local Qwen judge, original HERMES rubric, all 1,465 questions | 41.77% | 2.391 |
| Original rubric, three open-ended task categories only, 723 questions | 23.24% | 1.750 |
| Exact binary-reference subset, 698 questions | 63.32% | N/A |

The numerical difference between the paper and original-rubric local result is
11.23 percentage points, but it combines inference/data differences with judge
differences. The paper uses GPT-3.5-turbo-0125; these evaluations use
Qwen3.8-27B-FP8. Reusing the original rubric does not make the judge identical.
The directory named `original_prompt` preserves the semantic rubric but still
requests schema-constrained JSON (`correct`/`score`) instead of the repository's
Python dictionary (`pred`/`score`), changes the example score, and caps judge
output at 128 rather than 300 tokens. It is not byte-for-byte prompt parity.
Changing only the rubric decreased local overall accuracy by 0.75 points; it
did not close the gap. Subset metrics must not be compared to the full-table row.

`What event order` is the weakest category: 17/257 correct (6.61%) with the
original rubric. Binary answers have a strong yes bias: 352/380 exact yes
references are correct, versus 90/318 exact no references. These observations
identify useful ablation targets; they do not prove the cause of the errors.

## Settings and data checks

The following agree with the paper's setup or stated method: 0.5 FPS, 6,000
visual tokens per layer, 16-frame chunks, FP16, greedy decoding, 10%/60%/30%
layer partition, smoothing weights 0.1/0.3/0.4, lazy streaming re-indexing,
and the local/global guidance templates. See paper Sections 3–4 and Appendices
B–C. Integer layer rounding explains the exact partition above.

Additional checks against the saved artifacts and code:

- All 10 video IDs and every original conversation's order, content, and end
  timestamp match `data/rvs/ego/ego4d_oe.json`; there are 1,465 questions.
- Visual cache resets occur between videos. Predictions enter `conv_history`;
  only the last question/answer is used to condition later compression prompts.
  Full conversation text is not retained in the visual KV cache.
- The logs show 992 history-conditioned compression-query pairs and 134 generic
  pairs, as recorded by the existing comparison report.
- Cache lengths after answering are 6,013: 6,000 visual/summary tokens plus
  13 prefix tokens. This is not an accidental 13-token visual-budget overrun.
- Generation uses greedy selection, a 256-token answer cap, and repetition
  penalty 1.1. The latter two come from repository code; exact equivalence of
  those settings to the paper is not established.
- Inputs are existing 1-FPS JPEG sequences sampled every second image. Their
  frame count and annotation mapping are validated, but exact image/timestamp
  equivalence to decoding the original videos is not established.
- The current judge script differs from the original execution-manifest hash;
  use each judge directory's own manifest and prompt version when comparing
  scores. The original-rubric re-evaluation is a separate recorded evaluation.

## Secondary issues to keep separate from the main fixes

1. **Lazy re-indexing can happen after crossing the positional limit.**
   The threshold is `max_position_embeddings - 1024`, checked after encoding a
   chunk. A full chunk contains 3,136 visual tokens. Across the saved logs,
   41 of 119 compaction triggers have maximum positions at or above 32,768;
   the largest is 34,833. RoPE tables are dynamically enlarged, so this need
   not crash, but those prefills exceed the configured position range before
   compaction. Its accuracy impact is unmeasured. A separate follow-up should
   compact before a chunk would cross the limit and account for answer length.

2. **Compression scoring runs even below budget.**
   `pseudo_forward` computes three attention passes and ranking before checking
   whether pruning is needed. This adds processing cost, especially after the
   corrected full decoder computation. It is a throughput issue, not an
   established explanation of the accuracy gap.

3. **The printed TTFT is not a clean first-token measurement.**
   It is printed after the first generated token is fed back through the model,
   so it includes an extra decode step. CUDA timing is not explicitly
   synchronized. Do not directly compare these log timings to paper TTFT.

4. **Switching to eager attention is not a verified workaround.**
   The alternate `pseudo_forward` branch passes legacy caches with
   `use_cache=False`; the pinned Qwen2 implementation only converts legacy
   caches when `use_cache=True`. This branch needs separate cache-handling
   validation. The corrected manual path targets the selected SDPA backend.

## Validation and next experiment

`scripts/test_llava_attention.py` uses a small, randomly initialized Qwen2 on
CPU and compares against the pinned Transformers eager decoder. Before the
fix, full-forward equivalence failed at all three layers and the causal test
failed. After the fix, all four tests pass:

- Per-layer attention matches an ordinary full decoder forward.
- Future query tokens cannot change earlier attention.
- Scoring leaves the visual cache unchanged.
- Attention matches with different logical query positions at each layer.

Command:

```bash
PYTHONPATH=. .venv/hermes/bin/python scripts/test_llava_attention.py
```

`git diff --check` also passes. These tests establish attention correctness on
the small CPU fixture; they do not establish restored benchmark accuracy or
FP16/GPU throughput.

The next controlled experiment should use a new output directory and the same
frames, question grouping, checkpoint, and original-rubric judge. Compare the
saved baseline against attention-only correction, then attention plus the
Eq. (3) correction. Treat position-boundary handling as a separate ablation.
The current working tree includes both main corrections. Existing result
files have not been regenerated or relabeled as corrected results.
