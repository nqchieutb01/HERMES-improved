"""Time the stages of HERMES uniform inference (cuda-synchronised wall clock per stage).

Usage: python3 logs/batching/stage_timer.py <hermes_vqa args...>   (same arguments as python -m video_qa.hermes_vqa)
"""
import atexit
import collections
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/logs/", 1)[0])  # repository root
sys.argv[0] = "hermes_vqa"
from video_qa import base, hermes_vqa  # noqa: E402
from inference import batched_decoding, qwen3vl_hermes, qwenvl_hermes  # noqa: E402

T, N = collections.defaultdict(float), collections.Counter()


def timed(owner, name, label=None):
    fn = getattr(owner, name)

    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            T[label or name] += time.perf_counter() - t
            N[label or name] += 1
    setattr(owner, name, wrap)


timed(base.BaseVQA, "load_uniform_video")
timed(qwen3vl_hermes.Qwen3VL_Hermes, "encode_video_chunk")
timed(qwenvl_hermes.QwenVL_Hermes, "predict_and_compress")
timed(qwen3vl_hermes.Qwen3VL_Hermes, "question_answering")
timed(qwen3vl_hermes.Qwen3VL_Hermes, "contrastive_answering")
timed(hermes_vqa.HermesVQA, "_counterfactual_memory")
timed(hermes_vqa.HermesVQA, "analyze_a_video", "per-video loop (incl. flushes)")
timed(hermes_vqa.HermesVQA, "_flush_answers", "batched decode")
timed(batched_decoding, "prefill_job")
timed(base.BaseVQA, "analyze", "TOTAL")


@atexit.register
def report():
    print("\n=== PROFILE ===")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    for k, v in sorted(T.items(), key=lambda x: -x[1]):
        print(f"{k:28s} total {v:8.1f}s  calls {N[k]:4d}  per call {v / N[k]:6.2f}s")


base.work(hermes_vqa.HermesVQA)
