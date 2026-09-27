"""Build scheduler specs for phase 6: early-dense incremental sampling for Qwen3-VL-8B.

Usage: python3 logs/phase6/make_spec.py <round> > logs/phase6/<round>.json
Each config is run on S-EMBER MCQ (3 chunks) and grounding (4 chunks), 300 videos, with the same
settings as the fps0.2 KV6000 k=0 surviving-timestamp baselines; only frame sampling changes.
"""
import json
import sys

CODE = "/l/users/chieu.nguyen/HERMES-qwen-time-v3"
RESULTS = "/l/users/chieu.nguyen/HERMES/results/qwen3_vl_8b"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]

# name -> (sample_schedule or None, sample_fps, kv, k, strategy)
ROUNDS = {
    "round1": {
        "s1x60": ("1.0:60,0.2", 0.2, 6000, 0, "attention_weighted"),
        "s1x160": ("1.0:160,0.2", 0.2, 6000, 0, "attention_weighted"),
        "s1x80-05x160": ("1.0:80,0.5:160,0.2", 0.2, 6000, 0, "attention_weighted"),
        "s05x160": ("0.5:160,0.2", 0.2, 6000, 0, "attention_weighted"),
        "s2x32-1x80-05x160": ("2.0:32,1.0:80,0.5:160,0.2", 0.2, 6000, 0, "attention_weighted"),
        "fps1": (None, 1.0, 6000, 0, "attention_weighted"),
    },
    # Extra GPUs while round1 runs: do early-dense schedules pay off with a larger budget?
    "round1b": {
        "s1x160": ("1.0:160,0.2", 0.2, 10700, 0, "attention_weighted"),
        "s2x32-1x80-05x160": ("2.0:32,1.0:80,0.5:160,0.2", 0.2, 10700, 0, "attention_weighted"),
    },
}
ROUNDS["round2"] = {
    # Streaming counterpart of uniform sampling: dense early, old frames thinned onto a doubling grid.
    "grid32x16": ("grid:32:16", 0.2, 6000, 0, "attention_weighted"),
    "grid16x16": ("grid:16:16", 0.2, 6000, 0, "attention_weighted"),
    "grid32x8": ("grid:32:8", 0.2, 6000, 0, "attention_weighted"),
    "grid64x16": ("grid:64:16", 0.2, 6000, 0, "attention_weighted"),
}
ROUNDS["round3"] = {
    # Grid pairs spread over their cell (frames t, t + spacing/2): round 2 pairs were 1 s apart, leaving
    # 7-15 s blind gaps per group at large spacing, and grounding lost from 160 s on.
    "grid32x16s": ("grid:32:16:spread", 0.2, 6000, 0, "attention_weighted"),
    "grid32x8s": ("grid:32:8:spread", 0.2, 6000, 0, "attention_weighted"),
}
ROUNDS["round4"] = {
    # MCQ gains grow with frame count but grounding drops as more frames share a 6k budget: can a larger
    # budget or a per-frame floor (every frame keeps a token and its timestamp) recover grounding?
    "grid32x8": ("grid:32:8", 0.2, 10700, 0, "attention_weighted"),
    "grid16x16": ("grid:16:16", 0.2, 10700, 0, "attention_weighted"),
    "s1x60": ("1.0:60,0.2", 0.2, 10700, 0, "attention_weighted"),
    "grid32x8-k1": ("grid:32:8", 0.2, 6000, 1, "top_attention_patch"),
}
SLOTS_BY_ROUND = {"round1b": [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 6}]}
# Runs dropped after round 1's first result showed dense-early sampling without thinning floods the
# cache with early frames (late questions collapse): variants of the same idea add little.
DROPPED = {"m-q3-s1x80-05x160-kv6000-k0", "m-q3-s05x160-kv6000-k0", "g-q3-s05x160-kv6000-k0",
           "m-q3-s2x32-1x80-05x160-kv6000-k0", "g-q3-s2x32-1x80-05x160-kv6000-k0",
           "m-q3-s2x32-1x80-05x160-kv10700-k0", "g-q3-s2x32-1x80-05x160-kv10700-k0"}


def run_dir(name, fps, kv, k, strategy):
    rate = name.removesuffix("-k1") if name.startswith(("s", "grid")) else f"fps{fps}"
    return f"time-count-location-{rate}-kv{kv}-k{k}-{strategy}-native-time-keepsurv-v300"


def runs(round_name):
    out = []
    for name, (schedule, fps, kv, k, strategy) in ROUNDS[round_name].items():
        for task, prefix, chunks in (("sember_mcq", "m", 3), ("sember_grounding", "g", 4)):
            overrides = [
                f"experiment={task}_time_count_location",
                "model=qwen3_vl_8b",
                "dataset.max_videos=300",
                f"run.sample_fps={fps}",
                f"run.kv_size={kv}",
                f"run.min_tokens_per_frame={k}",
                f"run.frame_summary_strategy={strategy}",
                "run.keep_time_tokens=surviving",
            ]
            if schedule:
                overrides.append(f"run.sample_schedule='{schedule}'")
            if task == "sember_grounding":
                overrides.append("run.max_new_tokens=128")
            overrides.append(f"paths.save_dir={RESULTS}/{task}/{run_dir(name, fps, kv, k, strategy)}")
            out.append({"name": f"{prefix}-q3-{name.removesuffix('-k1')}-kv{kv}-k{k}", "code": CODE, "chunks": chunks,
                        "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}, "overrides": overrides})
    return out


if __name__ == "__main__":
    # Several round names are merged into one spec (one scheduler shares the job-submit limit).
    names = sys.argv[1:]
    all_runs = [r for n in names for r in runs(n) if r["name"] not in DROPPED]
    print(json.dumps({"slots": SLOTS_BY_ROUND.get(names[0], SLOTS) if len(names) == 1 else SLOTS,
                      "runs": all_runs}, indent=1))
