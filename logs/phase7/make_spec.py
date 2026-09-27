"""Scheduler specs for phase 7 (300 videos, S-EMBER).

Part 3: does grounding recover when frames cost fewer tokens (lower decode resolution)? Qwen3 at
native resolution spends ~340 visual tokens per frame, so a 6k cache holds ~18 frames' worth.
Part 2: does the dense-early/grid MCQ gain transfer to LLaVA-OV-7B (MCQ only: LLaVA grounding is
at the floor for every setting)?
Usage: python3 logs/phase7/make_spec.py <round>... > logs/phase7/<name>.json
"""
import json
import sys

CODE = "/home/chieu.nguyen/HERMES"
RESULTS = "/home/chieu.nguyen/HERMES/results"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
MODELS = {"q3": "qwen3_vl_8b", "lv": "llava_ov_7b"}

# name -> (model, tasks, sample_schedule or None, kv, k, strategy, frame_scale)
ROUNDS = {
    "r7a": {
        "q3-fps02-sc07": ("q3", ("mcq", "grounding"), None, 6000, 0, "attention_weighted", 0.7),
        "q3-fps02-sc05": ("q3", ("mcq", "grounding"), None, 6000, 0, "attention_weighted", 0.5),
        "q3-grid32x8-sc07": ("q3", ("mcq", "grounding"), "grid:32:8", 6000, 0, "attention_weighted", 0.7),
        "q3-grid32x8-sc05": ("q3", ("mcq", "grounding"), "grid:32:8", 6000, 0, "attention_weighted", 0.5),
        "lv-fps02": ("lv", ("mcq",), None, 6000, 0, "attention_weighted", 1.0),
        "lv-grid32x8": ("lv", ("mcq",), "grid:32:8", 6000, 0, "attention_weighted", 1.0),
        "lv-grid16x16": ("lv", ("mcq",), "grid:16:16", 6000, 0, "attention_weighted", 1.0),
        "lv-s1x60": ("lv", ("mcq",), "1.0:60,0.2", 6000, 0, "attention_weighted", 1.0),
    },
    # Qwen3 merges each frame pair into one token group; at 0.2 fps the pair is 5 s apart, blending two
    # moments. dup: streams every frame twice so a group holds one frame; scale 0.7 keeps the token
    # cost per real frame equal to the baseline's (660 tokens per pair at native resolution).
    "r7c": {
        "q3-dup02-sc07": ("q3", ("mcq", "grounding"), "dup:0.2", 6000, 0, "attention_weighted", 0.7),
        "q3-dup02": ("q3", ("mcq", "grounding"), "dup:0.2", 6000, 0, "attention_weighted", 1.0),
        "q3-dups1x60-sc07": ("q3", ("mcq", "grounding"), "dup:1.0:60,0.2", 6000, 0, "attention_weighted", 0.7),
    },
    # Combine the two MCQ wins: grid thinning (grid 32:8) and un-blended Qwen pairs (dup). Scale 0.7
    # keeps each one-frame group at ~336 tokens (a native two-frame group costs ~660).
    "r8": {
        "q3-grid32x8d-sc07": ("q3", ("mcq", "grounding"), "grid:32:8:dup", 6000, 0, "attention_weighted", 0.7),
        "q3-grid32x8d-sc07-kv10700": ("q3", ("mcq", "grounding"), "grid:32:8:dup", 10700, 0, "attention_weighted", 0.7),
        "q3-grid64x8d-sc07": ("q3", ("mcq", "grounding"), "grid:64:8:dup", 6000, 0, "attention_weighted", 0.7),
        "q3-dups1x60-sc07-kv10700": ("q3", ("mcq", "grounding"), "dup:1.0:60,0.2", 10700, 0, "attention_weighted", 0.7),
    },
}


def rate_tag(schedule):
    if schedule is None:
        return "fps0.2"
    if schedule.startswith("dup:"):
        return "dup" + rate_tag(schedule[4:] if ":" in schedule[4:] else None).replace("fps0.2", "0.2")
    if schedule.startswith("grid:"):
        _, n, smax, *rest = schedule.split(":")
        return f"grid{n}x{smax}" + {"spread": "s", "dup": "d"}.get(rest[0], "") if rest else f"grid{n}x{smax}"
    return {"1.0:60,0.2": "s1x60", "1.0:160,0.2": "s1x160"}[schedule]


def runs(round_name):
    out = []
    for name, (m, tasks, schedule, kv, k, strategy, scale) in ROUNDS[round_name].items():
        model = MODELS[m]
        for task in tasks:
            full = f"sember_{task}"
            tag = f"time-count-location-{rate_tag(schedule)}-kv{kv}-k{k}-{strategy}"
            if m == "q3":
                tag += "-native-time-keepsurv"
            if scale != 1.0:
                tag += f"-scale{scale}"
            tag += "-v300" if m == "q3" or schedule else "-v300-v3code"
            overrides = [f"experiment={full}_time_count_location", f"model={model}", "dataset.max_videos=300",
                         "run.sample_fps=0.2", f"run.kv_size={kv}", f"run.min_tokens_per_frame={k}",
                         f"run.frame_summary_strategy={strategy}"]
            if m == "q3":
                overrides.append("run.keep_time_tokens=surviving")
            if schedule:
                overrides.append(f"run.sample_schedule='{schedule}'")
            if scale != 1.0:
                overrides.append(f"run.frame_scale={scale}")
            if task == "grounding":
                overrides.append("run.max_new_tokens=128")
            overrides.append(f"paths.save_dir={RESULTS}/{model}/{full}/{tag}")
            out.append({"name": f"{task[0]}-{name}", "code": CODE, "chunks": 3 if task == "mcq" else 4,
                        "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}, "overrides": overrides})
    return out


def r7b():
    """Grounding with the full_span prompt (intervals are ~3x too short for every method)."""
    q3 = ["model=qwen3_vl_8b", "dataset.max_videos=300", "run.max_new_tokens=128", "dataset.grounding_prompt=full_span"]
    stream = ["experiment=sember_grounding_time_count_location", "run.sample_fps=0.2", "run.min_tokens_per_frame=0",
              "run.frame_summary_strategy=attention_weighted", "run.keep_time_tokens=surviving"]
    g = f"{RESULTS}/qwen3_vl_8b/sember_grounding"
    specs = {
        "fps02": stream + ["run.kv_size=6000",
                           f"paths.save_dir={g}/time-count-location-fps0.2-kv6000-k0-attention_weighted-native-time-keepsurv-fullspan-v300"],
        "uni32": ["experiment=sember_grounding_uniform", "run.uniform_num_frames=32",
                  f"paths.save_dir={g}/uniform-n32-time-count-location-fullspan-v300-native-time"],
        "grid16x16": stream + ["run.kv_size=6000", "run.sample_schedule='grid:16:16'",
                               f"paths.save_dir={g}/time-count-location-grid16x16-kv6000-k0-attention_weighted-native-time-keepsurv-fullspan-v300"],
        "grid32x8-kv10700": stream + ["run.kv_size=10700", "run.sample_schedule='grid:32:8'",
                                      f"paths.save_dir={g}/time-count-location-grid32x8-kv10700-k0-attention_weighted-native-time-keepsurv-fullspan-v300"],
    }
    return [{"name": f"g-q3-fullspan-{n}", "code": CODE, "chunks": 4,
             "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}, "overrides": q3 + o} for n, o in specs.items()]


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": [r for n in sys.argv[1:] for r in (r7b() if n == "r7b" else runs(n))]},
                     indent=1))
