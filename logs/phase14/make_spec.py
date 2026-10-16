"""Phase 14 spec: complete the timeline-prompt offline pruning grid (Qwen3-VL-8B, S-EMBER grounding).

Timeline prompt (dataset.grounding_prompt=timeline) after uniform 64 frames with one pruning pass, for every
keep ratio x selection that has an official-prompt baseline: keep 50% / 25% with HERMES, random, stratified
and spatial pooling (scale 0.7 / 0.5), keep 10% with random and pooling (scale 0.35), keep 5% with random.
HERMES and stratified at 10% / 5% ran in phase 13. Grounding only (the prompt does not touch MCQ).
Usage: python3 logs/phase14/make_spec.py > logs/phase14/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "dataset.grounding_prompt=timeline", "run.max_new_tokens=384",
        "experiment=sember_grounding_uniform", "run.uniform_num_frames=64", "run.keep_time_tokens=surviving"]


def run(name, overrides, tag):
    return {"name": f"g-tl-q3-{name}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + overrides + [f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}-timeline-v300-native-time"]}


def prune(score, keep):
    return run(f"off-{score}-{keep}", [f"run.offline_keep_ratio={keep}", f"run.prune_score={score}"],
               f"offline-{score}-keep{keep}")


def pool(scale):
    return run(f"pool-sc{scale}", [f"run.frame_scale={scale}"], f"scale{scale}")


def runs():
    out = []
    for keep, scale in ((0.5, 0.7), (0.25, 0.5)):
        out += [prune(s, keep) for s in ("hermes", "random", "stratified")] + [pool(scale)]
    out += [prune("random", 0.1), pool(0.35), prune("random", 0.05)]
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs()}, indent=1))
