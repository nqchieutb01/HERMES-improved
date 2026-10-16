"""Scheduler specs for phase 9: temporal grounding under token pruning (Qwen3-VL-8B, S-EMBER, 300 videos).

A2 (first): streaming runs with per-question retention snapshots (which frames' tokens survive).
A1: offline pruning after uniform-64 encoding at several keep ratios, with HERMES / random / recent
scores, against spatial pooling (lower decode resolution) at matched token cost.
Usage: python3 logs/phase9/make_spec.py > logs/phase9/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "run.keep_time_tokens=surviving"]


def run(name, task, overrides, save):
    extra = ["run.max_new_tokens=128"] if task == "grounding" else []
    return {"name": name, "code": CODE, "chunks": 4 if task == "grounding" else 3, "env": ENV,
            "overrides": BASE + overrides + extra + [f"paths.save_dir={R}/sember_{task}/{save}"]}


def a2():
    stream = ["experiment=sember_grounding_time_count_location", "run.sample_fps=0.2", "run.min_tokens_per_frame=0",
              "run.frame_summary_strategy=attention_weighted", "run.retention_snapshot=true"]
    tag = "-k0-attention_weighted-native-time-keepsurv-retention-v300"
    return [
        run("g-ret-fps02-kv6000", "grounding", stream + ["run.kv_size=6000"], f"time-count-location-fps0.2-kv6000{tag}"),
        run("g-ret-fps02-kv4000", "grounding", stream + ["run.kv_size=4000"], f"time-count-location-fps0.2-kv4000{tag}"),
        run("g-ret-grid32x8-kv6000", "grounding", stream + ["run.kv_size=6000", "run.sample_schedule='grid:32:8'"],
            f"time-count-location-grid32x8-kv6000{tag}"),
    ]


def a1():
    out = []
    for task in ("grounding", "mcq"):
        uni = [f"experiment=sember_{task}_uniform", "run.uniform_num_frames=64"]
        for score, ratios in (("hermes", (0.5, 0.25, 0.1)), ("random", (0.5, 0.25, 0.1)), ("recent", (0.25,))):
            for r in ratios:
                out.append(run(f"{task[0]}-off-{score}-{r}", task,
                               uni + [f"run.offline_keep_ratio={r}", f"run.prune_score={score}"],
                               f"uniform-n64-time-count-location-offline-{score}-keep{r}-v300-native-time"))
        for scale in (0.7, 0.5, 0.35):
            out.append(run(f"{task[0]}-off-pool-sc{scale}", task, uni + [f"run.frame_scale={scale}"],
                           f"uniform-n64-time-count-location-scale{scale}-v300-native-time"))
    return out


def b1():
    """Stratified (coverage-preserving) saliency pruning: equal share per frame, most salient within."""
    out = []
    for task in ("grounding", "mcq"):
        uni = [f"experiment=sember_{task}_uniform", "run.uniform_num_frames=64", "run.prune_score=stratified"]
        for r in (0.25, 0.1):
            out.append(run(f"{task[0]}-off-stratified-{r}", task, uni + [f"run.offline_keep_ratio={r}"],
                           f"uniform-n64-time-count-location-offline-stratified-keep{r}-v300-native-time"))
        stream = [f"experiment=sember_{task}_time_count_location", "run.sample_fps=0.2", "run.min_tokens_per_frame=0",
                  "run.frame_summary_strategy=attention_weighted", "run.prune_score=stratified"]
        for kv in (6000, 4000):
            out.append(run(f"{task[0]}-stream-stratified-kv{kv}", task, stream + [f"run.kv_size={kv}"],
                           f"time-count-location-fps0.2-kv{kv}-k0-attention_weighted-native-time-keepsurv-stratified-v300"))
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": a2() + a1() + b1()}, indent=1))
