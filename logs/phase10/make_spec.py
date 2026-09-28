"""Phase 10 spec: does coverage-preserving (stratified) pruning hold at harder budgets and on another model?

- Qwen3-VL-8B offline (uniform 64): keep 5% with hermes / random / stratified scores.
- Qwen3-VL-8B streaming (0.2 fps): KV 2k with hermes vs stratified.
- Qwen2.5-VL-7B offline (uniform 64): no pruning, and hermes / random / stratified at 25% and 10%.
Usage: python3 logs/phase10/make_spec.py > logs/phase10/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
RES = "/home/chieu.nguyen/HERMES/results"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
MODELS = {"q3": "qwen3_vl_8b", "q25": "qwen2.5_vl_7b"}


def run(name, m, task, overrides, save):
    extra = ["run.max_new_tokens=128"] if task == "grounding" else []
    return {"name": f"{task[0]}-{m}-{name}", "code": CODE, "chunks": 4 if task == "grounding" else 3, "env": ENV,
            "overrides": [f"model={MODELS[m]}", "dataset.max_videos=300", "run.keep_time_tokens=surviving"]
            + overrides + extra + [f"paths.save_dir={RES}/{MODELS[m]}/sember_{task}/{save}"]}


def offline(m, task, score, ratio):
    uni = [f"experiment=sember_{task}_uniform", "run.uniform_num_frames=64"]
    if ratio is None:
        return run("uni64", m, task, uni, "uniform-n64-time-count-location-v300-native-time")
    return run(f"off-{score}-{ratio}", m, task, uni + [f"run.offline_keep_ratio={ratio}", f"run.prune_score={score}"],
               f"uniform-n64-time-count-location-offline-{score}-keep{ratio}-v300-native-time")


def stream(m, task, score, kv):
    ov = [f"experiment=sember_{task}_time_count_location", "run.sample_fps=0.2", f"run.kv_size={kv}",
          "run.min_tokens_per_frame=0", "run.frame_summary_strategy=attention_weighted", f"run.prune_score={score}"]
    suffix = "" if score == "hermes" else f"-{score}"
    return run(f"stream-{score}-kv{kv}", m, task, ov,
               f"time-count-location-fps0.2-kv{kv}-k0-attention_weighted-native-time-keepsurv{suffix}-v300")


def runs():
    out = []
    for task in ("grounding", "mcq"):
        out += [offline("q3", task, s, 0.05) for s in ("hermes", "random", "stratified")]
        out += [stream("q3", task, s, 2000) for s in ("hermes", "stratified")]
        out += [offline("q25", task, None, None)]
        # Qwen2.5-VL grounding is at the floor without timestamp text (4.9 mIoU unpruned): prune MCQ only.
        if task == "mcq":
            out += [offline("q25", task, s, r) for s, r in (("hermes", 0.25), ("hermes", 0.1), ("random", 0.1),
                                                           ("stratified", 0.25), ("stratified", 0.1))]
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs()}, indent=1))
