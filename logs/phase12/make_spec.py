"""Phase 12 spec (Qwen3-VL-8B, S-EMBER, 300 videos).

A. Timeline-reasoning grounding prompt (dataset.grounding_prompt=timeline): the model lists the timestamped
   moments where the event is visible, then derives the answer and interval from them. Tested on the
   streaming baseline, uniform 32, and the best-answer streaming configuration. Scored with the official
   S-EMBER judge afterwards (logs/phase11).
B. Completing the offline pruning grid: keep 20% (HERMES / random / stratified), stratified at 50%, and
   spatial pooling at about 20% of tokens (scale 0.45).
Usage: python3 logs/phase12/make_spec.py > logs/phase12/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "run.keep_time_tokens=surviving"]
STREAM = ["run.min_tokens_per_frame=0", "run.frame_summary_strategy=attention_weighted"]


def run(name, task, overrides, save, max_new_tokens=128):
    extra = [f"run.max_new_tokens={max_new_tokens}"] if task == "grounding" else []
    return {"name": name, "code": CODE, "chunks": 4 if task == "grounding" else 3, "env": ENV,
            "overrides": BASE + overrides + extra + [f"paths.save_dir={R}/sember_{task}/{save}"]}


def timeline():
    tl = ["dataset.grounding_prompt=timeline"]
    ks = "-k0-attention_weighted-native-time-keepsurv"
    return [
        run("g-tl-fps02-kv6000", "grounding",
            tl + ["experiment=sember_grounding_time_count_location", "run.sample_fps=0.2", "run.kv_size=6000"] + STREAM,
            f"time-count-location-fps0.2-kv6000{ks}-timeline-v300", 384),
        run("g-tl-uni32", "grounding", tl + ["experiment=sember_grounding_uniform", "run.uniform_num_frames=32"],
            "uniform-n32-time-count-location-timeline-v300-native-time", 384),
        run("g-tl-dups1x60-kv10700", "grounding",
            tl + ["experiment=sember_grounding_time_count_location", "run.sample_fps=0.2", "run.kv_size=10700",
                  "run.sample_schedule='dup:1.0:60,0.2'", "run.frame_scale=0.7"] + STREAM,
            f"time-count-location-dups1x60-kv10700{ks}-scale0.7-timeline-v300", 384),
    ]


def pruning_grid():
    out = []
    for task in ("grounding", "mcq"):
        uni = [f"experiment=sember_{task}_uniform", "run.uniform_num_frames=64"]
        for score, keep in (("hermes", 0.2), ("random", 0.2), ("stratified", 0.2), ("stratified", 0.5)):
            out.append(run(f"{task[0]}-off-{score}-{keep}", task,
                           uni + [f"run.offline_keep_ratio={keep}", f"run.prune_score={score}"],
                           f"uniform-n64-time-count-location-offline-{score}-keep{keep}-v300-native-time"))
        out.append(run(f"{task[0]}-off-pool-sc0.45", task, uni + ["run.frame_scale=0.45"],
                       "uniform-n64-time-count-location-scale0.45-v300-native-time"))
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": timeline() + pruning_grid()}, indent=1))
