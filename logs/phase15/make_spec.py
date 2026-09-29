"""Phase 15 spec: diagnostic GPU runs (Qwen3-VL-8B, S-EMBER grounding, uniform 64 frames).

1. Where does the question attend? Real-question attention per frame over the unpruned cache.
2. What survives pruning? Per-frame retention (tokens and timestamps) for HERMES / random / stratified.
3. Is selection the bottleneck? Oracle pruning keeps gold-interval frames first (same budget).
4. Is HERMES's scorer the problem? hermes_exact propagates the probe questions through every layer.
5. Does the model read timestamps? Shift every timestamp by +200 s, or drop the timestamp text.
6. Is there a primacy bias? Frames start halfway through the window (the true start is not shown).
Usage: python3 logs/phase15/make_spec.py > logs/phase15/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
        "run.uniform_num_frames=64", "run.keep_time_tokens=surviving"]


def run(name, overrides, tag, timeline=False):
    prompt = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] if timeline else ["run.max_new_tokens=128"]
    t = "-timeline" if timeline else ""
    return {"name": f"g-dx-{name}{t}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + prompt + overrides
            + [f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}{t}-v300-native-time"]}


def runs():
    out = [run("qattn", ["run.question_attention=true", "run.retention_snapshot=true"], "dx-qattn")]
    for score, keep in (("hermes", 0.1), ("hermes", 0.05), ("stratified", 0.1), ("random", 0.1)):
        out.append(run(f"ret-{score}-{keep}", ["run.retention_snapshot=true", f"run.offline_keep_ratio={keep}",
                                               f"run.prune_score={score}"], f"dx-ret-{score}-keep{keep}"))
    for keep in (0.1, 0.05):
        for tl in (False, True):
            out.append(run(f"oracle-{keep}", ["run.retention_snapshot=true", f"run.offline_keep_ratio={keep}",
                                              "run.prune_score=oracle"], f"offline-oracle-keep{keep}", tl))
    for keep, tl in ((0.1, False), (0.05, False), (0.1, True)):
        out.append(run(f"exact-{keep}", ["run.retention_snapshot=true", f"run.offline_keep_ratio={keep}",
                                         "run.prune_score=hermes_exact"], f"offline-hermes_exact-keep{keep}", tl))
    for tl in (False, True):
        out.append(run("offset200", ["run.time_offset=200"], "dx-offset200", tl))
    out.append(run("nostamp", ["run.drop_timestamps=true"], "dx-nostamp"))
    for tl in (False, True):
        out.append(run("start50", ["run.uniform_start_frac=0.5"], "dx-start50", tl))
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs()}, indent=1))
