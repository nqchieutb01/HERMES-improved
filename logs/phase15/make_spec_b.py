"""Phase 15b spec: literature-inspired diagnostics (Qwen3-VL-8B, S-EMBER grounding, uniform 64 frames).

1. Perception vs generation (temporal grounding heads): per-head attention on gold frames while reading the
   prompt, writing the answer and writing the timestamps; unpruned and at 10% (HERMES, stratified).
2. Blind baseline: answers without video (language prior).
3. Shuffles: frame pairs moved with their timestamps (reads stamps or order?), and timestamps alone
   permuted (text vs visual order).
Usage: python3 logs/phase15/make_spec_b.py > logs/phase15/main_b.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 2}]
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
    out = [run("aattn", ["run.answer_attention=true"], "dx-aattn", tl) for tl in (False, True)]
    for score in ("hermes", "stratified"):
        out.append(run(f"aattn-{score}-0.1", ["run.answer_attention=true", "run.offline_keep_ratio=0.1",
                                              f"run.prune_score={score}"], f"dx-aattn-{score}-keep0.1", True))
    out += [run("blind", ["run.blind=true"], "dx-blind", tl) for tl in (False, True)]
    out += [run("shufframes", ["run.shuffle_mode=frames"], "dx-shufframes", tl) for tl in (False, True)]
    out.append(run("shufstamps", ["run.shuffle_mode=stamps"], "dx-shufstamps"))
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs()}, indent=1))
