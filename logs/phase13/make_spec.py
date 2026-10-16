"""Phase 13 spec: does coverage-preserving memory make the timeline-reasoning prompt work where it matters?

The timeline prompt (dataset.grounding_prompt=timeline) raised GQ@0.5 on uniform 32 (5.3 -> 8.8) but not
significantly on streaming at 6k. Hypothesis: streaming memory keeps too few timestamped moments to list
evidence from. Grounding runs only (the prompt does not touch MCQ); every run has a baseline with the
official prompt already scored.

A. Qwen3-VL-8B streaming (0.2 fps): timeline prompt with a per-frame floor (k=1) and with all timestamps
   kept, at 4k / 6k / 10.7k.
B. Qwen3-VL-8B offline (uniform 64): timeline prompt unpruned, and at keep 10% / 5% with HERMES vs
   stratified selection.
C. Qwen2.5-VL-7B offline: timeline prompt unpruned and with stratified 10%.
Usage: python3 logs/phase13/make_spec.py > logs/phase13/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
RES = "/home/chieu.nguyen/HERMES/results"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
MODELS = {"q3": "qwen3_vl_8b", "q25": "qwen2.5_vl_7b"}


def run(name, m, overrides, save):
    return {"name": f"g-tl-{m}-{name}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": [f"model={MODELS[m]}", "dataset.max_videos=300", "dataset.grounding_prompt=timeline",
                          "run.max_new_tokens=384"] + overrides
            + [f"paths.save_dir={RES}/{MODELS[m]}/sember_grounding/{save}"]}


def stream(kv, k, keep):
    strategy = "top_attention_patch" if k else "attention_weighted"
    ov = ["experiment=sember_grounding_time_count_location", "run.sample_fps=0.2", f"run.kv_size={kv}",
          f"run.min_tokens_per_frame={k}", f"run.frame_summary_strategy={strategy}", f"run.keep_time_tokens={keep}"]
    short = {"surviving": "surv", "all": "all"}[keep]
    return run(f"kv{kv}-k{k}-{short}", "q3", ov,
               f"time-count-location-fps0.2-kv{kv}-k{k}-{strategy}-native-time-keep{short}-timeline-v300")


def offline(m, score, ratio):
    ov = ["experiment=sember_grounding_uniform", "run.uniform_num_frames=64", "run.keep_time_tokens=surviving"]
    if ratio is None:
        return run("uni64", m, ov, "uniform-n64-time-count-location-timeline-v300-native-time")
    return run(f"off-{score}-{ratio}", m, ov + [f"run.offline_keep_ratio={ratio}", f"run.prune_score={score}"],
               f"uniform-n64-time-count-location-offline-{score}-keep{ratio}-timeline-v300-native-time")


def runs():
    out = [stream(4000, 0, "surviving"), stream(4000, 1, "surviving"), stream(4000, 0, "all"),
           stream(6000, 1, "surviving"), stream(6000, 0, "all"), stream(10700, 0, "all")]
    out += [offline("q3", None, None)] + [offline("q3", s, r) for r in (0.1, 0.05) for s in ("hermes", "stratified")]
    out += [offline("q25", None, None), offline("q25", "stratified", 0.1)]
    return out


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs()}, indent=1))
