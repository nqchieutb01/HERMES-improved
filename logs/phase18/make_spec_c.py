"""Phase 18c spec: combining the two methods, and prior-contrastive decoding in HERMES streaming.

  zoom+cd 10% / 5%  : coverage-then-zoom with self windows (random coarse pass) + prior-contrastive decoding
                      (contrastive_mode=blind, alpha 0.5), timeline prompt.
  stream 6k / 4k cd : HERMES streaming (0.2 fps, k=0, timestamps kept for surviving frames) + prior-contrastive decoding.
Usage: python3 logs/phase18/make_spec_c.py > logs/phase18/combo.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
W = "/home/chieu.nguyen/HERMES/logs/phase17"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
TL = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"]
CD = ["run.contrastive_mode=blind", "run.contrastive_alpha=0.5"]


def zoom(keep, windows):
    tag = f"zoom-self-keep{keep}-k32-a0.6-cd-blind-a0.5"
    return {"name": f"g-{tag}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
                          "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "run.prune_score=zoom",
                          f"run.offline_keep_ratio={keep}", f"run.zoom_windows={W}/{windows}", "run.zoom_frames=32",
                          "run.zoom_share=0.6"] + TL + CD
            + [f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}-timeline-v300-native-time"]}


def stream(kv):
    return {"name": f"g-stream{kv}-cd-blind-a0.5", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_time_count_location",
                          "run.sample_fps=0.2", f"run.kv_size={kv}", "run.min_tokens_per_frame=0",
                          "run.frame_summary_strategy=attention_weighted", "run.keep_time_tokens=surviving"] + TL + CD
            + [f"paths.save_dir={R}/time-count-location-fps0.2-kv{kv}-k0-attention_weighted-native-time-keepsurv-cd-blind-a0.5-timeline-v300"]}


RUNS = [zoom(0.1, "win_self_random10.json"), zoom(0.05, "win_self_random5.json"), stream(6000), stream(4000)]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
