"""Phase 17 spec: coverage-then-zoom (Qwen3-VL-8B, S-EMBER grounding, 300 videos, timeline prompt).

Zoom pass: 64 coarse uniform frames + zoom_frames frames sampled inside the zoom window, pruned to the absolute
budget of keep_ratio x (64-frame visual tokens) with prune_score=zoom (frames in the window share zoom_share of the
budget, the rest share the remainder; most salient tokens within each frame). Windows (logs/phase17/make_windows.py):
  self  : from the coarse pass's own output (random pruning at the same budget, timeline prompt); no labels.
  gold  : the gold interval (localisation upper bound).
  full  : the whole video (control: extra frames, no localisation).
Usage: python3 logs/phase17/make_spec.py > logs/phase17/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
W = "/home/chieu.nguyen/HERMES/logs/phase17"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
        "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "run.prune_score=zoom",
        "dataset.grounding_prompt=timeline", "run.max_new_tokens=384"]


def run(name, keep, windows, frames=32, share=0.6):
    tag = f"zoom-{name}-keep{keep}-k{frames}-a{share}"
    return {"name": f"g-{tag}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + [f"run.offline_keep_ratio={keep}", f"run.zoom_windows={W}/{windows}",
                                 f"run.zoom_frames={frames}", f"run.zoom_share={share}", "run.retention_snapshot=true",
                                 f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}-timeline-v300-native-time"]}


def runs():
    return [run("self", 0.1, "win_self_random10.json"), run("gold", 0.1, "win_gold.json"),
            run("full", 0.1, "win_full.json"),
            run("self", 0.05, "win_self_random5.json"), run("gold", 0.05, "win_gold.json")]


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": runs()}, indent=1))
