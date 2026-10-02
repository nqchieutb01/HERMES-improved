"""Phase 18 spec: generation-side and allocation methods at 10% of the tokens (Qwen3-VL-8B, S-EMBER grounding,
300 videos, random pruning of 64 uniform frames, timeline prompt).

  cd-stamps-a{1,0.5}: contrastive decoding against the same pruned frames with permuted timestamps.
  cd-blind-a0.5      : contrastive decoding against the question alone (language prior).
  vag4               : visual-attention gain 4 in all layers during answer generation.
  rzoom-t2-m0.3      : relevance-guided temporal importance sampling (grounding heads, tau 2, coverage mix 0.3).
Usage: python3 logs/phase18/make_spec.py > logs/phase18/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
        "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline",
        "run.max_new_tokens=384", "run.offline_keep_ratio=0.1", "run.prune_score=random"]


def run(name, overrides):
    return {"name": f"g-{name}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + overrides
            + [f"paths.save_dir={R}/uniform-n64-time-count-location-random10-{name}-timeline-v300-native-time"]}


RUNS = [run("cd-stamps-a1", ["run.contrastive_mode=stamps", "run.contrastive_alpha=1.0"]),
        run("cd-stamps-a0.5", ["run.contrastive_mode=stamps", "run.contrastive_alpha=0.5"]),
        run("cd-blind-a0.5", ["run.contrastive_mode=blind", "run.contrastive_alpha=0.5"]),
        run("vag4", ["run.visual_attention_gain=4.0"]),
        run("rzoom-t2-m0.3", ["run.relevance_heads=/home/chieu.nguyen/HERMES/logs/phase17/heads_top8.json",
                              "run.relevance_zoom=true", "run.zoom_temp=2.0", "run.zoom_mix=0.3"])]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
