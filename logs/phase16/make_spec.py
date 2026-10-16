"""Phase 16 spec: evidence-window oracle (upper bound on memory), Qwen3-VL-8B, S-EMBER grounding.

64 frames sampled uniformly inside the gold evidence interval (first annotator, clipped to the question
time); outside it, timestamp text alone every 5 s (0.2 fps) from 0 s to the question time. No pruning.
Official and timeline prompts. Usage: python3 logs/phase16/make_spec.py > logs/phase16/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
        "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "run.oracle_window=true"]


def run(timeline):
    prompt = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] if timeline else ["run.max_new_tokens=128"]
    t = "-timeline" if timeline else ""
    return {"name": f"g-oracle-window{t}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + prompt
            + [f"paths.save_dir={R}/uniform-n64-time-count-location-oracle-window{t}-v300-native-time"]}


if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": [run(False), run(True)]}, indent=1))
