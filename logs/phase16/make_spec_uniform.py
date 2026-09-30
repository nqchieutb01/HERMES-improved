"""Phase 16b spec: uniform frame dropping as a token selector (Qwen3-VL-8B, S-EMBER).

Fewer uniformly sampled frames at a token budget matched to offline pruning of 64 frames: 16 frames = 25%,
12 = 19% (~20%), 6 = 9% (~10%), 4 = 6% (~5%); Qwen3 encodes frame pairs, so counts are even. Grounding with the
official and timeline prompts, and MCQ. 32 frames (= 50%) already exists.
Usage: python3 logs/phase16/make_spec_uniform.py > logs/phase16/uniform.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 6},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}


def run(n, task, timeline=False):
    extra = []
    if task == "grounding":
        extra = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] if timeline else ["run.max_new_tokens=128"]
    t = "-timeline" if timeline else ""
    return {"name": f"{task[0]}-uni{n}{t}", "code": CODE, "chunks": 2, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", f"experiment=sember_{task}_uniform",
                          f"run.uniform_num_frames={n}", "run.keep_time_tokens=surviving"] + extra
            + [f"paths.save_dir={R}/sember_{task}/uniform-n{n}-time-count-location{t}-v300-native-time"]}


if __name__ == "__main__":
    runs = [r for n in (16, 12, 6, 4) for r in (run(n, "grounding"), run(n, "grounding", True), run(n, "mcq"))]
    print(json.dumps({"slots": SLOTS, "runs": runs}, indent=1))
