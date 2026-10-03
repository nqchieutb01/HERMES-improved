"""Phase 20 spec: confidence-adaptive prior-contrastive decoding (Qwen3-VL-8B, S-EMBER grounding, timeline prompt).

PCD lowers answer accuracy on questions whose evidence truly starts at the video start (random 10%: 19.2 -> 16.7;
unpruned: 21.8 -> 16.7; GQ@0.5 16.7 -> 12.8 unpruned), because the memory and the prior agree there and the contrast
pushes away from a correct early time. Adaptive strength alpha_t = alpha_max * (1 - max p(. | real memory)) removes the
contrast where the memory is confident. Usage: python3 logs/phase20/make_spec.py > logs/phase20/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}


def run(score, keep, amax, scope="all"):
    prune = [f"run.offline_keep_ratio={keep}", f"run.prune_score={score}"] if keep < 1 else []
    base = f"{score}{int(round(keep * 100))}" if keep < 1 else "unpruned"
    tag = f"{base}-cd-blind-adapt{amax}" + ("" if scope == "all" else f"-scope{scope}")
    return {"name": f"g-{tag}-timeline", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
                          "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "dataset.grounding_prompt=timeline",
                          "run.max_new_tokens=384"] + prune
            + ["run.contrastive_mode=blind", f"run.contrastive_alpha={amax}", "run.contrastive_adaptive=true",
               f"run.contrastive_scope={scope}",
               f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}-timeline-v300-native-time"]}


RUNS = [run("random", 0.1, 1.0), run("random", 0.1, 2.0), run("random", 0.1, 1.0, "time"),
        run("random", 0.25, 1.0), run("random", 1.0, 1.0)]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
