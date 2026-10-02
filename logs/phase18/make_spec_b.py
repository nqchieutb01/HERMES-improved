"""Phase 18b spec: robustness of prior-contrastive decoding (contrastive_mode=blind) on Qwen3-VL-8B, S-EMBER grounding.

Strength alpha (0.25, 1.0; 0.5 in phase 18), plausibility beta 0.2, budgets (5%, 25%, unpruned), selectors
(HERMES, stratified at 10%), and the official S-EMBER prompt (10% and unpruned).
Usage: python3 logs/phase18/make_spec_b.py > logs/phase18/robust.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
        "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "run.contrastive_mode=blind"]


def run(score, keep, alpha=0.5, beta=0.1, timeline=True):
    prompt = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] if timeline else ["run.max_new_tokens=128"]
    t = "-timeline" if timeline else ""
    prune = [f"run.offline_keep_ratio={keep}", f"run.prune_score={score}"] if keep < 1 else []
    base = f"{score}{int(round(keep * 100))}" if keep < 1 else "unpruned"
    tag = f"{base}-cd-blind-a{alpha}" + (f"-b{beta}" if beta != 0.1 else "")
    return {"name": f"g-{tag}{t}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + prompt + prune + [f"run.contrastive_alpha={alpha}", f"run.contrastive_beta={beta}",
                                                  f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}{t}-v300-native-time"]}


RUNS = [run("random", 0.1, alpha=0.25), run("random", 0.1, alpha=1.0), run("random", 0.1, beta=0.2),
        run("random", 0.05), run("random", 0.25), run("random", 1.0),
        run("hermes", 0.1), run("stratified", 0.1),
        run("random", 0.1, timeline=False), run("random", 1.0, timeline=False)]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
