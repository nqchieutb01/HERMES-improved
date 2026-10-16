"""Phase 21 spec: event-interval timeline, and evidence ablation (motivation for PCD).

(a) Event-interval prompt (dataset.grounding_prompt=events): each distinct occurrence on one line with start and end,
    targeting the run-on lists, over-counting and too-short extents found in logs/phase21/error_analysis.md.
(b) Evidence ablation (timeline prompt): remove the uniform frames inside the gold evidence ("gold") or as many outside
    it ("control"). If the model falls back on the temporal prior when evidence is missing, gold ablation should pull
    intervals to the start of the window, and PCD should resist it.
Baseline and final method (adaptive alpha_max 1, time scope) for each, at random 10% and unpruned. Batched answering and
frame prefetch (equivalent to batch size 1, logs/batching). Usage: cd logs/phase21 && python3 make_spec.py > main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
FINAL = ["run.contrastive_mode=blind", "run.contrastive_alpha=1.0", "run.contrastive_adaptive=true",
         "run.contrastive_scope=time"]


def run(keep, method, prompt="timeline", ablation="none"):
    memory = f"random{int(round(keep * 100))}" if keep < 1 else "unpruned"
    tag = f"{memory}-{method}-{prompt}" + ("" if ablation == "none" else f"-ablate{ablation}")
    prune = [f"run.offline_keep_ratio={keep}", "run.prune_score=random"] if keep < 1 else []
    return {"name": f"p21-{tag}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
                          "run.uniform_num_frames=64", "run.keep_time_tokens=surviving",
                          f"dataset.grounding_prompt={prompt}", "run.max_new_tokens=384"] + prune
            + (FINAL if method == "final" else [])
            + [f"run.batch_size={8 if keep < 1 else 3}", "run.prefetch_videos=4"]
            + ([f"run.evidence_ablation={ablation}"] if ablation != "none" else [])
            + [f"paths.save_dir={R}/uniform-n64-time-count-location-p21-{tag}-v300-native-time"]}


RUNS = [run(k, m, "events") for k in (0.1, 1.0) for m in ("base", "final")]
RUNS += [run(k, m, "timeline", a) for k in (0.1, 1.0) for m in ("base", "final") for a in ("gold", "control")]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
