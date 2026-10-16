"""Phase 19 spec: time-scoped prior-contrastive decoding (contrast only on time values) on Qwen3-VL-8B, S-EMBER.

Error analysis of PCD (logs/phase19/error_analysis.py): contrasting every token rewrites 85% of answers, makes 56 of
475 answers list more than 8 moments (12 without PCD) and doubles over-counting, while the diagnosed prior leaks into
time values. Scope "time" applies the contrast only to `Seen:` times and the `Time: [...]` interval; "time+answer"
also to the answer line. Same budgets and selectors as before for pairing.
Usage: python3 logs/phase19/make_spec.py > logs/phase19/main.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
CD = ["run.contrastive_mode=blind", "run.contrastive_alpha=0.5"]


def offline(score, keep, scope, timeline=True):
    prompt = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] if timeline else ["run.max_new_tokens=128"]
    t = "-timeline" if timeline else ""
    prune = [f"run.offline_keep_ratio={keep}", f"run.prune_score={score}"] if keep < 1 else []
    base = f"{score}{int(round(keep * 100))}" if keep < 1 else "unpruned"
    tag = f"{base}-cd-blind-a0.5-scope{scope.replace('+', '_')}"
    return {"name": f"g-{tag}{t}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
                          "run.uniform_num_frames=64", "run.keep_time_tokens=surviving"] + prompt + prune + CD
            + [f"run.contrastive_scope={scope}", f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}{t}-v300-native-time"]}


def stream(kv, scope):
    return {"name": f"g-stream{kv}-cd-blind-a0.5-scope{scope}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_time_count_location",
                          "run.sample_fps=0.2", f"run.kv_size={kv}", "run.min_tokens_per_frame=0",
                          "run.frame_summary_strategy=attention_weighted", "run.keep_time_tokens=surviving",
                          "dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] + CD
            + [f"run.contrastive_scope={scope}",
               f"paths.save_dir={R}/time-count-location-fps0.2-kv{kv}-k0-attention_weighted-native-time-keepsurv-cd-blind-a0.5-scope{scope}-timeline-v300"]}


RUNS = [offline("random", 0.1, "time"), offline("random", 0.1, "time+answer"), offline("random", 0.25, "time"),
        offline("random", 1.0, "time"), offline("hermes", 0.1, "time"), stream(4000, "time"),
        offline("random", 0.1, "time", timeline=False)]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
