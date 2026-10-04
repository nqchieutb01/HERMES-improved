"""Phase 20c: confidence-adaptive PCD across strengths, budgets, selectors and HERMES streaming.

Phase 20 result: with alpha_t = alpha_max * (1 - max p(. | M)), unpruned PCD keeps the overall gains and recovers the
early-evidence answers that constant-alpha PCD lost (Acc. 16.7 -> 23.1, GQ@0.5 12.8 -> 19.2; baseline 21.8 / 16.7);
at random 25% it raises GQ@0.5 over PCD (+2.3*); at random 10% the memory is rarely confident and nothing changes.
This round checks alpha_max 2 where evidence is richer, the other budgets/selectors, streaming, and the time scope.
Usage: cd logs/phase20 && python3 make_spec_c.py > c.json
"""
import json

from make_spec import CODE, ENV, R, SLOTS, run

AD = ["run.contrastive_mode=blind", "run.contrastive_adaptive=true"]


def stream(kv, amax):
    tag = f"cd-blind-adapt{amax}"
    return {"name": f"g-stream{kv}-{tag}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_time_count_location",
                          "run.sample_fps=0.2", f"run.kv_size={kv}", "run.min_tokens_per_frame=0",
                          "run.frame_summary_strategy=attention_weighted", "run.keep_time_tokens=surviving",
                          "dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] + AD
            + [f"run.contrastive_alpha={amax}",
               f"paths.save_dir={R}/time-count-location-fps0.2-kv{kv}-k0-attention_weighted-native-time-keepsurv-{tag}-timeline-v300"]}


RUNS = [run("random", 0.25, 2.0), run("random", 1.0, 2.0), run("random", 1.0, 1.0, "time"),
        run("hermes", 0.1, 1.0), run("stratified", 0.1, 1.0), run("random", 0.05, 1.0),
        stream(4000, 1.0), stream(6000, 1.0)]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
