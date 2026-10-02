"""Phase 17c spec: ablations of coverage-then-zoom with self-predicted windows (Qwen3-VL-8B, S-EMBER grounding).

  gold+margin          : gold window widened with the same margin rule as self windows (width-matched oracle).
  self m0.5 / m2       : margin 0.5x span (min 10 s) / 2x span (min 30 s) instead of 1x (min 30 s).
  self a0.4 / a0.8     : share of the budget given to the window.
  self strat10         : windows from a stratified (instead of random) coarse pass.
  self official        : official S-EMBER prompt in both passes.
  self 25%             : budget 25%, windows from the random-25% coarse pass.
Usage: python3 logs/phase17/make_spec_b.py > logs/phase17/ablation.json
"""
import json

CODE = "/home/chieu.nguyen/HERMES"
R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
W = "/home/chieu.nguyen/HERMES/logs/phase17"
SLOTS = [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 8},
         {"partition": "cscc-gpu-p", "qos": "cscc-gpu-qos", "max_jobs": 2}]
ENV = {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"}
BASE = ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
        "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "run.prune_score=zoom"]


def run(name, keep, windows, share=0.6, timeline=True):
    prompt = ["dataset.grounding_prompt=timeline", "run.max_new_tokens=384"] if timeline else ["run.max_new_tokens=128"]
    t = "-timeline" if timeline else ""
    tag = f"zoom-{name}-keep{keep}-k32-a{share}"
    return {"name": f"g-{tag}{t}", "code": CODE, "chunks": 4, "env": ENV,
            "overrides": BASE + prompt + [f"run.offline_keep_ratio={keep}", f"run.zoom_windows={W}/{windows}",
                                          "run.zoom_frames=32", f"run.zoom_share={share}",
                                          f"paths.save_dir={R}/uniform-n64-time-count-location-{tag}{t}-v300-native-time"]}


RUNS = [run("goldmargin", 0.1, "win_gold_margin.json"),
        run("self-m0.5", 0.1, "win_self_random10_m0.5.json"),
        run("self-m2", 0.1, "win_self_random10_m2.json"),
        run("self", 0.1, "win_self_random10.json", share=0.4),
        run("self", 0.1, "win_self_random10.json", share=0.8),
        run("self-strat", 0.1, "win_self_strat10.json"),
        run("self-official", 0.1, "win_self_random10_official.json", timeline=False),
        run("self", 0.25, "win_self_random25.json")]

if __name__ == "__main__":
    print(json.dumps({"slots": SLOTS, "runs": RUNS}, indent=1))
