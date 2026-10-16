"""Phase 16c spec: per-frame retention logs for HERMES at 50% and 25% (official prompt), to measure the share
of kept tokens that fall inside the gold evidence interval. Usage: python3 logs/phase16/make_spec_ret.py > logs/phase16/ret.json
"""
import json

R = "/home/chieu.nguyen/HERMES/results/qwen3_vl_8b/sember_grounding"
runs = [{"name": f"g-dx-ret-{s}-{k}", "code": "/home/chieu.nguyen/HERMES", "chunks": 4,
         "env": {"HERMES_QWEN3_ATTENTION_BACKEND": "flash_attention_2"},
         "overrides": ["model=qwen3_vl_8b", "dataset.max_videos=300", "experiment=sember_grounding_uniform",
                       "run.uniform_num_frames=64", "run.keep_time_tokens=surviving", "run.max_new_tokens=128",
                       "run.retention_snapshot=true", f"run.offline_keep_ratio={k}", f"run.prune_score={s}",
                       f"paths.save_dir={R}/uniform-n64-time-count-location-dx-ret-{s}-keep{k}-v300-native-time"]}
        for s, k in (("hermes", 0.5), ("hermes", 0.25))]
print(json.dumps({"slots": [{"partition": "long", "qos": "gpu-debug-qos", "max_jobs": 4}], "runs": runs}, indent=1))
