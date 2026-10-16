"""Phase 34b: ActivityNet Captions at 30% and 50% of the visual tokens (same 500 queries as phase 34).

This ActivityNet copy is re-encoded at 224 px / 3 fps: a memory of 64 frames has only ~1.2k-3k visual tokens
(S-EMBER: ~21k), so 10% keeps 120-300 tokens and every selector collapses (phase 34 = extreme-budget stress test).
30% (~400-900 tokens) and 50% are the regimes where per-frame selection is meaningful.
Usage: python3 p34b_anet.py > p34b_anet.json
"""
import importlib.util
import json

_spec = importlib.util.spec_from_file_location("p34", "/home/chieu.nguyen/hermes_pool/specs/p34_anet.py")
p34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p34)

KEEP = {"30": "run.offline_keep_ratio=0.3", "50": "run.offline_keep_ratio=0.5"}
CONFIGS = {  # name: overrides without the keep ratio (from phase 34)
    tag: [o for o in over if not o.startswith("run.offline_keep_ratio")]
    for tag, (over, _) in p34.CONFIGS.items()
    if tag in ("random10", "random10-pcd-ar64", "semvid10", "semvid10-pcd-ar64", "u64-qtopk10", "rc64-qtopk10",
               "rc64-qtopk10-pcd-ar64")}
runs = []
for k, keep in KEEP.items():
    for tag, over in CONFIGS.items():
        name = tag.replace("10", k, 1)
        runs.append({"name": f"p34b-{name}", "code": p34.CODE, "chunks": 8, "est_min": 35, "priority": 39,
                     "env": p34.ENV, "overrides": p34.BASE + over + [keep, f"paths.save_dir={p34.R}{name}"]})

if __name__ == "__main__":
    print(json.dumps({"slots": p34.SLOTS, "runs": runs}, indent=1))
