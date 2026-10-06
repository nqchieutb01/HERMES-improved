"""Phase 21c: two controls for the method (random 10%).

(a) Instruction-only debiasing: the timeline prompt plus "the relevant moments may occur at any point in the video;
    do not assume they are near the start", baseline and final method. Does stating the bias fix the leak without PCD?
(b) Which boundary of the contrast fragments events: events_merged prompt, 768 tokens, final method with the contrast
    on event start times only (time_start) or end times only (time_end); `Seen:` and `Time:` values stay contrasted.
(c) Token-budget control for timeline vs events: the timeline baseline at 768 tokens.
Usage: cd logs/phase21 && python3 make_spec_c.py > c.json
"""
import json

from make_spec import SLOTS, run


def variant(method, prompt, name, tokens=384, scope=None):
    r = run(0.1, method, prompt)
    r["name"] = f"p21c-random10-{method}-{name}"
    ov = [o.replace("run.max_new_tokens=384", f"run.max_new_tokens={tokens}") for o in r["overrides"][:-1]]
    if scope:
        ov = [o.replace("run.contrastive_scope=time", f"run.contrastive_scope={scope}") for o in ov]
    save = r["overrides"][-1].split("-p21-")[0] + f"-p21c-random10-{method}-{name}-v300-native-time"
    r["overrides"] = ov + [save]
    return r


RUNS = [variant("base", "timeline_debias", "timeline_debias"),
        variant("final", "timeline_debias", "timeline_debias"),
        variant("final", "events_merged", "events_merged-768-scopestart", 768, "time_start"),
        variant("final", "events_merged", "events_merged-768-scopeend", 768, "time_end"),
        variant("base", "timeline", "timeline-768", 768)]

if __name__ == "__main__":
    print(json.dumps({"slots": [dict(s, max_jobs=6 if s["partition"] == "long" else 2) for s in SLOTS],
                      "runs": RUNS}, indent=1))
