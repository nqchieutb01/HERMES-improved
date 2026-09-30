"""Collect the blind (no video) S-EMBER grounding outputs of Qwen3-VL-8B as readable JSON, grouped by video.

Each question: gold answer and interval, the blind output with the official and the timeline prompt (raw
response, parsed answer and interval, IoU, official-judge verdict) and, for comparison, the answer with all
64 uniform frames (same prompt). Summary: accuracy and mIoU per task, blind answers equal to the with-video
answer, questions answered correctly only blind / only with video, and the most common blind durations.
Writes logs/phase15/blind_by_question.json. Run from the repo root: python3 logs/phase15/collect_blind.py
"""
import collections
import json
import sys

sys.path.insert(0, "logs/phase11")
import rule_answer as ra  # noqa: E402

G = "results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
RUNS = {"official": ("dx-blind-v300-native-time", "v300-native-time"),
        "timeline": ("dx-blind-timeline-v300-native-time", "timeline-v300-native-time")}
OUT = "logs/phase15/blind_by_question.json"


def load(tag):
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(G + tag + "/sember_grounding_scored.jsonl")}
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(G + tag + "/answer_judgments.jsonl")}
    return rows, judge


def num(x):
    return None if x in (None, "") else round(float(x), 2)


def output(r, ok):
    return {"raw_response": (r["pred_raw"] or "").strip(),
            "answer": r["pred_answer_parsed"],
            "interval_s": [num(r["pred_start_time"]), num(r["pred_end_time"])] if r["interval_parseable"] else None,
            "iou": round(float(r["temporal_iou"] or 0), 3),
            "answer_correct": ok}


def main():
    videos = collections.defaultdict(dict)
    meta, summary = {}, {}
    for prompt, (blind_tag, video_tag) in RUNS.items():
        blind, bj = load(blind_tag)
        seen, vj = load(video_tag)
        per_task = collections.defaultdict(lambda: {"n": 0, "blind_ok": 0, "video_ok": 0, "blind_iou": 0.0, "video_iou": 0.0})
        same = only_blind = only_video = 0
        durations = collections.Counter()
        for q, r in blind.items():
            v = seen[q]
            meta[r["video_id"]] = {"duration_s": num(r["duration"]), "category": r["video_category"]}
            entry = videos[r["video_id"]].setdefault(q, {
                "question_id": q, "question": r["question"], "category": r["question_category"],
                "question_time_s": num(r["question_time"]), "gold_answer": r["answer"],
                "gold_interval_s": [num(r["answer_start_time"]), num(r["answer_end_time"])],
                "blind": {}, "with_64_frames": {}})
            entry["blind"][prompt] = output(r, bj[q])
            entry["with_64_frames"][prompt] = output(v, vj[q])
            t = per_task[r["question_category"]]
            t["n"] += 1
            t["blind_ok"] += bj[q]
            t["video_ok"] += vj[q]
            t["blind_iou"] += float(r["temporal_iou"] or 0)
            t["video_iou"] += float(v["temporal_iou"] or 0)
            same += (r["pred_answer_parsed"] or "").strip().lower() == (v["pred_answer_parsed"] or "").strip().lower()
            only_blind += bj[q] and not vj[q]
            only_video += vj[q] and not bj[q]
            if r["question_category"] == "time_duration":
                d = ra.parse_duration(ra.model_answer(r))
                if d:
                    durations[f"{round(d)} s"] += 1
        n = len(blind)
        summary[prompt] = {
            "questions": n,
            "accuracy blind / with 64 frames %": [round(100 * sum(t["blind_ok"] for t in per_task.values()) / n, 1),
                                                  round(100 * sum(t["video_ok"] for t in per_task.values()) / n, 1)],
            "per task (accuracy and mIoU, blind vs with 64 frames)": {
                cat: {"n": t["n"],
                      "accuracy %": [round(100 * t["blind_ok"] / t["n"], 1), round(100 * t["video_ok"] / t["n"], 1)],
                      "mIoU": [round(100 * t["blind_iou"] / t["n"], 1), round(100 * t["video_iou"] / t["n"], 1)]}
                for cat, t in sorted(per_task.items())},
            "blind answer identical to the with-video answer %": round(100 * same / n, 1),
            "correct only blind (questions)": only_blind,
            "correct only with video (questions)": only_video,
            "most common blind duration answers": durations.most_common(8),
        }
    out = {
        "description": "Qwen3-VL-8B on S-EMBER grounded QA answered without any video frame (blind), with the "
                       "official and the timeline prompt; with_64_frames gives the same model's answer from 64 "
                       "uniform frames for comparison. answer_correct is the official S-EMBER judge verdict; iou is "
                       "against the gold interval (0 when no interval could be parsed).",
        "summary": summary,
        "videos": [{"video_id": vid, **meta[vid], "num_questions": len(qs),
                    "questions": sorted(qs.values(), key=lambda q: q["question_time_s"])}
                   for vid, qs in sorted(videos.items())],
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
