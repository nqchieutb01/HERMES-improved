"""Collect the blind (no video) S-EMBER grounding outputs, grouped by video and question, as readable JSON.

Qwen3-VL-8B, 475 grounding questions (300 videos), answered from the question alone (run.blind=true), with the
official and the timeline prompt. Each question carries the gold answer and interval, the blind outputs (raw
response, parsed answer and interval, IoU, official-judge verdict), and for comparison the verdict and IoU of the
same prompt with 64 uniform frames. Writes logs/phase15/blind_by_question.json.
Run from the repo root: python3 logs/phase15/collect_blind.py
"""
import collections
import json

G = "results/qwen3_vl_8b/sember_grounding/uniform-n64-time-count-location-"
RUNS = {"official": ("dx-blind-v300-native-time", "v300-native-time"),
        "timeline": ("dx-blind-timeline-v300-native-time", "timeline-v300-native-time")}
OUT = "logs/phase15/blind_by_question.json"


def load(tag):
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{G}{tag}/sember_grounding_scored.jsonl")}
    judge = {json.loads(l)["question_id"]: bool(json.loads(l)["correct"]) for l in open(f"{G}{tag}/answer_judgments.jsonl")}
    return rows, judge


def num(x):
    return None if x in (None, "") else round(float(x), 2)


def main():
    videos = collections.defaultdict(dict)
    meta = {}
    summary = {}
    for prompt, (blind_tag, video_tag) in RUNS.items():
        rows, judge = load(blind_tag)
        vrows, vjudge = load(video_tag)
        for q, r in rows.items():
            meta[r["video_id"]] = {"duration_s": num(r["duration"]), "category": r["video_category"]}
            entry = videos[r["video_id"]].setdefault(q, {
                "question_id": q,
                "question": r["question"],
                "category": r["question_category"],
                "question_time_s": num(r["question_time"]),
                "gold_answer": r["answer"],
                "gold_interval_s": [num(r["answer_start_time"]), num(r["answer_end_time"])],
                "blind": {},
                "with_video_64_frames": {},
            })
            entry["blind"][prompt] = {
                "raw_response": (r["pred_raw"] or "").strip(),
                "answer": r["pred_answer_parsed"],
                "interval_s": [num(r["pred_start_time"]), num(r["pred_end_time"])] if r["interval_parseable"] else None,
                "iou": round(float(r["temporal_iou"] or 0), 3),
                "answer_correct": judge[q],
            }
            v = vrows.get(q)
            if v:
                entry["with_video_64_frames"][prompt] = {"answer": v["pred_answer_parsed"],
                                                         "iou": round(float(v["temporal_iou"] or 0), 3),
                                                         "answer_correct": vjudge.get(q)}
        by_cat = collections.defaultdict(list)
        for q, r in rows.items():
            by_cat[r["question_category"]].append(judge[q])
        both = [q for q in rows if q in vjudge]
        summary[prompt] = {
            "questions": len(rows),
            "blind accuracy %": round(100 * sum(judge.values()) / len(rows), 1),
            "blind accuracy by category %": {c: round(100 * sum(v) / len(v), 1) for c, v in sorted(by_cat.items())},
            "blind mIoU": round(100 * sum(float(r["temporal_iou"] or 0) for r in rows.values()) / len(rows), 1),
            "correct blind AND with video": sum(judge[q] and vjudge[q] for q in both),
            "correct blind only": sum(judge[q] and not vjudge[q] for q in both),
            "correct with video only": sum(vjudge[q] and not judge[q] for q in both),
        }
    out = {
        "description": "Qwen3-VL-8B on S-EMBER grounding, answered without any video frame (blind). "
                       "answer_correct is the official S-EMBER judge verdict; iou is against the gold interval "
                       "(0 when no interval could be parsed). with_video_64_frames: the same prompt with 64 "
                       "uniform frames, for comparison.",
        "summary": summary,
        "videos": [{"video_id": vid, **meta[vid], "questions": sorted(qs.values(), key=lambda e: e["question_time_s"])}
                   for vid, qs in sorted(videos.items())],
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
