"""Collect Qwen3-VL-8B uniform-64-frame results on S-EMBER, grouped by video and question, as readable JSON.

Grounding questions carry the gold answer and interval plus the model output with the official prompt and
with the timeline prompt (raw response, parsed answer and interval, IoU, official-judge verdict). MCQ
questions carry the options, the gold letter and the model's choice. Both are unpruned (no token pruning).
Writes logs/phase14/uniform64_by_question.json. Run from the repo root:
python3 logs/phase14/collect_uniform64.py
"""
import collections
import json

B = "results/qwen3_vl_8b"
RUNS = {"official": "uniform-n64-time-count-location-v300-native-time",
        "timeline": "uniform-n64-time-count-location-timeline-v300-native-time"}
OUT = "logs/phase14/uniform64_by_question.json"


def load(path):
    return [json.loads(l) for l in open(path)]


def num(x):
    return None if x in (None, "") else round(float(x), 2)


def main():
    videos = collections.defaultdict(lambda: {"grounding": {}, "mcq": []})
    meta = {}
    for prompt, tag in RUNS.items():
        d = f"{B}/sember_grounding/{tag}"
        judge = {j["question_id"]: bool(j["correct"]) for j in load(f"{d}/answer_judgments.jsonl")}
        for r in load(f"{d}/sember_grounding_scored.jsonl"):
            meta[r["video_id"]] = {"duration_s": num(r["duration"]), "category": r["video_category"]}
            q = videos[r["video_id"]]["grounding"].setdefault(r["question_id"], {
                "question_id": r["question_id"],
                "question": r["question"],
                "category": r["question_category"],
                "question_time_s": num(r["question_time"]),
                "gold_answer": r["answer"],
                "gold_interval_s": [num(r["answer_start_time"]), num(r["answer_end_time"])],
                "outputs": {},
            })
            q["outputs"][prompt] = {
                "raw_response": (r["pred_raw"] or "").strip(),
                "answer": r["pred_answer_parsed"],
                "interval_s": [num(r["pred_start_time"]), num(r["pred_end_time"])] if r["interval_parseable"] else None,
                "iou": round(float(r["temporal_iou"] or 0), 3),
                "answer_correct": judge[r["question_id"]],
            }
    for r in load(f"{B}/sember_mcq/{RUNS['official']}/sember_mcq_scored.jsonl"):
        meta.setdefault(r["video_id"], {"duration_s": num(r["duration"]), "category": r["video_category"]})
        videos[r["video_id"]]["mcq"].append({
            "question_id": r["question_id"],
            "question": r["question"],
            "category": r["question_category"],
            "question_time_s": num(r["question_time"]),
            "options": json.loads(r["options_json"]) if r.get("options_json") else r["choices"],
            "gold": r["correct_letter"],
            "predicted": r["pred_letter"],
            "correct": r["is_correct"] in (True, "True", "true", 1),
        })
    # Each task samples its own 300 videos, so the two sets only partly overlap.
    grounding_counts = collections.Counter(len(v["grounding"]) for v in videos.values() if v["grounding"])
    mcq_counts = collections.Counter(len(v["mcq"]) for v in videos.values() if v["mcq"])
    out = {
        "description": "Qwen3-VL-8B on S-EMBER, 64 uniformly sampled frames per question, no token pruning. "
                       "Grounding outputs with the official and the timeline prompt; answer_correct is the "
                       "official S-EMBER judge verdict; iou is against the gold interval (0 when no interval "
                       "could be parsed).",
        "summary": {
            "videos": len(videos),
            "grounding_videos": sum(grounding_counts.values()),
            "mcq_videos": sum(mcq_counts.values()),
            "videos_with_both_tasks": sum(1 for v in videos.values() if v["grounding"] and v["mcq"]),
            "grounding_questions": sum(len(v["grounding"]) for v in videos.values()),
            "mcq_questions": sum(len(v["mcq"]) for v in videos.values()),
            "grounding_questions_per_video (questions: videos)": {str(k): n for k, n in sorted(grounding_counts.items())},
            "mcq_questions_per_video (questions: videos)": {str(k): n for k, n in sorted(mcq_counts.items())},
        },
        "videos": [
            {"video_id": vid, **meta[vid], "num_grounding_questions": len(v["grounding"]),
             "num_mcq_questions": len(v["mcq"]),
             "grounding": sorted(v["grounding"].values(), key=lambda q: q["question_time_s"]),
             "mcq": sorted(v["mcq"], key=lambda q: q["question_time_s"])}
            for vid, v in sorted(videos.items())
        ],
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
