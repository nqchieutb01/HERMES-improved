"""Full TimeLens-Bench (Charades / ActivityNet / QVHighlights-TimeLens) in both layouts:
- full/data/TimeLens-Bench/<ds>-timelens.json + videos/<ds> -> the extracted shards: TimeLens' own layout, so their
  unmodified evaluation runs from repo_full/ (data -> full/data);
- <ds>_full.jsonl: our grounding format (question = TimeLens-normalised query, used with grounding_prompt=timelens;
  question time = video end; answer interval = the annotated span), one row per (video, query, span), as their loader.
Usage: python3 prepare_full.py
"""
import hashlib
import json
import os
import re

D = "/nfs-stor/chieu.nguyen/timelens-bench"
OUT = f"{D}/full/data/TimeLens-Bench"


def parse_query(query):  # TimeLens timelens/dataset/timelens_data.py
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


def main():
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    for ds in ("charades", "activitynet", "qvhighlights"):
        link = f"{OUT}/videos/{ds}"
        if not os.path.lexists(link):
            os.symlink(f"{D}/videos/{ds}/{ds}", link)
        anns = json.load(open(f"{D}/{ds}-timelens.json"))
        src = f"{OUT}/{ds}-timelens.json"
        if not os.path.lexists(src):
            os.symlink(f"{D}/{ds}-timelens.json", src)
        rows, missing = [], 0
        for vid, a in anns.items():
            if not os.path.exists(f"{D}/videos/{ds}/{ds}/{vid}.mp4"):
                missing += 1
                continue
            for span, query in zip(a["spans"], a["queries"]):
                q = parse_query(query)
                rows.append({"question_id": hashlib.sha1(f"tl-{ds}|{vid}|{q}|{span}".encode()).hexdigest(),
                             "video_id": vid, "question": q, "question_time": a["duration"],
                             "question_category": "moment_retrieval", "answer": q,
                             "answer_start_time": float(span[0]), "answer_end_time": float(span[1]),
                             "duration": a["duration"], "span": span})
        with open(f"{D}/{ds}_full.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{ds}: {len(rows)} queries on {len(anns) - missing} videos ({missing} videos missing)")


if __name__ == "__main__":
    main()
