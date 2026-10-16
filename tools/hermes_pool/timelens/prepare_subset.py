"""TimeLens-Bench QVHighlights subset: every query whose video is in video shard 01 (no sampling).

Writes, for the same queries:
- subset/data/TimeLens-Bench/qvhighlights-timelens.json + videos/qvhighlights (links): TimeLens' own layout, so
  their evaluation/eval_dataloader.py runs unchanged from repo/ with data -> subset/data;
- qvh_tl_subset.jsonl: our S-EMBER grounding format (question = the TimeLens-normalised query, used with
  grounding_prompt=timelens; question time = video end; answer interval = the annotated span);
- videos_flat/<video>.mp4 links for our loader.
Usage: python3 prepare_subset.py   (after extracting shard 01 into /nfs-stor/chieu.nguyen/timelens-bench/videos/qvhighlights)
"""
import hashlib
import json
import os
import re

D = "/nfs-stor/chieu.nguyen/timelens-bench"
OUT = f"{D}/subset"


def parse_query(query):  # TimeLens timelens/dataset/timelens_data.py
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


def main():
    videos = {}
    for root, _, files in os.walk(f"{D}/videos/qvhighlights"):
        for f in files:
            if f.endswith(".mp4"):
                videos[os.path.splitext(f)[0]] = os.path.join(root, f)
    anns = json.load(open(f"{D}/qvhighlights-timelens.json"))
    sub = {vid: a for vid, a in anns.items() if vid in videos}
    tl_dir = f"{OUT}/data/TimeLens-Bench"
    os.makedirs(f"{tl_dir}/videos/qvhighlights", exist_ok=True)
    os.makedirs(f"{D}/videos_flat", exist_ok=True)
    json.dump(sub, open(f"{tl_dir}/qvhighlights-timelens.json", "w"))
    rows = []
    for vid, a in sub.items():
        for link in (f"{tl_dir}/videos/qvhighlights/{vid}.mp4", f"{D}/videos_flat/{vid}.mp4"):
            if not os.path.lexists(link):
                os.symlink(videos[vid], link)
        for span, query in zip(a["spans"], a["queries"]):
            q = parse_query(query)
            rows.append({"question_id": hashlib.sha1(f"tl-qvh|{vid}|{q}|{span}".encode()).hexdigest(),
                         "video_id": vid, "question": q, "question_time": a["duration"],
                         "question_category": "moment_retrieval", "answer": q, "answer_start_time": float(span[0]),
                         "answer_end_time": float(span[1]), "duration": a["duration"], "span": span})
    with open(f"{D}/qvh_tl_subset.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    durs = [a["duration"] for a in sub.values()]
    print(f"{len(rows)} queries on {len(sub)} videos (of {len(anns)}); mean video {sum(durs) / len(durs):.1f} s")


if __name__ == "__main__":
    main()
