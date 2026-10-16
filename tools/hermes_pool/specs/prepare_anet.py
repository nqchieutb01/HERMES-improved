"""Phase 34: ActivityNet Captions (val_1) converted to the S-EMBER grounding JSONL, as prepare_qvh.py.

Videos: the first part (5.4 GB) of friedrichor/ActivityNet_Captions' video tar, extracted on its own (the last file,
cut at the part boundary, fails to decode and is skipped). Queries: one sentence per val_1 video (seed 0) among
the videos present, 500 queries; the question time is the end of the video, so the whole video is the window.
Usage: python prepare_anet.py
"""
import hashlib
import json
import os
import random

from decord import VideoReader

D = "/nfs-stor/chieu.nguyen/activitynet"
N = 500


def main():
    videos = {}
    for root, _, files in os.walk(f"{D}/videos"):
        for f in files:
            if f.endswith(".mp4"):
                videos[os.path.splitext(f)[0]] = os.path.join(root, f)
    anns = json.load(open(f"{D}/activitynet_captions_val1.json"))
    rng = random.Random(0)
    rng.shuffle(anns)
    os.makedirs(f"{D}/videos_flat", exist_ok=True)
    out, durations = [], []
    for a in anns:
        vid = a["video_id"]
        if vid not in videos:
            continue
        try:
            vr = VideoReader(videos[vid])
            dur = len(vr) / vr.get_avg_fps()
        except Exception:
            continue  # truncated at the tar part boundary
        k = rng.randrange(len(a["sentences"]))
        s, e = (float(x) for x in a["timestamps"][k])
        e = min(e, dur)
        if not 0 <= s < e:
            continue
        query = a["sentences"][k].strip()
        question = f'When does the following happen in the video: "{query}"?'
        qid = hashlib.sha1(f"anet|{vid}|{k}".encode()).hexdigest()
        out.append({"question_id": qid, "video_id": vid, "question": question, "question_time": round(dur, 3),
                    "question_category": "moment_retrieval", "answer": query, "answer_start_time": s,
                    "answer_end_time": e, "duration": round(dur, 3)})
        durations.append(dur)
        link = f"{D}/videos_flat/{vid}.mp4"
        if not os.path.lexists(link):
            os.symlink(videos[vid], link)
        if len(out) == N:
            break
    with open(f"{D}/anet_val1_500.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    lengths = [r["answer_end_time"] - r["answer_start_time"] for r in out]
    print(f"{len(out)} queries on {len({r['video_id'] for r in out})} videos; mean video "
          f"{sum(durations) / len(durations):.1f} s, mean moment {sum(lengths) / len(lengths):.1f} s")


if __name__ == "__main__":
    main()
