"""Phase 33: QVHighlights (val) converted to the S-EMBER grounding JSONL, as logs/phase28/prepare_charades.py.

Only queries with exactly one relevant window (1,020 of 1,550): the official R1 metric takes the best IoU over all
windows; with one window it is the plain IoU every run here reports. A fixed random subset of 500 queries (seed 0);
the question time is the end of the video, so the whole video (~150 s) is the window.
Usage: python prepare_qvh.py   (after extracting videos.tar.gz under /nfs-stor/chieu.nguyen/qvhighlights)
"""
import hashlib
import json
import os
import random

from decord import VideoReader

D = "/nfs-stor/chieu.nguyen/qvhighlights"
N = 500


def find_videos():
    paths = {}
    for root, _, files in os.walk(f"{D}/videos"):
        for f in files:
            if f.endswith(".mp4"):
                paths[os.path.splitext(f)[0]] = os.path.join(root, f)
    return paths


def main():
    rows = [json.loads(l) for l in open(f"{D}/highlight_val_release.jsonl")]
    rows = [r for r in rows if len(r["relevant_windows"]) == 1]
    random.Random(0).shuffle(rows)
    videos = find_videos()
    out, durations = [], {}
    for r in rows:
        vid = r["vid"]
        if vid not in videos:
            continue
        if vid not in durations:
            vr = VideoReader(videos[vid])
            durations[vid] = len(vr) / vr.get_avg_fps()
        dur = durations[vid]
        s, e = (float(x) for x in r["relevant_windows"][0])
        e = min(e, dur)
        if not 0 <= s < e:
            continue
        query = r["query"].strip()
        question = f'When does the following happen in the video: "{query}"?'
        qid = hashlib.sha1(f"qvh|{r['qid']}".encode()).hexdigest()
        out.append({"question_id": qid, "video_id": vid, "video_path": videos[vid], "question": question,
                    "question_time": round(dur, 3), "question_category": "moment_retrieval", "answer": query,
                    "answer_start_time": s, "answer_end_time": e, "duration": round(dur, 3), "qvh_qid": r["qid"]})
        if len(out) == N:
            break
    # The S-EMBER loader reads <video_root>/<video_id>.mp4: a flat directory of links, whatever the tar layout.
    os.makedirs(f"{D}/videos_flat", exist_ok=True)
    for r in out:
        link = f"{D}/videos_flat/{r['video_id']}.mp4"
        if not os.path.lexists(link):
            os.symlink(r["video_path"], link)
    with open(f"{D}/qvh_val_single_500.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    lengths = [r["answer_end_time"] - r["answer_start_time"] for r in out]
    print(f"{len(out)} queries on {len({r['video_id'] for r in out})} videos; mean video "
          f"{sum(durations.values()) / len(durations):.1f} s, mean moment {sum(lengths) / len(lengths):.1f} s")


if __name__ == "__main__":
    main()
