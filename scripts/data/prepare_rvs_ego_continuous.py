#!/usr/bin/env python3
"""Map the original 10-stream HERMES annotations onto available RVS frames."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare(dataset_root, output_dir):
    source = ROOT / "data/rvs/ego/ego4d_oe.json"
    annotations = json.loads(source.read_text())
    data = dataset_root / "vstream-realtime"
    clips = json.loads((data / "rvs_ego.json").read_text())
    questions = json.loads((data / "test_qa_ego4d.json").read_text())
    by_video = defaultdict(list)
    for clip in clips:
        by_video[clip["original_video"]].append(clip)
    metadata = defaultdict(deque)
    for qa in questions:
        metadata[(qa["video_name"], qa["question"], qa["answer"])].append(qa)
    provenance = []
    for video in annotations:
        video_id = video["video_id"]
        clip = max(by_video[video_id], key=lambda x: x["end_time"])
        assert clip["start_time"] == 0, clip
        frames = data / "ego4d_frames" / clip["video_id"]
        names = sorted(p.name for p in frames.glob("*.jpg"))
        assert names == [f"{i:06d}.jpg" for i in range(1, len(names) + 1)], frames
        assert len(names) >= max(q["end_time"] for q in video["conversations"]), video_id
        times = [q["end_time"] for q in video["conversations"]]
        assert times == sorted(times), video_id
        video.update(video_path=str(frames.resolve()), fps=1)
        for qa in video["conversations"]:
            info = metadata[(video_id, qa["question"], qa["answer"])].popleft()
            qa.update(task=info["answer_type"], question_id=info["id"])
            assert math.ceil(qa["end_time"] * 0.5) <= len(names[::2])
        provenance.append({"video_id": video_id, "frame_prefix_id": clip["video_id"],
                           "frames": len(names), "last_question_time": max(times),
                           "questions": len(times)})
    assert not any(metadata.values()), "Unmatched question metadata"
    assert len(annotations) == 10
    assert sum(len(v["conversations"]) for v in annotations) == 1465
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "annotations.json"
    target.write_text(json.dumps(annotations, indent=2) + "\n")
    manifest = {
        "protocol": "10 continuous original videos; original question order and timestamps; cache reset only between videos",
        "annotation_source": str(source),
        "annotation_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "model": "llava_ov_0.5b", "sample_fps": 0.5, "kv_size": 6000,
        "encode_chunk_size": 16, "precision": "float16", "decoding": "greedy",
        "judge": "Qwen/Qwen3.8-27B-FP8; unchanged local rubric",
        "remaining_paper_differences": ["Qwen judge instead of GPT-3.5-turbo-0125",
                                       "Existing extracted 1-FPS JPEG frames instead of decoding original videos"],
        "videos": provenance,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated 10 continuous streams / 1465 questions: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/nfs-stor/chieu.nguyen/VStream-QA"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset_root, args.output_dir)
