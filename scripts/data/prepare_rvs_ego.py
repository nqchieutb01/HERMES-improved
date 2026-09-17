#!/usr/bin/env python3
"""Extract the official RVS-Ego split ZIP and prepare HERMES annotations."""
import argparse
import bisect
import io
import json
from pathlib import Path
import shutil
import zipfile


class SplitArchive(io.RawIOBase):
    """Expose concatenated ZIP parts without creating another 28 GB file."""

    def __init__(self, paths):
        super().__init__()
        self.files = [path.open("rb") for path in paths]
        self.offsets = [0]
        for path in paths:
            self.offsets.append(self.offsets[-1] + path.stat().st_size)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.offsets[-1])
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.offsets[-1] - self.position)
        size = remaining if size < 0 else min(size, remaining)
        chunks = []
        while size:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            source = self.files[index]
            source.seek(self.position - self.offsets[index])
            data = source.read(min(size, self.offsets[index + 1] - self.position))
            if not data:
                raise EOFError("Truncated archive part")
            chunks.append(data)
            size -= len(data)
            self.position += len(data)
        return b"".join(chunks)

    def close(self):
        for source in self.files:
            source.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/nfs-stor/chieu.nguyen/VStream-QA"))
    parser.add_argument("--extract", action="store_true", help="Extract and CRC-check every RVS-Ego frame")
    args = parser.parse_args()
    root = args.dataset_root.resolve() / "vstream-realtime"
    parts = [root / f"ego4d_frames_online.part{suffix}" for suffix in ("aa", "ab", "ac")]
    with SplitArchive(parts) as stream, zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        print(f"Archive: {len(members)} entries, {sum(m.file_size for m in members) / 1e9:.2f} GB unpacked", flush=True)
        if args.extract:
            for index, member in enumerate(members):
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
                if index % 10000 == 0:
                    print(f"Extracted {index}/{len(members)} entries", flush=True)

    clips = {item["video_id"]: item for item in json.loads((root / "rvs_ego.json").read_text())}
    questions = json.loads((root / "test_qa_ego4d.json").read_text())
    grouped = {}
    for question in questions:
        video_id = question["video_id"]
        clip = clips[video_id]
        frames = root / "ego4d_frames" / video_id
        if video_id not in grouped:
            frame_files = sorted(frames.glob("*.jpg"))
            duration = clip["end_time"] - clip["start_time"]
            # Official frames are sampled at 1 FPS, with an optional endpoint.
            if len(frame_files) not in (duration, duration + 1):
                raise ValueError(f"{video_id}: expected {duration} or {duration + 1} frames at 1 FPS, found {len(frame_files)}")
            grouped[video_id] = {
                "video_id": video_id, "video_path": str(frames), "fps": 1,
                "duration": duration, "conversations": [],
            }
        grouped[video_id]["conversations"].append({
            "question": question["question"], "answer": question["answer"],
            "start_time": question["start_time"] - clip["start_time"],
            "end_time": question["end_time"] - clip["start_time"],
            "question_id": question["id"], "task": question["answer_type"],
        })
    for sample in grouped.values():
        sample["conversations"].sort(key=lambda item: item["end_time"])
    destination = root / "rvs_ego_hermes.json"
    destination.write_text(json.dumps(list(grouped.values()), indent=2) + "\n")
    print(f"Prepared {len(grouped)} clips / {len(questions)} questions: {destination}", flush=True)


if __name__ == "__main__":
    main()
