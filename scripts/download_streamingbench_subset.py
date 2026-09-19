#!/usr/bin/env python3
"""Download the StreamingBench Real-Time Visual Understanding CR/SU/EU subset.

The Hugging Face dataset stores videos in large ZIP archives. HTTP byte ranges
are used to inspect each ZIP directory and extract only the required members.
The ZIP archives themselves are never downloaded in full.
"""

import argparse
import binascii
import io
import json
import pathlib
import shutil
import urllib.parse
import urllib.request
import zipfile


DATASET = "mjuicem/StreamingBench"
DEFAULT_ANNOTATIONS = "data/streamingbench/streamingbench_realtime.json"
SELECTED_TASKS = frozenset({"Causal Reasoning", "Spatial Understanding", "Event Understanding"})
ARCHIVE_SIZE = 50
REQUEST_TIMEOUT = 120


class RemoteZip(io.RawIOBase):
    """A seekable, read-only file backed by HTTP Range requests."""

    def __init__(self, url):
        request = urllib.request.Request(url, headers={"Range": "bytes=-65536"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 206:
                raise RuntimeError("Server did not honor the byte-range request")
            content_range = response.headers.get("Content-Range", "")
            if "/" not in content_range:
                raise RuntimeError("Server omitted total size in Content-Range")
            self.url = response.url
            self.size = int(content_range.rsplit("/", 1)[1])
            self.tail = response.read()
        self.tail_start = self.size - len(self.tail)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the ZIP")
        self.position = position
        return position

    def read(self, size=-1):
        if self.position >= self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        start = self.position
        if start >= self.tail_start:
            data = self.tail[start - self.tail_start : start - self.tail_start + size]
        else:
            end = start + size - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-"):
                    raise RuntimeError("Server returned an unexpected byte range")
                data = response.read()
            if len(data) != size:
                raise RuntimeError("Incomplete HTTP range response")
        self.position += len(data)
        return data


def sample_name(record):
    return pathlib.PurePosixPath(record["video_path"]).name


def archive_for(sample):
    prefix, number, suffix = sample.rsplit("_", 2)
    if prefix != "sample" or suffix != "real.mp4":
        raise ValueError(f"Unexpected StreamingBench video name: {sample}")
    sample_number = int(number)
    if sample_number < 1:
        raise ValueError(f"Unexpected sample number: {sample_number}")
    first = ((sample_number - 1) // ARCHIVE_SIZE) * ARCHIVE_SIZE + 1
    return f"Real-Time Visual Understanding_{first}-{first + ARCHIVE_SIZE - 1}.zip"


def crc32(path):
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def download_member(source, info, target):
    if target.exists() and target.stat().st_size == info.file_size and crc32(target) == info.CRC:
        print(f"Ready: {target}", flush=True)
        return False
    if target.exists():
        print(f"Checksum mismatch, redownloading: {target}", flush=True)
    temporary = target.with_name(target.name + ".partial")
    print(f"Downloading {info.filename} ({info.file_size / 1e6:.1f} MB)", flush=True)
    with source.open(info) as member, temporary.open("wb") as output:
        shutil.copyfileobj(member, output, length=8 * 1024 * 1024)
    if temporary.stat().st_size != info.file_size or crc32(temporary) != info.CRC:
        temporary.unlink()
        raise RuntimeError(f"Checksum validation failed for {info.filename}")
    temporary.replace(target)
    return True


def parse_args():
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=pathlib.Path, default=root / DEFAULT_ANNOTATIONS,
                        help="Bundled StreamingBench annotation JSON")
    parser.add_argument("--output", type=pathlib.Path,
                        default=root / "data/streamingbench/realtime_cr_su_eu",
                        help="Output directory for videos, annotations, and manifest")
    parser.add_argument("--revision", help="Hugging Face commit; defaults to current main revision")
    return parser.parse_args()


def main():
    args = parse_args()
    annotations_path = args.annotations.resolve()
    destination = args.output.resolve()
    videos = destination / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    annotations = json.loads(annotations_path.read_text())
    selected = []
    for record in annotations:
        questions = [q for q in record["conversations"] if q.get("task") in SELECTED_TASKS]
        if questions:
            selected.append({**record, "conversations": questions})
    if not selected:
        raise RuntimeError(f"No questions found for tasks: {sorted(SELECTED_TASKS)}")

    archives = {}
    for record in selected:
        archives.setdefault(archive_for(sample_name(record)), set()).add(sample_name(record))
    revision = args.revision
    if revision is None:
        api_url = f"https://huggingface.co/api/datasets/{DATASET}"
        with urllib.request.urlopen(api_url, timeout=REQUEST_TIMEOUT) as response:
            revision = json.load(response)["sha"]

    manifest = {"dataset": DATASET, "revision": revision,
                "annotation_source": str(annotations_path), "tasks": sorted(SELECTED_TASKS),
                "videos": []}
    downloaded = 0
    for archive, requested_samples in sorted(archives.items()):
        url = (f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/"
               f"{urllib.parse.quote(archive)}")
        print(f"Reading ZIP directory: {archive}", flush=True)
        with zipfile.ZipFile(RemoteZip(url)) as source:
            entries = {entry.filename: entry for entry in source.infolist()}
            for sample in sorted(requested_samples):
                member_name = sample.removesuffix("_real.mp4") + "/video.mp4"
                if member_name not in entries:
                    raise RuntimeError(f"Missing ZIP member: {member_name}")
                info = entries[member_name]
                target = videos / sample
                downloaded += download_member(source, info, target)
                manifest["videos"].append({"file": str(target), "archive": archive,
                                           "member": member_name, "bytes": info.file_size,
                                           "zip_crc32": f"{info.CRC:08x}"})

    for record in selected:
        record["video_path"] = str(videos / sample_name(record))
    filtered_path = destination / "streamingbench_realtime_cr_su_eu.json"
    filtered_path.write_text(json.dumps(selected, indent=2) + "\n")
    manifest["question_count"] = sum(len(r["conversations"]) for r in selected)
    manifest["video_count"] = len(selected)
    manifest["downloaded_count"] = downloaded
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(selected)} videos / {manifest['question_count']} questions to {destination}", flush=True)


if __name__ == "__main__":
    main()
