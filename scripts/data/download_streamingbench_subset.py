#!/usr/bin/env python3
"""Download three original StreamingBench videos using HTTP ZIP byte ranges.

Only the selected ZIP members are transferred, not the multi-GB archives.
Uses Python's standard library; annotations come from HERMES's bundled JSON.
"""
import argparse
import io
import json
import pathlib
import shutil
import urllib.parse
import urllib.request
import zipfile


REPO = "mjuicem/StreamingBench"
SAMPLES = {
    "sample_9_real.mp4": "Real-Time Visual Understanding_1-50.zip",
    "sample_154_real.mp4": "Real-Time Visual Understanding_151-200.zip",
    "sample_158_real.mp4": "Real-Time Visual Understanding_151-200.zip",
}


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        req = urllib.request.Request(url, headers={"Range": "bytes=-65536"})
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError("Server did not honor the byte-range request")
            self.url = response.url
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.tail = response.read()
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        start = self.position
        if start >= self.size - len(self.tail):
            data = self.tail[start - (self.size - len(self.tail)):][:size]
        else:
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + size - 1}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise RuntimeError("Server returned an unexpected byte range")
                data = response.read()
            if len(data) != size:
                raise RuntimeError("Incomplete HTTP range response")
        self.position += len(data)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2] / "data/streamingbench/subset")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    videos = destination / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=120) as response:
        revision = json.load(response)["sha"]
    manifest = {"dataset": REPO, "revision": revision, "annotation_source": "data/streamingbench/streamingbench_realtime.json", "selection": "Three short original videos, first two chronological questions per video", "videos": []}
    for archive in dict.fromkeys(SAMPLES.values()):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(archive)}"
        print(f"Reading ZIP directory: {archive}", flush=True)
        with zipfile.ZipFile(RemoteZip(url)) as source:
            for basename, sample_archive in SAMPLES.items():
                if sample_archive != archive:
                    continue
                member_name = basename.replace("_real.mp4", "/video.mp4")
                matches = [entry for entry in source.infolist() if entry.filename == member_name]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one member for {basename}, found {[entry.filename for entry in matches]}")
                info = matches[0]
                target = videos / basename
                if not target.exists() or target.stat().st_size != info.file_size:
                    print(f"Downloading {info.filename}: {info.file_size / 1e6:.1f} MB", flush=True)
                    temporary = target.with_suffix(".mp4.partial")
                    with source.open(info) as member, temporary.open("wb") as output:
                        shutil.copyfileobj(member, output, length=8 * 1024 * 1024)
                    temporary.replace(target)
                manifest["videos"].append({"file": str(target), "archive": archive, "member": info.filename, "bytes": info.file_size, "zip_crc32": f"{info.CRC:08x}"})
                print(f"Ready: {target}", flush=True)
    annotations = json.loads((root / manifest["annotation_source"]).read_text())
    selected = []
    for basename in SAMPLES:
        record = next(record for record in annotations if pathlib.PurePosixPath(record["video_path"]).name == basename)
        record["video_path"] = str(videos / basename)
        record["conversations"] = sorted(record["conversations"], key=lambda question: question["end_time"])[:2]
        selected.append(record)
    (destination / "streamingbench_realtime_subset.json").write_text(json.dumps(selected, indent=2) + "\n")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(selected)} videos / {sum(len(r['conversations']) for r in selected)} questions to {destination}", flush=True)


if __name__ == "__main__":
    main()
