#!/usr/bin/env python3
"""Download and prepare selected StreamingBench Real-Time tasks.

The official StreamingBench repository stores the videos in large ZIP archives.
This script reads each archive's central directory with HTTP byte ranges and
extracts only videos referenced by the requested tasks.  It never downloads a
complete archive to disk.

Examples:
    python scripts/data/download_streamingbench_tasks.py --dry-run
    python scripts/data/download_streamingbench_tasks.py --tasks CR SU EU
    python scripts/data/download_streamingbench_tasks.py --tasks "Causal Reasoning" "Spatial Understanding"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import pathlib
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter


DATASET_ID = "mjuicem/StreamingBench"
REALTIME_PREFIX = "Real-Time Visual Understanding"
TASKS = {
    "CR": "Causal Reasoning",
    "SU": "Spatial Understanding",
    "EU": "Event Understanding",
}
TASK_ALIASES = {
    **{key.lower(): key for key in TASKS},
    **{name.lower(): key for key, name in TASKS.items()},
}


class RemoteZip(io.RawIOBase):
    """A seekable read-only file backed by HTTP byte-range requests."""

    def __init__(self, url: str, timeout: int = 120, retries: int = 3) -> None:
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.position = 0

        response = self._request("bytes=-65536")
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or "/" not in content_range:
            response.close()
            raise RuntimeError(
                "The dataset server did not honor the byte-range request "
                f"for {url} (status={response.status}, "
                f"content-range={content_range!r})"
            )
        self.url = response.url
        self.size = int(content_range.rsplit("/", 1)[1])
        self.tail = response.read()
        response.close()

    def _request(self, byte_range: str):
        last_error = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    self.url, headers={"Range": byte_range}
                )
                return urllib.request.urlopen(request, timeout=self.timeout)
            except Exception as error:  # urllib errors are transient in practice.
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Unable to read byte range {byte_range} from {self.url}"
        ) from last_error

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_position = offset
        elif whence == 1:
            new_position = self.position + offset
        elif whence == 2:
            new_position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if new_position < 0:
            raise ValueError("Cannot seek before the beginning of the archive")
        self.position = new_position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""

        start = self.position
        tail_start = self.size - len(self.tail)
        if start >= tail_start:
            data = self.tail[start - tail_start : start - tail_start + size]
        else:
            end = start + size - 1
            response = self._request(f"bytes={start}-{end}")
            content_range = response.headers.get("Content-Range", "")
            data = response.read()
            response.close()
            if response.status != 206 or not content_range.startswith(
                f"bytes {start}-"
            ):
                raise RuntimeError(
                    f"Unexpected byte range response for {self.url}: "
                    f"status={response.status}, content-range={content_range!r}"
                )
            if len(data) != size:
                raise RuntimeError(
                    f"Incomplete byte range response for {self.url}: "
                    f"expected {size}, got {len(data)}"
                )
        self.position += len(data)
        return data


def parse_tasks(values: list[str]) -> list[str]:
    selected = []
    for value in values:
        key = TASK_ALIASES.get(value.strip().lower())
        if key is None:
            valid = ", ".join(f"{key} ({name})" for key, name in TASKS.items())
            raise argparse.ArgumentTypeError(
                f"Unknown task {value!r}; choose from {valid}"
            )
        if key not in selected:
            selected.append(key)
    return selected


def sample_number(video_path: str) -> int:
    name = pathlib.PurePosixPath(video_path).name
    stem = name.removesuffix("_real.mp4")
    try:
        return int(stem.removeprefix("sample_"))
    except ValueError as error:
        raise ValueError(f"Cannot derive sample number from {video_path!r}") from error


def archive_name(number: int) -> str:
    start = ((number - 1) // 50) * 50 + 1
    return f"{REALTIME_PREFIX}_{start}-{start + 49}.zip"


def load_selection(annotation_path: pathlib.Path, task_keys: list[str]):
    annotations = json.loads(annotation_path.read_text())
    target_names = {TASKS[key] for key in task_keys}
    selected = []
    for record in annotations:
        conversations = [
            question
            for question in record.get("conversations", [])
            if question.get("task") in target_names
        ]
        if not conversations:
            continue
        # Hermes consumes a video prefix once and expects questions in time order.
        conversations = sorted(
            conversations,
            key=lambda question: question.get("end_time", float("inf")),
        )
        prepared = dict(record)
        prepared["conversations"] = conversations
        selected.append(prepared)
    return selected


def dataset_revision() -> str:
    url = f"https://huggingface.co/api/datasets/{DATASET_ID}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)["sha"]


def download_video(
    source: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    target: pathlib.Path,
) -> None:
    if target.exists() and target.stat().st_size == member.file_size:
        print(f"Ready: {target} ({member.file_size / 1e6:.1f} MB)", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        partial.unlink()
    print(
        f"Downloading {member.filename}: {member.file_size / 1e6:.1f} MB -> {target}",
        flush=True,
    )
    with source.open(member) as input_stream, partial.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
    if partial.stat().st_size != member.file_size:
        partial.unlink()
        raise RuntimeError(
            f"Downloaded size mismatch for {member.filename}: "
            f"expected {member.file_size}, got {partial.stat().st_size}"
        )
    partial.replace(target)


def download_member_from_url(
    url: str,
    member: zipfile.ZipInfo,
    target: pathlib.Path,
) -> dict:
    """Download one member with a ZIP reader dedicated to this worker."""
    with zipfile.ZipFile(RemoteZip(url)) as source:
        download_video(source, member, target)
    return {
        "file": str(target),
        "member": member.filename,
        "bytes": member.file_size,
        "zip_crc32": f"{member.CRC:08x}",
    }


def write_outputs(
    destination: pathlib.Path,
    selected: list[dict],
    task_keys: list[str],
    revision: str,
    files: list[dict],
    annotation_source: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    destination.mkdir(parents=True, exist_ok=True)
    videos = destination / "videos"
    prepared = []
    for record in selected:
        item = dict(record)
        basename = pathlib.PurePosixPath(record["video_path"]).name
        item["video_path"] = str(videos / basename)
        prepared.append(item)

    annotation_output = destination / "streamingbench_realtime_cr_su_eu.json"
    annotation_output.write_text(json.dumps(prepared, indent=2) + "\n")
    manifest = {
        "dataset": DATASET_ID,
        "revision": revision,
        "annotation_source": str(annotation_source),
        "annotation_output": str(annotation_output),
        "tasks": {key: TASKS[key] for key in task_keys},
        "video_count": len(prepared),
        "question_count": sum(len(record["conversations"]) for record in prepared),
        "archives": sorted({entry["archive"] for entry in files}),
        "videos": files,
    }
    manifest_output = destination / "manifest_cr_su_eu.json"
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    return annotation_output, manifest_output


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["CR", "SU", "EU"],
        metavar="TASK",
        help="Task acronyms or full names (default: CR SU EU)",
    )
    parser.add_argument(
        "--annotation",
        type=pathlib.Path,
        default=repo_root / "data/streamingbench/streamingbench_realtime.json",
        help="Bundled StreamingBench annotation JSON",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/nfs-stor/chieu.nguyen/StreamingBench"),
        help="Dataset output directory; keep large videos on shared storage",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for a small validation run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel video downloads per archive (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selection and archives without contacting Hugging Face",
    )
    args = parser.parse_args()
    try:
        task_keys = parse_tasks(args.tasks)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not args.annotation.is_file():
        parser.error(f"Annotation file does not exist: {args.annotation}")

    selected = load_selection(args.annotation, task_keys)
    if args.max_videos is not None:
        selected = selected[: args.max_videos]
    video_names = [pathlib.PurePosixPath(item["video_path"]).name for item in selected]
    archives = sorted({archive_name(sample_number(name)) for name in video_names})
    question_counts = Counter(
        question["task"]
        for record in selected
        for question in record["conversations"]
    )
    print(
        f"Selected {len(selected)} videos / "
        f"{sum(question_counts.values())} questions for "
        f"{', '.join(f'{key} ({TASKS[key]})' for key in task_keys)}",
        flush=True,
    )
    print(f"Questions by task: {dict(question_counts)}", flush=True)
    print(f"Archives: {', '.join(archives)}", flush=True)
    if args.dry_run:
        return

    revision = dataset_revision()
    print(f"Using Hugging Face revision {revision}", flush=True)
    videos = args.output.resolve() / "videos"
    files = []
    by_archive = {}
    for record in selected:
        basename = pathlib.PurePosixPath(record["video_path"]).name
        by_archive.setdefault(archive_name(sample_number(basename)), []).append(basename)

    for archive in archives:
        encoded = urllib.parse.quote(archive, safe="")
        url = (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
            f"{revision}/{encoded}"
        )
        print(f"Reading ZIP directory: {archive}", flush=True)
        with zipfile.ZipFile(RemoteZip(url)) as source:
            entries = {entry.filename: entry for entry in source.infolist()}
            jobs = []
            for basename in by_archive[archive]:
                member_name = basename.removesuffix("_real.mp4") + "/video.mp4"
                member = entries.get(member_name)
                if member is None:
                    raise RuntimeError(
                        f"Could not find {member_name} in {archive}; "
                        f"available matching entries: "
                        f"{[name for name in entries if basename.removesuffix('_real.mp4') in name]}"
                    )
                jobs.append((basename, member))

        def download_job(job):
            basename, member = job
            target = videos / basename
            result = download_member_from_url(url, member, target)
            result["archive"] = archive
            return result

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(jobs))
        ) as executor:
            futures = [executor.submit(download_job, job) for job in jobs]
            for future in futures:
                files.append(future.result())

    annotation_output, manifest_output = write_outputs(
        args.output.resolve(),
        selected,
        task_keys,
        revision,
        files,
        args.annotation.resolve(),
    )
    print(f"Wrote annotations: {annotation_output}", flush=True)
    print(f"Wrote provenance: {manifest_output}", flush=True)


if __name__ == "__main__":
    main()
