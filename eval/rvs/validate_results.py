#!/usr/bin/env python3
"""Validate RVS-Ego prediction coverage and CSV integrity without an answer judge.

Exit 0 on success, 1 on invalid results. Uses only the Python standard library.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_FIELDS = ("video_id", "question", "answer", "task")


def validate(annotations, rows, fieldnames):
    errors = []
    required = set(KEY_FIELDS) | {"pred_answer"}
    missing_columns = sorted(required - set(fieldnames or []))
    if missing_columns:
        errors.append(f"Missing CSV columns: {', '.join(missing_columns)}")

    expected = Counter(
        (str(video["video_id"]), qa["question"], qa["answer"], qa["task"])
        for video in annotations for qa in video["conversations"]
    )
    observed = Counter(tuple(row.get(key) for key in KEY_FIELDS) for row in rows)
    missing = expected - observed
    unexpected = observed - expected
    if missing:
        errors.append(f"Missing or mismatched expected records: {sum(missing.values())}")
    if unexpected:
        errors.append(f"Unexpected or excess duplicate records: {sum(unexpected.values())}")

    empty_rows = [i for i, row in enumerate(rows, 1) if not (row.get("pred_answer") or "").strip()]
    malformed_rows = [i for i, row in enumerate(rows, 1)
                      if None in row or any(value is None for value in row.values())]
    if empty_rows:
        errors.append(f"Empty predictions: {len(empty_rows)}")
    if malformed_rows:
        errors.append(f"Malformed CSV records: {len(malformed_rows)}")
    if not expected:
        errors.append("Annotation file contains no questions")

    def examples(counter):
        return [dict(zip(KEY_FIELDS, key), count=count)
                for key, count in list(counter.items())[:10]]

    return {
        "status": "FAIL" if errors else "PASS",
        "checks": "Coverage, annotation metadata, CSV structure, and nonempty predictions; no answer correctness judging",
        "expected_clips": len({str(v["video_id"]) for v in annotations}),
        "observed_clips": len({row.get("video_id") for row in rows}),
        "expected_questions": sum(expected.values()),
        "prediction_rows": len(rows),
        "nonempty_predictions": len(rows) - len(empty_rows),
        "missing_records": sum(missing.values()),
        "unexpected_records": sum(unexpected.values()),
        "empty_prediction_records": empty_rows,
        "malformed_records": malformed_rows,
        "missing_examples": examples(missing),
        "unexpected_examples": examples(unexpected),
        "task_counts": dict(sorted(Counter(row.get("task") or "<missing>" for row in rows).items())),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=REPO_ROOT / "results/llava_ov_0.5b/rvs_ego/fps0.5-kv1024/results.csv")
    default_anno = Path(
        os.environ.get(
            "VSTREAM_QA_ROOT",
            str(REPO_ROOT / "data/vstream-qa"),
        )
    ) / "vstream-realtime/rvs_ego_hermes.json"
    parser.add_argument("--anno-path", type=Path, default=default_anno)
    parser.add_argument("--report-path", type=Path, help="Defaults to validation.json beside the predictions")
    args = parser.parse_args()
    report_path = args.report_path or args.results_path.with_name("validation.json")
    try:
        annotations = json.loads(args.anno_path.read_text())
        with args.results_path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source, strict=True)
            rows = list(reader)
            report = validate(annotations, rows, reader.fieldnames)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        report = {"status": "FAIL", "errors": [str(exc)]}
    report.update(results_path=str(args.results_path.resolve()), anno_path=str(args.anno_path.resolve()))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}: RVS-Ego output validation (no judging)")
    if "prediction_rows" in report:
        print(f"Clips: {report['observed_clips']}/{report['expected_clips']}")
        print(f"Questions: {report['prediction_rows']}/{report['expected_questions']}")
        print(f"Nonempty predictions: {report['nonempty_predictions']}")
    for error in report["errors"]:
        print(f"  ERROR: {error}")
    print(f"Report: {report_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
