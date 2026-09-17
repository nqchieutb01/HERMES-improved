#!/usr/bin/env python3
"""Evaluate S-EMBER temporal grounding predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
from typing import Any


QUESTION_CATEGORIES = (
    "location_trace",
    "sequential_action",
    "counting_objects_events",
    "visual_detail_recall",
    "temporal_ordering_recognition",
    "time_duration",
    "object_comparison",
    "spatial_aware_reasoning",
)


def parse_grounding_response(
    prediction: str,
) -> tuple[str, float | None, float | None]:
    """Parse the answer and interval using the official S-EMBER format."""
    prediction = prediction or ""
    answer_text = prediction.strip()
    pred_start = None
    pred_end = None
    answer_match = re.search(r"Answer:\s*(.+?)(?:\n|Time:)", prediction, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    time_match = re.search(
        r"Time:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", prediction
    )
    if time_match:
        try:
            pred_start = float(time_match.group(1))
            pred_end = float(time_match.group(2))
        except ValueError:
            pass
    return answer_text, pred_start, pred_end


def temporal_iou(
    pred_start: float | None,
    pred_end: float | None,
    gt_start: float | None,
    gt_end: float | None,
) -> float:
    """Return temporal IoU, with missing or non-overlapping intervals at zero."""
    if any(value is None for value in (pred_start, pred_end, gt_start, gt_end)):
        return 0.0
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


def _required_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number} has invalid {field!r}: {row.get(field)!r}") from exc


def score_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Parse and score prediction rows."""
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        question_id = (row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Row {row_number} has no question_id")
        if question_id in seen:
            raise ValueError(f"Duplicate prediction question_id: {question_id}")
        seen.add(question_id)
        category = (row.get("question_category") or row.get("task") or "unknown").strip()
        prediction = row.get("pred_raw") or row.get("pred_answer") or ""
        pred_answer, pred_start, pred_end = parse_grounding_response(prediction)
        gt_start = _required_float(row, "answer_start_time", row_number)
        gt_end = _required_float(row, "answer_end_time", row_number)
        iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
        scored.append(
            {
                **row,
                "question_id": question_id,
                "question_category": category,
                "pred_raw": prediction,
                "pred_answer_parsed": pred_answer,
                "pred_start_time": pred_start,
                "pred_end_time": pred_end,
                "temporal_iou": iou,
                "recall_at_1_iou_0.5": bool(iou >= 0.5),
                "interval_parseable": pred_start is not None and pred_end is not None,
            }
        )
    if not scored:
        raise ValueError("Prediction file contains no rows")
    return scored


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parseable = sum(bool(row["interval_parseable"]) for row in rows)
    total_iou = sum(float(row["temporal_iou"]) for row in rows)
    recalled = sum(bool(row["recall_at_1_iou_0.5"]) for row in rows)
    return {
        "total": total,
        "parseable": parseable,
        "parse_rate": parseable / total,
        "mean_iou": total_iou / total,
        "mean_iou_percent": 100.0 * total_iou / total,
        "recall_at_1_iou_0.5": recalled / total,
        "recall_at_1_iou_0.5_percent": 100.0 * recalled / total,
    }


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overall and per-category S-EMBER grounding metrics."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["question_category"])].append(row)
    ordered_categories = list(QUESTION_CATEGORIES) + sorted(
        set(by_category) - set(QUESTION_CATEGORIES)
    )
    return {
        "metric": "sember_temporal_grounding",
        "overall": _aggregate_group(rows),
        "per_category": {
            category: _aggregate_group(by_category[category])
            for category in ordered_categories
            if category in by_category
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    args = parser.parse_args()

    with args.results_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source, strict=True))
    scored = score_rows(rows)
    summary = aggregate_scores(scored)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        with args.details_output.open("w", encoding="utf-8") as output:
            for row in scored:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    overall = summary["overall"]
    print(
        "S-EMBER temporal grounding: "
        f"mIoU={overall['mean_iou_percent']:.2f}% "
        f"R@1(IoU>=0.5)={overall['recall_at_1_iou_0.5_percent']:.2f}% "
        f"parseable={overall['parseable']}/{overall['total']}"
    )
    print(f"Metrics written to: {args.output_path}")


if __name__ == "__main__":
    main()
