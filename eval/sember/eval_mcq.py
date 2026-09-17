#!/usr/bin/env python3
"""Evaluate S-EMBER five-way multiple-choice predictions."""

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
CHOICE_LABELS = ("A", "B", "C", "D", "E")


def parse_mcq_choice(response_text: str, num_options: int = 5) -> str | None:
    """Extract a choice letter using the official S-EMBER parser semantics."""
    if not response_text:
        return None
    text = response_text.strip().upper()
    valid_labels = CHOICE_LABELS[:num_options]
    if text in valid_labels:
        return text
    match = re.search(r"\b([A-E])\b", text)
    if match and match.group(1) in valid_labels:
        return match.group(1)
    return None


def score_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Parse and score S-EMBER MCQ prediction rows."""
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        question_id = (row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Row {row_number} has no question_id")
        if question_id in seen:
            raise ValueError(f"Duplicate prediction question_id: {question_id}")
        seen.add(question_id)

        correct_letter = (
            row.get("correct_letter") or row.get("correct_choice") or ""
        ).strip().upper()
        if correct_letter not in CHOICE_LABELS:
            raise ValueError(
                f"Row {row_number} has invalid correct letter: {correct_letter!r}"
            )
        category = (
            row.get("question_category") or row.get("task") or "unknown"
        ).strip()
        prediction = row.get("pred_raw") or row.get("pred_answer") or ""
        pred_letter = parse_mcq_choice(prediction)
        scored.append(
            {
                **row,
                "question_id": question_id,
                "question_category": category,
                "correct_letter": correct_letter,
                "pred_raw": prediction,
                "pred_letter": pred_letter,
                "is_correct": pred_letter == correct_letter,
                "is_parseable": pred_letter is not None,
            }
        )
    if not scored:
        raise ValueError("Prediction file contains no rows")
    return scored


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parsed = sum(bool(row["is_parseable"]) for row in rows)
    correct = sum(bool(row["is_correct"]) for row in rows)
    return {
        "total": total,
        "parsed": parsed,
        "parse_rate": parsed / total,
        "correct": correct,
        "accuracy": correct / total,
        "accuracy_percent": 100.0 * correct / total,
    }


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overall, per-category, and choice-distribution metrics."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_distribution = {label: 0 for label in CHOICE_LABELS}
    for row in rows:
        by_category[str(row["question_category"])].append(row)
        if row["pred_letter"] in prediction_distribution:
            prediction_distribution[row["pred_letter"]] += 1
    ordered_categories = list(QUESTION_CATEGORIES) + sorted(
        set(by_category) - set(QUESTION_CATEGORIES)
    )
    return {
        "metric": "sember_mcq_accuracy",
        "overall": _aggregate_group(rows),
        "per_category": {
            category: _aggregate_group(by_category[category])
            for category in ordered_categories
            if category in by_category
        },
        "prediction_distribution": prediction_distribution,
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
        "S-EMBER MCQ: "
        f"accuracy={overall['accuracy_percent']:.2f}% "
        f"parsed={overall['parsed']}/{overall['total']}"
    )
    print(f"Metrics written to: {args.output_path}")


if __name__ == "__main__":
    main()
