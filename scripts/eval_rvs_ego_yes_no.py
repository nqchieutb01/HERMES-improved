#!/usr/bin/env python3
"""Evaluate explicit Yes/No answers with deterministic normalization; no judge API."""
import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata


DEFAULT_RESULTS = (Path(__file__).resolve().parents[1]
                   / "results/llava_ov_0.5b/rvs_ego/fps0.5-kv1024/results.csv")


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    # Ignore punctuation/Markdown wrappers while retaining word boundaries.
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def normalize_reference(text):
    text = normalize_text(text)
    return text if text in ("yes", "no") else None


def normalize_prediction(text):
    text = normalize_text(text)
    text = re.sub(r"^(?:the answer is|answer is|answer)\s+", "", text)
    # Treat alternatives such as 'yes/no' or 'yes or no' as unparseable.
    if re.match(r"^(yes|no)\s+(?:(?:or|and)\s+)?(yes|no)\b", text):
        return None
    match = re.match(r"^(yes|no)\b", text)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the predictions directory")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_path.parent
    with args.results_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, strict=True)
        required = {"question", "answer", "pred_answer", "video_id"}
        if not required.issubset(reader.fieldnames or []):
            parser.error(f"CSV must contain {sorted(required)}")
        rows = list(reader)

    selected = []
    for index, row in enumerate(rows, 1):
        reference = normalize_reference(row["answer"])
        if reference is None:
            continue
        prediction = normalize_prediction(row["pred_answer"])
        selected.append({
            "source_record": index, "video_id": row["video_id"],
            "question": row["question"], "task": row.get("task", ""),
            "answer": row["answer"], "pred_answer": row["pred_answer"],
            "normalized_answer": reference,
            "normalized_prediction": prediction or "unparseable",
            "correct": int(reference == prediction),
        })
    if not selected:
        parser.error("No exact Yes/No reference answers found after normalization")

    def metrics(records):
        correct = sum(r["correct"] for r in records)
        return {"questions": len(records), "correct": correct,
                "accuracy_percent": 100 * correct / len(records),
                "unparseable": sum(r["normalized_prediction"] == "unparseable" for r in records)}

    report = {
        "results_path": str(args.results_path.resolve()),
        "normalization": "Unicode NFKC, case folding, punctuation/symbol and whitespace normalization. Select exact yes/no references. Predictions use the leading whole yes/no word, optionally after an answer prefix; explanations are ignored. Unparseable predictions count as incorrect.",
        "total_input_rows": len(rows), "excluded_non_binary_rows": len(rows) - len(selected),
        **metrics(selected),
        "by_reference": {label: metrics(group) for label in ("yes", "no")
                         if (group := [r for r in selected if r["normalized_answer"] == label])},
        "by_task": {task: metrics([r for r in selected if r["task"] == task])
                    for task in sorted({r["task"] for r in selected})},
        "confusion_matrix": {
            reference: {prediction: sum(r["normalized_answer"] == reference and
                                       r["normalized_prediction"] == prediction for r in selected)
                        for prediction in ("yes", "no", "unparseable")}
            for reference in ("yes", "no")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "yes_no_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "yes_no_predictions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Yes/No accuracy: {report['accuracy_percent']:.2f}% ({report['correct']}/{report['questions']})")
    print(f"Unparseable predictions: {report['unparseable']}")
    for task, result in report["by_task"].items():
        print(f"{task}: {result['accuracy_percent']:.2f}% ({result['correct']}/{result['questions']})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
