#!/usr/bin/env python3
"""Compare completed continuous-stream results with the previous prefix runs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/llava_ov_0.5b/rvs_ego"
CURRENT = BASE / "continuous-fps0.5-kv6000"
JUDGE = "qwen3.8_27b_fp8_judge"


def main():
    runs = {}
    manifests = []
    for label, directory in (("prefix_kv1024", BASE / "fps0.5-kv1024"),
                             ("prefix_kv6072", BASE / "fps0.5-kv6072"),
                             ("continuous_kv6000", CURRENT)):
        summary = json.loads((directory / JUDGE / "summary.json").read_text())
        yes_no = json.loads((directory / "yes_no_evaluation.json").read_text())
        validation = json.loads((directory / "validation.json").read_text())
        assert validation["status"] == "PASS"
        assert summary["questions"] == 1465 and yes_no["questions"] == 698
        manifests.append(json.loads((directory / JUDGE / "manifest.json").read_text()))
        runs[label] = {"judge_accuracy_percent": summary["accuracy_percent"],
                       "judge_score": summary["score"],
                       "yes_no_accuracy_percent": yes_no["accuracy_percent"],
                       "judge_questions": summary["questions"],
                       "yes_no_questions": yes_no["questions"],
                       "by_task": summary["by_task"]}
    for field in ("model", "revision", "system_prompt", "schema", "temperature", "seed", "enable_thinking", "max_tokens"):
        assert all(m[field] == manifests[0][field] for m in manifests), field
    logs = [(CURRENT / f"inference-{i}.log").read_text() for i in range(2)]
    generic = sum(log.count("Local question: Describe the current scene") for log in logs)
    conditioned = sum(log.count("Local question: Find recent details related to:") for log in logs)
    assert conditioned > 0, "Conversation-conditioned compression did not activate"
    with (CURRENT / "results.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len({r["video_id"] for r in rows}) == 10
    report = {
        "runs": runs,
        "change_continuous_minus_prefix_kv6072": {
            k: runs["continuous_kv6000"][k] - runs["prefix_kv6072"][k]
            for k in ("judge_accuracy_percent", "judge_score", "yes_no_accuracy_percent")},
        "protocol_validation": {"continuous_videos": 10, "predictions": len(rows),
                                "generic_compression_queries": generic,
                                "history_conditioned_compression_queries": conditioned,
                                "same_judge_model_revision_and_rubric": True},
        "paper_reference": {"url": "https://arxiv.org/pdf/2601.14724", "table": 2,
                            "model": "LLaVA-OV-0.5B + HERMES (6K tokens)",
                            "accuracy_percent": 53.0, "score": 3.8,
                            "directly_comparable": False,
                            "reason": "Paper uses GPT-3.5-turbo-0125; local runs use Qwen3.8-27B-FP8 and a custom rubric. Existing extracted JPEG inputs also differ from original video decoding."},
    }
    destination = CURRENT / "comparison_vs_prefix_runs.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
