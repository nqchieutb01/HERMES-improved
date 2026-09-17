"""CPU-only tests for the S-EMBER adapter and grounding metrics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.sember.eval_grounding import (
    aggregate_scores as aggregate_grounding_scores,
    parse_grounding_response,
    score_rows as score_grounding_rows,
    temporal_iou,
)
from eval.sember.eval_mcq import (
    aggregate_scores as aggregate_mcq_scores,
    parse_mcq_choice,
    score_rows as score_mcq_rows,
)
from video_qa.adapters import (
    SEMBER_GROUNDING_PROMPT,
    load_sember_mcq,
    load_sember_grounding,
    sember_grounding_prompt,
    sember_mcq_prompt,
)


def sem_ber_row(
    question_id: str,
    video_id: str,
    question_time: float,
    *,
    category: str = "visual_detail_recall",
) -> dict:
    return {
        "question_id": question_id,
        "video_id": video_id,
        "video": f"videos/{video_id}.mp4",
        "video_category_broad": "Office",
        "video_category": "Packing",
        "question": f"Question {question_id}?",
        "question_time": question_time,
        "question_category": category,
        "memory_recency": question_time - 2,
        "duration": 30.0,
        "answer": f"Answer {question_id}.",
        "answer_start_time": 2.0,
        "answer_end_time": 4.0,
        "answer_range": 2.0,
        "answers": [{"answer_text": f"Answer {question_id}.", "start_ts": 2, "end_ts": 4}],
    }


def sem_ber_mcq_row(
    question_id: str,
    video_id: str,
    question_time: float,
    *,
    correct_index: int = 2,
    category: str = "visual_detail_recall",
) -> dict:
    options = [f"{label}. Option {label}" for label in "ABCDE"]
    return {
        "question_id": question_id,
        "video_id": video_id,
        "video": f"videos/{video_id}.mp4",
        "video_category_broad": "Office",
        "video_category": "Packing",
        "question": f"Question {question_id}?",
        "question_time": question_time,
        "question_category": category,
        "duration": 30.0,
        "options": options,
        "correct_index": correct_index,
        "correct_letter": "ABCDE"[correct_index],
        "correct_option_source": "GT1",
        "ground_truths": [f"Option {'ABCDE'[correct_index]}"],
        "answer_start_time": 2.0,
        "answer_end_time": 4.0,
    }


class SemberAdapterTest(unittest.TestCase):
    def test_formats_groups_and_sorts_questions_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "video-a.mp4").touch()
            (videos / "video-b.mp4").touch()
            rows = [
                sem_ber_row("later", "video-a", 20),
                sem_ber_row("other", "video-b", 5),
                sem_ber_row("earlier", "video-a", 10),
            ]
            annotation = root / "sember_grounding.jsonl"
            annotation.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            formatted = load_sember_grounding(
                annotation, video_root=videos, max_videos=1
            )

            self.assertEqual(len(formatted), 1)
            self.assertEqual(formatted[0]["video_id"], "video-a")
            conversations = formatted[0]["conversations"]
            self.assertEqual([item["question_id"] for item in conversations], ["earlier", "later"])
            self.assertEqual([item["end_time"] for item in conversations], [10.0, 20.0])
            self.assertTrue(conversations[0]["prompt"].startswith(SEMBER_GROUNDING_PROMPT))
            self.assertTrue(conversations[0]["prompt"].endswith("Question earlier?"))

    def test_prompt_matches_official_layout(self):
        prompt = sember_grounding_prompt("Where was the mug?")
        self.assertIn("Answer in 1-2 sentences.", prompt)
        self.assertIn("Answer: <your answer>\nTime: [<start_seconds>, <end_seconds>]", prompt)
        self.assertTrue(prompt.endswith("\n\nWhere was the mug?"))

    def test_rejects_duplicate_question_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "video-a.mp4").touch()
            rows = [sem_ber_row("same", "video-a", 10), sem_ber_row("same", "video-a", 20)]
            annotation = root / "annotations.jsonl"
            annotation.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Duplicate S-EMBER question_id"):
                load_sember_grounding(annotation, video_root=videos)

    def test_formats_mcq_and_uses_official_prompt_without_double_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "video-a.mp4").touch()
            rows = [
                sem_ber_mcq_row("later", "video-a", 20),
                sem_ber_mcq_row("earlier", "video-a", 10, correct_index=4),
            ]
            annotation = root / "sember_mcq.jsonl"
            annotation.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            formatted = load_sember_mcq(annotation, video_root=videos)
            conversations = formatted[0]["conversations"]
            self.assertEqual([item["question_id"] for item in conversations], ["earlier", "later"])
            self.assertEqual(conversations[0]["correct_letter"], "E")
            self.assertEqual(conversations[0]["answer"], "E. Option E")
            self.assertNotIn("A. A. Option A", conversations[0]["prompt"])
            self.assertTrue(conversations[0]["prompt"].endswith("Answer:"))

    def test_mcq_prompt_labels_unlabeled_options(self):
        prompt = sember_mcq_prompt("Pick one", ["one", "two", "three", "four", "five"])
        self.assertIn("  A. one", prompt)
        self.assertIn("ONLY a single letter (A/B/C/D/E)", prompt)

    def test_mcq_category_filter_runs_before_video_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            (videos / "video-a.mp4").touch()
            rows = [
                sem_ber_mcq_row(
                    "excluded",
                    "video-excluded",
                    1,
                    category="visual_detail_recall",
                ),
                sem_ber_mcq_row(
                    "selected",
                    "video-a",
                    2,
                    category="time_duration",
                ),
            ]
            annotation = root / "sember_mcq.jsonl"
            annotation.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            formatted = load_sember_mcq(
                annotation,
                video_root=videos,
                max_videos=1,
                question_categories=["time_duration"],
            )

            self.assertEqual([sample["video_id"] for sample in formatted], ["video-a"])
            self.assertEqual(formatted[0]["conversations"][0]["task"], "time_duration")

    def test_rejects_unknown_mcq_category_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            annotation = root / "sember_mcq.jsonl"
            annotation.write_text(
                json.dumps(sem_ber_mcq_row("q1", "video-a", 1)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown S-EMBER question categories"):
                load_sember_mcq(
                    annotation,
                    video_root=videos,
                    question_categories=["not_a_real_category"],
                )


class SemberMetricTest(unittest.TestCase):
    def test_parses_official_grounding_format(self):
        answer, start, end = parse_grounding_response(
            "Answer: The mug was on the table.\nTime: [12.5, 19]"
        )
        self.assertEqual(answer, "The mug was on the table.")
        self.assertEqual((start, end), (12.5, 19.0))

    def test_unparseable_interval_scores_zero(self):
        answer, start, end = parse_grounding_response("The mug was on the table.")
        self.assertEqual(answer, "The mug was on the table.")
        self.assertIsNone(start)
        self.assertIsNone(end)
        self.assertEqual(temporal_iou(start, end, 10, 20), 0.0)

    def test_temporal_iou(self):
        self.assertAlmostEqual(temporal_iou(10, 30, 20, 40), 1 / 3)
        self.assertEqual(temporal_iou(0, 5, 10, 20), 0.0)
        self.assertEqual(temporal_iou(10, 20, 10, 20), 1.0)

    def test_scores_and_aggregates_parse_failures_in_denominator(self):
        rows = [
            {
                "question_id": "q1",
                "question_category": "visual_detail_recall",
                "pred_answer": "Answer: Blue.\nTime: [10, 20]",
                "answer_start_time": "10",
                "answer_end_time": "20",
            },
            {
                "question_id": "q2",
                "question_category": "visual_detail_recall",
                "pred_answer": "Blue.",
                "answer_start_time": "10",
                "answer_end_time": "20",
            },
        ]
        scored = score_grounding_rows(rows)
        summary = aggregate_grounding_scores(scored)
        self.assertEqual(summary["overall"]["total"], 2)
        self.assertEqual(summary["overall"]["parseable"], 1)
        self.assertEqual(summary["overall"]["mean_iou"], 0.5)
        self.assertEqual(summary["overall"]["recall_at_1_iou_0.5"], 0.5)


class SemberMcqMetricTest(unittest.TestCase):
    def test_official_choice_parser(self):
        self.assertEqual(parse_mcq_choice("A"), "A")
        self.assertEqual(parse_mcq_choice("The answer is (D)."), "D")
        self.assertIsNone(parse_mcq_choice("I cannot tell"))

    def test_mcq_accuracy_includes_unparseable_predictions(self):
        rows = [
            {
                "question_id": "q1",
                "question_category": "visual_detail_recall",
                "correct_letter": "C",
                "pred_answer": "C",
            },
            {
                "question_id": "q2",
                "question_category": "visual_detail_recall",
                "correct_letter": "A",
                "pred_answer": "I cannot tell",
            },
        ]
        scored = score_mcq_rows(rows)
        summary = aggregate_mcq_scores(scored)
        self.assertEqual(summary["overall"]["total"], 2)
        self.assertEqual(summary["overall"]["parsed"], 1)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["prediction_distribution"]["C"], 1)


if __name__ == "__main__":
    unittest.main()
