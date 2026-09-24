"""Dataset adapters for the HERMES inference format."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
from typing import Any


SEMBER_GROUNDING_PROMPT = (
    "After reviewing the video, provide the best answer to the following question. "
    "Answer in 1-2 sentences.\n"
    "Important: You MUST provide the time interval (in seconds) where the answer evidence appears in the video.\n\n"
    "Use this exact format:\n"
    "Answer: <your answer>.\n"
    "Time: [<start_seconds>, <end_seconds>]"
)

# SEMBER_GROUNDING_PROMPT = (
#     "After reviewing the video, provide the best answer to the following question. "
#     "Answer in 1-2 sentences.\n"
#     "Important: You MUST provide the time interval (in seconds) where the evidence "
#     "supporting your answer appears in the video.\n\n"
#     "Use this exact format:\n"
#     "Time: [<start_seconds>, <end_seconds>]\n"
#     "Answer: <your answer>.\n\n"
#     "Example:\n"
#     "Question: What does the person do after entering the room?\n"
#     "Time: [12.5, 18.2]\n"
#     "Answer: The person walks to the table and picks up a book."
# )

SEMBER_CHOICE_LABELS = ("A", "B", "C", "D", "E")

SEMBER_QUESTION_CATEGORIES = (
    "counting_objects_events",
    "location_trace",
    "object_comparison",
    "sequential_action",
    "spatial_aware_reasoning",
    "temporal_ordering_recognition",
    "time_duration",
    "visual_detail_recall",
)

SEMBER_REQUIRED_FIELDS = {
    "question_id",
    "video_id",
    "question",
    "question_time",
    "question_category",
    "answer",
    "answer_start_time",
    "answer_end_time",
}

SEMBER_MCQ_REQUIRED_FIELDS = {
    "question_id",
    "video_id",
    "question",
    "question_time",
    "question_category",
    "options",
    "correct_index",
    "answer_start_time",
    "answer_end_time",
}

SEMBER_METADATA_FIELDS = (
    "question_id",
    "question_time",
    "question_category",
    "memory_recency",
    "answer_start_time",
    "answer_end_time",
    "answer_range",
    "duration",
    "video_category",
    "video_category_broad",
    "answers",
)

SEMBER_MCQ_METADATA_FIELDS = (
    "question_id",
    "question_time",
    "question_category",
    "correct_index",
    "correct_letter",
    "correct_option_source",
    "ground_truths",
    "answer_start_time",
    "answer_end_time",
    "duration",
    "video_category",
    "video_category_broad",
)


def sember_grounding_prompt(question: str) -> str:
    """Return the official S-EMBER grounded VideoQA prompt."""
    return f"{SEMBER_GROUNDING_PROMPT}\n\n{question}"


COUNTING_PROMPT_STYLES = ("official", "count_first")


def sember_mcq_prompt(question: str, options: list[str], style: str = "official") -> str:
    """Return the S-EMBER five-way MCQ prompt.

    ``official`` is the benchmark prompt (letter only). ``count_first`` asks the
    model to state its count before the letter; the MCQ evaluator reads the
    letter after ``Answer:``.
    """
    if len(options) != len(SEMBER_CHOICE_LABELS):
        raise ValueError(f"S-EMBER MCQ requires five options, got {len(options)}")
    labeled = []
    for label, option in zip(SEMBER_CHOICE_LABELS, options):
        option = str(option)
        if re.match(r"^[A-E]\.\s", option):
            labeled.append(option)
        else:
            labeled.append(f"{label}. {option}")
    options_text = "\n".join(f"  {option}" for option in labeled)
    if style == "count_first":
        return (
            "After reviewing the video, answer the following multiple-choice question.\n\n"
            f"Question: {question}\n\n"
            f"{options_text}\n\n"
            "First count carefully, then choose the option that matches your count. "
            "Respond in exactly this format and nothing else:\n"
            "Count: <number>\n"
            "Answer: <letter A/B/C/D/E>"
        )
    if style != "official":
        raise ValueError(f"Unknown S-EMBER MCQ prompt style: {style!r}")
    return (
        "After reviewing the video, answer the following multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"{options_text}\n\n"
        "IMPORTANT: Respond with ONLY a single letter (A/B/C/D/E). "
        "Do NOT include any explanation, reasoning, or additional text. "
        "Just the letter.\n\n"
        "Answer:"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"S-EMBER annotation file is empty: {path}")
    return rows


def _selected_sember_categories(
    question_categories: list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    """Validate an optional S-EMBER question-category filter."""
    if question_categories is None:
        return None
    selected = {str(category).strip() for category in question_categories}
    if not selected or "" in selected:
        raise ValueError("question_categories must contain at least one non-empty category")
    unknown = sorted(selected - set(SEMBER_QUESTION_CATEGORIES))
    if unknown:
        raise ValueError(
            "Unknown S-EMBER question categories: "
            f"{', '.join(unknown)}; expected one of "
            f"{', '.join(SEMBER_QUESTION_CATEGORIES)}"
        )
    return selected


def load_sember_grounding(
    annotation_path: str | Path,
    *,
    video_root: str | Path,
    max_videos: int | None = None,
    question_categories: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Format S-EMBER JSONL rows as chronological HERMES conversations.

    Videos are selected in first-appearance order. When ``max_videos`` is
    supplied, every question for the first N distinct videos is retained.
    """
    annotation_path = Path(annotation_path).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve()
    if max_videos is not None and max_videos <= 0:
        raise ValueError("max_videos must be positive when provided")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"S-EMBER annotations do not exist: {annotation_path}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"S-EMBER video directory does not exist: {video_root}")

    selected_categories = _selected_sember_categories(question_categories)
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    seen_question_ids: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(annotation_path), 1):
        missing = sorted(SEMBER_REQUIRED_FIELDS - row.keys())
        if missing:
            raise ValueError(
                f"S-EMBER row {line_number} is missing fields: {', '.join(missing)}"
            )

        question_category = str(row["question_category"])
        if selected_categories is not None and question_category not in selected_categories:
            continue

        question_id = str(row["question_id"])
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate S-EMBER question_id: {question_id}")
        seen_question_ids.add(question_id)

        video_id = str(row["video_id"])
        if video_id not in grouped:
            if max_videos is not None and len(grouped) >= max_videos:
                continue
            video_path = video_root / f"{video_id}.mp4"
            grouped[video_id] = {
                "video_id": video_id,
                "video_path": str(video_path),
                "duration": row.get("duration"),
                "conversations": [],
            }
        if video_id not in grouped:
            continue

        question_time = float(row["question_time"])
        answer_start = float(row["answer_start_time"])
        answer_end = float(row["answer_end_time"])
        if question_time < 0:
            raise ValueError(f"Negative question_time for {question_id}")
        if answer_start < 0 or answer_end < answer_start:
            raise ValueError(f"Invalid answer interval for {question_id}")

        conversation = {
            "question": str(row["question"]),
            "prompt": sember_grounding_prompt(str(row["question"])),
            "answer": str(row["answer"]),
            "end_time": question_time,
            "benchmark": "sember_grounding",
            "task": question_category,
        }
        for field in SEMBER_METADATA_FIELDS:
            if field in row:
                conversation[field] = row[field]
        grouped[video_id]["conversations"].append(conversation)

    formatted = list(grouped.values())
    if not formatted:
        raise ValueError("No S-EMBER videos were selected")
    missing_videos = [
        sample["video_path"]
        for sample in formatted
        if not Path(sample["video_path"]).is_file()
    ]
    if missing_videos:
        examples = ", ".join(missing_videos[:3])
        raise FileNotFoundError(
            f"{len(missing_videos)} selected S-EMBER video(s) are missing; examples: {examples}"
        )

    for sample in formatted:
        sample["conversations"].sort(
            key=lambda item: (float(item["question_time"]), str(item["question_id"]))
        )
    return formatted


def load_sember_mcq(
    annotation_path: str | Path,
    *,
    video_root: str | Path,
    max_videos: int | None = None,
    question_categories: list[str] | tuple[str, ...] | None = None,
    counting_prompt: str = "official",
) -> list[dict[str, Any]]:
    """Format S-EMBER MCQ JSONL rows as chronological HERMES conversations.

    ``counting_prompt`` selects the prompt style for counting_objects_events
    questions only; other categories always use the official prompt.
    """
    annotation_path = Path(annotation_path).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve()
    if max_videos is not None and max_videos <= 0:
        raise ValueError("max_videos must be positive when provided")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"S-EMBER MCQ annotations do not exist: {annotation_path}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"S-EMBER video directory does not exist: {video_root}")

    selected_categories = _selected_sember_categories(question_categories)
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    seen_question_ids: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(annotation_path), 1):
        missing = sorted(SEMBER_MCQ_REQUIRED_FIELDS - row.keys())
        if missing:
            raise ValueError(
                f"S-EMBER MCQ row {line_number} is missing fields: {', '.join(missing)}"
            )

        question_category = str(row["question_category"])
        if selected_categories is not None and question_category not in selected_categories:
            continue

        question_id = str(row["question_id"])
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate S-EMBER MCQ question_id: {question_id}")
        seen_question_ids.add(question_id)

        video_id = str(row["video_id"])
        if video_id not in grouped:
            if max_videos is not None and len(grouped) >= max_videos:
                continue
            video_path = video_root / f"{video_id}.mp4"
            grouped[video_id] = {
                "video_id": video_id,
                "video_path": str(video_path),
                "duration": row.get("duration"),
                "conversations": [],
            }
        if video_id not in grouped:
            continue

        options = row["options"]
        if not isinstance(options, list) or len(options) != len(SEMBER_CHOICE_LABELS):
            raise ValueError(f"S-EMBER MCQ question {question_id} must have five options")
        options = [str(option) for option in options]
        correct_index = int(row["correct_index"])
        if not 0 <= correct_index < len(options):
            raise ValueError(f"Invalid correct_index for S-EMBER MCQ question {question_id}")
        expected_letter = SEMBER_CHOICE_LABELS[correct_index]
        correct_letter = str(row.get("correct_letter", expected_letter)).upper()
        if correct_letter != expected_letter:
            raise ValueError(
                f"correct_letter/index mismatch for S-EMBER MCQ question {question_id}"
            )
        question_time = float(row["question_time"])
        answer_start = float(row["answer_start_time"])
        answer_end = float(row["answer_end_time"])
        if question_time < 0:
            raise ValueError(f"Negative question_time for {question_id}")
        if answer_start < 0 or answer_end < answer_start:
            raise ValueError(f"Invalid answer interval for {question_id}")

        conversation = {
            "question": str(row["question"]),
            "prompt": sember_mcq_prompt(
                str(row["question"]),
                options,
                counting_prompt
                if question_category == "counting_objects_events"
                else "official",
            ),
            "choices": options,
            "answer": options[correct_index],
            "end_time": question_time,
            "benchmark": "sember_mcq",
            "task": question_category,
        }
        for field in SEMBER_MCQ_METADATA_FIELDS:
            if field in row:
                conversation[field] = row[field]
        conversation["correct_letter"] = correct_letter
        grouped[video_id]["conversations"].append(conversation)

    formatted = list(grouped.values())
    if not formatted:
        raise ValueError("No S-EMBER MCQ videos were selected")
    missing_videos = [
        sample["video_path"]
        for sample in formatted
        if not Path(sample["video_path"]).is_file()
    ]
    if missing_videos:
        examples = ", ".join(missing_videos[:3])
        raise FileNotFoundError(
            f"{len(missing_videos)} selected S-EMBER video(s) are missing; examples: {examples}"
        )
    for sample in formatted:
        sample["conversations"].sort(
            key=lambda item: (float(item["question_time"]), str(item["question_id"]))
        )
    return formatted


def load_annotations(
    annotation_path: str | Path,
    *,
    adapter: str | None = None,
    video_root: str | Path | None = None,
    max_videos: int | None = None,
    question_categories: list[str] | tuple[str, ...] | None = None,
    counting_prompt: str = "official",
) -> list[dict[str, Any]]:
    """Load native HERMES JSON or adapt a supported external dataset."""
    if adapter in (None, "", "native"):
        with Path(annotation_path).open(encoding="utf-8") as source:
            annotations = json.load(source)
        if not isinstance(annotations, list):
            raise ValueError("Native HERMES annotations must be a JSON list")
        return annotations
    if adapter == "sember_grounding":
        if not video_root:
            raise ValueError("video_root is required for the sember_grounding adapter")
        return load_sember_grounding(
            annotation_path,
            video_root=video_root,
            max_videos=max_videos,
            question_categories=question_categories,
        )
    if adapter == "sember_mcq":
        if not video_root:
            raise ValueError("video_root is required for the sember_mcq adapter")
        return load_sember_mcq(
            annotation_path,
            video_root=video_root,
            max_videos=max_videos,
            question_categories=question_categories,
            counting_prompt=counting_prompt,
        )
    raise ValueError(f"Unknown dataset adapter: {adapter}")
