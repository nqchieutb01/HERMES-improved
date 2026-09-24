#!/usr/bin/env python3
"""Hydra entry point for HERMES inference, validation, and evaluation."""

from __future__ import annotations

import csv
from numbers import Integral, Real
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_MODES = {"infer", "evaluate", "full", "validate"}
VALID_FRAME_SAMPLING = {"incremental", "uniform"}
VALID_FRAME_SUMMARY_STRATEGIES = {
    "mean",
    "top_patch",
    "top_attention_patch",
    "attention_weighted",
    "softmax_score_weighted",
}


def _require_number(name: str, value: Any) -> Real:
    """Return a numeric config value or raise an actionable validation error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        hint = ""
        if isinstance(value, str) and "," in value:
            hint = (
                f" For a sweep, keep one scalar in YAML and pass "
                f"`-m {name}=value1,value2` on the command line."
            )
        raise ValueError(f"{name} must be a number, got {value!r}.{hint}")
    return value


def _require_integer(name: str, value: Any) -> int:
    """Return an integer config value or raise an actionable validation error."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        hint = ""
        if isinstance(value, str) and "," in value:
            hint = (
                f" For a sweep, keep one scalar in YAML and pass "
                f"`-m {name}=value1,value2` on the command line."
            )
        raise ValueError(f"{name} must be an integer, got {value!r}.{hint}")
    return int(value)


def _repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root / path


def _repo_root(cfg: DictConfig) -> Path:
    configured = cfg.paths.repo_root
    if configured in (None, ""):
        return REPO_ROOT
    return Path(to_absolute_path(str(configured))).resolve()


def _save_dir(cfg: DictConfig, repo_root: Path) -> Path:
    if cfg.paths.save_dir:
        return _repo_path(cfg.paths.save_dir, repo_root)
    output_root = _repo_path(cfg.paths.output_root, repo_root)
    name = f"fps{cfg.run.sample_fps}-kv{cfg.run.kv_size}"
    return output_root / cfg.model.name / cfg.dataset.name / name


def _results_path(cfg: DictConfig, save_dir: Path, repo_root: Path) -> Path:
    if cfg.paths.results_path:
        return _repo_path(cfg.paths.results_path, repo_root)
    return save_dir / "results.csv"


def _python(cfg: DictConfig, repo_root: Path) -> str:
    configured = cfg.runtime.python or cfg.model.get("python")
    if not configured:
        return sys.executable
    path = _repo_path(configured, repo_root)
    return str(path) if path.exists() else str(configured)


def validate_config(cfg: DictConfig) -> None:
    """Fail before launching a worker when a composed config is inconsistent."""
    if cfg.run.mode not in VALID_MODES:
        raise ValueError(
            f"run.mode must be one of {sorted(VALID_MODES)}, got {cfg.run.mode!r}"
        )
    num_chunks = _require_integer("run.num_chunks", cfg.run.num_chunks)
    sample_fps = _require_number("run.sample_fps", cfg.run.sample_fps)
    kv_size = _require_integer("run.kv_size", cfg.run.kv_size)
    encode_chunk_size = _require_integer(
        "run.encode_chunk_size", cfg.run.encode_chunk_size
    )
    max_new_tokens = _require_integer(
        "run.max_new_tokens", cfg.run.max_new_tokens
    )
    repetition_penalty = _require_number(
        "run.repetition_penalty", cfg.run.repetition_penalty
    )
    min_tokens_per_frame = _require_integer(
        "run.min_tokens_per_frame", cfg.run.min_tokens_per_frame
    )
    frame_summary_strategy = str(cfg.run.frame_summary_strategy)
    frame_summary_temperature = _require_number(
        "run.frame_summary_temperature", cfg.run.frame_summary_temperature
    )
    reindex_margin = _require_integer("run.reindex_margin", cfg.run.reindex_margin)
    recency_weight_decay = _require_number(
        "run.recency_weight_decay", cfg.run.recency_weight_decay
    )
    recency_weight_start = _require_number(
        "run.recency_weight_start", cfg.run.recency_weight_start
    )
    frame_sampling = str(cfg.run.frame_sampling)
    if frame_sampling not in VALID_FRAME_SAMPLING:
        raise ValueError(
            "run.frame_sampling must be one of "
            f"{sorted(VALID_FRAME_SAMPLING)}, got {frame_sampling!r}"
        )
    uniform_num_frames = cfg.run.uniform_num_frames
    if frame_sampling == "uniform":
        if uniform_num_frames is None:
            raise ValueError(
                "run.uniform_num_frames is required when run.frame_sampling=uniform"
            )
        uniform_num_frames = _require_integer(
            "run.uniform_num_frames", uniform_num_frames
        )
        if uniform_num_frames <= 0:
            raise ValueError("run.uniform_num_frames must be positive")

    if num_chunks <= 0:
        raise ValueError("run.num_chunks must be positive")
    if sample_fps <= 0:
        raise ValueError("run.sample_fps must be positive")
    if kv_size <= 0:
        raise ValueError("run.kv_size must be positive")
    if encode_chunk_size <= 0 or max_new_tokens <= 0:
        raise ValueError("run.encode_chunk_size and run.max_new_tokens must be positive")
    if repetition_penalty <= 0:
        raise ValueError("run.repetition_penalty must be positive")
    if min_tokens_per_frame < 0:
        raise ValueError("run.min_tokens_per_frame must be nonnegative")
    if frame_summary_strategy not in VALID_FRAME_SUMMARY_STRATEGIES:
        raise ValueError(
            "run.frame_summary_strategy must be one of "
            f"{sorted(VALID_FRAME_SUMMARY_STRATEGIES)}, got "
            f"{frame_summary_strategy!r}"
        )
    if frame_summary_temperature <= 0:
        raise ValueError("run.frame_summary_temperature must be positive")
    if reindex_margin < 0:
        raise ValueError("run.reindex_margin must be nonnegative")
    if not 0 <= recency_weight_decay <= recency_weight_start <= 1:
        raise ValueError(
            "require 0 <= run.recency_weight_decay <= "
            "run.recency_weight_start <= 1"
        )
    max_videos = cfg.dataset.get("max_videos")
    if max_videos is not None and int(max_videos) <= 0:
        raise ValueError("dataset.max_videos must be positive when provided")
    question_categories = cfg.dataset.get("question_categories")
    if question_categories is not None:
        categories = [str(category).strip() for category in question_categories]
        if not categories or any(not category for category in categories):
            raise ValueError(
                "dataset.question_categories must contain at least one non-empty category"
            )
        if len(set(categories)) != len(categories):
            raise ValueError("dataset.question_categories must not contain duplicates")


def _worker_devices(cfg: DictConfig) -> list[str | None]:
    """Return one CUDA_VISIBLE_DEVICES value per inference worker."""
    workers = int(cfg.run.num_chunks)
    per_worker = int(cfg.model.gpus_per_worker)
    if workers == 1 and not cfg.run.devices:
        return [None]

    visible = list(cfg.run.devices or [])
    if not visible:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible = inherited.split(",") if inherited else [str(i) for i in range(workers * per_worker)]
    required = workers * per_worker
    if len(visible) < required:
        raise ValueError(
            f"{workers} worker(s) x {per_worker} GPU(s) require {required} device(s); "
            f"configured/visible devices: {visible}"
        )
    return [",".join(visible[i * per_worker:(i + 1) * per_worker]) for i in range(workers)]


def inference_command(
    cfg: DictConfig,
    *,
    python: str,
    annotation: Path,
    save_dir: Path,
    chunk_index: int,
) -> list[str]:
    """Build one low-level worker command from the resolved Hydra config."""
    trace_path = cfg.run.token_trace_path
    if trace_path:
        trace = _repo_path(trace_path, _repo_root(cfg))
        if cfg.run.num_chunks > 1:
            trace = trace.with_name(f"{trace.stem}-{chunk_index}{trace.suffix}")
    else:
        trace = None

    command = [
        python,
        "-u",
        "-m",
        "video_qa.hermes_vqa",
        "--model",
        str(cfg.model.name),
        "--sample_fps",
        str(cfg.run.sample_fps),
        "--frame_sampling",
        str(cfg.run.frame_sampling),
        "--save_dir",
        str(save_dir),
        "--anno_path",
        str(annotation),
        "--debug",
        str(bool(cfg.run.debug)).lower(),
        "--num_chunks",
        str(cfg.run.num_chunks),
        "--chunk_idx",
        str(chunk_index),
        "--kv_size",
        str(cfg.run.kv_size),
        "--streaming",
        str(bool(cfg.dataset.streaming)).lower(),
        "--encode_chunk_size",
        str(cfg.run.encode_chunk_size),
        "--max_new_tokens",
        str(cfg.run.max_new_tokens),
        "--repetition_penalty",
        str(cfg.run.repetition_penalty),
        "--recency_weight_start",
        str(cfg.run.recency_weight_start),
        "--recency_weight_decay",
        str(cfg.run.recency_weight_decay),
        "--reindex_margin",
        str(cfg.run.reindex_margin),
        "--use_history",
        str(bool(cfg.run.use_history)).lower(),
        "--verbose_token_trace",
        str(bool(cfg.run.verbose_token_trace)).lower(),
        "--min_tokens_per_frame",
        str(cfg.run.min_tokens_per_frame),
        "--frame_summary_strategy",
        str(cfg.run.frame_summary_strategy),
        "--frame_summary_temperature",
        str(cfg.run.frame_summary_temperature),
        "--keep_time_tokens",
        str(cfg.run.get("keep_time_tokens", False)).lower(),
    ]
    model_path = cfg.model.get("model_path")
    if model_path:
        command.extend(
            ["--model_path", str(_repo_path(model_path, _repo_root(cfg)))]
        )
    if cfg.run.uniform_num_frames is not None:
        command.extend(["--uniform_num_frames", str(cfg.run.uniform_num_frames)])
    adapter = cfg.dataset.get("adapter")
    if adapter:
        command.extend(["--dataset_adapter", str(adapter)])
    video_root = cfg.dataset.get("video_root")
    if video_root:
        command.extend(["--video_root", str(_repo_path(video_root, _repo_root(cfg)))])
    max_videos = cfg.dataset.get("max_videos")
    if max_videos is not None:
        command.extend(["--max_videos", str(max_videos)])
    counting_prompt = cfg.dataset.get("counting_prompt")
    if counting_prompt:
        command.extend(["--counting_prompt", str(counting_prompt)])
    question_categories = cfg.dataset.get("question_categories")
    if question_categories:
        command.append("--question_categories")
        command.extend(str(category) for category in question_categories)
    if trace:
        command.extend(["--token_trace_path", str(trace)])
    return command


def merge_chunks(
    save_dir: Path,
    results_path: Path,
    num_chunks: int,
    *,
    keep_chunks: bool,
) -> None:
    """Merge worker CSV files atomically and verify that their schemas agree."""
    fieldnames: list[str] | None = None
    rows: list[dict[str, Any]] = []
    chunk_paths = [save_dir / f"{num_chunks}_{index}.csv" for index in range(num_chunks)]
    for chunk_path in chunk_paths:
        if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
            raise FileNotFoundError(f"Inference chunk is missing or empty: {chunk_path}")
        with chunk_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, strict=True)
            current_fields = list(reader.fieldnames or [])
            if not current_fields:
                raise ValueError(f"Inference chunk has no CSV header: {chunk_path}")
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(
                    f"CSV schema mismatch in {chunk_path}: {current_fields} != {fieldnames}"
                )
            rows.extend(reader)
    if not rows:
        raise ValueError("Inference produced no prediction rows")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = results_path.with_suffix(results_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results_path)
    if not keep_chunks:
        for chunk_path in chunk_paths:
            chunk_path.unlink()


def run_inference(
    cfg: DictConfig,
    *,
    repo_root: Path,
    python: str,
    annotation: Path,
    save_dir: Path,
    results_path: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        inference_command(
            cfg,
            python=python,
            annotation=annotation,
            save_dir=save_dir,
            chunk_index=index,
        )
        for index in range(cfg.run.num_chunks)
    ]
    if cfg.runtime.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return

    processes: list[subprocess.Popen] = []
    log_handles = []
    try:
        for index, (command, devices) in enumerate(zip(commands, _worker_devices(cfg))):
            env = os.environ.copy()
            if devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = devices
            process_kwargs: dict[str, Any] = {}
            log_path: Path | None = None
            if cfg.runtime.capture_worker_logs:
                log_path = (save_dir / f"inference-{index}.log").resolve()
                handle = log_path.open("w", encoding="utf-8")
                log_handles.append(handle)
                process_kwargs.update(stdout=handle, stderr=subprocess.STDOUT)
            process = subprocess.Popen(
                command, cwd=repo_root, env=env, **process_kwargs
            )
            processes.append(process)
            if log_path is not None:
                print(
                    f"Inference worker {index} started (PID {process.pid}). "
                    f"Progress log: {log_path}",
                    flush=True,
                )

        failures = []
        for index, process in enumerate(processes):
            return_code = process.wait()
            if return_code:
                failures.append((index, return_code))
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        for handle in log_handles:
            handle.close()
    if failures:
        details = []
        for index, return_code in failures:
            detail = f"chunk {index}, exit code {return_code}"
            if cfg.runtime.capture_worker_logs:
                detail += f", log: {(save_dir / f'inference-{index}.log').resolve()}"
            details.append(detail)
        raise RuntimeError(f"Inference workers failed: {'; '.join(details)}")
    merge_chunks(
        save_dir,
        results_path,
        int(cfg.run.num_chunks),
        keep_chunks=bool(cfg.run.keep_chunks),
    )


def evaluation_commands(
    cfg: DictConfig,
    *,
    python: str,
    results_path: Path,
    save_dir: Path,
    annotation: Path,
) -> list[list[str]]:
    """Translate declarative evaluator configs to internal commands."""
    commands: list[list[str]] = []
    for evaluator in cfg.dataset.evaluators:
        kind = evaluator.kind
        if kind == "multiple_choice":
            command = [
                python,
                "eval/eval_multiple_choice.py",
                str(evaluator.mode),
                "--results_path",
                str(results_path),
            ]
            if evaluator.get("debug", cfg.evaluation.debug):
                command.append("--debug")
        elif kind == "open_ended":
            output_dir = save_dir / str(evaluator.get("work_dir", "tmp"))
            output_json = save_dir / str(evaluator.get("output", "results.json"))
            command = [
                python,
                "eval/eval_open_ended.py",
                "--pred_path",
                str(results_path),
                "--output_dir",
                str(output_dir),
                "--output_json",
                str(output_json),
                "--num_tasks",
                str(evaluator.get("num_tasks", cfg.evaluation.num_tasks)),
            ]
        elif kind == "rvs_yes_no":
            command = [
                python,
                "eval/rvs/eval_yes_no.py",
                "--results-path",
                str(results_path),
                "--output-dir",
                str(save_dir),
            ]
        elif kind == "rvs_validate":
            command = [
                python,
                "eval/rvs/validate_results.py",
                "--results-path",
                str(results_path),
                "--anno-path",
                str(annotation),
            ]
        elif kind == "sember_grounding":
            output_json = save_dir / str(
                evaluator.get("output", "sember_grounding_metrics.json")
            )
            details_jsonl = save_dir / str(
                evaluator.get("details_output", "sember_grounding_scored.jsonl")
            )
            command = [
                python,
                "eval/sember/eval_grounding.py",
                "--results-path",
                str(results_path),
                "--output-path",
                str(output_json),
                "--details-output",
                str(details_jsonl),
            ]
        elif kind == "sember_mcq":
            output_json = save_dir / str(
                evaluator.get("output", "sember_mcq_metrics.json")
            )
            details_jsonl = save_dir / str(
                evaluator.get("details_output", "sember_mcq_scored.jsonl")
            )
            command = [
                python,
                "eval/sember/eval_mcq.py",
                "--results-path",
                str(results_path),
                "--output-path",
                str(output_json),
                "--details-output",
                str(details_jsonl),
            ]
        else:
            raise ValueError(f"Unknown evaluator kind: {kind}")
        commands.append(command)
    return commands


def run_evaluation(
    cfg: DictConfig,
    *,
    repo_root: Path,
    python: str,
    results_path: Path,
    save_dir: Path,
    annotation: Path,
    validation_only: bool = False,
) -> None:
    if not cfg.runtime.dry_run and not results_path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {results_path}")
    commands = evaluation_commands(
        cfg,
        python=python,
        results_path=results_path,
        save_dir=save_dir,
        annotation=annotation,
    )
    if validation_only:
        commands = [
            command
            for command in commands
            if any(part.endswith("validate_results.py") for part in command)
        ]
        if not commands:
            raise ValueError(
                f"Dataset {cfg.dataset.name!r} has no validation evaluator configured"
            )
    for index, command in enumerate(commands):
        print(subprocess.list2cmdline(command), flush=True)
        if not cfg.runtime.dry_run:
            if cfg.runtime.capture_evaluation_logs:
                suffix = "" if len(commands) == 1 else f"-{index}"
                with (save_dir / f"evaluation{suffix}.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            else:
                subprocess.run(command, cwd=repo_root, check=True)


def execute(cfg: DictConfig) -> None:
    validate_config(cfg)
    repo_root = _repo_root(cfg)
    python = _python(cfg, repo_root)
    annotation = _repo_path(cfg.dataset.annotation, repo_root)
    save_dir = _save_dir(cfg, repo_root)
    results_path = _results_path(cfg, save_dir, repo_root)
    if cfg.paths.results_path and not cfg.paths.save_dir:
        save_dir = results_path.parent

    print("Resolved HERMES configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Repository: {repo_root}")
    print(f"Predictions: {results_path}")
    needs_annotation = cfg.run.mode in {"infer", "full", "validate"}
    if not cfg.runtime.dry_run and needs_annotation and not annotation.is_file():
        raise FileNotFoundError(f"Dataset annotation does not exist: {annotation}")

    if cfg.run.mode in {"infer", "full"}:
        run_inference(
            cfg,
            repo_root=repo_root,
            python=python,
            annotation=annotation,
            save_dir=save_dir,
            results_path=results_path,
        )
    if cfg.run.mode in {"evaluate", "full", "validate"}:
        run_evaluation(
            cfg,
            repo_root=repo_root,
            python=python,
            results_path=results_path,
            save_dir=save_dir,
            annotation=annotation,
            validation_only=cfg.run.mode == "validate",
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    execute(cfg)


if __name__ == "__main__":
    main()
