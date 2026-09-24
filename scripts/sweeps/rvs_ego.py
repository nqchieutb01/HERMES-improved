"""Resumable full-dataset parameter search with a fixed local Qwen judge.

Prepare a source snapshot on the login node, then execute it under Slurm.
All subprocesses use the snapshot; historical predictions are never modified.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = dict(sample_fps=0.5, kv_size=6000, encode_chunk_size=16,
            max_new_tokens=256, repetition_penalty=1.1,
            recency_weight_start=0.75, recency_weight_decay=0.6,
            reindex_margin=1024, use_history=True)
INITIAL = [
    ("corrected_baseline", {}),
    ("earlier_reindex", dict(reindex_margin=4096)),
    ("kv4000", dict(kv_size=4000)),
    ("chunk8", dict(encode_chunk_size=8)),
    ("fps1", dict(sample_fps=1.0)),
    ("no_repetition_penalty", dict(repetition_penalty=1.0)),
    ("generic_guidance", dict(use_history=False)),
    ("previous_recency", dict(recency_weight_start=1.0, recency_weight_decay=0.9)),
]
JUDGE_FIELDS = ("model", "revision", "prompt_version", "system_prompt",
                "user_prompt_template", "schema", "temperature", "seed",
                "enable_thinking", "max_tokens")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for directory in ("configs", "eval", "inference", "scripts", "tests", "video_qa"):
        shutil.copytree(ROOT / directory, source / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for directory in ("models", "data", ".venv"):
        (source / directory).symlink_to(ROOT / directory, target_is_directory=True)
    baseline = ROOT / "results/llava_ov_0.5b/rvs_ego/continuous-fps0.5-kv6000"
    shutil.copy2(baseline / "annotations.json", output / "annotations.json")
    judge = json.loads((baseline / "qwen3.8_27b_fp8_judge_original_prompt/manifest.json").read_text())
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for d in ("configs", "eval", "inference", "scripts", "tests", "video_qa")
              for p in (source / d).rglob("*") if p.is_file()}
    plan = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                model="llava_ov_0.5b", target_accuracy=53.0, target_score=3.8,
                max_trials=24, partition="long", qos="gpu-debug-qos", gpus=1, source_sha256=hashes,
                annotation_sha256=hashlib.sha256((output / "annotations.json").read_bytes()).hexdigest(),
                fixed_judge={k: judge[k] for k in JUDGE_FIELDS},
                judge_execution=dict(tensor_parallel_size=1, batch_size=64),
                base=BASE, initial=[dict(name=n, parameters={**BASE, **p}) for n, p in INITIAL],
                interpretation="Exploratory tuning on RVS-Ego with a fixed Qwen judge, not an independent paper reproduction")
    write_json(output / "plan.json", plan)
    (output / "README.md").write_text(
        "# Full RVS-Ego parameter sweep\n\n"
        "Partition: `long`; QoS: `gpu-debug-qos`; 1 GPU; three-hour allocations.\n"
        "Inference processes all videos sequentially; Qwen judging uses tensor parallel size 1.\n"
        "The controller submits a continuation between trials after 90 minutes.\n"
        "Every scored trial uses all 1,465 questions.\n"
        "Fixed judge: Qwen3.8-27B-FP8, original HERMES semantic rubric.\n"
        "Targets: accuracy >=53% AND mean score >=3.8.\n"
        "Stop on both targets, 24 trials, or three consecutive failed trials.\n"
        "The first eight trials isolate parameters; later trials combine changes around the best result.\n"
        "This is exploratory benchmark tuning; judge differences remain.\n\n"
        "`status.json` records live progress; `leaderboard.json` and `leaderboard.csv` retain all trials.\n"
        "`source/` snapshots code; `plan.json` pins code hashes and judge settings.\n")
    print(output, flush=True)


def signature(parameters):
    return json.dumps(parameters, sort_keys=True)


def next_trial(plan, trials):
    tried = {signature(t["parameters"]) for t in trials}
    for trial in plan["initial"]:
        if signature(trial["parameters"]) not in tried:
            return trial
    successes = sorted((t for t in trials if t["state"] == "completed"),
                       key=lambda t: (t["accuracy_percent"], t["score"]), reverse=True)
    # Coordinate search followed by combinations; the judge itself is never tuned.
    changes = [dict(kv_size=k) for k in (4000, 6000, 8000, 12000)] + [
        dict(sample_fps=1.0), dict(sample_fps=0.5), dict(encode_chunk_size=8),
        dict(encode_chunk_size=16), dict(reindex_margin=4096),
        dict(repetition_penalty=1.0), dict(repetition_penalty=1.05),
        dict(use_history=False), dict(use_history=True),
        dict(recency_weight_start=0.5, recency_weight_decay=0.35),
        dict(recency_weight_start=0.75, recency_weight_decay=0.6),
        dict(max_new_tokens=128),
    ]
    for best in successes[:3]:
        for change in changes:
            params = {**best["parameters"], **change}
            if signature(params) not in tried:
                return dict(name="adaptive", parameters=params)
    for fps, kv, chunk in itertools.product((0.5, 1.0), (4000, 6000, 8000, 12000), (8, 16)):
        params = {**BASE, "sample_fps": fps, "kv_size": kv, "encode_chunk_size": chunk,
                  "reindex_margin": 4096}
        if signature(params) not in tried:
            return dict(name="grid", parameters=params)
    return None


def run(output):
    allocation_start = time.monotonic()
    source = output / "source"
    plan = json.loads((output / "plan.json").read_text())
    for name, expected in plan["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    if hashlib.sha256((output / "annotations.json").read_bytes()).hexdigest() != plan["annotation_sha256"]:
        raise ValueError("Annotations changed")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    num_workers = plan.get('gpus', 1)
    if len(devices) != num_workers or not all(devices):
        raise ValueError(f"Expected exactly {num_workers} allocated CUDA devices")
    python = source / ".venv/hermes/bin/python"
    env = {**os.environ, "PYTHONPATH": str(source), "OMP_NUM_THREADS": "8"}
    trials_path = output / "leaderboard.json"
    trials = json.loads(trials_path.read_text()) if trials_path.exists() else []

    def status(state, **kwargs):
        write_json(output / "status.json", dict(state=state, job_id=os.getenv("SLURM_JOB_ID"),
                   updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kwargs))
        print(state, kwargs, flush=True)

    def command(args, log, process_env=env):
        with log.open("a") as stream:
            subprocess.run([str(a) for a in args], cwd=source, env=process_env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def save_trials():
        write_json(trials_path, trials)
        with (output / "leaderboard.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "name", "state", "accuracy_percent",
                "score", "seconds", "error", *BASE])
            writer.writeheader()
            for trial in trials:
                writer.writerow({**{k: trial.get(k, "") for k in ("id", "name", "state",
                    "accuracy_percent", "score", "seconds", "error")}, **trial["parameters"]})

    status("preflight")
    try:
        command([python, "-c", "import os, torch; "
                 "print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True); "
                 f"assert torch.cuda.device_count() == {num_workers}; "
                 "[print(i, torch.cuda.get_device_name(i), torch.ones(1, device=f'cuda:{i}').item(), flush=True) "
                 f"for i in range({num_workers})]"], output / "gpu-preflight.log")
    except subprocess.CalledProcessError:
        status("infrastructure_failed", reason="Allocated GPU initialization failed; no parameter trial started")
        raise
    command([python, "-m", "unittest", "tests.test_llava_attention"], output / "regression-tests.log")
    failures = 0
    for previous in reversed(trials):
        if previous['state'] != 'failed':
            break
        failures += 1
    while len(trials) < plan["max_trials"]:
        # The account can use long only through the three-hour debug QoS.
        # Single-GPU trials take longer. Leave at least 90 minutes before
        # starting another full inference/judging cycle.
        if time.monotonic() - allocation_start > 90 * 60:
            submission = subprocess.run([
                "sbatch", "--parsable", "--dependency=afterany:" + os.environ["SLURM_JOB_ID"],
                "--export=ALL,HERMES_SWEEP_SOURCE=" + str(source),
                "--output=" + str(output / "slurm-%j.log"),
                str(source / "scripts/slurm/rvs_ego_sweep.sh"), "--output-dir", str(output)],
                text=True, capture_output=True, check=True)
            status("continuation_submitted", next_job_id=submission.stdout.strip(), trials=len(trials))
            return
        trial = next_trial(plan, trials)
        if trial is None:
            break
        # Persist an unfinished trial separately so a requeue resumes its files.
        pending = output / "pending.json"
        if pending.exists():
            trial = json.loads(pending.read_text())
            if any(previous["id"] == trial["id"] for previous in trials):
                pending.unlink()
                continue
        else:
            trial = {**trial, "id": f"{len(trials):02d}-{trial['name']}"}
            write_json(pending, trial)
        folder = output / trial["id"]
        folder.mkdir(exist_ok=True)
        write_json(folder / "parameters.json", trial["parameters"])
        start = time.monotonic()
        try:
            status("inference", trial=trial)
            args = []
            for key, value in trial["parameters"].items():
                args.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
            processes = []
            handles = []
            try:
                for chunk, gpu in enumerate(devices):
                    if (folder / f"{num_workers}_{chunk}.csv").exists():
                        continue
                    log = (folder / f"inference-{chunk}.log").open("a")
                    handles.append(log)
                    worker_command = [
                        str(python),
                        "-u",
                        "-m",
                        "video_qa.hermes_vqa",
                        "--model",
                        plan["model"],
                        "--anno_path",
                        str(output / "annotations.json"),
                        "--save_dir",
                        str(folder),
                        "--streaming",
                        "true",
                        "--num_chunks",
                        str(num_workers),
                        "--chunk_idx",
                        str(chunk),
                        "--debug",
                        "false",
                        *args,
                    ]
                    processes.append(
                        subprocess.Popen(
                            worker_command,
                            cwd=source,
                            env={**env, "CUDA_VISIBLE_DEVICES": gpu},
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    )
                codes = [process.wait() for process in processes]
                if any(codes):
                    raise RuntimeError(f"Inference worker exit codes: {codes}")
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                for handle in handles:
                    handle.close()
            rows = []
            for chunk in range(num_workers):
                with (folder / f"{num_workers}_{chunk}.csv").open(newline="") as stream:
                    rows.extend(csv.DictReader(stream))
            with (folder / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            command([
                python,
                "scripts/run.py",
                "dataset=rvs_ego_local",
                "run.mode=evaluate",
                "dataset.annotation=" + str(output / "annotations.json"),
                "paths.results_path=" + str(folder / "results.csv"),
                "paths.save_dir=" + str(folder),
            ], folder / "validation.log")
            status("judging", trial=trial)
            cuda_home = source / ".venv/judge/lib/python3.11/site-packages/nvidia/cu13"
            judge_env = {**env, "CUDA_HOME": str(cuda_home),
                "PATH": str(cuda_home / "bin") + ":" + env["PATH"],
                "LD_LIBRARY_PATH": str(source / ".venv/cuda-compat-13/usr/local/cuda-13.0/compat") +
                    ":" + env.get("LD_LIBRARY_PATH", ""),
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_CACHE_ROOT": str(output / "cache/vllm"),
                "TRITON_CACHE_DIR": str(output / "cache/triton")}
            judge_dir = folder / "qwen_judge"
            command([source / ".venv/judge/bin/python", "-u", "eval/rvs/judge.py",
                     "--results-path", folder / "results.csv", "--output-dir", judge_dir,
                     "--tensor-parallel-size", str(num_workers)],
                    folder / "judge.log", judge_env)
            manifest = json.loads((judge_dir / "manifest.json").read_text())
            if any(manifest[k] != plan["fixed_judge"][k] for k in JUDGE_FIELDS):
                raise ValueError("Judge protocol changed; refusing to rank this trial")
            summary = json.loads((judge_dir / "summary.json").read_text())
            if summary["questions"] != 1465:
                raise ValueError("Incomplete full-dataset evaluation")
            trial.update(state="completed", accuracy_percent=summary["accuracy_percent"], score=summary["score"])
            failures = 0
        except Exception as exc:
            trial.update(state="failed", error=str(exc))
            failures += 1
        trial["seconds"] = round(time.monotonic() - start, 1)
        trials.append(trial)
        save_trials()
        pending.unlink()
        if trial["state"] == "completed" and trial["accuracy_percent"] >= plan["target_accuracy"] and trial["score"] >= plan["target_score"]:
            status("target_met", best=trial, trials=len(trials))
            return
        if failures >= 3:
            status("failed", reason="Three consecutive failed trials; inspect logs", trials=len(trials))
            raise RuntimeError("Repeated failures")
    successes = [t for t in trials if t["state"] == "completed"]
    best = max(successes, key=lambda t: (t["accuracy_percent"], t["score"])) if successes else None
    status("search_exhausted", best=best, trials=len(trials))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.prepare:
        prepare(output)
    else:
        run(output)
