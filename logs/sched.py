"""Chunked Slurm scheduler for HERMES S-EMBER runs.

Each run is split into video chunks; each chunk runs as its own 1-GPU Slurm job
so no job nears the debug time limit. Up to N jobs run per partition slot; a
chunk is retried when Slurm fails to launch it or the worker dies. When all of
a run's chunks exist, they are merged into results.csv and scored.

Usage: python3 logs/sched.py <spec.json>
spec: {"slots": [{"partition":..., "qos":..., "max_jobs":...}, ...],
       "runs": [{"name":..., "code":..., "overrides":[...], "chunks":N, "env":{...}}, ...]}
"""
import csv, json, os, re, shlex, subprocess, sys, time

MAIN = "/l/users/chieu.nguyen/HERMES"
RUNNER = f"{MAIN}/.venv/hermes/bin/python3"
EVAL_PY = "/nfs-stor/chieu.nguyen/venvs/hermes-qwen/bin/python3"
MAX_ATTEMPTS = 3
# Nodes that cannot run jobs (gpu-24: task launch timeouts; gpu-23, gpu-62: /l/users not mounted).
EXCLUDED_NODES = {"gpu-23", "gpu-24", "gpu-62"}
SUBMIT_LIMIT_COOLDOWN = 120
METRIC_FILES = ("sember_mcq_metrics.json", "sember_grounding_metrics.json")
# One node name per line; re-read every loop so bad nodes can be excluded without a restart.
EXCLUDE_FILE = f"{MAIN}/logs/excluded_nodes.txt"
# A chunk whose log was written this recently at startup is still running from a previous
# scheduler; adopt it instead of relaunching. It is presumed dead once its log goes stale.
ADOPT_FRESH_S, ADOPT_STALE_S = 600, 1800
HARDWARE_ERRORS = ("hardware error", "Invalid access of peer GPU memory", "uncorrectable ECC", "CUDA driver error")

spec = json.load(open(sys.argv[1]))
status_path = sys.argv[1].replace(".json", ".status")


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def plan(run):
    """Dry-run run.py to get the per-chunk worker commands and evaluator commands."""
    args = [RUNNER, "scripts/run.py", *run["overrides"], f"run.num_chunks={run['chunks']}",
            "runtime.dry_run=true",
            # Hydra names its log dir after all overrides; keep it short.
            f"hydra.sweep.dir=outputs/hydra/sched-{run['name']}", "hydra.sweep.subdir=0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(run["chunks"])))
    out = subprocess.run(args, cwd=run["code"], env=env, capture_output=True, text=True, check=True).stdout
    workers = [l for l in out.splitlines() if " -m video_qa.hermes_vqa " in l]
    evals = [l for l in out.splitlines() if re.search(r"\seval/\S+\.py\s", l) and "hermes_vqa" not in l]
    save_dir = re.search(r"--save_dir (\S+)", workers[0]).group(1)
    assert len(workers) == run["chunks"], (run["name"], len(workers))
    return workers, evals, save_dir


def job_name(run, chunk):
    return f"h-{run['name']}-c{chunk['idx']}"


def queued_job_names():
    """Names of this user's queued or running Slurm jobs (for adopting jobs after a restart)."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        return set()
    return set(out.split())


QUEUED = queued_job_names()
runs = []
for r in spec["runs"]:
    workers, evals, save_dir = plan(r)
    os.makedirs(save_dir, exist_ok=True)
    chunks = [{"idx": i, "cmd": c, "attempts": 0, "state": "pending", "proc": None}
              for i, c in enumerate(workers)]
    for c in chunks:
        if os.path.getsize(f"{save_dir}/{r['chunks']}_{c['idx']}.csv") if os.path.exists(
                f"{save_dir}/{r['chunks']}_{c['idx']}.csv") else 0:
            c["state"] = "done"
        elif job_name(r, c) in QUEUED:
            # Submitted by a previous scheduler and still queued or running under Slurm.
            c["state"] = "adopted"
        else:
            log_path = f"{save_dir}/inference-{c['idx']}.log"
            if os.path.exists(log_path) and time.time() - os.path.getmtime(log_path) < ADOPT_FRESH_S \
                    and "host=" in open(log_path, errors="ignore").read(4096):
                text = open(log_path, errors="ignore").read()
                if "srun: error" not in text and "Traceback" not in text:
                    c["state"] = "adopted"
    runs.append({**r, "evals": evals, "save_dir": save_dir, "chunk_list": chunks,
                 "finished": any(os.path.exists(f"{save_dir}/{m}") for m in METRIC_FILES)})
    log(f"planned {r['name']}: {r['chunks']} chunks -> {save_dir}")

slots = [{**s, "running": 0, "cooldown_until": 0} for s in spec["slots"]]


def launch(run, chunk, slot):
    log_path = f"{run['save_dir']}/inference-{chunk['idx']}.log"
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in run.get("env", {}).items())
    inner = f"cd {shlex.quote(run['code'])} && echo host=$(hostname) && {env_prefix} {chunk['cmd']}"
    cmd = ["srun", f"--partition={slot['partition']}", "--gres=gpu:1", "--cpus-per-task=8",
           "--mem=40G", "--time=03:00:00", "--exclude=" + ",".join(sorted(EXCLUDED_NODES)),
           f"--job-name={job_name(run, chunk)}"]
    if slot.get("qos"):
        cmd.append(f"--qos={slot['qos']}")
    cmd += ["bash", "-c", inner]
    chunk["proc"] = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    chunk["state"], chunk["slot"] = "running", slot
    chunk["attempts"] += 1
    slot["running"] += 1
    log(f"launch {run['name']} chunk {chunk['idx']} on {slot['partition']} (attempt {chunk['attempts']})")


def finish_run(run):
    n = run["chunks"]
    paths = [f"{run['save_dir']}/{n}_{i}.csv" for i in range(n)]
    with open(f"{run['save_dir']}/results.csv", "w", newline="", encoding="utf-8") as out:
        writer = None
        for p in paths:
            with open(p, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    for e in run["evals"]:
        rc = subprocess.run(e, shell=True, cwd=run["code"], stdout=open(f"{run['save_dir']}/evaluation.log", "a"),
                            stderr=subprocess.STDOUT).returncode
        log(f"evaluated {run['name']} rc={rc}")
    run["finished"] = True
    try:
        if os.path.exists(f"{run['save_dir']}/sember_grounding_metrics.json"):
            o = json.load(open(f"{run['save_dir']}/sember_grounding_metrics.json"))["overall"]
            log(f"RESULT {run['name']}: mIoU {o['mean_iou_percent']:.1f} R@0.5 {o['recall_at_1_iou_0.5_percent']:.1f} "
                f"(n={o['total']})")
        else:
            o = json.load(open(f"{run['save_dir']}/sember_mcq_metrics.json"))["overall"]
            log(f"RESULT {run['name']}: {o['accuracy_percent']:.1f}% ({o['correct']}/{o['total']})")
    except Exception as exc:
        log(f"RESULT {run['name']}: metrics missing ({exc})")


def write_status():
    with open(status_path, "w") as f:
        for run in runs:
            states = [c["state"] for c in run["chunk_list"]]
            f.write(f"{run['name']:50s} {'finished' if run['finished'] else ''} "
                    f"done={states.count('done')}/{len(states)} running={states.count('running')} "
                    f"failed={states.count('failed')} adopted={states.count('adopted')}\n")


def refresh_excluded_nodes():
    try:
        EXCLUDED_NODES.update(l.strip() for l in open(EXCLUDE_FILE) if l.strip())
    except FileNotFoundError:
        pass


def exclude_node(node, reason):
    if node in EXCLUDED_NODES:
        return
    EXCLUDED_NODES.add(node)
    with open(EXCLUDE_FILE, "a") as f:
        f.write(node + "\n")
    log(f"excluding node {node} ({reason})")


def refresh_slot_limits():
    """Pick up edited max_jobs from the spec so GPU usage can be tuned without a restart."""
    try:
        fresh = {x["partition"]: x["max_jobs"] for x in json.load(open(sys.argv[1]))["slots"]}
    except Exception:
        return
    for s in slots:
        s["max_jobs"] = fresh.get(s["partition"], s["max_jobs"])


while True:
    refresh_slot_limits()
    refresh_excluded_nodes()
    QUEUED_NOW = queued_job_names() if any(c["state"] == "adopted" for r in runs for c in r["chunk_list"]) else set()
    for run in runs:
        for c in run["chunk_list"]:
            if c["state"] != "adopted":
                continue
            csv_path = f"{run['save_dir']}/{run['chunks']}_{c['idx']}.csv"
            log_path = f"{run['save_dir']}/inference-{c['idx']}.log"
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                c["state"] = "done"
                log(f"done {run['name']} chunk {c['idx']} (adopted)")
            elif job_name(run, c) in QUEUED_NOW:
                continue
            elif not os.path.exists(log_path) or time.time() - os.path.getmtime(log_path) > ADOPT_STALE_S:
                c["state"] = "pending"
                log(f"adopted chunk {run['name']} {c['idx']} went stale -> pending")
    for run in runs:
        for c in run["chunk_list"]:
            if c["state"] != "running" or c["proc"].poll() is None:
                continue
            c["slot"]["running"] -= 1
            csv_path = f"{run['save_dir']}/{run['chunks']}_{c['idx']}.csv"
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                c["state"] = "done"
                log(f"done {run['name']} chunk {c['idx']}")
            else:
                tail = open(f"{run['save_dir']}/inference-{c['idx']}.log", errors="ignore").read()[-600:]
                if "job submit limit" in tail or "Unable to allocate resources" in tail:
                    # Slurm refused the submission: not the chunk's fault.
                    c["attempts"] -= 1
                    c["slot"]["cooldown_until"] = time.time() + SUBMIT_LIMIT_COOLDOWN
                elif "couldn't chdir" in tail:
                    node = re.search(r"srun: error: (gpu-\d+)", tail)
                    if node:
                        exclude_node(node.group(1), "filesystem not mounted")
                    c["attempts"] -= 1
                elif any(e in open(f"{run['save_dir']}/inference-{c['idx']}.log", errors="ignore").read()
                         for e in HARDWARE_ERRORS):
                    node = re.search(r"srun: error: (gpu-\d+)", tail)
                    if node:
                        exclude_node(node.group(1), "GPU hardware error")
                    c["attempts"] -= 1
                c["state"] = "pending" if c["attempts"] < MAX_ATTEMPTS else "failed"
                log(f"chunk failed {run['name']} {c['idx']} -> {c['state']}: {tail.strip().splitlines()[-1] if tail.strip() else ''}")
        if not run["finished"] and all(c["state"] == "done" for c in run["chunk_list"]):
            finish_run(run)
    for run in runs:
        for c in run["chunk_list"]:
            if c["state"] != "pending":
                continue
            slot = next((s for s in slots if s["running"] < s["max_jobs"]
                         and time.time() >= s["cooldown_until"]), None)
            if slot is None:
                break
            launch(run, c, slot)
            time.sleep(2)
    write_status()
    if all(run["finished"] or any(c["state"] == "failed" for c in run["chunk_list"]) for run in runs) \
            and not any(c["state"] in ("running", "adopted") for run in runs for c in run["chunk_list"]):
        log("ALL DONE")
        break
    time.sleep(20)
