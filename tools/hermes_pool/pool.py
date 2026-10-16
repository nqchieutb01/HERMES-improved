"""Pipelined GPU pool for HERMES runs: every GPU pulls the next chunk as soon as its current one ends.

logs/sched.py packs G chunks into one job and the job ends when its slowest chunk ends, so the other GPUs sit idle
and each new chunk waits in the Slurm queue again. Here a Slurm job is a pool of per-GPU workers that keep claiming
chunks from a shared queue directory until the queue is empty or the job's time limit is near.

  python3 pool.py run <spec.json>     one manager per spec (same spec format as sched.py): queues the spec's chunks,
                                      re-queues failures, merges and scores each finished run
  python3 pool.py keeper <spec.json>  one per cluster: keeps pool jobs alive in the spec's slots while the queue has
                                      work (shares slot limits with sched.py jobs via live_jobs)
  python3 pool.py worker <gpu>        inside a pool job, one per GPU

State lives in state/: queue/<task>.json (pending), claimed/<task>.json (running; records job id and host),
failed/<task>.json (a worker's failure report, consumed by the manager). A chunk is done when its CSV exists.
A spec run may set "est_min" (minutes one chunk needs; default 45): a worker only claims a chunk that fits in the
time its job has left. A run may set "priority" (lower runs first; default 50).
"""
import csv, json, os, re, shlex, subprocess, sys, time

MAIN = "/home/chieu.nguyen/HERMES"
RUNNER = f"{MAIN}/.venv/hermes/bin/python3"
HERE = os.path.dirname(os.path.abspath(__file__))
POOL = f"{HERE}/state"
QUEUE, CLAIMED, FAILED, DONE, JOBS_DIR = (f"{POOL}/{d}" for d in ("queue", "claimed", "failed", "done", "jobs"))
EXCLUDE_FILE = f"{MAIN}/logs/excluded_nodes.txt"
METRIC_FILES = ("sember_mcq_metrics.json", "sember_grounding_metrics.json")
HARDWARE_ERRORS = ("hardware error", "Invalid access of peer GPU memory", "uncorrectable ECC", "CUDA driver error",
                   "No CUDA GPUs are available", "CUDA unknown error", "CUDA driver initialization failed")
MAX_ATTEMPTS = 3
MAX_REQUEUES = 6        # chunks lost without a failure report (job killed at its time limit, node lost)
ADOPT_FRESH_S, ADOPT_STALE_S = 600, 1800
IDLE_EXIT_S = 300       # a worker with an empty queue waits this long for new chunks before it exits
VANISH_GRACE_S = 90    # NFS: the login node can see a claim disappear before it sees the chunk's new CSV
for d in (QUEUE, CLAIMED, FAILED, DONE, JOBS_DIR):
    os.makedirs(d, exist_ok=True)


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def squeue_ids():
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%i"], capture_output=True, text=True, timeout=60,
                             check=True).stdout
    except Exception:
        return None  # unknown: treat every job as alive
    return set(out.split())


def job_alive(job_id):
    """Whether Slurm still lists the job (any state); unknown (squeue error) counts as alive."""
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True,
                             timeout=60)
    except Exception:
        return True
    if out.returncode != 0:
        return "Invalid job id" not in out.stderr  # finished jobs drop out of squeue with this error
    return bool(out.stdout.strip())


def csv_done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def excluded_nodes():
    try:
        return sorted({l.strip() for l in open(EXCLUDE_FILE) if l.strip()})
    except FileNotFoundError:
        return []


def exclude_node(node):
    if node and node not in excluded_nodes():
        with open(EXCLUDE_FILE, "a") as f:
            f.write(node + "\n")


# ---------------------------------------------------------------- manager
def plan(run):
    """Dry-run run.py to get the per-chunk worker commands and evaluator commands (as in sched.py)."""
    args = [RUNNER, "scripts/run.py", *run["overrides"], f"run.num_chunks={run['chunks']}", "runtime.dry_run=true",
            f"hydra.sweep.dir=outputs/hydra/sched-{run['name']}", "hydra.sweep.subdir=0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(run["chunks"])))
    out = subprocess.run(args, cwd=run["code"], env=env, capture_output=True, text=True, check=True).stdout
    workers = [l for l in out.splitlines() if " -m video_qa.hermes_vqa " in l]
    evals = [l for l in out.splitlines() if re.search(r"\seval/\S+\.py\s", l) and "hermes_vqa" not in l]
    save_dir = re.search(r"--save_dir (\S+)", workers[0]).group(1)
    assert len(workers) == run["chunks"], (run["name"], len(workers))
    return workers, evals, save_dir


def finish_run(run):
    n = run["chunks"]
    with open(f"{run['save_dir']}/results.csv", "w", newline="", encoding="utf-8") as out:
        writer = None
        for i in range(n):
            with open(f"{run['save_dir']}/{n}_{i}.csv", newline="", encoding="utf-8-sig") as f:
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


def manage(spec_path):
    spec = json.load(open(spec_path))
    status_path = spec_path.replace(".json", ".status")
    runs = []
    for r in spec["runs"]:
        workers, evals, save_dir = plan(r)
        os.makedirs(save_dir, exist_ok=True)
        prio = int(r.get("priority", spec.get("priority", 50)))
        chunks = []
        for i, cmd in enumerate(workers):
            csv_path, log_path = f"{save_dir}/{r['chunks']}_{i}.csv", f"{save_dir}/inference-{i}.log"
            chunks.append({"idx": i, "task": f"{prio:03d}-{r['name']}-c{i}.json", "attempts": 0, "requeues": 0,
                           "state": "pending", "csv": csv_path, "log": log_path,
                           "body": {"name": f"{r['name']}-c{i}", "code": r["code"], "env": r.get("env", {}),
                                    "cmd": cmd, "log": log_path, "csv": csv_path,
                                    "est_min": float(r.get("est_min", spec.get("est_min", 45)))}})
        runs.append({**r, "evals": evals, "save_dir": save_dir, "chunk_list": chunks,
                     "finished": any(os.path.exists(f"{save_dir}/{m}") for m in METRIC_FILES)})
        log(f"planned {r['name']}: {r['chunks']} chunks -> {save_dir}")

    for run in runs:
        for c in run["chunk_list"]:
            if csv_done(c["csv"]) or os.path.exists(f"{DONE}/{c['task']}"):
                c["state"] = "done"
            elif os.path.exists(f"{QUEUE}/{c['task']}"):
                c["state"] = "queued"
            elif os.path.exists(f"{CLAIMED}/{c['task']}"):
                c["state"] = "claimed"
            elif os.path.exists(c["log"]) and time.time() - os.path.getmtime(c["log"]) < ADOPT_FRESH_S \
                    and "Traceback" not in open(c["log"], errors="ignore").read():
                c["state"] = "external"  # still running under a sched.py job
                log(f"adopted running chunk {c['task']}")

    while True:
        alive = squeue_ids()
        for run in runs:
            for c in run["chunk_list"]:
                report = read_json(f"{FAILED}/{c['task']}")
                if report is not None:
                    os.remove(f"{FAILED}/{c['task']}")
                    if report.get("hardware"):
                        log(f"hardware failure {c['task']} on {report.get('host')}: requeue (no attempt used)")
                    else:
                        c["attempts"] += 1
                    c["state"] = "pending" if c["attempts"] < MAX_ATTEMPTS else "failed"
                    log(f"chunk failed {c['task']} -> {c['state']}: {report.get('last_line', '')}")
                if c["state"] in ("queued", "claimed", "external") and csv_done(c["csv"]):
                    c["state"] = "done"
                    log(f"done {c['task']}")
                elif c["state"] == "queued" and not os.path.exists(f"{QUEUE}/{c['task']}"):
                    if os.path.exists(f"{CLAIMED}/{c['task']}"):
                        c["state"] = "claimed"
                elif c["state"] == "claimed":
                    claim = read_json(f"{CLAIMED}/{c['task']}")
                    if claim is None:
                        os.listdir(run["save_dir"])  # refresh the NFS directory cache
                        if os.path.exists(f"{DONE}/{c['task']}") or csv_done(c["csv"]):
                            c["state"] = "done"
                            log(f"done {c['task']}")
                        elif not os.path.exists(f"{FAILED}/{c['task']}"):
                            c.setdefault("vanished", time.time())
                            if time.time() - c["vanished"] > VANISH_GRACE_S:
                                del c["vanished"]
                                c["requeues"] += 1
                                c["state"] = "pending" if c["requeues"] < MAX_REQUEUES else "failed"
                                log(f"claim of {c['task']} vanished without result -> {c['state']}")
                    elif alive is not None and claim.get("job") and claim["job"] not in alive:
                        # A job submitted after this loop's squeue snapshot (or a transiently empty squeue) also lands
                        # here: require 3 consecutive misses and a direct query of the job before requeueing.
                        c["missing"] = c.get("missing", 0) + 1
                        if c["missing"] >= 3 and not job_alive(claim["job"]) and not csv_done(c["csv"]):
                            try:
                                os.remove(f"{CLAIMED}/{c['task']}")
                            except FileNotFoundError:
                                pass
                            c["missing"] = 0
                            c["requeues"] += 1
                            c["state"] = "pending" if c["requeues"] < MAX_REQUEUES else "failed"
                            log(f"job {claim['job']} of {c['task']} ended without result -> {c['state']}")
                    else:
                        c["missing"] = 0
                elif c["state"] == "external":
                    if not os.path.exists(c["log"]) or time.time() - os.path.getmtime(c["log"]) > ADOPT_STALE_S \
                            or "Traceback" in open(c["log"], errors="ignore").read()[-20000:]:
                        c["state"] = "pending"
                        log(f"adopted chunk {c['task']} stopped without result -> pending")
                if c["state"] == "pending":
                    write_json(f"{QUEUE}/{c['task']}", {**c["body"], "attempt": c["attempts"] + 1})
                    c["state"] = "queued"
        for run in runs:
            if not run["finished"] and all(c["state"] == "done" for c in run["chunk_list"]):
                finish_run(run)
        with open(status_path, "w") as f:
            for run in runs:
                states = [c["state"] for c in run["chunk_list"]]
                f.write(f"{run['name']:40s} {'finished' if run['finished'] else '':8s} "
                        + " ".join(f"{s}={states.count(s)}" for s in sorted(set(states))) + "\n")
        if all(run["finished"] or any(c["state"] == "failed" for c in run["chunk_list"]) for run in runs) and \
                not any(c["state"] in ("queued", "claimed", "external") for run in runs for c in run["chunk_list"]):
            log("ALL DONE")
            return
        time.sleep(20)


# ---------------------------------------------------------------- keeper
def live_jobs(partition):
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-p", partition, "-o", "%i"], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception:
        return 10 ** 6
    return len(out.split())


def job_state(job_id):
    out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T %r"], capture_output=True, text=True).stdout.split()
    return out if len(out) == 2 else None


def submit(slot, gpus):
    mem = int(slot.get("mem_per_gpu", "16G").rstrip("G")) * gpus
    script = f"{JOBS_DIR}/pool.sh"
    with open(script, "w") as f:
        f.write("#!/bin/bash\n"
                "echo host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES\n"
                # A node whose filesystems do not serve the venv (autofs "No such device") kills every worker at once:
                # exclude it for later submissions and stop.
                f'"{RUNNER}" -c "" || {{ echo "runner unusable on $(hostname)"; '
                f'grep -qx "$(hostname -s)" {EXCLUDE_FILE} || hostname -s >> {EXCLUDE_FILE}; exit 3; }}\n'
                'IFS=, read -ra GPU <<< "$CUDA_VISIBLE_DEVICES"\n'
                f'for g in "${{GPU[@]}}"; do {RUNNER} -u {HERE}/pool.py worker $g '
                f'> {JOBS_DIR}/$SLURM_JOB_ID-gpu$g.log 2>&1 & done\n'
                "wait\n")
    cmd = ["sbatch", "--parsable", f"--partition={slot['partition']}", f"--gres=gpu:{gpus}",
           f"--cpus-per-task={8 * gpus}", f"--mem={mem}G", f"--time={slot.get('time', '03:00:00')}",
           "--exclude=" + ",".join(excluded_nodes()), f"--job-name=h-pool-{gpus}g", f"--output={JOBS_DIR}/%j.out"]
    if slot.get("qos"):
        cmd.append(f"--qos={slot['qos']}")
    res = subprocess.run(cmd + [script], capture_output=True, text=True)
    job_id = res.stdout.strip().split(";")[0]
    if res.returncode != 0 or not job_id.isdigit():
        log(f"submit failed on {slot['partition']}: {(res.stdout + res.stderr).strip()[-300:]}")
        slot["cooldown_until"] = time.time() + 120
        return None
    log(f"pool job {job_id} on {slot['partition']} ({gpus} GPU{'s' if gpus > 1 else ''})")
    return {"id": job_id, "slot": slot, "gpus": gpus, "start": time.time()}


def keep(spec_path):
    slots = [{**s, "cooldown_until": 0, "single_until": 0} for s in json.load(open(spec_path))["slots"]]
    jobs = []
    while True:
        alive = squeue_ids()
        if alive is not None:
            jobs = [j for j in jobs if j["id"] in alive]
        waiting_gpus = 0
        for j in jobs:
            st = job_state(j["id"])
            if st and st[0] == "PENDING":
                waiting_gpus += j["gpus"]
                wait_s = j["slot"].get("pack_wait", 600)
                if j["gpus"] > 1 and st[1] in ("Priority", "Resources") and time.time() - j["start"] > wait_s:
                    subprocess.run(["scancel", j["id"]])
                    j["slot"]["single_until"] = time.time() + j["slot"].get("single_hold", 1800)
                    log(f"pool job {j['id']} waited {wait_s}s for {j['gpus']} GPUs: cancelled, using 1-GPU jobs")
        queued = len([f for f in os.listdir(QUEUE) if f.endswith(".json")])
        for slot in slots:
            if queued <= waiting_gpus:
                break
            if time.time() < slot["cooldown_until"] or live_jobs(slot["partition"]) >= slot["max_jobs"]:
                continue
            g = 1 if time.time() < slot["single_until"] else max(1, int(slot.get("gpus_per_job", 1)))
            g = min(g, queued - waiting_gpus)
            job = submit(slot, g)
            if job:
                jobs.append(job)
                waiting_gpus += g
            time.sleep(2)
        time.sleep(30)


# ---------------------------------------------------------------- worker
def job_deadline():
    try:
        end = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%e"], capture_output=True,
                             text=True, timeout=60).stdout.strip()
        return time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return time.time() + 2.8 * 3600


def work(gpu):
    host = os.uname().nodename.split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    deadline = job_deadline()
    log(f"worker host={host} job={job} gpu={gpu}, {(deadline - time.time()) / 60:.0f} min left")
    idle_since = None
    while True:
        left_min = (deadline - time.time()) / 60
        names = sorted(f for f in os.listdir(QUEUE) if f.endswith(".json"))
        if not names:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > IDLE_EXIT_S:
                log("queue empty: exit")
                return
            time.sleep(20)
            continue
        idle_since = None
        task = None
        for name in names:
            body = read_json(f"{QUEUE}/{name}")
            if body is None or body["est_min"] * 1.2 + 5 > left_min:
                continue
            try:
                os.rename(f"{QUEUE}/{name}", f"{CLAIMED}/{name}")
            except FileNotFoundError:
                continue  # another worker claimed it
            task = (name, body)
            break
        if task is None:
            log(f"no queued chunk fits in the {left_min:.0f} min left: exit")
            return
        name, body = task
        write_json(f"{CLAIMED}/{name}", {**body, "job": job, "host": host, "gpu": gpu, "start": time.time()})
        log(f"run {name} (attempt {body.get('attempt')})")
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in body["env"].items())
        inner = (f"cd {shlex.quote(body['code'])} && echo host={host} gpu={gpu} job={job} && "
                 f"CUDA_VISIBLE_DEVICES={gpu} {env_prefix} {body['cmd']}")
        rc = subprocess.run(["bash", "-c", inner], stdout=open(body["log"], "w"), stderr=subprocess.STDOUT).returncode
        if csv_done(body["csv"]):
            log(f"done {name} rc={rc}")
            os.replace(f"{CLAIMED}/{name}", f"{DONE}/{name}")
            continue
        text = open(body["log"], errors="ignore").read()
        hardware = any(e in text for e in HARDWARE_ERRORS)
        lines = [l for l in text[-3000:].splitlines() if l.strip()]
        write_json(f"{FAILED}/{name}", {"rc": rc, "host": host, "hardware": hardware,
                                        "last_line": lines[-1][:400] if lines else ""})
        os.remove(f"{CLAIMED}/{name}")
        log(f"failed {name} rc={rc}{' (hardware)' if hardware else ''}")
        if hardware:
            exclude_node(host)
            return


if __name__ == "__main__":
    {"run": manage, "keeper": keep, "worker": work}[sys.argv[1]](sys.argv[2])
