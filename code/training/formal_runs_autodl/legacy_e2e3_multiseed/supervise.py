"""Run the nine validation jobs, then the fixed 27-job queue, under one lock."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import common as c

def make_smoke_data(req, adapter_sha):
    import yaml
    directory = c.ROOT / "smoke_data"
    manifest_path = directory / "manifest.json"
    if directory.exists():
        manifest = c.read(manifest_path)
        c.require(manifest["adapter_manifest_sha256"] == adapter_sha, "Existing smoke data uses another adapter")
        for path, expected in manifest["files"].items():
            c.require(c.sha(c.ROOT / path) == expected, "Existing smoke definition changed")
        return
    directory.mkdir()
    manifest = {"status": "prepared", "adapter_manifest_sha256": adapter_sha,
        "preflight_sha256": c.sha(c.ROOT / "preflight.json"), "datasets": {}, "files": {},
        "selection": "Before training: sorted original members; include the first eight small-target train images "
                     "and first four small-target val images, then fill to 128/8 in original order.",
        "smoke_only_changes": ["128 train images", "8 val images", "2 epochs", "separate data/output paths"]}
    for ds in req["datasets"]:
        selected = {}
        for split, total, small_count in (("train", 128, 8), ("val", 8, 4)):
            path = c.ROOT / "verified_data" / ds["key"] / (split + ".jsonl")
            records = sorted((json.loads(line) for line in path.read_text().splitlines()), key=lambda r: r["image"])
            small = []
            for record in records:
                width, height = record["width"], record["height"]
                scale = 960 / max(width, height)
                labels = [[float(x) for x in line.split()] for line in Path(record["label"]).read_text().splitlines()
                          if line.strip()]
                if any(row[3] * width * row[4] * height * scale * scale < 1024 for row in labels):
                    small.append(record["image"])
            c.require(len(small) >= small_count, "Smoke subset cannot cover small targets: " + ds["key"])
            selected[split] = list(dict.fromkeys(small[:small_count] + [r["image"] for r in records]))[:total]
            c.require(len(selected[split]) == total, "Not enough smoke images")
            text_path = directory / f"{ds['key']}_{split}.txt"
            text_path.write_text("\n".join(selected[split]) + "\n", encoding="utf-8")
            manifest["files"][str(text_path.relative_to(c.ROOT))] = c.sha(text_path)
        original = yaml.safe_load(Path(ds["yaml"]).read_text())
        original.update(train=str(directory / f"{ds['key']}_train.txt"),
                        val=str(directory / f"{ds['key']}_val.txt"))
        original.pop("test", None)
        yaml_path = directory / (ds["key"] + ".yaml")
        yaml_path.write_text(yaml.safe_dump(original, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest["files"][str(yaml_path.relative_to(c.ROOT))] = c.sha(yaml_path)
        manifest["datasets"][ds["key"]] = selected
    c.write(manifest_path, manifest)

def run_child(job, operation):
    logs = c.ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{job['phase']}_{job['key']}_{operation}.log"
    env = dict(os.environ, RING3_LEGACY_RELEASE_ROOT=str(c.ROOT), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-u", str(c.ROOT / "adapter" / (operation + ".py")),
               "--job", job["key"], "--phase", job["phase"]]
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=c.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        c.write(c.ROOT / "state.json", {"status": "training" if operation == "train" else "evaluating",
            "job": job, "updated_utc": c.now(), "supervisor_pid": os.getpid(), "child_pid": child.pid,
            "log": str(log_path), "run_directory": job["run_dir"]})
        try:
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    c.require(code == 0, f"{job['key']} {operation} failed with code {code}; preserve {log_path}")

def run_phase(req, phase, adapter_sha):
    fixed = c.job_list(req)
    selected = [job for job in fixed if job["seed"] == 0] if phase == "smoke" else fixed
    results = []
    for item in selected:
        job = c.resolve_job(req, item["key"], phase)
        run = Path(job["run_dir"])
        if (run / "completion.json").exists():
            c.verify_completion(req, job, adapter_sha)
        else:
            c.require(not run.exists(), "Incomplete run preserved; investigate before a new attempt: " + str(run))
            run_child(job, "train")
            c.verify_completion(req, job, adapter_sha)
        if not (run / "evaluation.json").exists():
            run_child(job, "evaluate")
        results.append(c.verify_evaluation(req, job, adapter_sha))
        payload = {"status": "running", "phase": phase, "updated_utc": c.now(),
            "request_sha256": c.REQUEST_SHA, "adapter_manifest_sha256": adapter_sha,
            "completed_jobs": len(results), "expected_jobs": len(selected), "results": results}
        if phase == "full":
            payload["paired_comparisons"] = c.paired_summary(results, req)
            payload["completed_datasets"] = [ds for ds in req["matrix"]["datasets"]
                if sum(r["job"]["dataset"] == ds for r in results) == 9]
        c.write(c.ROOT / ("smoke_summary.json" if phase == "smoke" else "summary.json"), payload)
    payload["status"] = "completed"
    payload["completed_utc"] = c.now()
    c.write(c.ROOT / ("smoke_summary.json" if phase == "smoke" else "summary.json"), payload)
    return results

def write_smoke_gate(req, adapter_sha, results):
    c.require(len(results) == 9, "All nine dataset/arm smoke jobs must complete")
    proof = {}
    for ds in req["matrix"]["datasets"]:
        initializations = []
        for arm in ("E0", "E2", "E3"):
            job = c.resolve_job(req, f"{ds}_{arm}_seed0", "smoke")
            run = Path(job["run_dir"])
            c.verify_evaluation(req, job, adapter_sha)
            runtime = c.read(run / "runtime_evidence.json")
            c.require(runtime["initialization"]["nc"] == len(c.dataset(req, ds)["names"]), "Wrong smoke head")
            c.require(runtime["finite_gradient_probe"]["finite"], "Smoke finite gradient is unverified")
            c.require(runtime["semantic_batches_checked"] > 0 and "validated_historical_runtime" in runtime,
                      "Smoke did not validate the historical method")
            initializations.append(runtime["initialization"]["state_sha256"])
            proof[job["key"]] = {"completion_sha256": c.sha(run / "completion.json"),
                                 "evaluation_sha256": c.sha(run / "evaluation.json")}
        c.require(len(set(initializations)) == 1, "Smoke arms had different starting tensors")
    c.write(c.ROOT / "smoke_gate.json", {"status": "passed", "completed_jobs": 9, "utc": c.now(),
        "request_sha256": c.REQUEST_SHA, "adapter_manifest_sha256": adapter_sha,
        "preflight_sha256": c.sha(c.ROOT / "preflight.json"),
        "smoke_data_manifest_sha256": c.sha(c.ROOT / "smoke_data/manifest.json"), "proof": proof})

def verify_smoke_gate(req, adapter_sha):
    gate = c.read(c.ROOT / "smoke_gate.json")
    c.require(gate["status"] == "passed" and gate["completed_jobs"] == 9, "Nine real-data smoke jobs are required")
    c.require(gate["request_sha256"] == c.REQUEST_SHA and gate["adapter_manifest_sha256"] == adapter_sha,
              "Smoke gate belongs to another release")
    c.require(gate["preflight_sha256"] == c.sha(c.ROOT / "preflight.json"), "Preflight changed after smoke")
    c.require(gate["smoke_data_manifest_sha256"] == c.sha(c.ROOT / "smoke_data/manifest.json"), "Smoke data changed")
    expected = {item["key"] for item in c.job_list(req) if item["seed"] == 0}
    c.require(set(gate["proof"]) == expected, "Smoke proof is incomplete")
    for key, proof in gate["proof"].items():
        job = c.resolve_job(req, key, "smoke")
        c.verify_evaluation(req, job, adapter_sha)
        for name in ("completion", "evaluation"):
            c.require(c.sha(Path(job["run_dir"]) / (name + ".json")) == proof[name + "_sha256"],
                      "Smoke completion/evaluation changed")

def run_owned_queue(phase):
    req = c.request()
    adapter_sha = c.verify_release(req)
    frozen = c.read(c.ROOT / "queue_manifest.json")
    c.require(frozen["request_sha256"] == c.REQUEST_SHA, "Queue request mismatch")
    c.require([{key: job[key] for key in ("key", "dataset", "seed", "arm")} for job in frozen["jobs"]]
              == c.job_list(req), "Frozen queue order changed")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        c.require(not any(str(c.ROOT / "adapter" / (name + ".py")).encode() in args for name in ("train", "evaluate")),
                  "An earlier training/evaluation child is still active")
    if phase in ("smoke", "all"):
        make_smoke_data(req, adapter_sha)
        results = run_phase(req, "smoke", adapter_sha)
        write_smoke_gate(req, adapter_sha, results)
    if phase in ("full", "all"):
        verify_smoke_gate(req, adapter_sha)
        results = run_phase(req, "full", adapter_sha)
        c.require(len(results) == 27, "Full queue is incomplete")
    c.write(c.ROOT / "state.json", {"status": "completed" if phase in ("full", "all") else "smoke_completed",
        "updated_utc": c.now(), "completed_scientific_jobs": 27 if phase in ("full", "all") else 0,
        "completed_smoke_jobs": 9, "request_sha256": c.REQUEST_SHA, "adapter_manifest_sha256": adapter_sha})

def main(phase):
    import fcntl
    c.require(c.ROOT.is_dir(), "Prepare the release before launching the queue")
    with (c.ROOT / "queue.lock").open("a+") as lock:
        # A second invocation must fail before it can overwrite the active queue state.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run_owned_queue(phase)
        except BaseException:
            c.write(c.ROOT / "state.json", {"status": "failed", "phase": phase,
                "updated_utc": c.now(), "traceback": traceback.format_exc(),
                "recovery": "Inspect retained evidence. No automatic recipe changes, overwrite, or retraining."})
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full", "all"), default="all")
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    main(arguments.phase)
