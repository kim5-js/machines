import json,os,signal,subprocess,sys
from pathlib import Path
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
        c.write(c.ROOT / "state.json", {"status": "training" if operation in ("train", "train_baseline") else "evaluating",
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

def write_smoke_gate(req, adapter_sha, results):
    c.require(len(results) == len(req['matrix']['arms']), 'All five FloW smoke branches must complete')
    proof, initializations = {}, []
    for arm in req['matrix']['arms']:
        job = c.resolve_job(req, 'FloW_' + arm + '_seed0', 'smoke')
        run = Path(job['run_dir'])
        c.verify_completion(req, job, adapter_sha)
        runtime = c.read(run / 'runtime_evidence.json')
        c.require(runtime['initialization']['nc'] == 1, 'Wrong smoke detection head')
        c.require(runtime['finite_gradient_probe']['finite'] and runtime['finite_gradient_probe']['nonzero'], 'Finite gradient not verified')
        c.require(runtime['semantic_batches_checked'] > 0 and 'validated_historical_runtime' in runtime, 'Historical semantics not verified')
        initializations.append(runtime['initialization']['nonattention_state_sha256'])
        if req['arms'][arm]['cbam']:c.require(runtime['attention_update_probe']['both_layers_nonzero'],'CBAM did not update both layers')
        proof[job['key']] = {'completion_sha256': c.sha(run / 'completion.json')}
    c.require(len(set(initializations)) == 1, 'Paired smoke arms initialized differently')
    c.write(c.ROOT / 'smoke_gate.json', {'status': 'passed', 'completed_jobs': len(req['matrix']['arms']), 'utc': c.now(),
        'request_sha256': c.REQUEST_SHA, 'adapter_manifest_sha256': adapter_sha,
        'preflight_sha256': c.sha(c.ROOT / 'preflight.json'),
        'smoke_data_manifest_sha256': c.sha(c.ROOT / 'smoke_data/manifest.json'), 'proof': proof,
        'scope': 'real-data training and native validation before full ablation queue'})

def verify_smoke_gate(req, adapter_sha):
    gate = c.read(c.ROOT / 'smoke_gate.json')
    c.require(gate['status'] == 'passed' and gate['completed_jobs'] == len(req['matrix']['arms']), 'All five real-data training smokes are required')
    c.require(gate['request_sha256'] == c.REQUEST_SHA and gate['adapter_manifest_sha256'] == adapter_sha, 'Smoke release changed')
    c.require(gate['preflight_sha256'] == c.sha(c.ROOT / 'preflight.json'), 'Preflight changed')
    c.require(gate['smoke_data_manifest_sha256'] == c.sha(c.ROOT / 'smoke_data/manifest.json'), 'Smoke subset changed')
    c.require(set(gate['proof']) == {'FloW_' + a + '_seed0' for a in req['matrix']['arms']}, 'Incomplete smoke proof')
    for key, proof in gate['proof'].items():
        job = c.resolve_job(req, key, 'smoke')
        c.verify_completion(req, job, adapter_sha)
        c.require(c.sha(Path(job['run_dir']) / 'completion.json') == proof['completion_sha256'], 'Smoke completion changed')
