"""Shared audit gates for the frozen original-split E0/E2/E3 reproduction."""
from pathlib import Path
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
import os
import sys

ROOT = Path(os.environ.get("RING3_LEGACY_RELEASE_ROOT", Path(__file__).resolve().parents[1]))
REQUEST_SHA = "513b51300ded9eb65db256ad727ca25eaa1b9cbbd8d0dd628d93d8febedd9780"
STAGING_SHA = "61ac46ece79800c18b06b5e0c8e7c7add0391bd99c99fc94a897c8a2a4c4ee5a"
METRICS = ("mAP50", "mAP50-95", "APsmall", "APmedium", "APlarge")

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def request():
    require(sha(ROOT / "request.json") == REQUEST_SHA, "Frozen request changed")
    return read(ROOT / "request.json")

def dataset(req, key):
    return next(item for item in req["datasets"] if item["key"] == key)

def job_list(req):
    baselines=[{'key':f'baseline_{model}_seed0','dataset':'FloW','seed':0,'arm':'baseline','model_key':model,'family':'baseline'} for model in req['baseline_order']]
    ablations=[{'key':f'FloW_{arm}_seed{seed}','dataset':'FloW','seed':seed,'arm':arm,'model_key':'yolov8s','family':'ablation'} for seed in req['matrix']['seeds'] for arm in req['matrix']['arms']]
    return baselines+ablations


def resolve_job(req,key,phase):
    job=next(dict(j) for j in job_list(req) if j['key']==key)
    require(phase in ('smoke','full'),'Invalid phase')
    require(phase!='smoke' or job['seed']==0,'Smoke uses seed 0')
    job['phase']=phase
    job['run_dir']=str(ROOT/('smoke_runs' if phase=='smoke' else 'runs')/key)
    job['data_yaml']=str(ROOT/'smoke_data/FloW.yaml') if phase=='smoke' else dataset(req,'FloW')['yaml']
    job['epochs']=2 if phase=='smoke' else 150
    return job


def variant(job):
    arm = job["arm"]
    return {"key": arm.lower(), "name": job["key"], "mdg_mode": "none" if arm == "E0" else "dual_view",
            "dual_view": arm != "E0", "degraded_small_weighting": arm == "E3",
            "degraded_small_target_weight": 1.5 if arm == "E3" else 1.0,
            "cbam": False, "focal_eiou": False, "consistency_loss": False}

def expected_args(req,job):
    args=dict(req['baselines'][job['model_key']]['args'] if job['family']=='baseline' else req['training_reference_args'])
    run=Path(job['run_dir'])
    args.update(data=job['data_yaml'],seed=job['seed'],project=str(run.parent),name=run.name,save_dir=str(run),epochs=job['epochs'])
    return args


def verify_args(actual, expected, defaults=None):
    missing = set(expected) - set(actual)
    differences = {key: {"actual": actual.get(key), "expected": value}
                   for key, value in expected.items() if actual.get(key) != value}
    extras = {key: value for key, value in actual.items()
              if key not in expected and (defaults is None or key not in defaults or defaults[key] != value)}
    require(not missing and not differences and not extras,
            f"Historical argument mismatch: missing={sorted(missing)} differences={differences} extras={extras}")

def bootstrap(req):
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "settings")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["RING3_AREA_CHUNK_SIZE"] = "4"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "historical_code")]
    os.chdir(ROOT)
    import ultralytics
    import torch
    import torchvision
    import numpy
    from ultralytics.utils import SETTINGS
    versions = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "numpy": numpy.__version__}
    for key, value in versions.items():
        require(value == req["runtime"][key], f"Runtime mismatch: {key}={value}")
    require(Path(ultralytics.__file__).resolve().is_relative_to(ROOT / "vendor"), "Wrong Ultralytics import")
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA is required by the old recipe")
    require("4090" in torch.cuda.get_device_name(0), "Unexpected GPU; inspect before changing the runtime")
    logging_keys = ("sync", "wandb", "mlflow", "clearml", "comet", "neptune", "raytune", "dvc", "tensorboard")
    SETTINGS.update({key: False for key in logging_keys if key in SETTINGS})
    return {**versions, "gpu": torch.cuda.get_device_name(0), "ultralytics_file": ultralytics.__file__}

def verify_release(req,require_preflight=True):
    require(sha(ROOT/'staging_manifest.json')==STAGING_SHA,'Historical scientific manifest changed')
    for e in read(ROOT/'staging_manifest.json')['files']:
        require(sha(ROOT/e['path'])==e['sha256'],'Historical scientific file changed: '+e['path'])
    bundle=read(ROOT/'execution_bundle_manifest.json')
    adapter=read(ROOT/'adapter_manifest.json')
    require(bundle['request_sha256']==REQUEST_SHA and sha(ROOT/'execution_bundle_manifest.json')==adapter['bundle_manifest_sha256'],'Bundle identity changed')
    expected=[{'path':e['path'],'sha256':e['sha256']} for e in bundle['files'] if e['path'].startswith('adapter/') or e['path'] in ('prepare.py','launch.py','probe.py','collect.py')]
    require(adapter['files']==expected,'Adapter manifest changed')
    for e in adapter['files']:
        require(sha(ROOT/e['path'])==e['sha256'],'Execution code changed: '+e['path'])
    for spec in req['baselines'].values():
        require(sha(spec['pretrained_weight']['path'])==spec['pretrained_weight']['sha256'],'Generic pretrained weight changed')
    if require_preflight:
        proof=read(ROOT/'preflight.json')
        require(proof['status']=='passed' and proof['request_sha256']==REQUEST_SHA,'Preflight missing or stale')
        require(proof['bundle_manifest_sha256']==adapter['bundle_manifest_sha256'],'Preflight bundle changed')
        for e in proof['runtime_assets']:
            require(sha(ROOT/e['path'])==e['sha256'],'Runtime asset changed: '+e['path'])
    return sha(ROOT/'adapter_manifest.json')


def data_records(req, job, split):
    path = ROOT / "verified_data" / job["dataset"] / (split + ".jsonl")
    details = read(ROOT / "preflight.json")["datasets"][job["dataset"]][split]
    require(sha(path) == details["sha256"], "Verified image manifest changed")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if job["phase"] == "smoke":
        smoke = read(ROOT / "smoke_data" / "manifest.json")
        require(smoke["adapter_manifest_sha256"] == sha(ROOT / "adapter_manifest.json"), "Smoke adapter changed")
        names = set(smoke["datasets"][job["dataset"]][split])
        records = [record for record in records if record["image"] in names]
        require(len(records) == len(names), "Smoke membership changed")
    return records

def verify_data(req, job):
    ds = dataset(req, job["dataset"])
    require(sha(ds["yaml"]) == ds["yaml_sha256"], "Original dataset YAML changed")
    if job["phase"] == "smoke":
        manifest = read(ROOT / "smoke_data" / "manifest.json")
        for path, expected in manifest["files"].items():
            require(sha(ROOT / path) == expected, "Smoke data definition changed: " + path)
    counts = {}
    for split in ("train", "val"):
        records = data_records(req, job, split)
        def check(record):
            require(sha(record["image"]) == record["image_sha256"], "Image content changed: " + record["image"])
            require(sha(record["label"]) == record["label_sha256"], "Label content changed: " + record["label"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check, records))
        counts[split] = len(records)
    return counts

def metric_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]

def verify_completion(req, job, adapter_sha):
    run = Path(job["run_dir"])
    done = read(run / "completion.json")
    require(done["status"] == "completed" and done["job"] == job, "Completion mismatch")
    require(done["request_sha256"] == REQUEST_SHA and done["adapter_manifest_sha256"] == adapter_sha,
            "Completion provenance changed")
    require(done["preflight_sha256"] == sha(ROOT / "preflight.json"), "Completed run uses another preflight")
    require(done["data_yaml_sha256"] == sha(job["data_yaml"]), "Completed data YAML changed")
    for path, expected in done["artifacts"].items():
        require(sha(run / path) == expected, "Completed artifact changed: " + path)
    return done

def verify_evaluation(req, job, adapter_sha):
    done = verify_completion(req, job, adapter_sha)
    run = Path(job["run_dir"])
    result = read(run / "evaluation.json")
    require(result["status"] == "completed" and result["job"] == job, "Evaluation mismatch")
    require(result["completion_sha256"] == sha(run / "completion.json"), "Evaluation uses another completion")
    require(result["checkpoint_sha256"] == done["artifacts"]["weights/best.pt"], "Evaluation checkpoint mismatch")
    require(result["request_sha256"] == REQUEST_SHA and result["adapter_manifest_sha256"] == adapter_sha,
            "Evaluation provenance changed")
    require(result["raw_evaluation_sha256"] == sha(run / "legacy_area_evaluation.json"), "Raw evaluation changed")
    return result

def paired_summary(results, req):
    import statistics
    lookup = {(r["job"]["dataset"], r["job"]["seed"], r["job"]["arm"]): r for r in results}
    output = {}
    for ds in req["matrix"]["datasets"]:
        comparisons = {}
        for left, right in (("E2", "E0"), ("E3", "E0"), ("E3", "E2")):
            metrics = {}
            for metric in METRICS:
                pairs = []
                for seed in req["matrix"]["seeds"]:
                    a, b = lookup.get((ds, seed, left)), lookup.get((ds, seed, right))
                    if a is None or b is None:
                        continue
                    av, bv = a["coco_metrics"][metric], b["coco_metrics"][metric]
                    if av is not None and bv is not None:
                        require(math.isfinite(av) and math.isfinite(bv), "Nonfinite summary input")
                        pairs.append({"seed": seed, "delta_pp": 100 * (av - bv)})
                values = [pair["delta_pp"] for pair in pairs]
                metrics[metric] = {"pairs": pairs, "n": len(values),
                    "mean_pp": statistics.mean(values) if values else None,
                    "sample_sd_pp": statistics.stdev(values) if len(values) > 1 else None,
                    "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values),
                    "zero": sum(x == 0 for x in values), "all_three_seeds": len(values) == 3}
            comparisons[left + "-" + right] = metrics
        output[ds] = comparisons
    return output


def process_record(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': int(pid), 'state': stat[0], 'ppid': int(stat[1]), 'pgid': int(stat[2]),
            'starttime': int(stat[19]), 'cpu_ticks': int(stat[11]) + int(stat[12])}

def verify_legacy_pause(req):
    raise RuntimeError("Historical pause control is not part of this release")


def resource_snapshot(req):
    import subprocess
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
    free,used,util=[int(v.strip()) for v in gpu.split(',')]
    others=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            state=(p/'stat').read_text().split(') ',1)[1].split()[0]
            if state in ('T','t','Z'):continue
            if any(a.endswith((b'/adapter/train.py',b'/adapter/train_baseline.py',b'/formal_baseline_queue.py',b'/formal_baseline_queue_resilient.py')) for a in args):
                others.append({'pid':int(p.name),'command':' '.join(a.decode(errors='replace') for a in args)})
        except (FileNotFoundError,PermissionError):pass
    return {'utc':now(),'free_mib':free,'used_mib':used,'gpu_utilization':util,'other_active_trainers':others,'ready':free>=req['execution']['minimum_free_gpu_mib'] and not others}

def wait_for_resources(req):
    import time
    while True:
        snapshot=resource_snapshot(req)
        if snapshot['ready']:return snapshot
        write(ROOT/'state.json',{'status':'waiting_for_gpu','updated_utc':now(),'resources':snapshot,'supervisor_pid':os.getpid()})
        time.sleep(30)

def verify_eval_data(req,job,split):
    records=data_records(req,job,split)
    for row in records:
        require(sha(row['image'])==row['image_sha256'] and sha(row['label'])==row['label_sha256'],'Evaluation data changed')
    return records
