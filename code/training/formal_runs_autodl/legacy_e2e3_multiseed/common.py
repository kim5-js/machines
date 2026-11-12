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
REQUEST_SHA = "ae66cd4e7e435be3777d860460a22455a7a9a75adc6d30c06a054cb323ed8c1e"
STAGING_SHA = "0c47c11d7c9539a7f08ec687bac0e556d18b4008e76369dec13a888714dcc948"
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
    return [{"key": f"{ds}_{arm}_seed{seed}", "dataset": ds, "seed": seed, "arm": arm}
            for seed in req["matrix"]["seeds"]
            for ds in req["matrix"]["datasets"]
            for arm in req["matrix"]["arms"]]

def resolve_job(req, key, phase):
    job = next(item for item in job_list(req) if item["key"] == key)
    require(phase in ("smoke", "full"), "Invalid phase")
    if phase == "smoke":
        require(job["seed"] == 0, "Smoke runs use the predeclared seed 0 only")
    job["phase"] = phase
    job["run_dir"] = str(ROOT / ("smoke_runs" if phase == "smoke" else "runs") / key)
    ds = dataset(req, job["dataset"])
    job["data_yaml"] = str(ROOT / "smoke_data" / (ds["key"] + ".yaml")) if phase == "smoke" else ds["yaml"]
    job["epochs"] = 2 if phase == "smoke" else req["training_reference_args"]["epochs"]
    return job

def variant(job):
    arm = job["arm"]
    return {"key": arm.lower(), "name": job["key"], "mdg_mode": "none" if arm == "E0" else "dual_view",
            "dual_view": arm != "E0", "degraded_small_weighting": arm == "E3",
            "degraded_small_target_weight": 1.5 if arm == "E3" else 1.0,
            "cbam": False, "focal_eiou": False, "consistency_loss": False}

def expected_args(req, job):
    args = dict(req["training_reference_args"])
    run = Path(job["run_dir"])
    args.update(data=job["data_yaml"], seed=job["seed"], project=str(run.parent), name=run.name,
                save_dir=str(run), epochs=job["epochs"])
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

def verify_release(req, require_preflight=True):
    require(sha(ROOT / "staging_manifest.json") == STAGING_SHA, "Historical staging manifest changed")
    for entry in read(ROOT / "staging_manifest.json")["files"]:
        require(sha(ROOT / entry["path"]) == entry["sha256"], "Frozen file changed: " + entry["path"])
    adapter = read(ROOT / "adapter_manifest.json")
    bundle = read(ROOT / "execution_bundle_manifest.json")
    require(sha(ROOT / "execution_bundle_manifest.json") == adapter["bundle_manifest_sha256"],
            "Uploaded bundle manifest changed")
    code = [{"path": e["path"], "sha256": e["sha256"]} for e in bundle["files"]
            if e["path"].startswith("adapter/") or e["path"] == "prepare.py"]
    require(adapter["files"] == code, "Adapter manifest differs from the reviewed execution bundle")
    for entry in adapter["files"]:
        require(sha(ROOT / entry["path"]) == entry["sha256"], "Adapter changed: " + entry["path"])
    require(sha(Path(req["pretrained_weight"]["path"])) == req["pretrained_weight"]["sha256"],
            "Initial pretrained weight changed")
    proof = read(ROOT / "predecessor_gate.json")
    require(proof["status"] == "passed" and proof["first_stage_complete"], "Predecessor is incomplete")
    if require_preflight:
        preflight = read(ROOT / "preflight.json")
        require(preflight["status"] == "passed", "Image preflight did not pass")
        require(preflight["request_sha256"] == REQUEST_SHA, "Preflight belongs to another request")
        require(preflight["bundle_manifest_sha256"] == adapter["bundle_manifest_sha256"],
                "Preflight uses another execution bundle")
        for entry in preflight["runtime_assets"]:
            require(sha(ROOT / entry["path"]) == entry["sha256"], "Runtime asset changed")
    return sha(ROOT / "adapter_manifest.json")

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
