from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


SCHEMA = "ring3.wsodd_single_checkpoint_fixed_test_metrics.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_metric_rows(path: Path) -> list[dict[str, str]]:
    required = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    output: list[dict[str, str]] = []
    for row in rows:
        try:
            if all(math.isfinite(float(row[key])) for key in required):
                output.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if not output:
        raise RuntimeError(f"No finite metric rows: {path}")
    return output


def resolve_test_images(config: dict[str, Any]) -> list[Path]:
    dataset = config["dataset"]
    manifest = Path(dataset["test_manifest"])
    images = [Path(line.strip()).resolve() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(images) != int(dataset["test_images"]):
        raise RuntimeError(f"Fixed test count changed: {len(images)}")
    if len(images) != len(set(images)) or not all(path.is_file() for path in images):
        raise RuntimeError("Fixed test manifest has duplicates or missing images")
    return images


def metric_row(row: dict[str, str]) -> dict[str, float | int]:
    return {
        "epoch": int(float(row["epoch"])),
        "Precision": float(row["metrics/precision(B)"]),
        "Recall": float(row["metrics/recall(B)"]),
        "mAP50": float(row["metrics/mAP50(B)"]),
        "mAP50-95": float(row["metrics/mAP50-95(B)"]),
    }


def run(config_path: Path, run_dir: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    evaluation = config["post_training_validation"]
    assets = {
        "dataset_yaml": (Path(dataset["yaml"]), dataset["yaml_sha256"]),
        "test_manifest": (Path(dataset["test_manifest"]), dataset["test_manifest_sha256"]),
        "base_evaluator": (Path(evaluation["base_evaluator"]), evaluation["base_evaluator_sha256"]),
        "robust_helper": (Path(evaluation["robust_helper"]), evaluation["robust_helper_sha256"]),
    }
    fingerprints: dict[str, Any] = {"config": {"path": str(config_path), "sha256": sha256(config_path)}}
    for key, (path, expected_hash) in assets.items():
        observed = sha256(path) if path.is_file() else None
        if observed != expected_hash:
            raise RuntimeError(f"Locked fixed-test asset missing or changed: {key}={path}: {observed}")
        fingerprints[key] = {"path": str(path), "sha256": observed}

    best_weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    if not best_weights.is_file() or not results_csv.is_file():
        raise RuntimeError(f"Completed E3 artifacts are missing: {run_dir}")
    rows = finite_metric_rows(results_csv)
    best_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    final_row = rows[-1]
    fingerprints["best_checkpoint"] = {
        "path": str(best_weights),
        "sha256": sha256(best_weights),
        "results_csv_sha256": sha256(results_csv),
    }

    importlib.import_module("pretrained_sgd_mainline_models")
    evaluator = load_module("ring3_locked_wsodd_area_evaluator", Path(evaluation["base_evaluator"]))
    robust = load_module("ring3_locked_wsodd_robust_helper", Path(evaluation["robust_helper"]))
    evaluator_self_test = evaluator.synthetic_self_test()
    image_paths = resolve_test_images(config)
    prediction = {
        "imgsz": int(evaluation["imgsz"]),
        "batch": int(evaluation["batch"]),
        "chunk_size": int(evaluation["chunk_size"]),
        "confidence_floor": float(evaluation["confidence_floor"]),
        "nms_iou": float(evaluation["nms_iou"]),
        "prediction_max_det": int(evaluation["prediction_max_det"]),
        "evaluation_max_det": int(evaluation["evaluation_max_det"]),
        "device": 0,
    }
    model = YOLO(str(best_weights))
    identity_equivalence = robust.identity_source_equivalence(model, image_paths, prediction)
    detections, ground_truths, counts = robust.predict_dataset(
        model=model,
        image_paths=image_paths,
        transform=lambda image, _path: image.copy(),
        evaluator=evaluator,
        prediction=prediction,
    )
    data = yaml.safe_load(Path(dataset["yaml"]).read_text(encoding="utf-8"))
    if int(data["nc"]) != int(dataset["classes"]):
        raise RuntimeError(f"Class count changed: {data['nc']} != {dataset['classes']}")
    metrics = robust.evaluate_metrics(
        evaluator,
        detections,
        ground_truths,
        list(range(int(data["nc"]))),
        prediction["evaluation_max_det"],
    )
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "experiment": "E3",
        "degraded_small_target_weight": 1.5,
        "dataset_protocol": dataset["protocol"],
        "fixed_split": "test",
        "fixed_test_images": len(image_paths),
        "same_evaluator_and_prediction_parameters_as_existing_sequence_disjoint_e0_e3": True,
        "fixed_test": robust.metric_summary(metrics),
        "training_validation": {
            "best": metric_row(best_row),
            "final": metric_row(final_row),
        },
        "prediction_parameters": prediction,
        "prediction_counts": counts,
        "evaluator_self_test": evaluator_self_test,
        "identity_source_equivalence": identity_equivalence,
        "fingerprints": fingerprints,
        "single_seed_mechanism_evidence_only": True,
        "e4_started": False,
    }
    write_json_atomic(output_path, report)
    robust.release_gpu(model)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one completed WSODD checkpoint on the locked sequence-disjoint fixed test")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_dir, args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
