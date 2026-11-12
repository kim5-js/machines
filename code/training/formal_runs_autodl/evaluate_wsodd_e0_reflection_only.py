from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def finite_metric_rows(path: Path) -> list[dict[str, str]]:
    required = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    output = []
    for row in read_rows(path):
        try:
            if all(math.isfinite(float(row[key])) for key in required):
                output.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if not output:
        raise RuntimeError(f"No finite metric rows: {path}")
    return output


def resolve_test_images(config: dict[str, Any]) -> list[Path]:
    manifest = Path(config["dataset"]["test_manifest"])
    images = [Path(line.strip()).resolve() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(images) != int(config["dataset"]["test_images"]):
        raise RuntimeError(f"Fixed test count changed: {len(images)}")
    if len(images) != len(set(images)) or not all(path.is_file() for path in images):
        raise RuntimeError("Fixed test manifest has duplicates or missing images")
    return images


def find_baseline(summary: dict[str, Any], expected: dict[str, float]) -> dict[str, float]:
    candidates: list[dict[str, Any]] = []
    for value in summary.values():
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict) and item.get("experiment") == "E0")
    metric_keys = tuple(expected)
    for item in candidates:
        if all(key in item and item[key] is not None for key in metric_keys):
            observed = {key: float(item[key]) for key in metric_keys}
            if all(math.isclose(observed[key], expected[key], rel_tol=0.0, abs_tol=1e-6) for key in metric_keys):
                return observed
    raise RuntimeError("Formal E0 fixed-test metrics were not found exactly in the locked baseline summary")


def run(config_path: Path, run_dir: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    assets = {
        "dataset_yaml": (Path(config["dataset"]["yaml"]), config["dataset"]["yaml_sha256"]),
        "test_manifest": (Path(config["dataset"]["test_manifest"]), config["dataset"]["test_manifest_sha256"]),
        "evaluator": (Path(evaluation["evaluator"]), evaluation["evaluator_sha256"]),
        "robust_helper": (Path(evaluation["robust_helper"]), evaluation["robust_helper_sha256"]),
        "baseline_summary": (Path(evaluation["baseline_summary"]), evaluation["baseline_summary_sha256"]),
    }
    fingerprints: dict[str, Any] = {"config": {"path": str(config_path), "sha256": sha256(config_path)}}
    for key, (path, expected_hash) in assets.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Locked evaluation asset missing or changed: {key}={path}")
        fingerprints[key] = {"path": str(path), "sha256": expected_hash}

    best_weights = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    if not best_weights.is_file() or not results_csv.is_file():
        raise RuntimeError(f"Completed reflection-only artifacts are missing: {run_dir}")
    fingerprints["reflection_only_best"] = {
        "path": str(best_weights),
        "sha256": sha256(best_weights),
        "results_csv_sha256": sha256(results_csv),
    }
    rows = finite_metric_rows(results_csv)
    best_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    final_row = rows[-1]

    evaluator = load_module("ring3_locked_wsodd_area_evaluator", Path(evaluation["evaluator"]))
    robust = load_module("ring3_locked_wsodd_robust_helper", Path(evaluation["robust_helper"]))
    self_test = evaluator.synthetic_self_test()
    image_paths = resolve_test_images(config)
    prediction = {
        "imgsz": evaluation["imgsz"],
        "batch": evaluation["batch"],
        "chunk_size": evaluation["chunk_size"],
        "confidence_floor": evaluation["confidence_floor"],
        "nms_iou": evaluation["nms_iou"],
        "prediction_max_det": evaluation["prediction_max_det"],
        "evaluation_max_det": evaluation["evaluation_max_det"],
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
    dataset = yaml.safe_load(Path(config["dataset"]["yaml"]).read_text(encoding="utf-8"))
    metrics = robust.evaluate_metrics(
        evaluator,
        detections,
        ground_truths,
        list(range(int(dataset["nc"]))),
        int(evaluation["evaluation_max_det"]),
    )
    fixed_test = robust.metric_summary(metrics)
    baseline_summary = json.loads(Path(evaluation["baseline_summary"]).read_text(encoding="utf-8"))
    baseline = find_baseline(baseline_summary, {key: float(value) for key, value in evaluation["formal_e0_fixed_test"].items()})
    delta = {key: float(fixed_test[key]) - baseline[key] for key in baseline}
    report = {
        "schema": "ring3.wsodd_e0_reflection_only_fixed_test_metrics.v1",
        "status": "completed",
        "evaluation_only": true,
        "checkpoint_updated": false,
        "fixed_split": "test",
        "fixed_test_images": len(image_paths),
        "same_evaluator_and_prediction_parameters_as_formal_e0": true,
        "reflection_only_fixed_test": fixed_test,
        "formal_e0_fixed_test": baseline,
        "reflection_only_minus_formal_e0": delta,
        "training_validation": {
            "best_epoch": int(float(best_row["epoch"])),
            "best": {
                "Precision": float(best_row["metrics/precision(B)"]),
                "Recall": float(best_row["metrics/recall(B)"]),
                "mAP50": float(best_row["metrics/mAP50(B)"]),
                "mAP50-95": float(best_row["metrics/mAP50-95(B)"]),
            },
            "final_epoch": int(float(final_row["epoch"])),
            "final": {
                "Precision": float(final_row["metrics/precision(B)"]),
                "Recall": float(final_row["metrics/recall(B)"]),
                "mAP50": float(final_row["metrics/mAP50(B)"]),
                "mAP50-95": float(final_row["metrics/mAP50-95(B)"]),
            },
        },
        "prediction_parameters": prediction,
        "prediction_counts": counts,
        "evaluator_self_test": self_test,
        "identity_source_equivalence": identity_equivalence,
        "fingerprints": fingerprints,
        "single_seed_mechanism_evidence_only": true,
        "interpretation_guard": "This isolates the locked reflection-only overlay recipe on the current pretrained-SGD protocol; it does not establish variance robustness.",
    }
    write_json_atomic(output_path, report)
    robust.release_gpu(model)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WSODD E0 plus reflection-only on the locked fixed test split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_dir, args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
