from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_e3_coco_areas as evaluator


EXPECTED_BASE_EVALUATOR_SHA256 = "c1eeb2e5bdf72e627a856ac14f4f679f844de2a9ff0a205cfce50323de9b61cc"
EXPECTED_VALIDATION_IMAGES = 1245
EXPECTED_CLASSES = 14
RUN_KEYS = ("e0", "e1", "e2", "e3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def release_gpu(model: Any | None = None) -> None:
    if model is not None:
        try:
            model.predictor = None
        except Exception:
            pass
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_one(
    *,
    run_key: str,
    run_dir: Path,
    data_yaml: Path,
    output_path: Path,
    imgsz: int,
    batch: int,
    device: int,
) -> dict[str, Any]:
    from ultralytics import YOLO
    import ultralytics

    results_path = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"
    if not results_path.is_file() or not best_weights.is_file():
        raise RuntimeError(f"Completed run artifacts are missing from {run_dir}")

    best_row, final_row = evaluator.read_metric_rows(results_path)
    image_paths, data = evaluator.resolve_validation_images(data_yaml)
    if len(image_paths) != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(
            f"Expected locked WSODD legacy validation count {EXPECTED_VALIDATION_IMAGES}, found {len(image_paths)}"
        )
    if int(data["nc"]) != EXPECTED_CLASSES:
        raise RuntimeError(f"Expected {EXPECTED_CLASSES} WSODD classes, found {data['nc']}")

    model = YOLO(str(best_weights))
    detections_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ground_truths_by_image: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.70,
        max_det=300,
        device=device,
        save=False,
        verbose=False,
    )
    for result in predictions:
        image_path = Path(result.path)
        image_id = str(image_path.resolve())
        seen.add(image_id)
        image_height, image_width = (int(value) for value in result.orig_shape)
        ground_truths_by_image[image_id] = evaluator.read_ground_truths(
            image_path, image_height, image_width
        )
        if result.boxes is None:
            continue
        for coordinates, class_value, score in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = coordinates
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            detections_by_image[image_id].append(
                {
                    "class_id": int(class_value),
                    "bbox": [float(x1), float(y1), width, height],
                    "area": width * height,
                    "score": float(score),
                }
            )

    expected = {str(path.resolve()) for path in image_paths}
    if seen != expected:
        raise RuntimeError(
            f"WSODD validation prediction stream mismatch: missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    area_metrics = evaluator.evaluate_area_ranges(
        dict(detections_by_image),
        ground_truths_by_image,
        list(range(EXPECTED_CLASSES)),
        max_detections=100,
    )
    report = {
        "schema": "ring3.wsodd_legacy_split_coco_area_metrics.v1",
        "status": "completed",
        "experiment": run_key.upper(),
        "dataset_protocol": "wsodd_legacy_image_split_v1",
        "evidence_class": "legacy_protocol_reproduction_not_formal_generalization_evidence",
        "selection": "best validation metrics/mAP50-95(B)",
        "best_epoch": int(float(best_row["epoch"])),
        "best_metrics": {
            "Precision": float(best_row["metrics/precision(B)"]),
            "Recall": float(best_row["metrics/recall(B)"]),
            "mAP50": float(best_row["metrics/mAP50(B)"]),
            "mAP50-95": float(best_row["metrics/mAP50-95(B)"]),
            "APsmall": area_metrics["small"]["AP50_95"],
            "APmedium": area_metrics["medium"]["AP50_95"],
            "APlarge": area_metrics["large"]["AP50_95"],
        },
        "final_epoch": int(float(final_row["epoch"])),
        "final_metrics": {
            "Precision": float(final_row["metrics/precision(B)"]),
            "Recall": float(final_row["metrics/recall(B)"]),
            "mAP50": float(final_row["metrics/mAP50(B)"]),
            "mAP50-95": float(final_row["metrics/mAP50-95(B)"]),
        },
        "area_evaluation": {
            "protocol": "COCO-style 10 IoU thresholds and 101 recall points",
            "area_coordinates": "original validation image pixels",
            "area_ranges": {
                "small": "[0, 32^2)",
                "medium": "[32^2, 96^2)",
                "large": "[96^2, inf)",
            },
            "max_detections_per_image": 100,
            "prediction_confidence_floor": 0.001,
            "prediction_nms_iou": 0.70,
            "prediction_max_detections_per_image": 300,
            "metrics": area_metrics,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "results_csv": str(results_path),
            "results_csv_sha256": evaluator.sha256(results_path),
            "best_weights": str(best_weights),
            "best_weights_sha256": evaluator.sha256(best_weights),
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": evaluator.sha256(data_yaml),
            "validation_image_count": len(image_paths),
            "ground_truth_count": sum(len(items) for items in ground_truths_by_image.values()),
            "prediction_count": sum(len(items) for items in detections_by_image.values()),
            "ultralytics_version": ultralytics.__version__,
        },
        "evaluator_self_test": evaluator.synthetic_self_test(),
        "single_seed_mechanism_evidence_only": True,
        "sequence_leakage_disclosed": True,
        "e4_started": False,
    }
    write_json_atomic(output_path, report)
    release_gpu(model)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover WSODD legacy-split E0-E3 COCO area evaluation without retraining"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    base_evaluator_path = Path(evaluator.__file__).resolve()
    observed_base_hash = evaluator.sha256(base_evaluator_path)
    if observed_base_hash != EXPECTED_BASE_EVALUATOR_SHA256:
        raise RuntimeError(
            f"Base evaluator hash mismatch: {observed_base_hash} != {EXPECTED_BASE_EVALUATOR_SHA256}"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("dataset", {}).get("val_images") != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError("Legacy WSODD config no longer declares 1245 validation images")
    if config.get("dataset", {}).get("classes") != EXPECTED_CLASSES:
        raise RuntimeError("Legacy WSODD config no longer declares 14 classes")
    run_order = config.get("run_order", [])
    if tuple(str(item.get("key")) for item in run_order) != RUN_KEYS:
        raise RuntimeError("Legacy WSODD E0-E3 run order changed")

    original_state = json.loads(args.state.read_text(encoding="utf-8"))
    if original_state.get("completed_runs") != list(RUN_KEYS) or original_state.get("failed_runs"):
        raise RuntimeError("Training completion evidence is not intact for E0-E3")

    recovery_state_path = args.output_root / "wsodd_legacy_area_recovery_state.json"
    reports: dict[str, Any] = {}
    write_json_atomic(
        recovery_state_path,
        {
            "schema": "ring3.wsodd_legacy_area_recovery_state.v1",
            "status": "running",
            "started_utc": utc_now(),
            "completed_runs": [],
            "training_started": False,
            "checkpoint_updates": False,
            "e4_started": False,
        },
    )
    data_yaml = Path(config["dataset"]["yaml"])
    runs_root = Path(config["execution"]["output_root"])
    for item in run_order:
        run_key = str(item["key"])
        run_dir = runs_root / str(item["name"])
        output_path = args.output_root / f"wsodd_legacy_split_sgd_{run_key}_area_metrics.json"
        print(f"[area-recovery:{run_key}] evaluating {run_dir}", flush=True)
        reports[run_key] = evaluate_one(
            run_key=run_key,
            run_dir=run_dir,
            data_yaml=data_yaml,
            output_path=output_path,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        write_json_atomic(
            recovery_state_path,
            {
                "schema": "ring3.wsodd_legacy_area_recovery_state.v1",
                "status": "running",
                "updated_utc": utc_now(),
                "current_run": run_key,
                "completed_runs": list(reports),
                "training_started": False,
                "checkpoint_updates": False,
                "e4_started": False,
            },
        )

    original_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [
        {"experiment": key.upper(), "best_epoch": reports[key]["best_epoch"], **reports[key]["best_metrics"]}
        for key in RUN_KEYS
    ]
    recovered_summary = {
        "schema": "ring3.wsodd_legacy_area_recovery_summary.v1",
        "status": "completed",
        "completed_utc": utc_now(),
        "dataset_protocol": "wsodd_legacy_image_split_v1",
        "evidence_class": "legacy_protocol_reproduction_not_formal_generalization_evidence",
        "original_training_summary": str(args.summary),
        "original_training_summary_sha256": evaluator.sha256(args.summary),
        "original_queue_status_before_recovery": original_state.get("status"),
        "failure_recovered": "SeaDronesSee-specific validation-count guard replaced by locked WSODD count",
        "base_evaluator": str(base_evaluator_path),
        "base_evaluator_sha256": observed_base_hash,
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": evaluator.sha256(Path(__file__).resolve()),
        "prediction_parameters": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "confidence_floor": 0.001,
            "nms_iou": 0.70,
            "prediction_max_det": 300,
            "evaluation_max_det": 100,
            "device": args.device,
        },
        "rows": rows,
        "reports": reports,
        "training_started": False,
        "checkpoint_updates": False,
        "e4_started": False,
        "single_seed_mechanism_evidence_only": True,
        "sequence_leakage_disclosed": True,
    }
    summary_path = args.output_root / "wsodd_legacy_area_recovery_summary.json"
    write_json_atomic(summary_path, recovered_summary)
    write_json_atomic(
        recovery_state_path,
        {
            "schema": "ring3.wsodd_legacy_area_recovery_state.v1",
            "status": "completed",
            "completed_utc": utc_now(),
            "completed_runs": list(RUN_KEYS),
            "summary": str(summary_path),
            "summary_sha256": evaluator.sha256(summary_path),
            "training_started": False,
            "checkpoint_updates": False,
            "e4_started": False,
        },
    )
    print(json.dumps(recovered_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
