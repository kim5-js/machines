from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate_e3_coco_areas as evaluator
import recover_wsodd_legacy_coco_areas as recovery_v1


def evaluate_one_chunked(
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
    if len(image_paths) != recovery_v1.EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(
            "Expected locked WSODD legacy validation count "
            f"{recovery_v1.EXPECTED_VALIDATION_IMAGES}, found {len(image_paths)}"
        )
    if int(data["nc"]) != recovery_v1.EXPECTED_CLASSES:
        raise RuntimeError(f"Expected {recovery_v1.EXPECTED_CLASSES} WSODD classes, found {data['nc']}")

    chunk_size = int(os.environ.get("RING3_AREA_CHUNK_SIZE", "4"))
    if chunk_size <= 0:
        raise RuntimeError(f"RING3_AREA_CHUNK_SIZE must be positive, found {chunk_size}")
    model = YOLO(str(best_weights))
    detections_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ground_truths_by_image: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        predictions = model.predict(
            source=[str(path) for path in chunk_paths],
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
        chunk_seen = 0
        for result in predictions:
            chunk_seen += 1
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
        if chunk_seen != len(chunk_paths):
            raise RuntimeError(
                f"Prediction chunk mismatch at {start}: {chunk_seen} != {len(chunk_paths)}"
            )

    expected = {str(path.resolve()) for path in image_paths}
    if seen != expected:
        raise RuntimeError(
            f"WSODD validation prediction stream mismatch: missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    area_metrics = evaluator.evaluate_area_ranges(
        dict(detections_by_image),
        ground_truths_by_image,
        list(range(recovery_v1.EXPECTED_CLASSES)),
        max_detections=100,
    )
    report = {
        "schema": "ring3.wsodd_legacy_split_coco_area_metrics.v2",
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
            "prediction_chunk_size": chunk_size,
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
        "training_started": False,
        "checkpoint_updates": False,
        "e4_started": False,
    }
    recovery_v1.write_json_atomic(output_path, report)
    recovery_v1.release_gpu(model)
    return report


def main() -> int:
    recovery_v1.evaluate_one = evaluate_one_chunked
    result = recovery_v1.main()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
