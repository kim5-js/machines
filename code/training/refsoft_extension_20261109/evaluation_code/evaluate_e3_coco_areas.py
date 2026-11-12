from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
AREA_RANGES = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metric_rows(results_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    valid = [
        row
        for row in rows
        if row.get("metrics/mAP50-95(B)") and math.isfinite(float(row["metrics/mAP50-95(B)"]))
    ]
    if not valid:
        raise RuntimeError(f"No finite validation metric rows in {results_path}")
    return max(valid, key=lambda row: float(row["metrics/mAP50-95(B)"])), valid[-1]


def xywh_iou(detections: list[dict[str, Any]], ground_truths: list[dict[str, Any]]) -> np.ndarray:
    if not detections or not ground_truths:
        return np.zeros((len(detections), len(ground_truths)), dtype=np.float64)
    dt = np.asarray([item["bbox"] for item in detections], dtype=np.float64)
    gt = np.asarray([item["bbox"] for item in ground_truths], dtype=np.float64)
    dt_xyxy = np.column_stack((dt[:, 0], dt[:, 1], dt[:, 0] + dt[:, 2], dt[:, 1] + dt[:, 3]))
    gt_xyxy = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]))
    left_top = np.maximum(dt_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    right_bottom = np.minimum(dt_xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
    intersection = np.clip(right_bottom - left_top, 0.0, None).prod(axis=2)
    dt_area = np.clip(dt[:, 2], 0.0, None) * np.clip(dt[:, 3], 0.0, None)
    gt_area = np.clip(gt[:, 2], 0.0, None) * np.clip(gt[:, 3], 0.0, None)
    union = dt_area[:, None] + gt_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def evaluate_class_area(
    detections_by_image: dict[str, list[dict[str, Any]]],
    ground_truths_by_image: dict[str, list[dict[str, Any]]],
    class_id: int,
    area_range: tuple[float, float],
    *,
    max_detections: int,
) -> dict[str, Any] | None:
    all_scores: list[np.ndarray] = []
    all_matches: list[np.ndarray] = []
    all_ignores: list[np.ndarray] = []
    positive_gt_count = 0
    minimum_area, maximum_area = area_range
    image_ids = sorted(set(detections_by_image) | set(ground_truths_by_image))

    for image_id in image_ids:
        detections = sorted(
            (item for item in detections_by_image.get(image_id, []) if int(item["class_id"]) == class_id),
            key=lambda item: -float(item["score"]),
        )[:max_detections]
        ground_truths = [
            item for item in ground_truths_by_image.get(image_id, []) if int(item["class_id"]) == class_id
        ]
        gt_ignore = np.asarray(
            [
                not (minimum_area <= float(item["area"]) < maximum_area)
                for item in ground_truths
            ],
            dtype=bool,
        )
        if ground_truths:
            gt_order = np.argsort(gt_ignore, kind="mergesort")
            ground_truths = [ground_truths[index] for index in gt_order]
            gt_ignore = gt_ignore[gt_order]
        positive_gt_count += int((~gt_ignore).sum())

        scores = np.asarray([float(item["score"]) for item in detections], dtype=np.float64)
        matches = np.zeros((len(IOU_THRESHOLDS), len(detections)), dtype=bool)
        ignores = np.zeros_like(matches)
        ious = xywh_iou(detections, ground_truths)
        detection_areas = np.asarray([float(item["area"]) for item in detections], dtype=np.float64)

        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            gt_matched = np.zeros(len(ground_truths), dtype=bool)
            for detection_index in range(len(detections)):
                best_gt = -1
                best_iou = min(float(threshold), 1.0 - 1e-10)
                for gt_index in range(len(ground_truths)):
                    if gt_matched[gt_index]:
                        continue
                    if best_gt >= 0 and not gt_ignore[best_gt] and gt_ignore[gt_index]:
                        break
                    candidate_iou = float(ious[detection_index, gt_index])
                    if candidate_iou < best_iou:
                        continue
                    best_iou = candidate_iou
                    best_gt = gt_index
                if best_gt >= 0:
                    gt_matched[best_gt] = True
                    matches[threshold_index, detection_index] = True
                    ignores[threshold_index, detection_index] = bool(gt_ignore[best_gt])
            if detections:
                unmatched = ~matches[threshold_index]
                outside_area = (detection_areas < minimum_area) | (detection_areas >= maximum_area)
                ignores[threshold_index, unmatched & outside_area] = True

        if detections:
            all_scores.append(scores)
            all_matches.append(matches)
            all_ignores.append(ignores)

    if positive_gt_count <= 0:
        return None
    if all_scores:
        scores = np.concatenate(all_scores)
        matches = np.concatenate(all_matches, axis=1)
        ignores = np.concatenate(all_ignores, axis=1)
        order = np.argsort(-scores, kind="mergesort")
        matches = matches[:, order]
        ignores = ignores[:, order]
    else:
        matches = np.zeros((len(IOU_THRESHOLDS), 0), dtype=bool)
        ignores = np.zeros_like(matches)

    threshold_aps: list[float] = []
    for threshold_index in range(len(IOU_THRESHOLDS)):
        true_positive = matches[threshold_index] & ~ignores[threshold_index]
        false_positive = ~matches[threshold_index] & ~ignores[threshold_index]
        tp_cumulative = np.cumsum(true_positive, dtype=np.float64)
        fp_cumulative = np.cumsum(false_positive, dtype=np.float64)
        recall = tp_cumulative / positive_gt_count
        precision = np.divide(
            tp_cumulative,
            tp_cumulative + fp_cumulative,
            out=np.zeros_like(tp_cumulative),
            where=(tp_cumulative + fp_cumulative) > 0.0,
        )
        if precision.size:
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            indices = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
            sampled = np.zeros_like(RECALL_THRESHOLDS)
            valid = indices < precision.size
            sampled[valid] = precision[indices[valid]]
        else:
            sampled = np.zeros_like(RECALL_THRESHOLDS)
        threshold_aps.append(float(sampled.mean()))

    return {
        "class_id": class_id,
        "positive_gt_count": positive_gt_count,
        "AP50": threshold_aps[0],
        "AP50_95": float(np.mean(threshold_aps)),
        "AP_by_iou": {
            f"{threshold:.2f}": value for threshold, value in zip(IOU_THRESHOLDS, threshold_aps, strict=True)
        },
    }


def evaluate_area_ranges(
    detections_by_image: dict[str, list[dict[str, Any]]],
    ground_truths_by_image: dict[str, list[dict[str, Any]]],
    class_ids: list[int],
    *,
    max_detections: int = 100,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for area_name, area_range in AREA_RANGES.items():
        per_class = [
            result
            for class_id in class_ids
            if (
                result := evaluate_class_area(
                    detections_by_image,
                    ground_truths_by_image,
                    class_id,
                    area_range,
                    max_detections=max_detections,
                )
            )
            is not None
        ]
        if not per_class:
            output[area_name] = {"AP50": None, "AP50_95": None, "per_class": []}
            continue
        output[area_name] = {
            "AP50": float(np.mean([item["AP50"] for item in per_class])),
            "AP50_95": float(np.mean([item["AP50_95"] for item in per_class])),
            "positive_gt_count": sum(int(item["positive_gt_count"]) for item in per_class),
            "per_class": per_class,
        }
    return output


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        raise RuntimeError(f"Cannot map image path to labels: {image_path}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_ground_truths(image_path: Path, image_height: int, image_width: int) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Malformed YOLO label at {label_path}:{line_number}")
        class_id = int(fields[0])
        x_center, y_center, width, height = (float(value) for value in fields[1:])
        pixel_width = width * image_width
        pixel_height = height * image_height
        records.append(
            {
                "class_id": class_id,
                "bbox": [
                    (x_center - width / 2.0) * image_width,
                    (y_center - height / 2.0) * image_height,
                    pixel_width,
                    pixel_height,
                ],
                "area": pixel_width * pixel_height,
            }
        )
    return records


def resolve_validation_images(data_yaml_path: Path) -> tuple[list[Path], dict[str, Any]]:
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    val = Path(data["val"])
    val_path = val if val.is_absolute() else root / val
    if val_path.is_dir():
        images = sorted(
            (path for path in val_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: str(path),
        )
    elif val_path.is_file():
        images = [
            Path(line.strip()) if Path(line.strip()).is_absolute() else root / line.strip()
            for line in val_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise RuntimeError(f"Locked validation source does not exist: {val_path}")
    return images, data


def synthetic_self_test() -> dict[str, Any]:
    ground_truths: dict[str, list[dict[str, Any]]] = {
        "small": [{"class_id": 0, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0}],
        "medium": [{"class_id": 0, "bbox": [10.0, 10.0, 50.0, 50.0], "area": 2500.0}],
        "large": [{"class_id": 0, "bbox": [10.0, 10.0, 120.0, 120.0], "area": 14400.0}],
    }
    detections: dict[str, list[dict[str, Any]]] = {
        key: [
            {
                "class_id": int(value[0]["class_id"]),
                "bbox": list(value[0]["bbox"]),
                "area": float(value[0]["area"]),
                "score": 0.99,
            }
        ]
        for key, value in ground_truths.items()
    }
    result = evaluate_area_ranges(detections, ground_truths, [0])
    failures = [
        name
        for name in AREA_RANGES
        if result[name]["AP50_95"] is None or float(result[name]["AP50_95"]) < 0.999999
    ]
    if failures:
        raise RuntimeError(f"COCO-area evaluator perfect-match self-test failed for {failures}: {result}")
    return {"status": "passed", "perfect_match_area_metrics": result}


def run_evaluation(
    *,
    run_dir: Path,
    data_yaml_path: Path,
    output_path: Path,
    imgsz: int,
    batch_size: int,
    device: int,
) -> dict[str, Any]:
    from ultralytics import YOLO
    import ultralytics

    self_test = synthetic_self_test()
    results_path = run_dir / "results.csv"
    best_weights = run_dir / "weights" / "best.pt"
    if not results_path.is_file() or not best_weights.is_file():
        raise RuntimeError(f"Completed E3 artifacts are missing from {run_dir}")
    best_row, final_row = read_metric_rows(results_path)
    image_paths, data = resolve_validation_images(data_yaml_path)
    if len(image_paths) != 1555:
        raise RuntimeError(f"Expected locked SeaDronesSee val count 1555, found {len(image_paths)}")

    model = YOLO(str(best_weights))
    detections_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ground_truths_by_image: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        imgsz=imgsz,
        batch=batch_size,
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
        ground_truths_by_image[image_id] = read_ground_truths(image_path, image_height, image_width)
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
            f"E3 validation prediction stream mismatch: missing={len(expected - seen)}, extra={len(seen - expected)}"
        )
    area_metrics = evaluate_area_ranges(
        dict(detections_by_image),
        ground_truths_by_image,
        list(range(int(data["nc"]))),
        max_detections=100,
    )
    report = {
        "schema": "ring3.e3_coco_area_metrics.v1",
        "status": "completed",
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
            "metrics": area_metrics,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "results_csv": str(results_path),
            "results_csv_sha256": sha256(results_path),
            "best_weights": str(best_weights),
            "best_weights_sha256": sha256(best_weights),
            "data_yaml": str(data_yaml_path),
            "data_yaml_sha256": sha256(data_yaml_path),
            "validation_image_count": len(image_paths),
            "ground_truth_count": sum(len(items) for items in ground_truths_by_image.values()),
            "prediction_count": sum(len(items) for items in detections_by_image.values()),
            "ultralytics_version": ultralytics.__version__,
        },
        "evaluator_self_test": self_test,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report formal E3 best metrics and COCO-style area AP")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.run_dir is None or args.data is None or args.output is None:
        parser.error("--run-dir, --data, and --output are required unless --self-test is used")
    report = run_evaluation(
        run_dir=args.run_dir,
        data_yaml_path=args.data,
        output_path=args.output,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
