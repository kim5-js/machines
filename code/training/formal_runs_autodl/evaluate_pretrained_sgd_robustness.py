from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


Transform = Callable[[np.ndarray, Path], np.ndarray]


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
    temporary.replace(path)


def load_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ring3_locked_coco_area_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_assets(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "ring3.pretrained_sgd_robustness_eval_config.v1":
        raise RuntimeError(f"Unexpected config schema: {config.get('schema')!r}")
    if any(value is not True for value in config["invariants"].values()):
        raise RuntimeError(f"Evaluation invariants are not locked true: {config['invariants']}")
    checks: dict[str, Any] = {}
    assets = {
        "dataset_yaml": (Path(config["dataset"]["yaml"]), config["dataset"]["yaml_sha256"]),
        "evaluator": (Path(config["evaluator"]["path"]), config["evaluator"]["sha256"]),
        "mdg_config": (
            Path(config["controlled_degradation"]["source_config"]),
            config["controlled_degradation"]["source_config_sha256"],
        ),
    }
    for key, (path, expected) in assets.items():
        if not path.is_file():
            raise RuntimeError(f"Required evaluation asset is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Evaluation asset hash mismatch for {key}: {actual} != {expected}")
        checks[key] = {"path": str(path), "sha256": actual}
    for key, model in config["models"].items():
        run_dir = Path(model["run_dir"])
        best = run_dir / "weights" / "best.pt"
        results = run_dir / "results.csv"
        if not best.is_file() or not results.is_file():
            raise RuntimeError(f"Existing best checkpoint artifacts are missing for {key}: {run_dir}")
        actual = sha256(best)
        if actual != model["best_weights_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for {key}: {actual} != {model['best_weights_sha256']}")
        checks[key] = {
            "run_dir": str(run_dir),
            "best_weights": str(best),
            "best_weights_sha256": actual,
            "results_csv_sha256": sha256(results),
        }
    return checks


def load_scene_labels(path: Path, expected_names: set[str], allowed: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Manual scene labels do not exist: {path}")
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            name = str(row.get("image_name") or "").strip()
            label = str(row.get("manual_label") or "").strip()
            if not name or label not in allowed:
                raise RuntimeError(f"Invalid scene label at {path}:{line_number}: {row}")
            if name in labels:
                raise RuntimeError(f"Duplicate scene label for {name}")
            labels[name] = label
    if set(labels) != expected_names:
        raise RuntimeError(
            f"Scene label coverage mismatch: missing={len(expected_names - set(labels))}, extra={len(set(labels) - expected_names)}"
        )
    counts = {label: sum(value == label for value in labels.values()) for label in sorted(allowed)}
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"Every scene label must contain images: {counts}")
    return labels, {"path": str(path), "sha256": sha256(path), "counts": counts}


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read validation image: {path}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise RuntimeError(f"Unexpected image representation for {path}: shape={image.shape} dtype={image.dtype}")
    return image


def midpoint(bounds: list[float]) -> float:
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise RuntimeError(f"Expected a two-element severity band, got {bounds!r}")
    low, high = float(bounds[0]), float(bounds[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise RuntimeError(f"Invalid severity band: {bounds!r}")
    return (low + high) / 2.0


def deterministic_rng(namespace: str, family: str, image_name: str) -> random.Random:
    material = f"{namespace}|{family}|{image_name}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(seed)


def low_light_transform(spec: dict[str, Any]) -> Transform:
    brightness = midpoint(spec["brightness"])
    gamma = midpoint(spec["gamma"])
    levels = np.arange(256, dtype=np.float32) / np.float32(255.0)
    lookup = np.clip(
        np.rint(np.power(np.clip(levels, 0.0, 1.0), gamma) * brightness * np.float32(255.0)),
        0.0,
        255.0,
    ).astype(np.uint8)

    def apply(image: np.ndarray, _: Path) -> np.ndarray:
        return lookup[image]

    return apply


def glare_transform(
    *,
    spec: dict[str, Any],
    glare_config: dict[str, Any],
    namespace: str,
) -> Transform:
    target_ratio = midpoint(spec["target_area_ratio"])
    intensity = midpoint(spec["intensity_255"])
    aspect = midpoint(glare_config["aspect_ratio"])
    softness = midpoint(glare_config["softness"])

    def apply(image: np.ndarray, image_path: Path) -> np.ndarray:
        height, width = image.shape[:2]
        target_area = target_ratio * float(height * width)
        radius_y = max(2, int(math.sqrt(target_area / (math.pi * aspect))))
        radius_x = max(3, int(radius_y * aspect))
        radius_y = min(radius_y, max(2, height // 2 - 1))
        radius_x = min(radius_x, max(3, width // 2 - 1))
        rng = deterministic_rng(namespace, "glare", image_path.name)
        x_low, x_high = radius_x, width - radius_x - 1
        y_low, y_high = radius_y, height - radius_y - 1
        center_x = rng.randint(x_low, x_high) if x_high >= x_low else width // 2
        center_y = rng.randint(y_low, y_high) if y_high >= y_low else height // 2
        # The mask is exactly zero outside the ellipse. Restricting the
        # calculation to its bounding box preserves every output pixel while
        # avoiding full-resolution float grids for high-resolution images.
        x0, x1 = max(0, center_x - radius_x), min(width, center_x + radius_x + 1)
        y0, y1 = max(0, center_y - radius_y), min(height, center_y + radius_y + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance = ((xx - float(center_x)) / float(max(radius_x, 1))) ** 2 + (
            (yy - float(center_y)) / float(max(radius_y, 1))
        ) ** 2
        mask = np.exp(-distance * softness) * (distance <= 1.0)
        output = image.copy()
        region = image[y0:y1, x0:x1].astype(np.float32) + mask[..., None].astype(np.float32) * intensity
        output[y0:y1, x0:x1] = np.clip(np.rint(region), 0.0, 255.0).astype(np.uint8)
        return output

    return apply


def build_conditions(config: dict[str, Any]) -> tuple[dict[str, Transform], dict[str, Any]]:
    source_path = Path(config["controlled_degradation"]["source_config"])
    mdg = json.loads(source_path.read_text(encoding="utf-8"))
    namespace = str(config["controlled_degradation"]["random_seed_namespace"])
    conditions: dict[str, Transform] = {"normal": lambda image, _: image.copy()}
    parameters: dict[str, Any] = {"normal": {"identity": True}}
    for severity in config["controlled_degradation"]["severities"]:
        low_spec = mdg["low_light"][severity]
        glare_spec = mdg["glare"][severity]
        conditions[f"low_light_{severity}"] = low_light_transform(low_spec)
        conditions[f"glare_{severity}"] = glare_transform(
            spec=glare_spec,
            glare_config=mdg["glare"],
            namespace=namespace,
        )
        parameters[f"low_light_{severity}"] = {
            "brightness": midpoint(low_spec["brightness"]),
            "gamma": midpoint(low_spec["gamma"]),
        }
        parameters[f"glare_{severity}"] = {
            "target_area_ratio": midpoint(glare_spec["target_area_ratio"]),
            "intensity_255": midpoint(glare_spec["intensity_255"]),
            "aspect_ratio": midpoint(mdg["glare"]["aspect_ratio"]),
            "softness": midpoint(mdg["glare"]["softness"]),
            "center_policy": "deterministic per image and shared across severities",
            "random_seed_namespace": namespace,
        }
    return conditions, parameters


def transform_self_test(conditions: dict[str, Transform], parameters: dict[str, Any]) -> dict[str, Any]:
    height, width = 360, 640
    base = np.full((height, width, 3), 110, dtype=np.uint8)
    image_path = Path("synthetic_scene.jpg")
    results: dict[str, Any] = {}
    for name, transform in conditions.items():
        output = transform(base, image_path)
        if output.shape != base.shape or output.dtype != np.uint8 or not np.isfinite(output).all():
            raise RuntimeError(f"Controlled transform self-test failed for {name}")
        optimized_reference_equivalent = None
        if name.startswith("glare_"):
            spec = parameters[name]
            target_ratio = float(spec["target_area_ratio"])
            intensity = float(spec["intensity_255"])
            aspect = float(spec["aspect_ratio"])
            softness = float(spec["softness"])
            height, width = base.shape[:2]
            target_area = target_ratio * float(height * width)
            radius_y = max(2, int(math.sqrt(target_area / (math.pi * aspect))))
            radius_x = max(3, int(radius_y * aspect))
            radius_y = min(radius_y, max(2, height // 2 - 1))
            radius_x = min(radius_x, max(3, width // 2 - 1))
            rng = deterministic_rng(str(spec["random_seed_namespace"]), "glare", image_path.name)
            x_low, x_high = radius_x, width - radius_x - 1
            y_low, y_high = radius_y, height - radius_y - 1
            center_x = rng.randint(x_low, x_high) if x_high >= x_low else width // 2
            center_y = rng.randint(y_low, y_high) if y_high >= y_low else height // 2
            yy, xx = np.ogrid[:height, :width]
            distance = ((xx - float(center_x)) / float(max(radius_x, 1))) ** 2 + (
                (yy - float(center_y)) / float(max(radius_y, 1))
            ) ** 2
            mask = np.exp(-distance * softness) * (distance <= 1.0)
            dense = base.astype(np.float32) + mask[..., None].astype(np.float32) * intensity
            dense = np.clip(np.rint(dense), 0.0, 255.0).astype(np.uint8)
            optimized_reference_equivalent = bool(np.array_equal(output, dense))
            if not optimized_reference_equivalent:
                raise RuntimeError(f"Sparse glare transform changed pixels for {name}")
        elif name.startswith("low_light_"):
            spec = parameters[name]
            values = base.astype(np.float32) / 255.0
            dense = np.power(np.clip(values, 0.0, 1.0), float(spec["gamma"])) * float(spec["brightness"])
            dense = np.clip(np.rint(dense * 255.0), 0.0, 255.0).astype(np.uint8)
            optimized_reference_equivalent = bool(np.array_equal(output, dense))
            if not optimized_reference_equivalent:
                raise RuntimeError(f"Low-light lookup changed pixels for {name}")
        results[name] = {
            "mean": float(output.mean()),
            "mean_absolute_delta": float(np.abs(output.astype(np.float32) - base.astype(np.float32)).mean()),
            "min": int(output.min()),
            "max": int(output.max()),
            "parameters": parameters[name],
            "optimized_reference_bitwise_equivalent": optimized_reference_equivalent,
        }
    for family in ("glare", "low_light"):
        deltas = [results[f"{family}_{severity}"]["mean_absolute_delta"] for severity in ("mild", "medium", "strong")]
        if not deltas[0] < deltas[1] < deltas[2]:
            raise RuntimeError(f"Controlled severity is not monotonic for {family}: {deltas}")
    return {"status": "passed", "conditions": results}


def release_gpu(model: Any | None = None) -> None:
    if model is not None:
        try:
            model.predictor = None
        except Exception:
            pass
        try:
            model.model.to("cpu")
        except Exception:
            pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def result_records(result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if result.boxes is None:
        return records
    for coordinates, class_value, score in zip(
        result.boxes.xyxy.detach().cpu().tolist(),
        result.boxes.cls.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
        strict=True,
    ):
        x1, y1, x2, y2 = coordinates
        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))
        records.append(
            {
                "class_id": int(class_value),
                "bbox": [float(x1), float(y1), width, height],
                "area": width * height,
                "score": float(score),
            }
        )
    return records


def predict_dataset(
    *,
    model: Any,
    image_paths: list[Path],
    transform: Transform,
    evaluator: Any,
    prediction: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    detections: dict[str, list[dict[str, Any]]] = {}
    ground_truths: dict[str, list[dict[str, Any]]] = {}
    seen: list[str] = []
    chunk_size = int(prediction["chunk_size"])
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        sources: list[np.ndarray] = []
        original_shapes: list[tuple[int, int]] = []
        for image_path in chunk_paths:
            original = load_bgr(image_path)
            transformed = transform(original, image_path)
            if transformed.shape != original.shape or transformed.dtype != np.uint8:
                raise RuntimeError(f"Photometric transform changed geometry or dtype for {image_path}")
            sources.append(np.ascontiguousarray(transformed))
            original_shapes.append((int(original.shape[0]), int(original.shape[1])))
        results = list(
            model.predict(
                source=sources,
                stream=True,
                imgsz=int(prediction["imgsz"]),
                batch=int(prediction["batch"]),
                conf=float(prediction["confidence_floor"]),
                iou=float(prediction["nms_iou"]),
                max_det=int(prediction["prediction_max_det"]),
                device=int(prediction.get("device", 0)),
                save=False,
                verbose=False,
            )
        )
        if len(results) != len(chunk_paths):
            raise RuntimeError(f"Prediction chunk mismatch at {start}: {len(results)} != {len(chunk_paths)}")
        for image_path, original_shape, result in zip(chunk_paths, original_shapes, results, strict=True):
            if tuple(int(value) for value in result.orig_shape) != original_shape:
                raise RuntimeError(f"Prediction geometry mismatch for {image_path}: {result.orig_shape} != {original_shape}")
            key = image_path.name
            if key in detections:
                raise RuntimeError(f"Duplicate prediction key: {key}")
            detections[key] = result_records(result)
            ground_truths[key] = evaluator.read_ground_truths(image_path, original_shape[0], original_shape[1])
            seen.append(key)
    expected = [path.name for path in image_paths]
    if seen != expected:
        raise RuntimeError("Prediction order or coverage changed")
    return detections, ground_truths, {
        "images": len(seen),
        "ground_truths": sum(len(value) for value in ground_truths.values()),
        "predictions": sum(len(value) for value in detections.values()),
    }


def identity_source_equivalence(model: Any, image_paths: list[Path], prediction: dict[str, Any]) -> dict[str, Any]:
    sample_paths = image_paths[:4]
    arrays = [load_bgr(path) for path in sample_paths]
    kwargs = {
        "stream": True,
        "imgsz": int(prediction["imgsz"]),
        "batch": int(prediction["batch"]),
        "conf": float(prediction["confidence_floor"]),
        "iou": float(prediction["nms_iou"]),
        "max_det": int(prediction["prediction_max_det"]),
        "device": int(prediction.get("device", 0)),
        "save": False,
        "verbose": False,
    }
    path_results = list(model.predict(source=[str(path) for path in sample_paths], **kwargs))
    array_results = list(model.predict(source=arrays, **kwargs))
    if len(path_results) != len(array_results):
        raise RuntimeError("Path/array identity source test returned different result counts")
    max_box_delta = 0.0
    max_conf_delta = 0.0
    for left, right in zip(path_results, array_results, strict=True):
        left_boxes = left.boxes.xyxy.detach().cpu().numpy() if left.boxes is not None else np.empty((0, 4))
        right_boxes = right.boxes.xyxy.detach().cpu().numpy() if right.boxes is not None else np.empty((0, 4))
        left_conf = left.boxes.conf.detach().cpu().numpy() if left.boxes is not None else np.empty((0,))
        right_conf = right.boxes.conf.detach().cpu().numpy() if right.boxes is not None else np.empty((0,))
        left_cls = left.boxes.cls.detach().cpu().numpy() if left.boxes is not None else np.empty((0,))
        right_cls = right.boxes.cls.detach().cpu().numpy() if right.boxes is not None else np.empty((0,))
        if left_boxes.shape != right_boxes.shape or not np.array_equal(left_cls, right_cls):
            raise RuntimeError("Path/array identity source test changed detections")
        if left_boxes.size:
            max_box_delta = max(max_box_delta, float(np.abs(left_boxes - right_boxes).max()))
            max_conf_delta = max(max_conf_delta, float(np.abs(left_conf - right_conf).max()))
    if max_box_delta > 1e-4 or max_conf_delta > 1e-6:
        raise RuntimeError(f"Path/array identity source drift: boxes={max_box_delta}, conf={max_conf_delta}")
    return {
        "status": "passed",
        "images": len(sample_paths),
        "max_box_absolute_delta": max_box_delta,
        "max_confidence_absolute_delta": max_conf_delta,
    }


def evaluate_metrics(
    evaluator: Any,
    detections: dict[str, list[dict[str, Any]]],
    ground_truths: dict[str, list[dict[str, Any]]],
    class_ids: list[int],
    max_detections: int,
) -> dict[str, Any]:
    return evaluator.evaluate_area_ranges(detections, ground_truths, class_ids, max_detections=max_detections)


def metric_summary(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "mAP50": metrics["all"]["AP50"],
        "mAP50-95": metrics["all"]["AP50_95"],
        "APsmall": metrics["small"]["AP50_95"],
        "APmedium": metrics["medium"]["AP50_95"],
        "APlarge": metrics["large"]["AP50_95"],
        "ground_truth_count": metrics["all"]["positive_gt_count"],
        "small_ground_truth_count": metrics["small"]["positive_gt_count"],
    }


def valid_cached(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    fingerprints = payload.get("fingerprints", {})
    if any(fingerprints.get(key) != value for key, value in expected.items()):
        return None
    return payload


def import_checkpoint_modules() -> None:
    for name in ("cbam_focal_formal_models", "pretrained_sgd_mainline_models"):
        importlib.import_module(name)


def evaluate_scenes(
    *,
    config: dict[str, Any],
    checks: dict[str, Any],
    evaluator: Any,
    image_paths: list[Path],
    scene_labels: dict[str, str],
    label_audit: dict[str, Any],
    output_root: Path,
    self_tests: dict[str, Any],
) -> dict[str, Any]:
    from ultralytics import YOLO

    prediction = {**config["evaluator"], "device": 0}
    class_ids = list(range(5))
    scene_reports: dict[str, Any] = {}
    for experiment in config["scene_evaluation"]["models"]:
        output_path = output_root / "scene_metrics" / f"{experiment}.json"
        expected = {
            "config_sha256": checks["config_sha256"],
            "evaluator_sha256": checks["evaluator"]["sha256"],
            "dataset_yaml_sha256": checks["dataset_yaml"]["sha256"],
            "scene_labels_sha256": label_audit["sha256"],
            "best_weights_sha256": checks[experiment]["best_weights_sha256"],
        }
        cached = valid_cached(output_path, expected)
        if cached is not None:
            print(f"[scene:{experiment}] reusing {output_path}", flush=True)
            scene_reports[experiment] = cached
            continue
        print(f"[scene:{experiment}] predicting locked original val", flush=True)
        model = YOLO(checks[experiment]["best_weights"])
        if experiment == "E0":
            self_tests["identity_source_equivalence"] = identity_source_equivalence(model, image_paths, prediction)
        detections, ground_truths, counts = predict_dataset(
            model=model,
            image_paths=image_paths,
            transform=lambda image, _: image.copy(),
            evaluator=evaluator,
            prediction=prediction,
        )
        full_metrics = evaluate_metrics(
            evaluator,
            detections,
            ground_truths,
            class_ids,
            int(prediction["evaluation_max_det"]),
        )
        scenes: dict[str, Any] = {}
        for scene in config["scene_evaluation"]["allowed_labels"]:
            names = [path.name for path in image_paths if scene_labels[path.name] == scene]
            subset_detections = {name: detections[name] for name in names}
            subset_ground_truths = {name: ground_truths[name] for name in names}
            metrics = evaluate_metrics(
                evaluator,
                subset_detections,
                subset_ground_truths,
                class_ids,
                int(prediction["evaluation_max_det"]),
            )
            scenes[scene] = {
                "image_count": len(names),
                "prediction_count": sum(len(value) for value in subset_detections.values()),
                "summary": metric_summary(metrics),
                "metrics": metrics,
            }
        report = {
            "schema": "ring3.pretrained_sgd_scene_robustness.v1",
            "status": "completed",
            "experiment": experiment,
            "condition": "original_locked_val",
            "fingerprints": expected,
            "prediction_parameters": prediction,
            "counts": counts,
            "full_summary": metric_summary(full_metrics),
            "full_metrics": full_metrics,
            "scenes": scenes,
            "single_seed_mechanism_evidence_only": True,
        }
        write_json_atomic(output_path, report)
        scene_reports[experiment] = report
        release_gpu(model)
    return scene_reports


def evaluate_degradations(
    *,
    config: dict[str, Any],
    checks: dict[str, Any],
    evaluator: Any,
    image_paths: list[Path],
    output_root: Path,
    scene_reports: dict[str, Any],
    conditions: dict[str, Transform],
    condition_parameters: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    from ultralytics import YOLO

    prediction = {**config["evaluator"], "device": 0}
    class_ids = list(range(5))
    reports: dict[str, dict[str, Any]] = defaultdict(dict)
    for experiment in config["controlled_degradation"]["models"]:
        normal_summary = scene_reports[experiment]["full_summary"]
        normal_report = {
            "schema": "ring3.pretrained_sgd_controlled_degradation.v1",
            "status": "completed",
            "experiment": experiment,
            "condition": "normal",
            "family": "normal",
            "severity": "normal",
            "parameters": condition_parameters["normal"],
            "fingerprints": {
                "config_sha256": checks["config_sha256"],
                "evaluator_sha256": checks["evaluator"]["sha256"],
                "dataset_yaml_sha256": checks["dataset_yaml"]["sha256"],
                "best_weights_sha256": checks[experiment]["best_weights_sha256"],
                "source_scene_report_sha256": sha256(output_root / "scene_metrics" / f"{experiment}.json"),
            },
            "prediction_parameters": prediction,
            "counts": scene_reports[experiment]["counts"],
            "summary": normal_summary,
            "metrics": scene_reports[experiment]["full_metrics"],
        }
        normal_path = output_root / "degradation_metrics" / experiment / "normal.json"
        write_json_atomic(normal_path, normal_report)
        reports[experiment]["normal"] = normal_report
        model = YOLO(checks[experiment]["best_weights"])
        for family in config["controlled_degradation"]["families"]:
            for severity in config["controlled_degradation"]["severities"]:
                condition = f"{family}_{severity}"
                output_path = output_root / "degradation_metrics" / experiment / f"{condition}.json"
                expected = {
                    "config_sha256": checks["config_sha256"],
                    "evaluator_sha256": checks["evaluator"]["sha256"],
                    "dataset_yaml_sha256": checks["dataset_yaml"]["sha256"],
                    "mdg_config_sha256": checks["mdg_config"]["sha256"],
                    "best_weights_sha256": checks[experiment]["best_weights_sha256"],
                }
                cached = valid_cached(output_path, expected)
                if cached is not None:
                    print(f"[degradation:{experiment}:{condition}] reusing {output_path}", flush=True)
                    reports[experiment][condition] = cached
                    continue
                print(f"[degradation:{experiment}:{condition}] predicting", flush=True)
                detections, ground_truths, counts = predict_dataset(
                    model=model,
                    image_paths=image_paths,
                    transform=conditions[condition],
                    evaluator=evaluator,
                    prediction=prediction,
                )
                metrics = evaluate_metrics(
                    evaluator,
                    detections,
                    ground_truths,
                    class_ids,
                    int(prediction["evaluation_max_det"]),
                )
                summary = metric_summary(metrics)
                report = {
                    "schema": "ring3.pretrained_sgd_controlled_degradation.v1",
                    "status": "completed",
                    "experiment": experiment,
                    "condition": condition,
                    "family": family,
                    "severity": severity,
                    "parameters": condition_parameters[condition],
                    "fingerprints": expected,
                    "prediction_parameters": prediction,
                    "counts": counts,
                    "summary": summary,
                    "drop_from_normal": {
                        key: float(normal_summary[key]) - float(summary[key])
                        for key in ("mAP50-95", "APsmall")
                    },
                    "metrics": metrics,
                }
                write_json_atomic(output_path, report)
                reports[experiment][condition] = report
                del detections, ground_truths
                release_gpu()
        release_gpu(model)
    return dict(reports)


def write_summaries(
    *,
    output_root: Path,
    config: dict[str, Any],
    checks: dict[str, Any],
    label_audit: dict[str, Any],
    self_tests: dict[str, Any],
    scene_reports: dict[str, Any],
    degradation_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scene_rows: list[dict[str, Any]] = []
    for experiment in config["scene_evaluation"]["models"]:
        for scene in config["scene_evaluation"]["allowed_labels"]:
            item = scene_reports[experiment]["scenes"][scene]
            scene_rows.append(
                {
                    "experiment": experiment,
                    "scene": scene,
                    "image_count": item["image_count"],
                    "ground_truth_count": item["summary"]["ground_truth_count"],
                    "small_ground_truth_count": item["summary"]["small_ground_truth_count"],
                    "mAP50-95": item["summary"]["mAP50-95"],
                    "APsmall": item["summary"]["APsmall"],
                }
            )
    degradation_rows: list[dict[str, Any]] = []
    for experiment in config["controlled_degradation"]["models"]:
        normal = degradation_reports[experiment]["normal"]["summary"]
        for family in config["controlled_degradation"]["families"]:
            for level, severity in enumerate(("normal", *config["controlled_degradation"]["severities"])):
                report = degradation_reports[experiment]["normal" if severity == "normal" else f"{family}_{severity}"]
                summary = report["summary"]
                degradation_rows.append(
                    {
                        "experiment": experiment,
                        "family": family,
                        "severity": severity,
                        "level": level,
                        "mAP50-95": summary["mAP50-95"],
                        "APsmall": summary["APsmall"],
                        "mAP50-95_drop": float(normal["mAP50-95"]) - float(summary["mAP50-95"]),
                        "APsmall_drop": float(normal["APsmall"]) - float(summary["APsmall"]),
                    }
                )
    output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scene_metrics.csv", scene_rows), ("degradation_curves.csv", degradation_rows)):
        with (output_root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema": "ring3.pretrained_sgd_robustness_summary.v1",
        "status": "completed",
        "evaluation_only": True,
        "training_started": False,
        "e3_continued": False,
        "e4_started": False,
        "fingerprints": checks,
        "scene_labels": label_audit,
        "self_tests": self_tests,
        "prediction_parameters": config["evaluator"],
        "scene_rows": scene_rows,
        "degradation_rows": degradation_rows,
        "single_seed_mechanism_evidence_only": True,
    }
    write_json_atomic(output_root / "robustness_summary.json", summary)
    return summary


def plot_curves(output_root: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    levels = ["Normal", "Mild", "Medium", "Strong"]
    for column, family in enumerate(("glare", "low_light")):
        for row_index, metric in enumerate(("mAP50-95", "APsmall")):
            axis = axes[row_index, column]
            for experiment, marker in (("E0", "o"), ("E2", "s")):
                subset = sorted(
                    (row for row in rows if row["experiment"] == experiment and row["family"] == family),
                    key=lambda row: int(row["level"]),
                )
                axis.plot([int(row["level"]) for row in subset], [float(row[metric]) for row in subset], marker=marker, label=experiment)
            axis.set_title(f"{family.replace('_', ' ').title()} - {metric}")
            axis.set_ylabel(metric)
            axis.grid(True, alpha=0.3)
            axis.set_xticks(range(4), levels)
            axis.legend()
    fig.suptitle("Pretrained-SGD E0 vs E2 controlled degradation robustness")
    fig.tight_layout()
    fig.savefig(output_root / "degradation_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing pretrained-SGD checkpoints under manual scenes and controlled degradation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checks = verify_assets(config)
    checks["config_path"] = str(args.config)
    checks["config_sha256"] = sha256(args.config)
    evaluator = load_evaluator(Path(config["evaluator"]["path"]))
    evaluator_self_test = evaluator.synthetic_self_test()
    conditions, condition_parameters = build_conditions(config)
    transform_test = transform_self_test(conditions, condition_parameters)
    self_tests: dict[str, Any] = {
        "evaluator": evaluator_self_test,
        "controlled_transforms": transform_test,
    }
    if args.self_test_only:
        print(json.dumps({"status": "passed", "checks": checks, "self_tests": self_tests}, ensure_ascii=False, indent=2))
        return

    image_paths, data = evaluator.resolve_validation_images(Path(config["dataset"]["yaml"]))
    if len(image_paths) != int(config["dataset"]["images"]) or int(data["nc"]) != 5:
        raise RuntimeError(f"Locked validation contract mismatch: images={len(image_paths)} nc={data['nc']}")
    scene_labels, label_audit = load_scene_labels(
        Path(config["scene_evaluation"]["labels_csv"]),
        {path.name for path in image_paths},
        set(config["scene_evaluation"]["allowed_labels"]),
    )
    import_checkpoint_modules()
    output_root = Path(config["output"])
    scene_reports = evaluate_scenes(
        config=config,
        checks=checks,
        evaluator=evaluator,
        image_paths=image_paths,
        scene_labels=scene_labels,
        label_audit=label_audit,
        output_root=output_root,
        self_tests=self_tests,
    )
    degradation_reports = evaluate_degradations(
        config=config,
        checks=checks,
        evaluator=evaluator,
        image_paths=image_paths,
        output_root=output_root,
        scene_reports=scene_reports,
        conditions=conditions,
        condition_parameters=condition_parameters,
    )
    summary = write_summaries(
        output_root=output_root,
        config=config,
        checks=checks,
        label_audit=label_audit,
        self_tests=self_tests,
        scene_reports=scene_reports,
        degradation_reports=degradation_reports,
    )
    plot_curves(output_root, summary["degradation_rows"])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
