from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

import formal_baseline_queue as base
from maritime_degradation import MaritimeDegradation, load_config as load_mdg_config


CONFIG_SCHEMA = "ring3.p3_region_cosine_diagnostic_config.v1"
SUMMARY_SCHEMA = "ring3.p3_region_cosine_diagnostic_summary.v1"
EXPECTED_DETECT_INPUTS = [15, 18, 21]
EXPECTED_FEATURE_LAYER_TYPE = "C2f"
DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_diagnostic_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Unexpected diagnostic schema: {config.get('schema')!r}")
    if int(config.get("formal_batches", 0)) not in range(100, 201):
        raise ValueError("formal_batches must stay within the mentor-requested 100-200 range")
    if int(config.get("batch", 0)) <= 0 or int(config.get("imgsz", 0)) <= 0:
        raise ValueError("batch and imgsz must be positive")
    threshold = float(config["region_definition"]["small_object_area_ratio_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("small-object area-ratio threshold must be in (0, 1)")
    gate = config["gate"]
    if not 0.5 < float(gate["bootstrap_confidence"]) < 1.0:
        raise ValueError("bootstrap confidence must be in (0.5, 1)")
    if int(gate["bootstrap_samples"]) < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    return config


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_mean_ci(
    values: Iterable[float], *, samples: int, confidence: float, seed: int
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "samples": int(samples), "confidence": float(confidence)}
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = int(samples)
    while remaining > 0:
        chunk = min(512, remaining)
        indices = rng.integers(0, array.size, size=(chunk, array.size))
        means.append(array[indices].mean(axis=1))
        remaining -= chunk
    bootstrap = np.concatenate(means)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "count": int(array.size),
        "samples": int(samples),
        "confidence": float(confidence),
        "mean": float(array.mean()),
        "lower": float(np.quantile(bootstrap, alpha)),
        "upper": float(np.quantile(bootstrap, 1.0 - alpha)),
    }


def build_region_masks(
    bboxes_xywhn: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    small_area_ratio_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Project normalized post-augmentation GT boxes onto overlapping P3 grid cells."""
    boxes = bboxes_xywhn.detach().float().cpu()
    indices = batch_indices.detach().long().view(-1).cpu()
    foreground = torch.zeros((batch_size, 1, height, width), dtype=torch.bool)
    small = torch.zeros_like(foreground)
    object_counts = [0 for _ in range(batch_size)]
    small_object_counts = [0 for _ in range(batch_size)]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or indices.numel() != boxes.shape[0]:
        raise ValueError(
            f"Expected normalized xywh boxes and aligned batch indices, got boxes={tuple(boxes.shape)}, "
            f"indices={tuple(indices.shape)}"
        )

    for box, batch_index_tensor in zip(boxes, indices):
        batch_index = int(batch_index_tensor)
        if batch_index < 0 or batch_index >= batch_size:
            raise ValueError(f"Invalid batch index {batch_index} for batch size {batch_size}")
        x, y, box_width, box_height = [float(value) for value in box]
        if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
            raise FloatingPointError(f"Non-finite GT box: {box.tolist()}")
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        x1 = max(0.0, min(1.0, x - box_width / 2.0))
        y1 = max(0.0, min(1.0, y - box_height / 2.0))
        x2 = max(0.0, min(1.0, x + box_width / 2.0))
        y2 = max(0.0, min(1.0, y + box_height / 2.0))
        x_start = min(width - 1, max(0, int(math.floor(x1 * width))))
        y_start = min(height - 1, max(0, int(math.floor(y1 * height))))
        x_end = min(width - 1, max(x_start, int(math.ceil(x2 * width) - 1)))
        y_end = min(height - 1, max(y_start, int(math.ceil(y2 * height) - 1)))
        foreground[batch_index, 0, y_start : y_end + 1, x_start : x_end + 1] = True
        object_counts[batch_index] += 1
        if box_width * box_height <= small_area_ratio_threshold:
            small[batch_index, 0, y_start : y_end + 1, x_start : x_end + 1] = True
            small_object_counts[batch_index] += 1
    return foreground, small, object_counts, small_object_counts


def masked_cosine(
    clean: torch.Tensor, degraded: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean.shape != degraded.shape or clean.ndim != 4:
        raise ValueError(f"Expected paired BCHW features, got {tuple(clean.shape)} and {tuple(degraded.shape)}")
    if mask.shape != (clean.shape[0], 1, clean.shape[2], clean.shape[3]):
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match feature shape {tuple(clean.shape)}")
    clean_float = clean.float()
    degraded_float = degraded.float()
    mask_float = mask.to(device=clean.device, dtype=torch.float32)
    dot = (clean_float * degraded_float * mask_float).sum(dim=(1, 2, 3))
    clean_norm = (clean_float.square() * mask_float).sum(dim=(1, 2, 3)).sqrt()
    degraded_norm = (degraded_float.square() * mask_float).sum(dim=(1, 2, 3)).sqrt()
    cell_counts = mask_float.sum(dim=(1, 2, 3))
    valid = (cell_counts > 0) & (clean_norm > eps) & (degraded_norm > eps)
    cosine = torch.full_like(dot, float("nan"), dtype=torch.float32)
    cosine[valid] = dot[valid] / (clean_norm[valid] * degraded_norm[valid]).clamp_min(eps)
    return cosine.clamp(min=-1.0, max=1.0), cell_counts


def mean_finite(values: Iterable[float]) -> float:
    return float(finite_stats(values).get("mean", float("nan")))


def validate_source_results(run_dir: Path, expected_epochs: int) -> dict[str, Any]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise RuntimeError(f"Source results are missing: {results_path}")
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != expected_epochs:
        raise RuntimeError(f"Source results row count changed: expected={expected_epochs}, observed={len(rows)}")
    final = rows[-1]
    required = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    for key in required:
        if key not in final or not math.isfinite(float(final[key])):
            raise RuntimeError(f"Source final metric is missing or non-finite: {key}")
    return {key: float(final[key]) for key in required}


def preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    lock_path = SETUP_ROOT / "formal_experiment_lock.json"
    amendment_path = SETUP_ROOT / "formal_runtime_amendment.json"
    mdg_path = SETUP_ROOT / "e1_mdg_config.json"
    expected_hashes = {
        lock_path: str(config["formal_lock_sha256"]),
        amendment_path: str(config["runtime_amendment_sha256"]),
        mdg_path: str(config["mdg_config_sha256"]),
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise RuntimeError(f"Required locked file is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Locked hash mismatch for {path.name}: expected={expected}, observed={observed}")

    lock = base.load_lock()
    base.verify_environment(lock)
    base.validate_training_lock(lock)
    dataset = lock["dataset_protocols"]["seadronessee"]
    data_yaml = Path(dataset["yaml"])
    manifest_path = Path(dataset["manifest_path"])
    for path, expected in (
        (data_yaml, dataset["yaml_sha256"]),
        (manifest_path, dataset["manifest_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"SeaDronesSee locked dataset evidence changed: {path}")

    source_run = Path(config["source_run"])
    source_args_path = source_run / "args.yaml"
    if not source_args_path.is_file() or sha256(source_args_path) != config["source_args_sha256"]:
        raise RuntimeError("The recorded lambda=0 source args are missing or changed")
    source_args = yaml.safe_load(source_args_path.read_text(encoding="utf-8"))
    required_args = {
        "data": str(data_yaml),
        "epochs": int(lock["common_train"]["epochs"]),
        "batch": int(config["batch"]),
        "imgsz": int(config["imgsz"]),
        "seed": int(config["seed"]),
        "pretrained": True,
        "deterministic": True,
        "optimizer": lock["common_train"]["optimizer"],
    }
    mismatches = {
        key: {"expected": expected, "observed": source_args.get(key)}
        for key, expected in required_args.items()
        if source_args.get(key) != expected
    }
    if mismatches:
        raise RuntimeError("Lambda=0 source args mismatch: " + json.dumps(mismatches, ensure_ascii=False))

    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError(f"Lambda=0 checkpoint is missing: {checkpoint}")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise RuntimeError(
            f"Lambda=0 checkpoint hash mismatch: expected={config['checkpoint_sha256']}, observed={checkpoint_hash}"
        )
    final_metrics = validate_source_results(source_run, int(lock["common_train"]["epochs"]))
    return {
        "lock": lock,
        "data_yaml": data_yaml,
        "dataset_manifest": manifest_path,
        "source_run": source_run,
        "source_args": source_args,
        "source_args_path": source_args_path,
        "checkpoint": checkpoint,
        "mdg_path": mdg_path,
        "config_sha256": sha256(config_path),
        "source_final_metrics": final_metrics,
    }


def summarize_region(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    cosines = [
        float(row[f"{prefix}_cosine"])
        for row in records
        if math.isfinite(float(row[f"{prefix}_cosine"]))
    ]
    return {
        "cosine": finite_stats(cosines),
        "discrepancy": finite_stats(1.0 - value for value in cosines),
    }


def run_diagnostic(config_path: Path, *, requested_batches: int | None, run_name: str | None) -> Path:
    config = load_diagnostic_config(config_path)
    formal_batches = int(config["formal_batches"])
    batches = formal_batches if requested_batches is None else int(requested_batches)
    if batches <= 0 or batches > formal_batches:
        raise ValueError(f"Requested batches must be within 1..{formal_batches}, got {batches}")
    context = preflight(config, config_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "formal" if batches == formal_batches else "smoke"
    name = run_name or f"{timestamp}_{mode}_{batches}b_lambda0_best"
    run_dir = Path(config["output_root"]) / name
    if run_dir.exists():
        raise FileExistsError(f"Diagnostic run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    append_jsonl(events_path, {"timestamp_utc": utc_now(), "event": "diagnostic_started", "mode": mode})

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_args = dict(context["source_args"])
    source_args.pop("save_dir", None)
    source_args.update(
        {
            "model": str(context["checkpoint"]),
            "data": str(context["data_yaml"]),
            "epochs": 1,
            "save": False,
            "val": False,
            "plots": False,
            "resume": False,
            "project": str(run_dir),
            "name": "loader_state",
            "exist_ok": True,
            "device": str(config["device"]),
        }
    )
    trainer = DetectionTrainer(overrides=source_args)
    model = YOLO(str(context["checkpoint"])).model
    model.to(trainer.device)
    model.requires_grad_(False)
    model.train()
    trainer.model = model
    trainer.stride = max(int(model.stride.max()), 32)

    detect = model.model[-1]
    detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
    feature_layer_index = int(detect_inputs[0])
    feature_layer_type = type(model.model[feature_layer_index]).__name__
    if detect_inputs != EXPECTED_DETECT_INPUTS:
        raise RuntimeError(f"Detect inputs changed: expected={EXPECTED_DETECT_INPUTS}, observed={detect_inputs}")
    if feature_layer_type != EXPECTED_FEATURE_LAYER_TYPE:
        raise RuntimeError(
            f"P3 feature layer changed: expected={EXPECTED_FEATURE_LAYER_TYPE}, observed={feature_layer_type}"
        )

    loader = trainer.get_dataloader(
        trainer.data["train"],
        batch_size=int(config["batch"]),
        rank=-1,
        mode="train",
    )
    mdg = MaritimeDegradation(load_mdg_config(context["mdg_path"]))
    random.seed(seed + 1729)
    captured: list[torch.Tensor] = []

    def capture_feature(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"P3 diagnostic hook expected Tensor, got {type(output).__name__}")
        captured.append(output)

    hook = model.model[feature_layer_index].register_forward_hook(capture_feature)
    per_image: list[dict[str, Any]] = []
    per_batch: list[dict[str, Any]] = []
    mdg_counts: Counter[str] = Counter()
    observed_feature_shape: list[int] | None = None
    image_counter = 0
    amp_enabled = bool(config["amp"]) and trainer.device.type == "cuda"
    small_threshold = float(config["region_definition"]["small_object_area_ratio_threshold"])

    try:
        for batch_index, raw_batch in enumerate(loader, start=1):
            if batch_index > batches:
                break
            batch = trainer.preprocess_batch(raw_batch)
            clean_images = batch["img"]
            clean_batch_size = int(clean_images.shape[0])
            degraded_images, counts = mdg.apply_batch(clean_images.clone())
            mdg_counts.update(counts)
            paired_images = torch.cat((clean_images, degraded_images), dim=0)
            captured.clear()
            with torch.inference_mode(), torch.autocast(
                device_type=trainer.device.type,
                enabled=amp_enabled,
            ):
                model(paired_images)
            if len(captured) != 1:
                raise RuntimeError(f"Expected exactly one P3 hook event, observed {len(captured)}")
            feature = captured[0]
            if feature.ndim != 4 or int(feature.shape[0]) != clean_batch_size * 2:
                raise RuntimeError(
                    f"Unexpected paired P3 shape {tuple(feature.shape)} for clean batch {clean_batch_size}"
                )
            observed_feature_shape = [int(value) for value in feature.shape]
            clean_feature = feature[:clean_batch_size]
            degraded_feature = feature[clean_batch_size:]
            global_cosine = F.cosine_similarity(
                clean_feature.float().flatten(1),
                degraded_feature.float().flatten(1),
                dim=1,
                eps=1e-8,
            ).clamp(min=-1.0, max=1.0)

            foreground_cpu, small_cpu, object_counts, small_object_counts = build_region_masks(
                batch["bboxes"],
                batch["batch_idx"],
                batch_size=clean_batch_size,
                height=int(feature.shape[2]),
                width=int(feature.shape[3]),
                small_area_ratio_threshold=small_threshold,
            )
            foreground = foreground_cpu.to(trainer.device)
            small = small_cpu.to(trainer.device)
            background = ~foreground
            foreground_cosine, foreground_cells = masked_cosine(clean_feature, degraded_feature, foreground)
            background_cosine, background_cells = masked_cosine(clean_feature, degraded_feature, background)
            small_cosine, small_cells = masked_cosine(clean_feature, degraded_feature, small)

            global_values = global_cosine.detach().cpu().tolist()
            foreground_values = foreground_cosine.detach().cpu().tolist()
            background_values = background_cosine.detach().cpu().tolist()
            small_values = small_cosine.detach().cpu().tolist()
            foreground_cell_values = foreground_cells.detach().cpu().tolist()
            background_cell_values = background_cells.detach().cpu().tolist()
            small_cell_values = small_cells.detach().cpu().tolist()
            image_files = batch.get("im_file", [None] * clean_batch_size)

            batch_rows: list[dict[str, Any]] = []
            for local_index in range(clean_batch_size):
                image_counter += 1
                row = {
                    "batch_index": batch_index,
                    "image_index": image_counter,
                    "image_file": str(image_files[local_index]) if local_index < len(image_files) else "",
                    "objects": int(object_counts[local_index]),
                    "small_objects": int(small_object_counts[local_index]),
                    "p3_height": int(feature.shape[2]),
                    "p3_width": int(feature.shape[3]),
                    "foreground_cells": int(foreground_cell_values[local_index]),
                    "background_cells": int(background_cell_values[local_index]),
                    "small_object_cells": int(small_cell_values[local_index]),
                    "global_cosine": float(global_values[local_index]),
                    "foreground_cosine": float(foreground_values[local_index]),
                    "background_cosine": float(background_values[local_index]),
                    "small_object_cosine": float(small_values[local_index]),
                }
                per_image.append(row)
                batch_rows.append(row)

            batch_record = {
                "batch_index": batch_index,
                "images": clean_batch_size,
                "objects": int(sum(object_counts)),
                "small_objects": int(sum(small_object_counts)),
                "global_cosine_mean": mean_finite(row["global_cosine"] for row in batch_rows),
                "foreground_cosine_mean": mean_finite(row["foreground_cosine"] for row in batch_rows),
                "background_cosine_mean": mean_finite(row["background_cosine"] for row in batch_rows),
                "small_object_cosine_mean": mean_finite(row["small_object_cosine"] for row in batch_rows),
                "mdg_counts": counts,
            }
            batch_record["foreground_minus_background_discrepancy"] = (
                (1.0 - batch_record["foreground_cosine_mean"])
                - (1.0 - batch_record["background_cosine_mean"])
            )
            per_batch.append(batch_record)
            append_jsonl(run_dir / "per_batch.jsonl", batch_record)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == batches:
                print(
                    json.dumps({"progress": f"{batch_index}/{batches}", **batch_record}, ensure_ascii=False),
                    flush=True,
                )
    finally:
        hook.remove()

    if len(per_batch) != batches:
        raise RuntimeError(f"Diagnostic ended early: requested={batches}, completed={len(per_batch)}")
    if not per_image or observed_feature_shape is None:
        raise RuntimeError("Diagnostic produced no feature observations")

    with (run_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)

    global_summary = summarize_region(per_image, "global")
    foreground_summary = summarize_region(per_image, "foreground")
    background_summary = summarize_region(per_image, "background")
    small_summary = summarize_region(per_image, "small_object")
    paired_deltas = [
        (1.0 - float(row["foreground_cosine"])) - (1.0 - float(row["background_cosine"]))
        for row in per_image
        if math.isfinite(float(row["foreground_cosine"]))
        and math.isfinite(float(row["background_cosine"]))
    ]
    gate_cfg = config["gate"]
    ci = bootstrap_mean_ci(
        paired_deltas,
        samples=int(gate_cfg["bootstrap_samples"]),
        confidence=float(gate_cfg["bootstrap_confidence"]),
        seed=seed + 404,
    )
    batch_positive_fraction = float(
        np.mean([float(row["foreground_minus_background_discrepancy"]) > 0.0 for row in per_batch])
    )
    foreground_discrepancy_mean = float(foreground_summary["discrepancy"]["mean"])
    background_discrepancy_mean = float(background_summary["discrepancy"]["mean"])
    discrepancy_ratio = foreground_discrepancy_mean / max(background_discrepancy_mean, 1e-12)
    reference = config["reference"]
    global_anchor_delta = float(global_summary["cosine"]["mean"]) - float(reference["global_cosine_mean"])

    conditions = {
        "formal_200_batches": batches == formal_batches,
        "global_anchor_reproduced": abs(global_anchor_delta) <= float(reference["absolute_tolerance"]),
        "valid_foreground_pairs": int(foreground_summary["cosine"]["count"])
        >= int(gate_cfg["minimum_valid_foreground_pairs"]),
        "foreground_batch_fraction": batch_positive_fraction
        >= float(gate_cfg["minimum_batch_fraction_foreground_discrepancy_gt_background"]),
        "foreground_bootstrap_lower_gt_zero": float(ci.get("lower", float("nan"))) > 0.0,
        "foreground_mean_discrepancy_ratio": discrepancy_ratio
        >= float(gate_cfg["minimum_mean_discrepancy_ratio_foreground_over_background"]),
    }
    gate_passed = all(conditions.values())
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "mode": mode,
        "completed_utc": utc_now(),
        "run_dir": str(run_dir),
        "batches": len(per_batch),
        "images": len(per_image),
        "checkpoint": str(context["checkpoint"]),
        "checkpoint_sha256": sha256(context["checkpoint"]),
        "source_run": str(context["source_run"]),
        "source_args_sha256": sha256(context["source_args_path"]),
        "source_final_metrics": context["source_final_metrics"],
        "formal_lock_sha256": config["formal_lock_sha256"],
        "runtime_amendment_sha256": config["runtime_amendment_sha256"],
        "mdg_config_sha256": config["mdg_config_sha256"],
        "diagnostic_config_sha256": context["config_sha256"],
        "diagnostic_code_sha256": sha256(Path(__file__).resolve()),
        "dataset_yaml": str(context["data_yaml"]),
        "dataset_yaml_sha256": sha256(context["data_yaml"]),
        "dataset_manifest": str(context["dataset_manifest"]),
        "dataset_manifest_sha256": sha256(context["dataset_manifest"]),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": str(trainer.device),
            "gpu": torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None,
            "amp": amp_enabled,
            "model_mode": "train_no_grad",
            "optimizer_created": False,
            "backward_calls": 0,
            "trainable_parameter_tensors": sum(parameter.requires_grad for parameter in model.parameters()),
        },
        "architecture": {
            "detect_input_indices": detect_inputs,
            "target_feature_layer_index": feature_layer_index,
            "target_feature_layer_type": feature_layer_type,
            "paired_feature_shape_last_batch": observed_feature_shape,
        },
        "region_definition": config["region_definition"],
        "mdg_counts": dict(mdg_counts),
        "regions": {
            "global": global_summary,
            "foreground": foreground_summary,
            "background": background_summary,
            "small_object": small_summary,
        },
        "foreground_vs_background": {
            "paired_discrepancy_delta": finite_stats(paired_deltas),
            "paired_bootstrap_mean_ci": ci,
            "positive_batch_fraction": batch_positive_fraction,
            "mean_discrepancy_ratio": discrepancy_ratio,
        },
        "reference_global_cosine": {
            **reference,
            "observed": float(global_summary["cosine"]["mean"]),
            "observed_minus_reference": global_anchor_delta,
        },
        "e3_object_aware_gate": {
            "passed": gate_passed,
            "conditions": conditions,
            "decision": "eligible_to_implement_e3_object_aware" if gate_passed else "do_not_start_e3",
        },
    }
    write_json(run_dir / "diagnostic_summary.json", summary)
    write_json(
        run_dir / "diagnostic_protocol.json",
        {
            "schema": CONFIG_SCHEMA,
            "activated_utc": utc_now(),
            "config": config,
            "config_sha256": context["config_sha256"],
            "code_sha256": sha256(Path(__file__).resolve()),
        },
    )
    append_jsonl(
        events_path,
        {
            "timestamp_utc": utc_now(),
            "event": "diagnostic_completed",
            "gate_passed": gate_passed,
            "summary_path": str(run_dir / "diagnostic_summary.json"),
        },
    )
    print(
        json.dumps(
            {"summary": str(run_dir / "diagnostic_summary.json"), "gate": summary["e3_object_aware_gate"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return run_dir


def self_test() -> dict[str, Any]:
    boxes = torch.tensor(
        [[0.25, 0.25, 0.20, 0.20], [0.75, 0.75, 0.02, 0.02]],
        dtype=torch.float32,
    )
    indices = torch.tensor([0, 0], dtype=torch.long)
    foreground, small, object_counts, small_counts = build_region_masks(
        boxes,
        indices,
        batch_size=1,
        height=4,
        width=4,
        small_area_ratio_threshold=0.0025,
    )
    if object_counts != [2] or small_counts != [1] or not bool(small.any()) or not bool(foreground.any()):
        raise RuntimeError("Region mask self-test failed")
    clean = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    clean[:, 0] = 1.0
    degraded = clean.clone()
    degraded[:, 0][foreground[:, 0]] = 0.0
    degraded[:, 1][foreground[:, 0]] = 1.0
    foreground_cosine, _ = masked_cosine(clean, degraded, foreground)
    background_cosine, _ = masked_cosine(clean, degraded, ~foreground)
    if not float(foreground_cosine[0]) < float(background_cosine[0]):
        raise RuntimeError("Masked cosine self-test failed to isolate the foreground discrepancy")
    ci = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=1000, confidence=0.95, seed=0)
    if not float(ci["lower"]) > 0.0:
        raise RuntimeError("Bootstrap self-test failed")
    return {
        "status": "passed",
        "foreground_cells": int(foreground.sum()),
        "small_cells": int(small.sum()),
        "foreground_cosine": float(foreground_cosine[0]),
        "background_cosine": float(background_cosine[0]),
        "bootstrap": ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Short lambda=0 checkpoint diagnostic for regional P3 cosine")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("p3_region_cosine_diagnostic_config.json"),
    )
    parser.add_argument("--batches", type=int, default=None, help="Use <200 only for a non-gating smoke run")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    run_diagnostic(args.config, requested_batches=args.batches, run_name=args.run_name)


if __name__ == "__main__":
    main()
