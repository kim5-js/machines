#!/usr/bin/env python3
"""Evaluate matched E0/E1 seeds on locked WSODD, SeaDronesSee, and FloW splits.

This is an evaluation-only entry point.  It reuses the project's locked
COCO-area evaluator and controlled photometric transforms, evaluates the clean
fixed split plus strong low-light and glare, and writes paired three-seed
summaries suitable for the cross-dataset table requested by the user.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

import evaluate_pretrained_sgd_robustness as robust


CONFIG_SCHEMA = "ring3.cross_dataset_e0_e1_multiseed_eval_config.v1"
RELEASE_SCHEMA = "ring3.cross_dataset_e0_e1_multiseed_eval_release.v1"
STATE_SCHEMA = "ring3.cross_dataset_e0_e1_multiseed_eval_state.v1"
REPORT_SCHEMA = "ring3.cross_dataset_e0_e1_condition_report.v1"
SUMMARY_SCHEMA = "ring3.cross_dataset_e0_e1_multiseed_eval_summary.v1"
SEEDS = (0, 1, 2)
FAMILIES = ("E0", "E1")
CONDITIONS = ("normal", "low_light_strong", "glare_strong")
METRICS = ("mAP50", "mAP50-95", "APsmall", "APmedium", "APlarge")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def verify_release(config_path: Path) -> dict[str, Any]:
    manifest_path = config_path.parent / "cross_dataset_e0_e1_multiseed_eval_release_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Release manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != RELEASE_SCHEMA or manifest.get("status") != "locked":
        raise RuntimeError("Unexpected cross-dataset evaluation release manifest")
    verified: list[dict[str, str]] = []
    for item in manifest.get("files", []):
        path = config_path.parent / str(item["path"])
        observed = sha256(path) if path.is_file() else "missing"
        if observed != item.get("sha256"):
            raise RuntimeError(f"Release hash mismatch for {item.get('path')}: {observed}")
        verified.append({"path": str(path), "sha256": observed})
    if len(verified) < 12:
        raise RuntimeError(f"Release manifest is incomplete: {len(verified)} files")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "verified_files": verified,
    }


def assert_not_stopped(config: dict[str, Any]) -> None:
    sentinel = Path(config["execution"]["stop_sentinel"])
    if sentinel.exists():
        raise RuntimeError(f"Evaluation stop sentinel exists: {sentinel}")


def verify_training_states(config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for item in config["training_states"]:
        path = Path(item["path"])
        if not path.is_file():
            raise RuntimeError(f"Training state is missing: {path}")
        payload = load_json(path)
        if payload.get("status") != "completed" or payload.get("failed_runs") not in (None, []):
            raise RuntimeError(
                f"Training state is not a clean completed queue: {path} "
                f"status={payload.get('status')} failed_runs={payload.get('failed_runs')}"
            )
        checks[item["key"]] = {
            "path": str(path),
            "sha256": sha256(path),
            "schema": payload.get("schema"),
            "status": payload.get("status"),
            "completed_runs": payload.get("completed_runs"),
        }
    return checks


def verify_args(args_path: Path, dataset: dict[str, Any], model: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if not args_path.is_file():
        raise RuntimeError(f"Training args are missing: {args_path}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = {
        "data": dataset["yaml"],
        "seed": int(model["seed"]),
        **contract,
    }
    mismatches: dict[str, Any] = {}
    for key, value in expected.items():
        observed = args.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not close(observed, value):
                mismatches[key] = {"expected": value, "observed": observed}
        elif observed != value:
            mismatches[key] = {"expected": value, "observed": observed}
    if mismatches:
        raise RuntimeError(f"Training-argument mismatch for {args_path}: {mismatches}")
    return {"path": str(args_path), "sha256": sha256(args_path)}


def verify_models(config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    contract = config["training_contract"]
    for dataset in config["datasets"]:
        dataset_models = dataset["models"]
        expected_keys = [f"{family.lower()}_seed{seed}" for seed in SEEDS for family in FAMILIES]
        if [item["key"] for item in dataset_models] != expected_keys:
            raise RuntimeError(f"Model order/coverage changed for {dataset['key']}")
        for model in dataset_models:
            checkpoint = Path(model["checkpoint"])
            if not checkpoint.is_file():
                raise RuntimeError(f"Checkpoint is missing: {checkpoint}")
            observed = sha256(checkpoint)
            if observed != model["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch for {model['key']}: {observed}")
            args_check = verify_args(checkpoint.parents[1] / "args.yaml", dataset, model, contract)
            checks[f"{dataset['key']}:{model['key']}"] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed,
                "args": args_check,
                "family": model["family"],
                "seed": int(model["seed"]),
            }
    return checks


def parse_manifest(path: Path, manifest_format: str) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if manifest_format == "basename":
        names = rows
    elif manifest_format == "tsv_last_column":
        names = [row.split("\t")[-1].strip() for row in rows]
    else:
        raise RuntimeError(f"Unsupported manifest format: {manifest_format}")
    if any(Path(name).name != name for name in names):
        raise RuntimeError(f"Manifest must resolve to basenames only: {path}")
    return names


def resolve_images(dataset: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    yaml_path = Path(dataset["yaml"])
    manifest_path = Path(dataset["manifest"])
    if sha256(yaml_path) != dataset["yaml_sha256"]:
        raise RuntimeError(f"Dataset YAML hash mismatch for {dataset['key']}")
    if sha256(manifest_path) != dataset["manifest_sha256"]:
        raise RuntimeError(f"Split manifest hash mismatch for {dataset['key']}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    split_source = Path(data[dataset["split"]])
    split_dir = split_source if split_source.is_absolute() else root / split_source
    names = parse_manifest(manifest_path, dataset["manifest_format"])
    if len(names) != int(dataset["images"]) or len(names) != len(set(names)):
        raise RuntimeError(
            f"Split count/uniqueness changed for {dataset['key']}: "
            f"{len(names)} != {dataset['images']}"
        )
    images = [split_dir / name for name in names]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise RuntimeError(f"Fixed-split images are missing for {dataset['key']}: {missing[:5]}")
    if int(data.get("nc", -1)) != int(dataset["classes"]):
        raise RuntimeError(f"Class count changed for {dataset['key']}: {data.get('nc')}")
    return images, data


def prediction_parameters(config: dict[str, Any]) -> dict[str, Any]:
    values = config["prediction"]
    return {
        "imgsz": int(values["imgsz"]),
        "batch": int(values["batch"]),
        "chunk_size": int(values["chunk_size"]),
        "confidence_floor": float(values["confidence_floor"]),
        "nms_iou": float(values["nms_iou"]),
        "prediction_max_det": int(values["prediction_max_det"]),
        "evaluation_max_det": int(values["evaluation_max_det"]),
        "device": int(values["device"]),
    }


def validate_metric_row(row: dict[str, Any], context: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        value = row.get(metric)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"Missing/non-finite {metric} in {context}: {value}")
        result[metric] = float(value)
    for count_key in ("ground_truth_count", "small_ground_truth_count"):
        if row.get(count_key) is not None:
            result[count_key] = int(row[count_key])
    return result


def load_clean_anchors(dataset: dict[str, Any], prediction: dict[str, Any]) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, Any]]:
    anchors = dataset.get("clean_anchors")
    if not anchors:
        return {}, {}
    rows: dict[str, dict[int, dict[str, float]]] = {family: {} for family in FAMILIES}
    checks: dict[str, Any] = {}
    for family in FAMILIES:
        spec = anchors[family]
        path = Path(spec["path"])
        observed = sha256(path) if path.is_file() else "missing"
        if observed != spec["sha256"]:
            raise RuntimeError(f"Clean anchor hash mismatch for {dataset['key']} {family}: {observed}")
        payload = load_json(path)
        if (
            payload.get("status") != "completed"
            or payload.get("fixed_split") != dataset["split"]
            or int(payload.get("fixed_test_images", -1)) != int(dataset["images"])
            or payload.get("prediction_parameters") != prediction
        ):
            raise RuntimeError(f"Clean anchor contract mismatch for {dataset['key']} {family}")
        source_rows = payload.get(spec["field"], [])
        for row in source_rows:
            seed = int(row.get("seed", -1))
            if seed in SEEDS:
                rows[family][seed] = validate_metric_row(row, f"{dataset['key']} {family} seed{seed} anchor")
        if sorted(rows[family]) != list(SEEDS):
            raise RuntimeError(f"Clean anchor seed coverage mismatch for {dataset['key']} {family}")
        checks[family] = {"path": str(path), "sha256": observed, "schema": payload.get("schema")}
    return rows, checks


def condition_bundle(config: dict[str, Any], dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    controlled = config["controlled_degradation"]
    helper_config = {
        "controlled_degradation": {
            "source_config": controlled["source_config"],
            "severities": ["mild", "medium", "strong"],
            "random_seed_namespace": f"{controlled['namespace_root']}/{dataset['key']}",
        }
    }
    all_conditions, parameters = robust.build_conditions(helper_config)
    self_test = robust.transform_self_test(all_conditions, parameters)
    return (
        {name: all_conditions[name] for name in CONDITIONS},
        {name: parameters[name] for name in CONDITIONS},
        self_test,
    )


def report_fingerprints(
    *,
    config_path: Path,
    release: dict[str, Any],
    config: dict[str, Any],
    dataset: dict[str, Any],
    model: dict[str, Any],
    condition: str,
    condition_parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_sha256": sha256(config_path),
        "release_manifest_sha256": release["sha256"],
        "dataset_yaml_sha256": dataset["yaml_sha256"],
        "split_manifest_sha256": dataset["manifest_sha256"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "base_evaluator_sha256": config["assets"]["base_evaluator_sha256"],
        "robust_helper_sha256": config["assets"]["robust_helper_sha256"],
        "mdg_config_sha256": config["controlled_degradation"]["source_config_sha256"],
        "condition": condition,
        "condition_parameters": condition_parameters,
        "prediction_parameters": prediction_parameters(config),
    }


def cached_report(path: Path, fingerprints: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = load_json(path)
    if payload.get("status") != "completed" or payload.get("fingerprints") != fingerprints:
        return None
    validate_metric_row(payload.get("summary", {}), str(path))
    return payload


def write_anchor_report(
    path: Path,
    *,
    dataset: dict[str, Any],
    model: dict[str, Any],
    summary: dict[str, Any],
    fingerprints: dict[str, Any],
    anchor_check: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "completed_utc": utc_now(),
        "evaluation_only": True,
        "dataset": dataset["key"],
        "dataset_protocol": dataset["protocol"],
        "fixed_split": dataset["split"],
        "fixed_images": int(dataset["images"]),
        "experiment": model["family"],
        "seed": int(model["seed"]),
        "condition": "normal",
        "summary": summary,
        "source": "locked_clean_fixed_test_anchor",
        "anchor": anchor_check,
        "fingerprints": fingerprints,
        "training_started": False,
        "e2_e3_e4_started": False,
    }
    write_json_atomic(path, payload)
    return payload


def evaluate_condition(
    *,
    model: Any,
    dataset: dict[str, Any],
    model_spec: dict[str, Any],
    condition: str,
    transform: Any,
    condition_parameters: dict[str, Any],
    images: list[Path],
    evaluator: Any,
    prediction: dict[str, Any],
    fingerprints: dict[str, Any],
    output_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    detections, ground_truths, counts = robust.predict_dataset(
        model=model,
        image_paths=images,
        transform=transform,
        evaluator=evaluator,
        prediction=prediction,
    )
    metrics = robust.evaluate_metrics(
        evaluator,
        detections,
        ground_truths,
        list(range(int(dataset["classes"]))),
        int(prediction["evaluation_max_det"]),
    )
    summary = validate_metric_row(robust.metric_summary(metrics), f"{dataset['key']} {model_spec['key']} {condition}")
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        "completed_utc": utc_now(),
        "evaluation_only": True,
        "dataset": dataset["key"],
        "dataset_protocol": dataset["protocol"],
        "fixed_split": dataset["split"],
        "fixed_images": len(images),
        "experiment": model_spec["family"],
        "seed": int(model_spec["seed"]),
        "condition": condition,
        "condition_parameters": condition_parameters,
        "summary": summary,
        "full_metrics": metrics,
        "prediction_counts": counts,
        "identity_source_equivalence": identity,
        "fingerprints": fingerprints,
        "training_started": False,
        "e2_e3_e4_started": False,
    }
    write_json_atomic(output_path, payload)
    return payload


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def summarize_dataset(dataset: dict[str, Any], reports: dict[str, dict[int, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for family in FAMILIES:
        aggregates[family] = {}
        for condition in CONDITIONS:
            aggregates[family][condition] = {
                metric: aggregate(
                    [float(reports[family][seed][condition]["summary"][metric]) for seed in SEEDS]
                )
                for metric in METRICS
            }

    paired_clean: list[dict[str, Any]] = []
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            e0 = float(reports["E0"][seed]["normal"]["summary"][metric])
            e1 = float(reports["E1"][seed]["normal"]["summary"][metric])
            row[f"{metric}_E0"] = e0
            row[f"{metric}_E1"] = e1
            row[f"{metric}_delta_E1_minus_E0"] = e1 - e0
        paired_clean.append(row)
    paired_clean_summary = {
        metric: {
            **aggregate([float(row[f"{metric}_delta_E1_minus_E0"]) for row in paired_clean]),
            "positive_seed_count": sum(
                float(row[f"{metric}_delta_E1_minus_E0"]) > 0.0 for row in paired_clean
            ),
        }
        for metric in METRICS
    }

    robustness_rows: dict[str, list[dict[str, Any]]] = {}
    robustness_summary: dict[str, Any] = {}
    for condition in ("low_light_strong", "glare_strong"):
        family_name = condition.removesuffix("_strong")
        rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            row = {"seed": seed}
            for metric in ("mAP50-95", "APsmall"):
                e0_normal = float(reports["E0"][seed]["normal"]["summary"][metric])
                e0_strong = float(reports["E0"][seed][condition]["summary"][metric])
                e1_normal = float(reports["E1"][seed]["normal"]["summary"][metric])
                e1_strong = float(reports["E1"][seed][condition]["summary"][metric])
                e0_drop = e0_normal - e0_strong
                e1_drop = e1_normal - e1_strong
                row[f"{metric}_E0_drop"] = e0_drop
                row[f"{metric}_E1_drop"] = e1_drop
                row[f"{metric}_E1_drop_reduction"] = e0_drop - e1_drop
            rows.append(row)
        robustness_rows[family_name] = rows
        robustness_summary[family_name] = {
            metric: {
                "E0_drop": aggregate([float(row[f"{metric}_E0_drop"]) for row in rows]),
                "E1_drop": aggregate([float(row[f"{metric}_E1_drop"]) for row in rows]),
                "E1_drop_reduction": {
                    **aggregate([float(row[f"{metric}_E1_drop_reduction"]) for row in rows]),
                    "positive_seed_count": sum(
                        float(row[f"{metric}_E1_drop_reduction"]) > 0.0 for row in rows
                    ),
                    "interpretation": "positive means E1 drops less than matched E0",
                },
            }
            for metric in ("mAP50-95", "APsmall")
        }

    return {
        "dataset": dataset["key"],
        "dataset_protocol": dataset["protocol"],
        "fixed_split": dataset["split"],
        "fixed_images": int(dataset["images"]),
        "per_seed_reports": reports,
        "aggregate": aggregates,
        "paired_clean_E1_minus_E0_per_seed": paired_clean,
        "paired_clean_delta_aggregate": paired_clean_summary,
        "strong_degradation_drop_per_seed": robustness_rows,
        "strong_degradation_drop_aggregate": robustness_summary,
    }


def table_row(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate_block = summary["aggregate"]
    paired = summary["paired_clean_delta_aggregate"]
    robustness_block = summary["strong_degradation_drop_aggregate"]
    return {
        "dataset": summary["dataset"],
        "fixed_split": summary["fixed_split"],
        "fixed_images": summary["fixed_images"],
        "E0_mAP50-95_mean": aggregate_block["E0"]["normal"]["mAP50-95"]["mean"],
        "E0_mAP50-95_SD": aggregate_block["E0"]["normal"]["mAP50-95"]["sample_standard_deviation"],
        "E1_mAP50-95_mean": aggregate_block["E1"]["normal"]["mAP50-95"]["mean"],
        "E1_mAP50-95_SD": aggregate_block["E1"]["normal"]["mAP50-95"]["sample_standard_deviation"],
        "paired_mAP50-95_delta_pp": paired["mAP50-95"]["mean"] * 100.0,
        "paired_mAP50-95_positive_seeds": paired["mAP50-95"]["positive_seed_count"],
        "E0_APsmall_mean": aggregate_block["E0"]["normal"]["APsmall"]["mean"],
        "E0_APsmall_SD": aggregate_block["E0"]["normal"]["APsmall"]["sample_standard_deviation"],
        "E1_APsmall_mean": aggregate_block["E1"]["normal"]["APsmall"]["mean"],
        "E1_APsmall_SD": aggregate_block["E1"]["normal"]["APsmall"]["sample_standard_deviation"],
        "paired_APsmall_delta_pp": paired["APsmall"]["mean"] * 100.0,
        "paired_APsmall_positive_seeds": paired["APsmall"]["positive_seed_count"],
        "strong_low_light_E0_mAP50-95_drop_pp": robustness_block["low_light"]["mAP50-95"]["E0_drop"]["mean"] * 100.0,
        "strong_low_light_E1_mAP50-95_drop_pp": robustness_block["low_light"]["mAP50-95"]["E1_drop"]["mean"] * 100.0,
        "strong_low_light_E1_drop_reduction_pp": robustness_block["low_light"]["mAP50-95"]["E1_drop_reduction"]["mean"] * 100.0,
        "strong_low_light_E1_slower_drop_seeds": robustness_block["low_light"]["mAP50-95"]["E1_drop_reduction"]["positive_seed_count"],
        "strong_glare_E0_mAP50-95_drop_pp": robustness_block["glare"]["mAP50-95"]["E0_drop"]["mean"] * 100.0,
        "strong_glare_E1_mAP50-95_drop_pp": robustness_block["glare"]["mAP50-95"]["E1_drop"]["mean"] * 100.0,
        "strong_glare_E1_drop_reduction_pp": robustness_block["glare"]["mAP50-95"]["E1_drop_reduction"]["mean"] * 100.0,
        "strong_glare_E1_slower_drop_seeds": robustness_block["glare"]["mAP50-95"]["E1_drop_reduction"]["positive_seed_count"],
    }


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def update_state(path: Path, **updates: Any) -> None:
    payload = load_json(path) if path.is_file() else {"schema": STATE_SCHEMA}
    payload.update(updates, last_progress_utc=utc_now())
    write_json_atomic(path, payload)


def run(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise RuntimeError("Unexpected cross-dataset evaluation config")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise RuntimeError("Cross-dataset evaluation invariants are not all locked true")
    assert_not_stopped(config)
    release = verify_release(config_path)
    state_path = Path(config["output"]["state"])
    events_path = Path(config["output"]["events"])
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    training_checks = verify_training_states(config)
    model_checks = verify_models(config)

    assets = config["assets"]
    for key, path_key, hash_key in (
        ("base_evaluator", "base_evaluator", "base_evaluator_sha256"),
        ("robust_helper", "robust_helper", "robust_helper_sha256"),
        ("mdg_config", "source_config", "source_config_sha256"),
    ):
        source = assets if key != "mdg_config" else config["controlled_degradation"]
        path = Path(source[path_key])
        observed = sha256(path) if path.is_file() else "missing"
        if observed != source[hash_key]:
            raise RuntimeError(f"Locked asset changed for {key}: {observed}")

    total_tasks = sum(len(dataset["models"]) * len(CONDITIONS) for dataset in config["datasets"])
    completed_task_keys: list[str] = []
    update_state(
        state_path,
        status="running",
        started_utc=utc_now(),
        config_sha256=sha256(config_path),
        release_manifest_sha256=release["sha256"],
        current_dataset=None,
        current_model=None,
        current_condition=None,
        completed_tasks=0,
        total_tasks=total_tasks,
        completed_task_keys=[],
        training_started=False,
        e2_e3_e4_started=False,
    )
    append_event(events_path, "evaluation_started", total_tasks=total_tasks)

    evaluator = robust.load_evaluator(Path(assets["base_evaluator"]))
    evaluator_self_test = evaluator.synthetic_self_test()
    robust.import_checkpoint_modules()
    prediction = prediction_parameters(config)
    dataset_summaries: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "release": release,
        "training_states": training_checks,
        "models": model_checks,
        "evaluator_self_test": evaluator_self_test,
        "datasets": {},
    }

    for dataset in config["datasets"]:
        assert_not_stopped(config)
        images, _ = resolve_images(dataset)
        conditions, condition_parameters, transform_test = condition_bundle(config, dataset)
        clean_anchors, anchor_checks = load_clean_anchors(dataset, prediction)
        audit["datasets"][dataset["key"]] = {
            "yaml": {"path": dataset["yaml"], "sha256": dataset["yaml_sha256"]},
            "manifest": {"path": dataset["manifest"], "sha256": dataset["manifest_sha256"]},
            "images": len(images),
            "transform_self_test": transform_test,
            "clean_anchors": anchor_checks,
        }
        reports: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
            family: {seed: {} for seed in SEEDS} for family in FAMILIES
        }
        dataset_dir = output_root / dataset["key"]
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for model_spec in dataset["models"]:
            family = str(model_spec["family"])
            seed = int(model_spec["seed"])
            model_object: Any | None = None
            identity: dict[str, Any] | None = None
            try:
                for condition in CONDITIONS:
                    assert_not_stopped(config)
                    task_key = f"{dataset['key']}:{model_spec['key']}:{condition}"
                    output_path = dataset_dir / f"{model_spec['key']}__{condition}.json"
                    fingerprints = report_fingerprints(
                        config_path=config_path,
                        release=release,
                        config=config,
                        dataset=dataset,
                        model=model_spec,
                        condition=condition,
                        condition_parameters=condition_parameters[condition],
                    )
                    update_state(
                        state_path,
                        status="running",
                        current_dataset=dataset["key"],
                        current_model=model_spec["key"],
                        current_condition=condition,
                        completed_tasks=len(completed_task_keys),
                        total_tasks=total_tasks,
                        completed_task_keys=completed_task_keys,
                    )
                    report = cached_report(output_path, fingerprints)
                    if report is None and condition == "normal" and clean_anchors:
                        report = write_anchor_report(
                            output_path,
                            dataset=dataset,
                            model=model_spec,
                            summary=clean_anchors[family][seed],
                            fingerprints=fingerprints,
                            anchor_check=anchor_checks[family],
                        )
                    if report is None:
                        if model_object is None:
                            model_object = YOLO(model_spec["checkpoint"])
                        if identity is None:
                            identity = robust.identity_source_equivalence(model_object, images, prediction)
                        report = evaluate_condition(
                            model=model_object,
                            dataset=dataset,
                            model_spec=model_spec,
                            condition=condition,
                            transform=conditions[condition],
                            condition_parameters=condition_parameters[condition],
                            images=images,
                            evaluator=evaluator,
                            prediction=prediction,
                            fingerprints=fingerprints,
                            output_path=output_path,
                            identity=identity,
                        )
                    reports[family][seed][condition] = report
                    if task_key not in completed_task_keys:
                        completed_task_keys.append(task_key)
                    update_state(
                        state_path,
                        status="running",
                        current_dataset=dataset["key"],
                        current_model=model_spec["key"],
                        current_condition=None,
                        completed_tasks=len(completed_task_keys),
                        total_tasks=total_tasks,
                        completed_task_keys=completed_task_keys,
                    )
                    append_event(
                        events_path,
                        "condition_completed",
                        task_key=task_key,
                        completed_tasks=len(completed_task_keys),
                        total_tasks=total_tasks,
                        summary=report["summary"],
                    )
            finally:
                robust.release_gpu(model_object)

        dataset_summary = summarize_dataset(dataset, reports)
        write_json_atomic(dataset_dir / "dataset_summary.json", dataset_summary)
        dataset_summaries.append(dataset_summary)
        append_event(
            events_path,
            "dataset_completed",
            dataset=dataset["key"],
            completed_tasks=len(completed_task_keys),
            total_tasks=total_tasks,
        )

    table_rows = [table_row(summary) for summary in dataset_summaries]
    table_path = Path(config["output"]["table_csv"])
    summary_path = Path(config["output"]["summary"])
    write_table_csv(table_path, table_rows)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_utc": utc_now(),
        "evaluation_only": True,
        "training_started": False,
        "e2_e3_e4_started": False,
        "cohort": [0, 1, 2],
        "same_evaluator_and_prediction_parameters_for_all_datasets_models_and_seeds": True,
        "prediction_parameters": prediction,
        "controlled_conditions": list(CONDITIONS),
        "dataset_summaries": dataset_summaries,
        "table_rows": table_rows,
        "audit": audit,
        "interpretation_guard": "Three paired seeds provide cross-dataset mechanism evidence; they do not establish broad distributional robustness.",
    }
    write_json_atomic(summary_path, payload)
    update_state(
        state_path,
        status="completed",
        completed_utc=utc_now(),
        current_dataset=None,
        current_model=None,
        current_condition=None,
        completed_tasks=total_tasks,
        total_tasks=total_tasks,
        completed_task_keys=completed_task_keys,
        summary_path=str(summary_path),
        summary_sha256=sha256(summary_path),
        table_csv_path=str(table_path),
        table_csv_sha256=sha256(table_path),
        training_started=False,
        e2_e3_e4_started=False,
    )
    append_event(events_path, "evaluation_completed", summary_path=str(summary_path), table_csv_path=str(table_path))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate matched E0/E1 seeds across three locked datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("RING3_SETUP_ROOT", "/root/autodl-tmp/ring3_setup/autodl"))
        / "cross_dataset_e0_e1_multiseed_eval_config.json",
    )
    args = parser.parse_args()
    try:
        payload = run(args.config)
        print(json.dumps({"status": payload["status"], "table_rows": payload["table_rows"]}, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        try:
            config = load_json(args.config)
            state_path = Path(config["output"]["state"])
            events_path = Path(config["output"]["events"])
            update_state(
                state_path,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                training_started=False,
                e2_e3_e4_started=False,
            )
            append_event(events_path, "evaluation_failed", error_type=type(exc).__name__, error=str(exc))
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
