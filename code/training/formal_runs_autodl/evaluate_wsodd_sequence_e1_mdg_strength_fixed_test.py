#!/usr/bin/env python3
"""Evaluate the locked WSODD E1 MDG-dose ablation on the fixed test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


CONFIG_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_ablation_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_ablation_release.v1"
REPORT_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_fixed_test_report.v1"
SUMMARY_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_fixed_test_summary.v1"
STATE_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_ablation_state.v1"
EXPECTED_RUN_KEYS = ["e1_lite", "e1_strong"]
EXPECTED_BASELINE_SUMMARY_SHA256 = "5620ef7572ed723f7977e3d5562d9b9e22886a76686eae36f344b9448811c0f4"
EXPECTED_BASELINE = {
    "E0": {
        "mAP50": 0.4500705571954184,
        "mAP50-95": 0.22872770220034228,
        "APsmall": 0.11991360563516992,
        "APmedium": 0.18204148635662917,
        "APlarge": 0.3363743228114737,
        "ground_truth_count": 6297,
        "small_ground_truth_count": 1126,
    },
    "E1": {
        "mAP50": 0.4458277258702253,
        "mAP50-95": 0.22699865162299138,
        "APsmall": 0.14533887083614097,
        "APmedium": 0.17864654436550956,
        "APlarge": 0.33342423081094236,
        "ground_truth_count": 6297,
        "small_ground_truth_count": 1126,
    },
}


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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def finite_metric_rows(path: Path) -> list[dict[str, str]]:
    required = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    finite: list[dict[str, str]] = []
    for row in rows:
        try:
            if all(math.isfinite(float(row[key])) for key in required):
                finite.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    if not finite:
        raise RuntimeError(f"No finite metric rows: {path}")
    return finite


def metric_row(row: dict[str, str]) -> dict[str, float | int]:
    return {
        "epoch": int(float(row["epoch"])),
        "Precision": float(row["metrics/precision(B)"]),
        "Recall": float(row["metrics/recall(B)"]),
        "mAP50": float(row["metrics/mAP50(B)"]),
        "mAP50-95": float(row["metrics/mAP50-95(B)"]),
    }


def training_metrics(run_dir: Path) -> dict[str, Any]:
    rows = finite_metric_rows(run_dir / "results.csv")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {"best": metric_row(best), "final": metric_row(rows[-1])}


def verify_release_manifest(config_path: Path) -> dict[str, Any]:
    manifest_path = config_path.parent / "wsodd_sequence_e1_mdg_strength_ablation_release_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "locked":
        raise RuntimeError("Unexpected E1 MDG strength release manifest")
    verified: list[dict[str, str]] = []
    for item in manifest.get("files", []):
        path = config_path.parent / str(item["path"])
        observed = sha256(path) if path.is_file() else "missing"
        if observed != item.get("sha256"):
            raise RuntimeError(f"Release hash mismatch for {item.get('path')}: {observed}")
        verified.append({"path": str(path), "sha256": observed})
    if len(verified) < 20:
        raise RuntimeError(f"Release manifest is incomplete: {len(verified)} files")
    return {"path": str(manifest_path), "sha256": sha256(manifest_path), "verified_files": verified}


def resolve_fixed_test_images(config: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    dataset = config["dataset"]
    yaml_path = Path(dataset["yaml"])
    manifest_path = Path(dataset["test_manifest"])
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    test_source = Path(data["test"])
    test_dir = test_source if test_source.is_absolute() else root / test_source
    names = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != int(dataset["test_images"]) or len(names) != len(set(names)):
        raise RuntimeError(f"Fixed test manifest count or uniqueness changed: {len(names)}")
    if any(Path(name).name != name for name in names):
        raise RuntimeError("Fixed test manifest must contain audited basenames only")
    images = [test_dir / name for name in names]
    if not test_dir.is_dir() or not all(path.is_file() for path in images):
        missing = [str(path) for path in images if not path.is_file()][:5]
        raise RuntimeError(f"Fixed test images are missing: {missing}")
    if int(data.get("nc", -1)) != int(dataset["classes"]):
        raise RuntimeError(f"WSODD class count changed: {data.get('nc')}")
    return images, data


def verify_baseline(config: dict[str, Any], prediction: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    anchors = config["reused_anchors"]
    path = Path(anchors["fixed_test_summary"])
    observed_hash = sha256(path) if path.is_file() else "missing"
    if observed_hash != EXPECTED_BASELINE_SUMMARY_SHA256 or observed_hash != anchors["fixed_test_summary_sha256"]:
        raise RuntimeError(f"Locked E0/E1-Mid fixed-test summary changed: {observed_hash}")
    summary = load_json(path)
    if summary.get("status") != "completed" or summary.get("fixed_split") != "test" or summary.get("e4_started") is not False:
        raise RuntimeError("Locked baseline fixed-test summary is not valid")
    for key, expected in prediction.items():
        if key == "device":
            continue
        if summary.get("prediction_parameters", {}).get(key) != expected:
            raise RuntimeError(f"Prediction parameter differs from locked baseline for {key}")
    rows = {str(row["experiment"]): row for row in summary.get("area_rows", [])}
    for experiment, expected in EXPECTED_BASELINE.items():
        row = rows.get(experiment)
        if row is None or any(not close(row.get(key), value) for key, value in expected.items()):
            raise RuntimeError(f"Locked baseline row changed for {experiment}: {row}")
    if summary.get("fingerprints", {}).get("E0", {}).get("best_weights_sha256") != anchors["E0"]["best_weights_sha256"]:
        raise RuntimeError("E0 checkpoint fingerprint differs from the reused anchor")
    if summary.get("fingerprints", {}).get("E1", {}).get("best_weights_sha256") != anchors["E1-Mid"]["best_weights_sha256"]:
        raise RuntimeError("E1-Mid checkpoint fingerprint differs from the reused anchor")
    return {"E0": rows["E0"], "E1-Mid": rows["E1"]}, {
        "path": str(path),
        "sha256": observed_hash,
        "source_schema": summary.get("schema"),
    }


def verify_completed_runs(
    config: dict[str, Any], config_path: Path, release: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Any]]:
    data_root = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
    output_root = data_root / "formal_wsodd_sequence_e1_mdg_strength_ablation"
    state_path = data_root / "formal_state" / "wsodd_sequence_e1_mdg_strength_ablation_state.json"
    summary_path = data_root / "formal_state" / "wsodd_sequence_e1_mdg_strength_ablation_summary.json"
    if not state_path.is_file() or not summary_path.is_file():
        raise RuntimeError("E1 MDG strength training state/summary is missing")
    state = load_json(state_path)
    training_summary = load_json(summary_path)
    expected_config_hash = sha256(config_path)
    anchors = {
        "schema": STATE_SCHEMA,
        "status": "completed",
        "config_sha256": expected_config_hash,
        "release_manifest_sha256": release["sha256"],
        "completed_runs": EXPECTED_RUN_KEYS,
        "failed_runs": [],
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    mismatches = {key: {"expected": value, "observed": state.get(key)} for key, value in anchors.items() if state.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Training queue is not a valid completed queue: {mismatches}")
    runs: dict[str, dict[str, Any]] = {}
    for variant in config["run_order"]:
        key = str(variant["key"])
        run_dir = output_root / str(variant["name"])
        files = {
            "best_checkpoint": run_dir / "weights" / "best.pt",
            "results_csv": run_dir / "results.csv",
            "args": run_dir / "args.yaml",
            "runtime_evidence": run_dir / "wsodd_sequence_e1_mdg_strength_runtime_evidence.json",
            "run_metadata": run_dir / "wsodd_sequence_e1_mdg_strength_run_metadata.json",
            "completion_evidence": run_dir / "wsodd_sequence_e1_mdg_strength_completion_evidence.json",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Completed run artifacts are missing for {key}: {missing}")
        entry = training_summary.get("metrics", {}).get(key, {})
        if entry.get("status") != "completed" or Path(entry.get("run_dir", "")).resolve() != run_dir.resolve():
            raise RuntimeError(f"Training summary does not validate {key}")
        metadata = load_json(files["run_metadata"])
        runtime = load_json(files["runtime_evidence"])
        expected_metadata = {
            "variant_key": key,
            "variant": variant,
            "pretrained": True,
            "pretrained_source_sha256": config["weights"]["sha256"],
            "optimizer": "SGD",
            "config_sha256": expected_config_hash,
            "release_manifest_sha256": release["sha256"],
            "cross_run_checkpoint_loaded": False,
            "consistency_loss_enabled": False,
            "cbam_enabled": False,
            "focal_eiou_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "e4_started": False,
        }
        metadata_mismatches = {
            name: {"expected": value, "observed": metadata.get(name)}
            for name, value in expected_metadata.items()
            if metadata.get(name) != value
        }
        runtime_required = {
            "variant": variant,
            "mdg_mode": "single_view",
            "dual_view_enabled": False,
            "degraded_small_weighting_enabled": False,
            "cbam_enabled": False,
            "focal_eiou_enabled": False,
            "no_consistency_loss": True,
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
        }
        runtime_mismatches = {
            name: {"expected": value, "observed": runtime.get(name)}
            for name, value in runtime_required.items()
            if runtime.get(name) != value
        }
        mdg_images = int(runtime.get("mdg_counts_current_batch", {}).get("images_total", -1))
        target_images = int(runtime.get("target_counts_current_batch", {}).get("images_total", -2))
        if mdg_images <= 0 or mdg_images != target_images:
            runtime_mismatches["image_accounting"] = {"mdg": mdg_images, "targets": target_images}
        if metadata_mismatches or runtime_mismatches:
            raise RuntimeError(f"Completed-run provenance mismatch for {key}: metadata={metadata_mismatches}, runtime={runtime_mismatches}")
        args = yaml.safe_load(files["args"].read_text(encoding="utf-8"))
        locked_args = {**config["training"], "model": config["weights"]["path"], "data": config["dataset"]["yaml"]}
        ignored = {"device"}
        args_mismatches = {
            name: {"expected": value, "observed": args.get(name)}
            for name, value in locked_args.items()
            if name not in ignored and args.get(name) != value
        }
        if str(args.get("device")) != str(config["training"]["device"]):
            args_mismatches["device"] = {"expected": config["training"]["device"], "observed": args.get("device")}
        if args_mismatches:
            raise RuntimeError(f"Locked args mismatch for {key}: {args_mismatches}")
        runs[key] = {
            "variant": variant,
            "run_dir": str(run_dir),
            "training_validation": training_metrics(run_dir),
            "fingerprints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()},
        }
    queue_fingerprints = {
        "state": {"path": str(state_path), "sha256": sha256(state_path)},
        "training_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
    }
    return runs, state_path, queue_fingerprints


def prediction_parameters(config: dict[str, Any]) -> dict[str, Any]:
    values = config["post_training_validation"]
    return {
        "imgsz": int(values["imgsz"]),
        "batch": int(values["batch"]),
        "chunk_size": int(values["chunk_size"]),
        "confidence_floor": float(values["confidence_floor"]),
        "nms_iou": float(values["nms_iou"]),
        "prediction_max_det": int(values["prediction_max_det"]),
        "evaluation_max_det": int(values["evaluation_max_det"]),
        "device": 0,
    }


def evaluate_variant(
    *,
    variant: dict[str, Any],
    run: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    release: dict[str, Any],
    evaluator: Any,
    robust: Any,
    images: list[Path],
    prediction: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    key = str(variant["key"])
    output_path = output_root / f"{key}_fixed_test.json"
    checkpoint = Path(run["run_dir"]) / "weights" / "best.pt"
    fingerprints = {
        "config_sha256": sha256(config_path),
        "release_manifest_sha256": release["sha256"],
        "dataset_yaml_sha256": config["dataset"]["yaml_sha256"],
        "test_manifest_sha256": config["dataset"]["test_manifest_sha256"],
        "base_evaluator_sha256": config["post_training_validation"]["base_evaluator_sha256"],
        "robust_helper_sha256": config["post_training_validation"]["robust_helper_sha256"],
        "best_checkpoint_sha256": sha256(checkpoint),
    }
    if output_path.is_file():
        cached = load_json(output_path)
        if cached.get("status") == "completed" and cached.get("fingerprints") == fingerprints:
            return cached
    model = YOLO(str(checkpoint))
    try:
        identity = robust.identity_source_equivalence(model, images, prediction)
        detections, ground_truths, counts = robust.predict_dataset(
            model=model,
            image_paths=images,
            transform=lambda image, _path: image.copy(),
            evaluator=evaluator,
            prediction=prediction,
        )
        metrics = robust.evaluate_metrics(
            evaluator,
            detections,
            ground_truths,
            list(range(int(config["dataset"]["classes"]))),
            int(prediction["evaluation_max_det"]),
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "completed",
            "completed_utc": utc_now(),
            "experiment": variant["label"],
            "mdg_probability_and_strength_scale": variant["mdg_scale"],
            "dataset_protocol": config["dataset"]["protocol"],
            "fixed_split": "test",
            "fixed_test_images": len(images),
            "same_evaluator_and_prediction_parameters_as_E0_and_E1_Mid": True,
            "fixed_test": robust.metric_summary(metrics),
            "full_metrics": metrics,
            "training_validation": run["training_validation"],
            "prediction_parameters": prediction,
            "prediction_counts": counts,
            "identity_source_equivalence": identity,
            "fingerprints": fingerprints,
            "single_seed_mechanism_evidence_only": True,
            "e2_e3_e4_started": False,
            "e4_started": False,
        }
        write_json_atomic(output_path, report)
        return report
    finally:
        robust.release_gpu(model)


def summarize(
    *,
    config: dict[str, Any],
    config_path: Path,
    release: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    baseline_fingerprint: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    queue_fingerprints: dict[str, Any],
    evaluator_self_test: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {"experiment": "E0", "mdg_scale": 0.0, **{key: baseline["E0"][key] for key in EXPECTED_BASELINE["E0"]}},
        {"experiment": "E1-Lite", "mdg_scale": 0.5, **reports["e1_lite"]["fixed_test"]},
        {"experiment": "E1-Mid", "mdg_scale": 1.0, **{key: baseline["E1-Mid"][key] for key in EXPECTED_BASELINE["E1"]}},
        {"experiment": "E1-Strong", "mdg_scale": 1.5, **reports["e1_strong"]["fixed_test"]},
    ]
    e0 = rows[0]
    gate = config["balance_gate"]
    candidates: list[dict[str, Any]] = []
    for row in rows[1:]:
        delta_map = float(row["mAP50-95"]) - float(e0["mAP50-95"])
        delta_small = float(row["APsmall"]) - float(e0["APsmall"])
        passed = (
            delta_small >= float(gate["minimum_APsmall_gain_vs_E0"])
            and delta_map >= float(gate["minimum_mAP50_95_delta_vs_E0"])
        )
        candidates.append(
            {
                "experiment": row["experiment"],
                "mdg_scale": row["mdg_scale"],
                "mAP50-95_delta_vs_E0": delta_map,
                "APsmall_delta_vs_E0": delta_small,
                "passed": passed,
            }
        )
    passers = [candidate for candidate in candidates if candidate["passed"]]
    passers.sort(
        key=lambda candidate: (
            float(next(row["mAP50-95"] for row in rows if row["experiment"] == candidate["experiment"])),
            float(next(row["APsmall"] for row in rows if row["experiment"] == candidate["experiment"])),
        ),
        reverse=True,
    )
    training_rows = [
        {"experiment": "E0", **training_metrics(Path(config["reused_anchors"]["E0"]["run_dir"]))},
        {"experiment": "E1-Lite", **runs["e1_lite"]["training_validation"]},
        {"experiment": "E1-Mid", **training_metrics(Path(config["reused_anchors"]["E1-Mid"]["run_dir"]))},
        {"experiment": "E1-Strong", **runs["e1_strong"]["training_validation"]},
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_utc": utc_now(),
        "purpose": config["purpose"],
        "dataset_protocol": config["dataset"]["protocol"],
        "fixed_split": "test",
        "fixed_test_images": int(config["dataset"]["test_images"]),
        "same_evaluator_and_prediction_parameters_for_all_rows": True,
        "prediction_parameters": prediction,
        "fixed_test_rows": rows,
        "training_validation_rows": training_rows,
        "balance_gate": gate,
        "balance_candidates": candidates,
        "selected_balance": passers[0] if passers else None,
        "selection_status": "passed" if passers else "no_candidate_passed",
        "evaluator_self_test": evaluator_self_test,
        "fingerprints": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "release_manifest": release,
            "baseline_summary": baseline_fingerprint,
            "new_queue": queue_fingerprints,
            "new_reports": {
                key: {
                    "path": str(Path(config["post_training_validation"]["output_root"]) / f"{key}_fixed_test.json"),
                    "sha256": sha256(Path(config["post_training_validation"]["output_root"]) / f"{key}_fixed_test.json"),
                }
                for key in EXPECTED_RUN_KEYS
            },
        },
        "single_seed_mechanism_evidence_only": True,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }


def update_state(state_path: Path, **updates: Any) -> None:
    state = load_json(state_path)
    state.update(updates, last_progress_utc=utc_now())
    write_json_atomic(state_path, state)


def run(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise RuntimeError("Unexpected E1 MDG strength config")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise RuntimeError("E1 MDG strength invariants are not all locked true")
    release = verify_release_manifest(config_path)
    static_assets = {
        "dataset_yaml": (Path(config["dataset"]["yaml"]), config["dataset"]["yaml_sha256"]),
        "test_manifest": (Path(config["dataset"]["test_manifest"]), config["dataset"]["test_manifest_sha256"]),
        "base_evaluator": (
            Path(config["post_training_validation"]["base_evaluator"]),
            config["post_training_validation"]["base_evaluator_sha256"],
        ),
        "robust_helper": (
            Path(config["post_training_validation"]["robust_helper"]),
            config["post_training_validation"]["robust_helper_sha256"],
        ),
    }
    for name, (path, expected) in static_assets.items():
        observed = sha256(path) if path.is_file() else "missing"
        if observed != expected:
            raise RuntimeError(f"Locked evaluator asset changed for {name}: {observed}")
    prediction = prediction_parameters(config)
    baseline, baseline_fingerprint = verify_baseline(config, prediction)
    runs, state_path, queue_fingerprints = verify_completed_runs(config, config_path, release)
    update_state(state_path, post_training_validation_status="running", validation_current_run=None)
    importlib.import_module("cbam_focal_formal_models")
    importlib.import_module("pretrained_sgd_mainline_models")
    evaluator = load_module("ring3_locked_wsodd_area_evaluator", static_assets["base_evaluator"][0])
    robust = load_module("ring3_locked_wsodd_robust_helper", static_assets["robust_helper"][0])
    evaluator_self_test = evaluator.synthetic_self_test()
    images, _data = resolve_fixed_test_images(config)
    output_root = Path(config["post_training_validation"]["output_root"])
    reports: dict[str, dict[str, Any]] = {}
    for variant in config["run_order"]:
        key = str(variant["key"])
        update_state(state_path, post_training_validation_status="running", validation_current_run=key)
        reports[key] = evaluate_variant(
            variant=variant,
            run=runs[key],
            config=config,
            config_path=config_path,
            release=release,
            evaluator=evaluator,
            robust=robust,
            images=images,
            prediction=prediction,
            output_root=output_root,
        )
    payload = summarize(
        config=config,
        config_path=config_path,
        release=release,
        baseline=baseline,
        baseline_fingerprint=baseline_fingerprint,
        reports=reports,
        runs=runs,
        queue_fingerprints=queue_fingerprints,
        evaluator_self_test=evaluator_self_test,
        prediction=prediction,
    )
    summary_path = Path(config["post_training_validation"]["summary_path"])
    write_json_atomic(summary_path, payload)
    update_state(
        state_path,
        post_training_validation_status="completed",
        validation_current_run=None,
        fixed_test_metrics_path=str(summary_path),
        fixed_test_metrics_sha256=sha256(summary_path),
        selected_balance=payload["selected_balance"],
        e2_e3_e4_started=False,
        e4_started=False,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WSODD E1 MDG dose ablation on the locked fixed test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("RING3_SETUP_ROOT", "/root/autodl-tmp/ring3_setup/autodl"))
        / "wsodd_sequence_e1_mdg_strength_ablation_config.json",
    )
    args = parser.parse_args()
    state_path = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp")) / "formal_state" / "wsodd_sequence_e1_mdg_strength_ablation_state.json"
    try:
        payload = run(args.config)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        if state_path.is_file():
            update_state(
                state_path,
                post_training_validation_status="failed",
                validation_error_type=type(exc).__name__,
                validation_error=str(exc),
                validation_traceback=traceback.format_exc(),
                e2_e3_e4_started=False,
                e4_started=False,
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
