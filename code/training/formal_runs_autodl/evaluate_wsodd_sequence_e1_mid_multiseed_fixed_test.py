#!/usr/bin/env python3
"""Evaluate WSODD fixed-split E1-Mid seeds 1/2 and aggregate seeds 0/1/2."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import traceback
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

import evaluate_wsodd_sequence_e1_mdg_strength_fixed_test as common


CONFIG_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_release.v1"
REPORT_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_fixed_test_report.v1"
SUMMARY_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_fixed_test_summary.v1"
STATE_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_state.v1"
COMPLETION_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_completion.v1"
EXPECTED_RUN_KEYS = ["e1_mid_seed1", "e1_mid_seed2"]
METRIC_KEYS = ["mAP50", "mAP50-95", "APsmall", "APmedium", "APlarge"]


def verify_release_manifest(config_path: Path) -> dict[str, Any]:
    manifest_path = config_path.parent / "wsodd_sequence_e1_mid_multiseed_release_manifest.json"
    manifest = common.load_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "locked":
        raise RuntimeError("Unexpected E1-Mid multiseed release manifest")
    verified: list[dict[str, str]] = []
    for item in manifest.get("files", []):
        path = config_path.parent / str(item["path"])
        observed = common.sha256(path) if path.is_file() else "missing"
        if observed != item.get("sha256"):
            raise RuntimeError(f"Release hash mismatch for {item.get('path')}: {observed}")
        verified.append({"path": str(path), "sha256": observed})
    if len(verified) < 20:
        raise RuntimeError(f"Release manifest is incomplete: {len(verified)} files")
    return {"path": str(manifest_path), "sha256": common.sha256(manifest_path), "verified_files": verified}


def verify_baseline(config: dict[str, Any], prediction: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    adapted = copy.deepcopy(config)
    adapted["reused_anchors"]["E1-Mid"] = adapted["reused_anchors"]["E1-Mid-seed0"]
    return common.verify_baseline(adapted, prediction)


def verify_completed_runs(
    config: dict[str, Any], config_path: Path, release: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Any]]:
    data_root = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
    output_root = data_root / "formal_wsodd_sequence_e1_mid_multiseed"
    state_path = data_root / "formal_state" / "wsodd_sequence_e1_mid_multiseed_state.json"
    summary_path = data_root / "formal_state" / "wsodd_sequence_e1_mid_multiseed_training_summary.json"
    if not state_path.is_file() or not summary_path.is_file():
        raise RuntimeError("E1-Mid multiseed training state/summary is missing")
    state = common.load_json(state_path)
    training_summary = common.load_json(summary_path)
    expected_config_hash = common.sha256(config_path)
    expected_state = {
        "schema": STATE_SCHEMA,
        "status": "completed",
        "config_sha256": expected_config_hash,
        "release_manifest_sha256": release["sha256"],
        "completed_runs": EXPECTED_RUN_KEYS,
        "failed_runs": [],
        "reused_seed0": True,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in expected_state.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Training queue is not a valid completed queue: {mismatches}")

    runs: dict[str, dict[str, Any]] = {}
    for variant in config["run_order"]:
        key = str(variant["key"])
        seed = int(variant["seed"])
        run_dir = output_root / str(variant["name"])
        files = {
            "best_checkpoint": run_dir / "weights" / "best.pt",
            "results_csv": run_dir / "results.csv",
            "args": run_dir / "args.yaml",
            "runtime_evidence": run_dir / "wsodd_sequence_e1_mid_multiseed_runtime_evidence.json",
            "run_metadata": run_dir / "wsodd_sequence_e1_mid_multiseed_run_metadata.json",
            "completion_evidence": run_dir / "wsodd_sequence_e1_mid_multiseed_completion_evidence.json",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Completed run artifacts are missing for {key}: {missing}")
        entry = training_summary.get("metrics", {}).get(key, {})
        if entry.get("status") != "completed" or int(entry.get("seed", -1)) != seed:
            raise RuntimeError(f"Training summary does not validate {key}")
        if Path(str(entry.get("run_dir", ""))).resolve() != run_dir.resolve():
            raise RuntimeError(f"Training summary run directory differs for {key}")

        metadata = common.load_json(files["run_metadata"])
        runtime = common.load_json(files["runtime_evidence"])
        completion = common.load_json(files["completion_evidence"])
        required_metadata = {
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
            for name, value in required_metadata.items()
            if metadata.get(name) != value
        }
        if metadata.get("training_kwargs", {}).get("seed") != seed:
            metadata_mismatches["training_kwargs.seed"] = {
                "expected": seed,
                "observed": metadata.get("training_kwargs", {}).get("seed"),
            }
        required_runtime = {
            "variant": variant,
            "pretrained": True,
            "pretrained_source_sha256": config["weights"]["sha256"],
            "optimizer": "SGD",
            "mdg_mode": "single_view",
            "dual_view_enabled": False,
            "degraded_small_weighting_enabled": False,
            "cbam_enabled": False,
            "focal_eiou_enabled": False,
            "no_consistency_loss": True,
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "cross_run_checkpoint_loaded": False,
        }
        runtime_mismatches = {
            name: {"expected": value, "observed": runtime.get(name)}
            for name, value in required_runtime.items()
            if runtime.get(name) != value
        }
        mdg_images = int(runtime.get("mdg_counts_current_batch", {}).get("images_total", -1))
        target_images = int(runtime.get("target_counts_current_batch", {}).get("images_total", -2))
        if mdg_images <= 0 or mdg_images != target_images:
            runtime_mismatches["image_accounting"] = {"mdg": mdg_images, "targets": target_images}
        required_completion = {
            "schema": COMPLETION_SCHEMA,
            "status": "passed",
            "variant_key": key,
            "normal_training_return_confirmed": True,
            "config_sha256": expected_config_hash,
            "release_manifest_sha256": release["sha256"],
        }
        completion_mismatches = {
            name: {"expected": value, "observed": completion.get(name)}
            for name, value in required_completion.items()
            if completion.get(name) != value
        }
        if metadata_mismatches or runtime_mismatches or completion_mismatches:
            raise RuntimeError(
                f"Completed-run provenance mismatch for {key}: metadata={metadata_mismatches}, "
                f"runtime={runtime_mismatches}, completion={completion_mismatches}"
            )

        args = yaml.safe_load(files["args"].read_text(encoding="utf-8"))
        locked_args = {**config["training"], "seed": seed, "model": config["weights"]["path"], "data": config["dataset"]["yaml"]}
        args_mismatches = {
            name: {"expected": value, "observed": args.get(name)}
            for name, value in locked_args.items()
            if name != "device" and args.get(name) != value
        }
        if str(args.get("device")) != str(config["training"]["device"]):
            args_mismatches["device"] = {"expected": config["training"]["device"], "observed": args.get("device")}
        if args_mismatches:
            raise RuntimeError(f"Locked args mismatch for {key}: {args_mismatches}")
        runs[key] = {
            "variant": variant,
            "run_dir": str(run_dir),
            "training_validation": common.training_metrics(run_dir),
            "fingerprints": {name: {"path": str(path), "sha256": common.sha256(path)} for name, path in files.items()},
        }
    return runs, state_path, {
        "state": {"path": str(state_path), "sha256": common.sha256(state_path)},
        "training_summary": {"path": str(summary_path), "sha256": common.sha256(summary_path)},
    }


def evaluate_variant(
    variant: dict[str, Any], run: dict[str, Any], config: dict[str, Any], config_path: Path,
    release: dict[str, Any], evaluator: Any, robust: Any, images: list[Path],
    prediction: dict[str, Any], output_root: Path,
) -> dict[str, Any]:
    key = str(variant["key"])
    output_path = output_root / f"{key}_fixed_test.json"
    checkpoint = Path(run["run_dir"]) / "weights" / "best.pt"
    fingerprints = {
        "config_sha256": common.sha256(config_path),
        "release_manifest_sha256": release["sha256"],
        "dataset_yaml_sha256": config["dataset"]["yaml_sha256"],
        "test_manifest_sha256": config["dataset"]["test_manifest_sha256"],
        "base_evaluator_sha256": config["post_training_validation"]["base_evaluator_sha256"],
        "robust_helper_sha256": config["post_training_validation"]["robust_helper_sha256"],
        "best_checkpoint_sha256": common.sha256(checkpoint),
    }
    if output_path.is_file():
        cached = common.load_json(output_path)
        if cached.get("status") == "completed" and cached.get("fingerprints") == fingerprints:
            return cached
    model = YOLO(str(checkpoint))
    try:
        identity = robust.identity_source_equivalence(model, images, prediction)
        detections, ground_truths, counts = robust.predict_dataset(
            model=model, image_paths=images, transform=lambda image, _path: image.copy(),
            evaluator=evaluator, prediction=prediction,
        )
        metrics = robust.evaluate_metrics(
            evaluator, detections, ground_truths, list(range(int(config["dataset"]["classes"]))),
            int(prediction["evaluation_max_det"]),
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "completed",
            "completed_utc": common.utc_now(),
            "experiment": variant["label"],
            "seed": int(variant["seed"]),
            "dataset_protocol": config["dataset"]["protocol"],
            "fixed_split": "test",
            "fixed_test_images": len(images),
            "same_evaluator_and_prediction_parameters_as_seed0": True,
            "fixed_test": robust.metric_summary(metrics),
            "full_metrics": metrics,
            "training_validation": run["training_validation"],
            "prediction_parameters": prediction,
            "prediction_counts": counts,
            "identity_source_equivalence": identity,
            "fingerprints": fingerprints,
            "e2_e3_e4_started": False,
            "e4_started": False,
        }
        common.write_json_atomic(output_path, report)
        return report
    finally:
        robust.release_gpu(model)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: {
            "mean": statistics.mean(float(row[key]) for row in rows),
            "sample_standard_deviation": statistics.stdev(float(row[key]) for row in rows),
        }
        for key in METRIC_KEYS
    }


def update_state(state_path: Path, **updates: Any) -> None:
    state = common.load_json(state_path)
    state.update(updates, last_progress_utc=common.utc_now())
    common.write_json_atomic(state_path, state)


def run(config_path: Path) -> dict[str, Any]:
    config = common.load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise RuntimeError("Unexpected E1-Mid multiseed config")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise RuntimeError("E1-Mid multiseed invariants are not all locked true")
    release = verify_release_manifest(config_path)
    static_assets = {
        "dataset_yaml": (Path(config["dataset"]["yaml"]), config["dataset"]["yaml_sha256"]),
        "test_manifest": (Path(config["dataset"]["test_manifest"]), config["dataset"]["test_manifest_sha256"]),
        "base_evaluator": (Path(config["post_training_validation"]["base_evaluator"]), config["post_training_validation"]["base_evaluator_sha256"]),
        "robust_helper": (Path(config["post_training_validation"]["robust_helper"]), config["post_training_validation"]["robust_helper_sha256"]),
    }
    for name, (path, expected) in static_assets.items():
        observed = common.sha256(path) if path.is_file() else "missing"
        if observed != expected:
            raise RuntimeError(f"Locked evaluator asset changed for {name}: {observed}")
    prediction = common.prediction_parameters(config)
    baseline, baseline_fingerprint = verify_baseline(config, prediction)
    runs, state_path, queue_fingerprints = verify_completed_runs(config, config_path, release)
    update_state(state_path, post_training_validation_status="running", validation_current_run=None)
    common.importlib.import_module("cbam_focal_formal_models")
    common.importlib.import_module("pretrained_sgd_mainline_models")
    evaluator = common.load_module("ring3_locked_wsodd_area_evaluator", static_assets["base_evaluator"][0])
    robust = common.load_module("ring3_locked_wsodd_robust_helper", static_assets["robust_helper"][0])
    evaluator_self_test = evaluator.synthetic_self_test()
    images, _ = common.resolve_fixed_test_images(config)
    output_root = Path(config["post_training_validation"]["output_root"])
    reports: dict[str, dict[str, Any]] = {}
    for variant in config["run_order"]:
        key = str(variant["key"])
        update_state(state_path, post_training_validation_status="running", validation_current_run=key)
        reports[key] = evaluate_variant(
            variant, runs[key], config, config_path, release, evaluator, robust, images, prediction, output_root
        )

    seed0 = {"seed": 0, **config["reused_anchors"]["E1-Mid-seed0"]["fixed_test"]}
    rows = [seed0] + [
        {"seed": int(variant["seed"]), **reports[str(variant["key"])]["fixed_test"]}
        for variant in config["run_order"]
    ]
    e0 = baseline["E0"]
    gate = config["replication_gate"]
    per_seed_gate = []
    for row in rows:
        delta_small = float(row["APsmall"]) - float(e0["APsmall"])
        delta_map = float(row["mAP50-95"]) - float(e0["mAP50-95"])
        per_seed_gate.append({
            "seed": row["seed"], "APsmall_delta_vs_E0_seed0": delta_small,
            "mAP50-95_delta_vs_E0_seed0": delta_map,
            "passed_descriptive_gate": delta_small >= float(gate["minimum_APsmall_gain_vs_E0"])
            and delta_map >= float(gate["minimum_mAP50_95_delta_vs_E0"]),
        })
    aggregated = aggregate(rows)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_utc": common.utc_now(),
        "purpose": config["purpose"],
        "dataset_protocol": config["dataset"]["protocol"],
        "fixed_split": "test",
        "fixed_test_images": int(config["dataset"]["test_images"]),
        "same_evaluator_and_prediction_parameters_for_all_seeds": True,
        "prediction_parameters": prediction,
        "E0_seed0_reference": {key: e0[key] for key in common.EXPECTED_BASELINE["E0"]},
        "E1_Mid_per_seed": rows,
        "E1_Mid_three_seed_aggregate": aggregated,
        "descriptive_gate_against_E0_seed0": per_seed_gate,
        "all_three_seeds_passed_descriptive_gate": all(item["passed_descriptive_gate"] for item in per_seed_gate),
        "E0_not_replicated_across_seeds": True,
        "no_seed_controlled_significance_claim": True,
        "evaluator_self_test": evaluator_self_test,
        "fingerprints": {
            "config": {"path": str(config_path), "sha256": common.sha256(config_path)},
            "release_manifest": release,
            "baseline_summary": baseline_fingerprint,
            "new_queue": queue_fingerprints,
            "new_reports": {
                key: {"path": str(output_root / f"{key}_fixed_test.json"), "sha256": common.sha256(output_root / f"{key}_fixed_test.json")}
                for key in EXPECTED_RUN_KEYS
            },
        },
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    summary_path = Path(config["post_training_validation"]["summary_path"])
    common.write_json_atomic(summary_path, payload)
    update_state(
        state_path, post_training_validation_status="completed", validation_current_run=None,
        fixed_test_metrics_path=str(summary_path), fixed_test_metrics_sha256=common.sha256(summary_path),
        all_three_seeds_passed_descriptive_gate=payload["all_three_seeds_passed_descriptive_gate"],
        e2_e3_e4_started=False, e4_started=False,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WSODD E1-Mid seeds 0/1/2 on the locked fixed test")
    parser.add_argument(
        "--config", type=Path,
        default=Path(os.environ.get("RING3_SETUP_ROOT", "/root/autodl-tmp/ring3_setup/autodl"))
        / "wsodd_sequence_e1_mid_multiseed_config.json",
    )
    args = parser.parse_args()
    state_path = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp")) / "formal_state" / "wsodd_sequence_e1_mid_multiseed_state.json"
    try:
        payload = run(args.config)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        if state_path.is_file():
            update_state(
                state_path, post_training_validation_status="failed", validation_error_type=type(exc).__name__,
                validation_error=str(exc), validation_traceback=traceback.format_exc(),
                e2_e3_e4_started=False, e4_started=False,
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
