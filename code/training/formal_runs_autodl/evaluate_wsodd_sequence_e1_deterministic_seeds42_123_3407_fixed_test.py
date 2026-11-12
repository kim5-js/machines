#!/usr/bin/env python3
"""Evaluate deterministic-image-id E1 seeds 42/123/3407 on the locked WSODD test."""

from __future__ import annotations

import argparse
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


CONFIG_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_release.v1"
REPORT_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_fixed_test_report.v1"
SUMMARY_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_fixed_test_summary.v1"
STATE_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_state.v1"
TRAINING_SUMMARY_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_training_summary.v1"
COMPLETION_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_completion.v1"
EXPECTED_RUN_KEYS = [
    "e1_deterministic_seed42",
    "e1_deterministic_seed123",
    "e1_deterministic_seed3407",
]
METRIC_KEYS = ["mAP50", "mAP50-95", "APsmall", "APmedium", "APlarge"]


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def verify_release_manifest(config_path: Path) -> dict[str, Any]:
    manifest_path = config_path.parent / "wsodd_sequence_e1_deterministic_seeds42_123_3407_release_manifest.json"
    manifest = common.load_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "locked":
        raise RuntimeError("Unexpected deterministic E1 release manifest")
    verified: list[dict[str, str]] = []
    for item in manifest.get("files", []):
        path = config_path.parent / str(item["path"])
        observed = common.sha256(path) if path.is_file() else "missing"
        if observed != item.get("sha256"):
            raise RuntimeError(f"Release hash mismatch for {item.get('path')}: {observed}")
        verified.append({"path": str(path), "sha256": observed})
    if len(verified) < 20:
        raise RuntimeError(f"Release manifest is incomplete: {len(verified)} files")
    return {
        "path": str(manifest_path),
        "sha256": common.sha256(manifest_path),
        "verified_files": verified,
    }


def verify_comparison_anchors(
    config: dict[str, Any],
    prediction: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchors = config["reused_anchors"]
    e0_anchor = anchors["E0-multiseed-paired-summary"]
    e0_path = Path(e0_anchor["path"])
    e0_hash = common.sha256(e0_path) if e0_path.is_file() else "missing"
    if e0_hash != e0_anchor["sha256"]:
        raise RuntimeError(f"Paired E0 summary changed: {e0_hash}")
    e0_summary = common.load_json(e0_path)
    if (
        e0_summary.get("status") != "completed"
        or e0_summary.get("dataset_protocol") != config["dataset"]["protocol"]
        or e0_summary.get("fixed_split") != "test"
        or int(e0_summary.get("fixed_test_images", -1)) != int(config["dataset"]["test_images"])
        or e0_summary.get("same_evaluator_and_prediction_parameters_for_all_models_and_seeds") is not True
        or e0_summary.get("prediction_parameters") != prediction
        or e0_summary.get("e4_started") is not False
    ):
        raise RuntimeError("Paired E0 multiseed summary is not a compatible fixed-test anchor")
    e0_rows = e0_summary.get("E0_per_seed", [])
    if [int(row.get("seed", -1)) for row in e0_rows] != [0, 1, 2]:
        raise RuntimeError("E0 anchor does not contain seeds 42/123/3407")

    random_anchor = anchors["random-E1-Mid-multiseed-summary"]
    random_path = Path(random_anchor["path"])
    random_hash = common.sha256(random_path) if random_path.is_file() else "missing"
    if random_hash != random_anchor["sha256"]:
        raise RuntimeError(f"Random E1-Mid summary changed: {random_hash}")
    random_summary = common.load_json(random_path)
    if (
        random_summary.get("status") != "completed"
        or random_summary.get("dataset_protocol") != config["dataset"]["protocol"]
        or random_summary.get("fixed_split") != "test"
        or int(random_summary.get("fixed_test_images", -1)) != int(config["dataset"]["test_images"])
        or random_summary.get("same_evaluator_and_prediction_parameters_for_all_seeds") is not True
        or random_summary.get("prediction_parameters") != prediction
        or random_summary.get("e4_started") is not False
    ):
        raise RuntimeError("Random E1-Mid summary is not a compatible fixed-test anchor")
    random_rows = random_summary.get("E1_Mid_per_seed", [])
    if [int(row.get("seed", -1)) for row in random_rows] != [0, 1, 2]:
        raise RuntimeError("Random E1-Mid anchor does not contain seeds 42/123/3407")
    e0_embedded_random = e0_summary.get("E1_Mid_per_seed", [])
    for left, right in zip(random_rows, e0_embedded_random):
        if int(left["seed"]) != int(right["seed"]) or any(
            not close(left[key], right[key]) for key in METRIC_KEYS
        ):
            raise RuntimeError("The E0 and random-E1 anchor summaries disagree")
    return e0_rows, random_rows, {
        "E0_paired_multiseed_summary": {"path": str(e0_path), "sha256": e0_hash},
        "random_E1_Mid_multiseed_summary": {
            "path": str(random_path),
            "sha256": random_hash,
        },
    }


def verify_completed_runs(
    config: dict[str, Any],
    config_path: Path,
    release: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], Path, dict[str, Any]]:
    data_root = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
    output_root = data_root / "formal_wsodd_sequence_e1_deterministic_seeds42_123_3407"
    state_path = data_root / "formal_state" / "wsodd_sequence_e1_deterministic_seeds42_123_3407_state.json"
    summary_path = data_root / "formal_state" / "wsodd_sequence_e1_deterministic_seeds42_123_3407_training_summary.json"
    if not state_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Deterministic E1 training state/summary is missing")
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
        "reused_seed0": False,
        "all_three_seeds_fresh": True,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in expected_state.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Deterministic training queue is not valid and complete: {mismatches}")
    if training_summary.get("schema") != TRAINING_SUMMARY_SCHEMA:
        raise RuntimeError("Unexpected deterministic E1 training summary schema")
    assignment_map_sha256 = state.get("deterministic_assignment_map_sha256")
    if assignment_map_sha256 != config["deterministic_assignment"]["expected_assignment_map_sha256"]:
        raise RuntimeError("Training state lacks a deterministic assignment-map hash")

    runs: dict[str, dict[str, Any]] = {}
    for variant in config["run_order"]:
        key = str(variant["key"])
        seed = int(variant["seed"])
        run_dir = output_root / str(variant["name"])
        files = {
            "best_checkpoint": run_dir / "weights" / "best.pt",
            "results_csv": run_dir / "results.csv",
            "args": run_dir / "args.yaml",
            "runtime_evidence": run_dir / "wsodd_sequence_e1_deterministic_seeds42_123_3407_runtime_evidence.json",
            "run_metadata": run_dir / "wsodd_sequence_e1_deterministic_seeds42_123_3407_run_metadata.json",
            "completion_evidence": run_dir / "wsodd_sequence_e1_deterministic_seeds42_123_3407_completion_evidence.json",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Completed deterministic E1 artifacts are missing for {key}: {missing}")
        entry = training_summary.get("metrics", {}).get(key, {})
        if (
            entry.get("status") != "completed"
            or int(entry.get("seed", -1)) != seed
            or Path(str(entry.get("run_dir", ""))).resolve() != run_dir.resolve()
        ):
            raise RuntimeError(f"Training summary does not validate {key}")

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
            "deterministic_degradation_assignment": True,
            "deterministic_assignment_schema": config["deterministic_assignment"]["schema"],
            "deterministic_assignment_namespace": config["deterministic_assignment"]["namespace"],
            "deterministic_assignment_seed_independent": True,
            "deterministic_assignment_epoch_independent": True,
            "deterministic_assignment_batch_order_independent": True,
            "deterministic_assignment_conflict_count": 0,
            "deterministic_assignment_map_sha256": assignment_map_sha256,
        }
        runtime_mismatches = {
            name: {"expected": value, "observed": runtime.get(name)}
            for name, value in required_runtime.items()
            if runtime.get(name) != value
        }
        if int(runtime.get("mdg_counts_current_batch", {}).get("images_total", -1)) <= 0:
            runtime_mismatches["mdg_counts_current_batch.images_total"] = runtime.get(
                "mdg_counts_current_batch"
            )
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
        locked_args = {
            **config["training"],
            "seed": seed,
            "model": config["weights"]["path"],
            "data": config["dataset"]["yaml"],
        }
        args_mismatches = {
            name: {"expected": value, "observed": args.get(name)}
            for name, value in locked_args.items()
            if name != "device" and args.get(name) != value
        }
        if str(args.get("device")) != str(config["training"]["device"]):
            args_mismatches["device"] = {
                "expected": config["training"]["device"],
                "observed": args.get("device"),
            }
        if args_mismatches:
            raise RuntimeError(f"Locked args mismatch for {key}: {args_mismatches}")
        runs[key] = {
            "variant": variant,
            "run_dir": str(run_dir),
            "training_validation": common.training_metrics(run_dir),
            "fingerprints": {
                name: {"path": str(path), "sha256": common.sha256(path)}
                for name, path in files.items()
            },
        }
    return runs, state_path, {
        "state": {"path": str(state_path), "sha256": common.sha256(state_path)},
        "training_summary": {"path": str(summary_path), "sha256": common.sha256(summary_path)},
        "deterministic_assignment_map_sha256": assignment_map_sha256,
    }


def evaluate_variant(
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
            "completed_utc": common.utc_now(),
            "experiment": variant["label"],
            "seed": int(variant["seed"]),
            "dataset_protocol": config["dataset"]["protocol"],
            "fixed_split": "test",
            "fixed_test_images": len(images),
            "same_evaluator_and_prediction_parameters_as_E0_and_random_E1": True,
            "fixed_test": robust.metric_summary(metrics),
            "full_metrics": metrics,
            "training_validation": run["training_validation"],
            "prediction_parameters": prediction,
            "prediction_counts": counts,
            "identity_source_equivalence": identity,
            "fingerprints": fingerprints,
            "deterministic_mdg_used_during_training_only": True,
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


def cohort_mean_deltas(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Compare means without pretending different seed cohorts are paired."""
    reference = aggregate(reference_rows)
    candidate = aggregate(candidate_rows)
    return {
        key: {
            "reference_label": reference_label,
            "reference_seed_set": [int(row["seed"]) for row in reference_rows],
            "reference_mean": reference[key]["mean"],
            "candidate_label": candidate_label,
            "candidate_seed_set": [int(row["seed"]) for row in candidate_rows],
            "candidate_mean": candidate[key]["mean"],
            f"mean_delta_{candidate_label}_minus_{reference_label}": (
                candidate[key]["mean"] - reference[key]["mean"]
            ),
            "paired": False,
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
        raise RuntimeError("Unexpected deterministic E1 multiseed config")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise RuntimeError("Deterministic E1 invariants are not all locked true")
    release = verify_release_manifest(config_path)
    static_assets = {
        "dataset_yaml": (Path(config["dataset"]["yaml"]), config["dataset"]["yaml_sha256"]),
        "test_manifest": (
            Path(config["dataset"]["test_manifest"]),
            config["dataset"]["test_manifest_sha256"],
        ),
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
        observed = common.sha256(path) if path.is_file() else "missing"
        if observed != expected:
            raise RuntimeError(f"Locked evaluator asset changed for {name}: {observed}")
    prediction = common.prediction_parameters(config)
    e0_rows, random_rows, anchor_fingerprints = verify_comparison_anchors(
        config,
        prediction,
    )
    runs, state_path, queue_fingerprints = verify_completed_runs(
        config,
        config_path,
        release,
    )
    update_state(
        state_path,
        post_training_validation_status="running",
        validation_current_run=None,
    )
    common.importlib.import_module("cbam_focal_formal_models")
    common.importlib.import_module("pretrained_sgd_mainline_models")
    evaluator = common.load_module(
        "ring3_locked_wsodd_e1_deterministic_area_evaluator",
        static_assets["base_evaluator"][0],
    )
    robust = common.load_module(
        "ring3_locked_wsodd_e1_deterministic_robust_helper",
        static_assets["robust_helper"][0],
    )
    evaluator_self_test = evaluator.synthetic_self_test()
    images, _ = common.resolve_fixed_test_images(config)
    output_root = Path(config["post_training_validation"]["output_root"])
    reports: dict[str, dict[str, Any]] = {}
    for variant in config["run_order"]:
        key = str(variant["key"])
        update_state(
            state_path,
            post_training_validation_status="running",
            validation_current_run=key,
        )
        reports[key] = evaluate_variant(
            variant,
            runs[key],
            config,
            config_path,
            release,
            evaluator,
            robust,
            images,
            prediction,
            output_root,
        )

    deterministic_rows = [
        {
            "seed": int(variant["seed"]),
            **reports[str(variant["key"])]["fixed_test"],
        }
        for variant in config["run_order"]
    ]
    versus_e0_summary = cohort_mean_deltas(
        e0_rows,
        deterministic_rows,
        reference_label="E0",
        candidate_label="E1_Deterministic",
    )
    versus_random_summary = cohort_mean_deltas(
        random_rows,
        deterministic_rows,
        reference_label="E1_Random",
        candidate_label="E1_Deterministic",
    )
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_utc": common.utc_now(),
        "purpose": config["purpose"],
        "dataset_protocol": config["dataset"]["protocol"],
        "fixed_split": "test",
        "fixed_test_images": int(config["dataset"]["test_images"]),
        "same_evaluator_and_prediction_parameters_for_all_models_and_seeds": True,
        "prediction_parameters": prediction,
        "deterministic_assignment_map_sha256": queue_fingerprints[
            "deterministic_assignment_map_sha256"
        ],
        "E0_reference_seeds_0_1_2_per_seed": e0_rows,
        "E0_reference_three_seed_aggregate": aggregate(e0_rows),
        "E1_Random_reference_seeds_0_1_2_per_seed": random_rows,
        "E1_Random_reference_three_seed_aggregate": aggregate(random_rows),
        "E1_Deterministic_seeds_42_123_3407_per_seed": deterministic_rows,
        "E1_Deterministic_new_three_seed_aggregate": aggregate(deterministic_rows),
        "unpaired_cohort_mean_E1_Deterministic_minus_E0": versus_e0_summary,
        "unpaired_cohort_mean_E1_Deterministic_minus_E1_Random": versus_random_summary,
        "reference_seed_set": [0, 1, 2],
        "candidate_seed_set": [42, 123, 3407],
        "unpaired_cross_seed_cohort_comparison_only": True,
        "no_significance_claim": True,
        "evaluator_self_test": evaluator_self_test,
        "fingerprints": {
            "config": {"path": str(config_path), "sha256": common.sha256(config_path)},
            "release_manifest": release,
            "anchors": anchor_fingerprints,
            "new_queue": queue_fingerprints,
            "new_reports": {
                key: {
                    "path": str(output_root / f"{key}_fixed_test.json"),
                    "sha256": common.sha256(output_root / f"{key}_fixed_test.json"),
                }
                for key in EXPECTED_RUN_KEYS
            },
        },
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    summary_path = Path(config["post_training_validation"]["summary_path"])
    common.write_json_atomic(summary_path, payload)
    update_state(
        state_path,
        post_training_validation_status="completed",
        validation_current_run=None,
        fixed_test_metrics_path=str(summary_path),
        fixed_test_metrics_sha256=common.sha256(summary_path),
        cohort_comparison_status="completed",
        e2_e3_e4_started=False,
        e4_started=False,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic-image-id WSODD E1 seeds 42/123/3407 on the locked fixed test"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get(
                "RING3_SETUP_ROOT",
                "/root/autodl-tmp/ring3_setup/autodl",
            )
        )
        / "wsodd_sequence_e1_deterministic_seeds42_123_3407_config.json",
    )
    args = parser.parse_args()
    state_path = (
        Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
        / "formal_state"
        / "wsodd_sequence_e1_deterministic_seeds42_123_3407_state.json"
    )
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


