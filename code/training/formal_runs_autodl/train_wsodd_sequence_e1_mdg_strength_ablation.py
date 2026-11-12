#!/usr/bin/env python3
"""Locked WSODD sequence-disjoint E1 MDG dose ablation.

Only E1-Lite (0.5x probability and strength) and E1-Strong (1.5x) are
trained.  The completed E0 and E1-Mid checkpoints remain immutable anchors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO

import pretrained_sgd_mainline_entry as mainline_entry
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_pretrained_sgd_mainline as wsodd
from cbam_focal_formal_models import FactorialDetectionModel


CONFIG_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_ablation_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_ablation_release.v1"
SUPERVISOR_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_ablation_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.wsodd_sequence_e1_mdg_strength_completion.v1"
COMPLETION_EVIDENCE_NAME = "wsodd_sequence_e1_mdg_strength_completion_evidence.json"

EXPECTED_WEIGHT_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_DATA_YAML_SHA256 = "180965bbca7a5416d467b0d089965db2c8fc22751843795df295d1376c62907e"
EXPECTED_DATA_MANIFEST_SHA256 = "a2edac79e4b7782d6b39d3d91bf22bfa30469811c095310e5f258714fe9869a8"
EXPECTED_ANNOTATION_AUDIT_SHA256 = "e53493e4a791595e7e7f18f53a5614f53106f09e6ad61a60b4dd23fb1b2d0efd"
EXPECTED_BASE_MDG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_SPLIT_COUNTS = {"train": 3977, "val": 1084, "test": 2406}
EXPECTED_CLASSES = 14

BASE_VALIDATE_RUNTIME_EVIDENCE = impl.validate_runtime_evidence
BASE_VERIFY_RESUME = impl.verify_resume
BASE_SUPERVISOR_RUN_VARIANT = supervisor.run_variant
BASE_TRAINER = impl.MainlineSGDTrainer


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _assert_pair(actual: Any, expected: list[float], label: str) -> None:
    if not isinstance(actual, list) or len(actual) != 2 or not all(
        _close(actual[index], expected[index]) for index in range(2)
    ):
        raise RuntimeError(f"Scaled MDG range mismatch for {label}: expected={expected}, observed={actual}")


def verify_scaled_mdg(base: dict[str, Any], scaled: dict[str, Any], factor: float) -> dict[str, Any]:
    base_probs = base["branch_probabilities"]
    active_names = ("low_light", "glare", "mixed")
    neutral_names = ("clean", "standard_aug_only")
    active_total = sum(float(base_probs[name]) for name in active_names)
    scaled_active_total = active_total * factor
    if scaled_active_total >= 1.0:
        raise RuntimeError(f"MDG active probability is not valid after scaling: {scaled_active_total}")
    base_neutral_total = sum(float(base_probs[name]) for name in neutral_names)
    expected_probabilities = {
        name: float(base_probs[name]) * factor for name in active_names
    }
    for name in neutral_names:
        expected_probabilities[name] = (1.0 - scaled_active_total) * float(base_probs[name]) / base_neutral_total
    observed_probabilities = scaled["branch_probabilities"]
    for name, expected in expected_probabilities.items():
        if not _close(observed_probabilities.get(name), expected):
            raise RuntimeError(
                f"Scaled MDG probability mismatch for {name}: expected={expected}, observed={observed_probabilities.get(name)}"
            )
    if scaled["severity_probabilities"] != base["severity_probabilities"]:
        raise RuntimeError("Severity probabilities changed in the dose ablation")

    for severity in ("mild", "medium", "strong"):
        source_low = base["low_light"][severity]
        expected_brightness = [1.0 - factor * (1.0 - float(value)) for value in source_low["brightness"]]
        expected_gamma = [1.0 + factor * (float(value) - 1.0) for value in source_low["gamma"]]
        _assert_pair(scaled["low_light"][severity]["brightness"], expected_brightness, f"low_light.{severity}.brightness")
        _assert_pair(scaled["low_light"][severity]["gamma"], expected_gamma, f"low_light.{severity}.gamma")

        source_glare = base["glare"][severity]
        expected_area = [factor * float(value) for value in source_glare["target_area_ratio"]]
        expected_intensity = [factor * float(value) for value in source_glare["intensity_255"]]
        _assert_pair(scaled["glare"][severity]["target_area_ratio"], expected_area, f"glare.{severity}.area")
        _assert_pair(scaled["glare"][severity]["intensity_255"], expected_intensity, f"glare.{severity}.intensity")

    if scaled["glare"]["aspect_ratio"] != base["glare"]["aspect_ratio"]:
        raise RuntimeError("Glare aspect ratio changed in the dose ablation")
    if scaled["glare"]["softness"] != base["glare"]["softness"]:
        raise RuntimeError("Glare softness changed in the dose ablation")
    provenance = scaled.get("provenance", {})
    if not _close(provenance.get("probability_scale"), factor) or not _close(
        provenance.get("strength_scale"), factor
    ):
        raise RuntimeError("Scaled MDG provenance does not record the requested factor")
    return {
        "factor": factor,
        "active_probability": scaled_active_total,
        "neutral_probability": 1.0 - scaled_active_total,
        "probabilities": observed_probabilities,
    }


def mdg_set_sha256(runs: list[dict[str, Any]]) -> str:
    payload = {str(run["key"]): str(run["mdg_config_sha256"]) for run in runs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected WSODD E1 MDG strength config schema/status")
    runs = config.get("run_order", [])
    observed = [
        (
            run.get("key"),
            run.get("mdg_scale"),
            run.get("mdg_mode"),
            run.get("dual_view"),
            run.get("degraded_small_weighting"),
            run.get("degraded_small_target_weight"),
            run.get("cbam"),
            run.get("focal_eiou"),
            run.get("consistency_loss"),
        )
        for run in runs
    ]
    expected = [
        ("e1_lite", 0.5, "single_view", False, False, 1.0, False, False, False),
        ("e1_strong", 1.5, "single_view", False, False, 1.0, False, False, False),
    ]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 2:
        raise ValueError(f"Unexpected E1 MDG strength run matrix: {observed!r}")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("E1 MDG strength invariants are incomplete")
    gate = config.get("balance_gate", {})
    if not _close(gate.get("minimum_APsmall_gain_vs_E0"), 0.01) or not _close(
        gate.get("minimum_mAP50_95_delta_vs_E0"), -0.003
    ):
        raise ValueError("The pre-registered balance gate changed")


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    checks = {
        "source_mainline_config": (
            Path(anchors["source_mainline_config"]), anchors["source_mainline_config_sha256"]
        ),
        "source_mainline_release_manifest": (
            Path(anchors["source_mainline_release_manifest"]), anchors["source_mainline_release_manifest_sha256"]
        ),
        "baseline_mdg_config": (Path(anchors["baseline_mdg_config"]), anchors["baseline_mdg_config_sha256"]),
        "fixed_test_summary": (Path(anchors["fixed_test_summary"]), anchors["fixed_test_summary_sha256"]),
        "E0_best": (Path(anchors["E0"]["run_dir"]) / "weights" / "best.pt", anchors["E0"]["best_weights_sha256"]),
        "E1_Mid_best": (
            Path(anchors["E1-Mid"]["run_dir"]) / "weights" / "best.pt",
            anchors["E1-Mid"]["best_weights_sha256"],
        ),
    }
    verified: dict[str, Any] = {}
    for name, (path, expected) in checks.items():
        observed = impl.sha256(path) if path.is_file() else "missing"
        if observed != expected:
            raise RuntimeError(f"Reused anchor changed: {name}: expected={expected}, observed={observed}")
        verified[name] = {"path": str(path), "sha256": observed}
    source_config = impl.load_json(Path(anchors["source_mainline_config"]))
    if config["training"] != source_config["training"]:
        raise RuntimeError("Training recipe differs from the completed fixed-split E0/E1 mainline")
    if config["dataset"]["yaml_sha256"] != source_config["dataset"]["yaml_sha256"]:
        raise RuntimeError("Dataset YAML differs from the completed fixed-split mainline")
    return verified


def preflight() -> dict[str, Any]:
    if impl.STOP_SENTINEL.exists():
        raise RuntimeError(f"Stop sentinel exists: {impl.STOP_SENTINEL}")
    config = impl.load_json(impl.CONFIG_PATH)
    validate_config(config)
    release = impl.verify_release_manifest()
    if impl.sha256(impl.HISTORICAL_LOCK_PATH) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Historical formal lock changed")
    if impl.sha256(impl.AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Runtime amendment changed")
    if ultralytics.__version__ != "8.4.124" or torch.__version__ != "2.5.1+cu124":
        raise RuntimeError(f"Locked runtime changed: ultralytics={ultralytics.__version__}, torch={torch.__version__}")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Locked RTX 4090 runtime is unavailable")
    weight_path = Path(config["weights"]["path"])
    if not weight_path.is_file() or impl.sha256(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt is missing or changed")
    dataset_report = wsodd.verify_dataset(config)
    anchor_report = verify_reused_anchors(config)
    base_path = Path(config["reused_anchors"]["baseline_mdg_config"])
    if impl.sha256(base_path) != EXPECTED_BASE_MDG_SHA256:
        raise RuntimeError("Baseline E1-Mid MDG config changed")
    base_mdg = impl.load_mdg_config(base_path)
    mdg_reports: dict[str, Any] = {}
    for run in config["run_order"]:
        path = Path(run["mdg_config"])
        observed = impl.sha256(path) if path.is_file() else "missing"
        if observed != run["mdg_config_sha256"]:
            raise RuntimeError(f"MDG config hash changed for {run['key']}: {observed}")
        scaled = impl.load_mdg_config(path)
        mdg_reports[run["key"]] = {
            "path": str(path),
            "sha256": observed,
            **verify_scaled_mdg(base_mdg, scaled, float(run["mdg_scale"])),
        }
    set_hash = mdg_set_sha256(config["run_order"])
    return {
        "config": config,
        "config_sha256": impl.sha256(impl.CONFIG_PATH),
        "release": release,
        "historical_lock_sha256": EXPECTED_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": set_hash,
        "mdg_config_set_sha256": set_hash,
        "mdg_reports": mdg_reports,
        "dataset_report": dataset_report,
        "reused_anchors": anchor_report,
        "environment": {
            "python": impl.sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    BASE_VALIDATE_RUNTIME_EVIDENCE(evidence, variant)
    recorded = evidence.get("variant", {})
    required = {
        "key": variant["key"],
        "mdg_scale": variant["mdg_scale"],
        "mdg_config": variant["mdg_config"],
        "mdg_config_sha256": variant["mdg_config_sha256"],
        "mdg_mode": "single_view",
        "dual_view": False,
        "degraded_small_weighting": False,
    }
    mismatches = {key: {"expected": value, "observed": recorded.get(key)} for key, value in required.items() if recorded.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Runtime evidence uses the wrong MDG dose: {mismatches}")


def verify_resume(run_dir: Path, variant: dict[str, Any], context: dict[str, Any]) -> None:
    BASE_VERIFY_RESUME(run_dir, variant, context)
    metadata = impl.load_json(run_dir / impl.RUN_METADATA_NAME)
    if metadata.get("variant") != variant:
        raise RuntimeError("Refusing resume because the recorded MDG dose differs")
    copied = run_dir / Path(variant["mdg_config"]).name
    if not copied.is_file() or impl.sha256(copied) != variant["mdg_config_sha256"]:
        raise RuntimeError("Refusing resume because the run-local MDG config is missing or changed")


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.wsodd_sequence_e1_mdg_strength_ablation_protocol.v1",
        "status": "active_locked",
        "created_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_set_sha256": context["mdg_config_set_sha256"],
        "mdg_reports": context["mdg_reports"],
        "dataset_report": context["dataset_report"],
        "reused_anchors": context["reused_anchors"],
        "environment": context["environment"],
        "run_order": context["config"]["run_order"],
        "training": context["config"]["training"],
        "balance_gate": context["config"]["balance_gate"],
        "invariants": context["config"]["invariants"],
    }


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.wsodd_sequence_e1_mdg_strength_ablation_state.v1",
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_set_sha256"],
        "mdg_config_set_sha256": context["mdg_config_set_sha256"],
        "protocol_sha256": impl.sha256(impl.PROTOCOL_PATH),
        "run_order": [run["key"] for run in context["config"]["run_order"]],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": 6.0,
        "eta_utc": (now + timedelta(hours=6)).isoformat(),
        "post_training_validation_status": "pending",
        "reused_E0": True,
        "reused_E1_Mid": True,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    source = YOLO(context["config"]["weights"]["path"]).model
    source_state = source.state_dict()
    tested: list[dict[str, Any]] = []
    for variant in context["config"]["run_order"]:
        model = FactorialDetectionModel(
            source.yaml,
            ch=3,
            nc=EXPECTED_CLASSES,
            verbose=False,
            use_cbam=False,
            use_focal_eiou=False,
        )
        model.load(source)
        target_state = model.state_dict()
        compatible = [key for key, tensor in source_state.items() if key in target_state and target_state[key].shape == tensor.shape]
        mismatched = [key for key in compatible if not torch.equal(target_state[key].cpu(), source_state[key].cpu())]
        if mismatched or len(compatible) < 300:
            raise RuntimeError(f"Pretrained transfer failed for {variant['key']}: compatible={len(compatible)}, mismatched={mismatched[:5]}")
        random.seed(0)
        torch.manual_seed(0)
        transform = impl.MaritimeDegradation(impl.load_mdg_config(Path(variant["mdg_config"])))
        images = torch.rand((128, 3, 96, 96), device=device)
        original = images.clone()
        degraded, counts = transform.apply_batch(images)
        active = sum(int(counts.get(f"branch_{name}", 0)) for name in ("low_light", "glare", "mixed"))
        if not bool(torch.isfinite(degraded).all()) or active <= 0:
            raise RuntimeError(f"MDG self-test failed for {variant['key']}: {counts}")
        tested.append(
            {
                "variant": variant,
                "compatible_pretrained_tensors": len(compatible),
                "mismatched_pretrained_tensors": len(mismatched),
                "mdg_counts": counts,
                "active_images": active,
                "mean_absolute_pixel_delta": float((degraded - original).abs().mean()),
                "finite": True,
                "model_evidence": model.factorial_evidence(),
            }
        )
        del model, degraded, images, original
        torch.cuda.empty_cache()
    payload = {
        "schema": "ring3.wsodd_sequence_e1_mdg_strength_ablation_self_test.v1",
        "status": "passed",
        "verified_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": EXPECTED_WEIGHT_SHA256,
        "tested_variants": tested,
        "e2_e3_e4_started": False,
    }
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def _context_for_variant(context: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    output = dict(context)
    output["mdg_config_sha256"] = variant["mdg_config_sha256"]
    return output


def run_variant_scaled(variant: dict[str, Any], context: dict[str, Any], state: dict[str, Any]) -> bool:
    mdg_path = Path(variant["mdg_config"])
    mdg_hash = variant["mdg_config_sha256"]
    if not mdg_path.is_file() or impl.sha256(mdg_path) != mdg_hash:
        raise RuntimeError(f"Active MDG config changed before {variant['key']}")
    impl.MDG_CONFIG_PATH = mdg_path
    impl.EXPECTED_MDG_SHA256 = mdg_hash
    state.update(
        active_mdg_config=str(mdg_path),
        active_mdg_config_sha256=mdg_hash,
        active_mdg_scale=variant["mdg_scale"],
        last_progress_utc=impl.utc_now(),
    )
    impl.write_json_atomic(impl.STATE_PATH, state)
    return BASE_SUPERVISOR_RUN_VARIANT(variant, _context_for_variant(context, variant), state)


def summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        details = supervisor.run_completion(
            run_dir,
            epochs=int(context["config"]["training"]["epochs"]),
            patience=int(context["config"]["training"]["patience"]),
        )
        if details.get("complete") is not True:
            metrics[variant["key"]] = {"status": "incomplete", "run_dir": str(run_dir), "completion": details}
            continue
        supervisor.validate_completed_variant(run_dir, variant, _context_for_variant(context, variant))
        metrics[variant["key"]] = {
            "status": "completed",
            "label": variant["label"],
            "mdg_scale": variant["mdg_scale"],
            "run_dir": str(run_dir),
            "completion_kind": details["completion_kind"],
            "final_epoch": details["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": details["final_row"],
        }
    payload = {
        "schema": "ring3.wsodd_sequence_e1_mdg_strength_ablation_summary.v1",
        "created_utc": impl.utc_now(),
        "supervisor_schema": SUPERVISOR_SCHEMA,
        "metrics": metrics,
        "reused_anchors": context["reused_anchors"],
        "post_training_validation_status": "pending",
        "single_seed_mechanism_evidence_only": True,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def configure_runtime() -> None:
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT
    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "wsodd_sequence_e1_mdg_strength_ablation_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_sequence_e1_mdg_strength_ablation_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_sequence_e1_mdg_strength_ablation.py"
    impl.OUTPUT_ROOT = data_root / "formal_wsodd_sequence_e1_mdg_strength_ablation"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_sequence_e1_mdg_strength_ablation"
    impl.STATE_PATH = state_root / "wsodd_sequence_e1_mdg_strength_ablation_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_sequence_e1_mdg_strength_ablation_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_sequence_e1_mdg_strength_ablation_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_sequence_e1_mdg_strength_ablation_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_sequence_e1_mdg_strength_ablation_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_SEQUENCE_E1_MDG_STRENGTH_ABLATION"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_sequence_e1_mdg_strength_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_sequence_e1_mdg_strength_run_metadata.json"
    impl.EXPECTED_HISTORICAL_LOCK_SHA256 = EXPECTED_LOCK_SHA256
    impl.EXPECTED_AMENDMENT_SHA256 = EXPECTED_AMENDMENT_SHA256
    impl.EXPECTED_WEIGHT_SHA256 = EXPECTED_WEIGHT_SHA256
    impl.EXPECTED_DATA_YAML_SHA256 = EXPECTED_DATA_YAML_SHA256
    impl.EXPECTED_DATA_MANIFEST_SHA256 = EXPECTED_DATA_MANIFEST_SHA256
    impl.validate_config = validate_config
    impl.verify_dataset = wsodd.verify_dataset
    impl.validate_runtime_evidence = validate_runtime_evidence
    impl.verify_resume = verify_resume
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = static_self_test
    impl.summary = summary
    impl.MainlineSGDTrainer = BASE_TRAINER
    impl.verify_args = mainline_entry.verify_args_with_serialized_device

    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_sequence_e1_mdg_strength_ablation_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = completion_v2.validate_completed_variant_v2
    supervisor.summary = summary
    supervisor.install()
    supervisor.run_variant = run_variant_scaled
    supervisor.summary = summary
    impl.run_variant = run_variant_scaled
    impl.summary = summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked WSODD fixed-split E1 MDG strength ablation")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = preflight()
        print("WSODD_E1_MDG_STRENGTH_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("WSODD_E1_MDG_STRENGTH_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("WSODD_E1_MDG_STRENGTH_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.wsodd_sequence_e1_mdg_strength_ablation_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e2_e3_e4_started": False,
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, failure)
        if impl.STATE_PATH.is_file():
            state = impl.load_json(impl.STATE_PATH)
            state.update(status="supervisor_failed", last_progress_utc=impl.utc_now(), error_type=type(exc).__name__, error=str(exc))
            impl.write_json_atomic(impl.STATE_PATH, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
