from __future__ import annotations

import argparse
import copy
import json
import math
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ultralytics.utils.tal import TaskAlignedAssigner

import evaluate_wsodd_single_fixed_test as fixed_test
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_legacy_e3_alpha15 as released_alpha15
import train_wsodd_pretrained_sgd_mainline as strict_wsodd
import wsodd_pretrained_sgd_hotfix_v3_entry as amp_hotfix


CONFIG_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_release_manifest.v1"
PROTOCOL_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_protocol.v1"
STATE_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_state.v1"
SUMMARY_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_summary.v1"
SUPERVISOR_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.wsodd_sequence_e3_alpha15_completion_evidence.v1"
COMPLETION_EVIDENCE_NAME = "wsodd_sequence_e3_alpha15_completion_evidence.json"
EXPECTED_ALPHA = 1.5
EXPECTED_OUTPUT_ROOT = "/root/autodl-tmp/formal_wsodd_sequence_e3_alpha15"
EXPECTED_REFERENCE_CONFIG_SHA256 = "b89049d9c3ef7ae95e3d6007777597779d07ad4e17d9e9decb85425f21ce1ec2"
EXPECTED_REFERENCE_TRAINER_SHA256 = "8d8d46976d2be9d2473886f7de25e91bbd788a4ed368bd51b7998d42fcc2e413"
EXPECTED_REFERENCE_LOSS_SHA256 = "5df5e0d4b39f17c0abb08288ff0d826b4894f6245bdd90dd6be6458590ecd67b"
EXPECTED_BASE_EVALUATOR_SHA256 = "c1eeb2e5bdf72e627a856ac14f4f679f844de2a9ff0a205cfce50323de9b61cc"
EXPECTED_ROBUST_HELPER_SHA256 = "cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f"

RELEASED_MAINLINE_TRAINER = released_alpha15.RELEASED_MAINLINE_TRAINER
RELEASED_VALIDATE_CONFIG = released_alpha15.RELEASED_VALIDATE_CONFIG
RELEASED_VALIDATE_RUNTIME_EVIDENCE = released_alpha15.RELEASED_VALIDATE_RUNTIME_EVIDENCE


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected WSODD sequence-disjoint E3 alpha=1.5 config schema/status")
    runs = config.get("run_order", [])
    observed = [
        (
            run.get("key"), run.get("mdg_mode"), run.get("dual_view"),
            run.get("degraded_small_weighting"), run.get("degraded_small_target_weight"),
            run.get("cbam"), run.get("focal_eiou"), run.get("consistency_loss"),
        )
        for run in runs
    ]
    expected = [("e3", "dual_view", True, True, EXPECTED_ALPHA, False, False, False)]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 1:
        raise ValueError(f"Unexpected WSODD sequence-disjoint E3 alpha=1.5 run matrix: {observed!r}")

    proxy = copy.deepcopy(config)
    proxy["schema"] = "ring3.pretrained_sgd_mainline_config.v1"
    proxy["run_order"] = [
        {"key": "e0", "name": "e0", "mdg_mode": "none", "dual_view": False, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e1", "name": "e1", "mdg_mode": "single_view", "dual_view": False, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e2", "name": "e2", "mdg_mode": "dual_view", "dual_view": True, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e3", "name": "e3", "mdg_mode": "dual_view", "dual_view": True, "degraded_small_weighting": True, "cbam": False, "focal_eiou": False, "consistency_loss": False},
    ]
    rebound_schema = impl.CONFIG_SCHEMA
    try:
        impl.CONFIG_SCHEMA = "ring3.pretrained_sgd_mainline_config.v1"
        RELEASED_VALIDATE_CONFIG(proxy)
    finally:
        impl.CONFIG_SCHEMA = rebound_schema

    mismatches: dict[str, Any] = {}
    dataset = config.get("dataset", {})
    expected_dataset = {
        "yaml_sha256": strict_wsodd.EXPECTED_DATA_YAML_SHA256,
        "manifest_sha256": strict_wsodd.EXPECTED_DATA_MANIFEST_SHA256,
        "annotation_semantics_audit_sha256": strict_wsodd.EXPECTED_ANNOTATION_AUDIT_SHA256,
        "protocol": "wsodd_sequence_disjoint_v1",
        "train_images": 3977,
        "val_images": 1084,
        "test_images": 2406,
        "classes": strict_wsodd.EXPECTED_CLASSES,
        "sequence_groups": 147,
        "group_overlap": {"train-val": 0, "train-test": 0, "val-test": 0},
    }
    for key, expected_value in expected_dataset.items():
        if dataset.get(key) != expected_value:
            mismatches[f"dataset.{key}"] = {"expected": expected_value, "observed": dataset.get(key)}

    reference = config.get("released_seadronessee_e3_reference", {})
    expected_reference = {
        "config_sha256": EXPECTED_REFERENCE_CONFIG_SHA256,
        "trainer_sha256": EXPECTED_REFERENCE_TRAINER_SHA256,
        "loss_sha256": EXPECTED_REFERENCE_LOSS_SHA256,
        "degraded_small_target_weight": EXPECTED_ALPHA,
    }
    for key, expected_value in expected_reference.items():
        if reference.get(key) != expected_value:
            mismatches[f"released_seadronessee_e3_reference.{key}"] = {
                "expected": expected_value, "observed": reference.get(key),
            }

    weighting = config.get("small_object_weighting", {})
    expected_weighting = {
        "ordinary_target_weight": 1.0,
        "degraded_small_target_weight": EXPECTED_ALPHA,
        "small_area_lt_px2": 1024,
        "area_coordinates": "post-augmentation training pixels",
        "scope": "degraded-small assigned positive anchors only",
        "normalization_denominator_unchanged": True,
        "background_anchor_weight_unchanged": True,
    }
    for key, expected_value in expected_weighting.items():
        if weighting.get(key) != expected_value:
            mismatches[f"small_object_weighting.{key}"] = {"expected": expected_value, "observed": weighting.get(key)}

    execution = config.get("execution", {})
    if execution.get("output_root") != EXPECTED_OUTPUT_ROOT:
        mismatches["execution.output_root"] = execution.get("output_root")
    for key in (
        "independent_fresh_start", "resume_only_from_same_run_last_pt",
        "numeric_failure_restarts_from_locked_pretrained", "normal_early_stopping_is_completion",
        "previous_alpha12_run_is_not_reused", "previous_e0_e2_runs_are_not_rerun",
    ):
        if execution.get(key) is not True:
            mismatches[f"execution.{key}"] = execution.get(key)
    if execution.get("max_attempts_per_run") != 3:
        mismatches["execution.max_attempts_per_run"] = execution.get("max_attempts_per_run")

    validation = config.get("post_training_validation", {})
    expected_validation = {
        "fixed_split": "test",
        "fixed_images": 2406,
        "same_evaluator_and_prediction_parameters": True,
        "base_evaluator_sha256": EXPECTED_BASE_EVALUATOR_SHA256,
        "robust_helper_sha256": EXPECTED_ROBUST_HELPER_SHA256,
        "imgsz": 960,
        "batch": 8,
        "chunk_size": 4,
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "prediction_max_det": 300,
        "evaluation_max_det": 100,
    }
    for key, expected_value in expected_validation.items():
        if validation.get(key) != expected_value:
            mismatches[f"post_training_validation.{key}"] = {"expected": expected_value, "observed": validation.get(key)}
    if mismatches:
        raise ValueError(f"WSODD sequence-disjoint E3 alpha=1.5 contract changed: {mismatches}")


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "active_locked",
        "created_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "dataset_report": context["dataset_report"],
        "environment": context["environment"],
        "run_order": context["config"]["run_order"],
        "training": context["config"]["training"],
        "released_seadronessee_e3_reference": context["config"]["released_seadronessee_e3_reference"],
        "small_object_weighting": context["config"]["small_object_weighting"],
        "post_training_validation": context["config"]["post_training_validation"],
        "invariants": context["config"]["invariants"],
    }


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": STATE_SCHEMA,
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "protocol_sha256": impl.sha256(impl.PROTOCOL_PATH),
        "dataset_protocol": "wsodd_sequence_disjoint_v1",
        "run_order": ["e3"],
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
        "released_e3_alpha": EXPECTED_ALPHA,
        "alpha12_checkpoint_reused": False,
        "previous_e0_e2_runs_untouched": True,
        "single_seed_mechanism_evidence_only": True,
        "e4_started": False,
    }


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    payload = released_alpha15.static_self_test(context)
    payload.update(
        schema="ring3.wsodd_sequence_e3_alpha15_self_test.v1",
        dataset_protocol="wsodd_sequence_disjoint_v1",
        fixed_test_images=2406,
        amp_foreground_mask_hotfix_installed=TaskAlignedAssigner.forward is amp_hotfix.task_aligned_assigner_forward_bool_mask,
    )
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    variant = context["config"]["run_order"][0]
    run_dir = impl.OUTPUT_ROOT / variant["name"]
    details = supervisor.run_completion(
        run_dir,
        epochs=int(context["config"]["training"]["epochs"]),
        patience=int(context["config"]["training"]["patience"]),
    )
    if details.get("complete") is True:
        supervisor.validate_completed_variant(run_dir, variant, context)
        metric = {
            "status": "completed",
            "run_dir": str(run_dir),
            "completion_kind": details["completion_kind"],
            "final_epoch": details["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": details["final_row"],
        }
    else:
        metric = {"status": "incomplete", "run_dir": str(run_dir), "completion": details}
    payload: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "created_utc": impl.utc_now(),
        "supervisor_schema": SUPERVISOR_SCHEMA,
        "formal_e3_definition": "released SeaDronesSee E3 degraded-small alpha=1.5",
        "dataset_protocol": "wsodd_sequence_disjoint_v1",
        "metrics": {"e3": metric},
        "post_training_validation_status": "pending",
        "single_seed_mechanism_evidence_only": True,
        "alpha12_checkpoint_reused": False,
        "e4_started": False,
    }
    fixed_path = impl.STATE_ROOT / "wsodd_sequence_e3_alpha15_fixed_test_metrics.json"
    if fixed_path.is_file():
        payload["fixed_test_metrics"] = impl.load_json(fixed_path)
        payload["post_training_validation_status"] = "completed"
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def run_fixed_test(context: dict[str, Any]) -> dict[str, Any]:
    state = impl.load_json(impl.STATE_PATH)
    state.update(status="evaluating", post_training_validation_status="running", last_progress_utc=impl.utc_now())
    impl.write_json_atomic(impl.STATE_PATH, state)
    variant = context["config"]["run_order"][0]
    output_path = impl.STATE_ROOT / "wsodd_sequence_e3_alpha15_fixed_test_metrics.json"
    report = fixed_test.run(impl.CONFIG_PATH, impl.OUTPUT_ROOT / variant["name"], output_path)
    final_summary = summary(context)
    state = impl.load_json(impl.STATE_PATH)
    state.update(
        status="completed",
        post_training_validation_status="completed",
        fixed_test_metrics_path=str(output_path),
        fixed_test_metrics_sha256=impl.sha256(output_path),
        summary_path=str(impl.SUMMARY_PATH),
        summary_sha256=impl.sha256(impl.SUMMARY_PATH),
        completed_utc=impl.utc_now(),
        last_progress_utc=impl.utc_now(),
        e4_started=False,
    )
    impl.write_json_atomic(impl.STATE_PATH, state)
    impl.append_event("fixed_test_evaluation_completed", report=report, summary=final_summary)
    return report


def configure_runtime() -> None:
    released_alpha15.configure_runtime()
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT
    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "wsodd_sequence_e3_alpha15_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_sequence_e3_alpha15_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_sequence_e3_alpha15.py"
    impl.OUTPUT_ROOT = data_root / "formal_wsodd_sequence_e3_alpha15"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_sequence_e3_alpha15"
    impl.STATE_PATH = state_root / "wsodd_sequence_e3_alpha15_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_sequence_e3_alpha15_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_sequence_e3_alpha15_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_sequence_e3_alpha15_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_sequence_e3_alpha15_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_SEQUENCE_E3_ALPHA15"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_sequence_e3_alpha15_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_sequence_e3_alpha15_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = strict_wsodd.EXPECTED_DATA_YAML_SHA256
    impl.validate_config = validate_config
    impl.verify_dataset = strict_wsodd.verify_dataset
    impl.validate_runtime_evidence = RELEASED_VALIDATE_RUNTIME_EVIDENCE
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = static_self_test
    impl.MainlineSGDTrainer = RELEASED_MAINLINE_TRAINER
    impl.verify_args = amp_hotfix.verify_args_allow_canonical_own_last_pt

    TaskAlignedAssigner.forward = amp_hotfix.task_aligned_assigner_forward_bool_mask
    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_sequence_e3_alpha15_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = completion_v2.validate_completed_variant_v2
    supervisor.summary = summary
    supervisor.install()
    supervisor.summary = summary
    impl.summary = summary
    os.environ["RING3_WSODD_E3_HOTFIX"] = amp_hotfix.HOTFIX_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one locked WSODD sequence-disjoint E3 alpha=1.5 experiment")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = impl.preflight()
        print("WSODD_SEQUENCE_E3_ALPHA15_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        early_stop_guard = completion_v2.guard_self_test()
        amp_mask_guard = amp_hotfix.mask_hotfix_self_test()
        print("WSODD_SEQUENCE_E3_ALPHA15_EARLY_STOP_GUARD " + json.dumps(early_stop_guard, sort_keys=True), flush=True)
        print("WSODD_SEQUENCE_E3_ALPHA15_AMP_MASK_GUARD " + json.dumps(amp_mask_guard, sort_keys=True), flush=True)
        self_test = static_self_test(context)
        print("WSODD_SEQUENCE_E3_ALPHA15_SELF_TEST " + json.dumps(self_test, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        result = supervisor.execute_queue(context)
        if result != 0:
            return result
        report = run_fixed_test(context)
        print("WSODD_SEQUENCE_E3_ALPHA15_FIXED_TEST " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        state_status = "supervisor_failed"
        training_results_preserved = False
        if impl.STATE_PATH.is_file() and impl.load_json(impl.STATE_PATH).get("status") == "evaluating":
            state_status = "evaluation_failed"
            training_results_preserved = True
        failure = {
            "schema": "ring3.wsodd_sequence_e3_alpha15_failure.v1",
            "status": state_status,
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_results_preserved": training_results_preserved,
            "alpha12_checkpoint_reused": False,
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, failure)
        if impl.STATE_PATH.is_file():
            state = impl.load_json(impl.STATE_PATH)
            state.update(status=state_status, last_progress_utc=impl.utc_now(), error_type=type(exc).__name__, error=str(exc))
            impl.write_json_atomic(impl.STATE_PATH, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
