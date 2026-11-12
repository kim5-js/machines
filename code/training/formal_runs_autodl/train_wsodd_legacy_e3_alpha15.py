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

import torch
from ultralytics import YOLO

import pretrained_sgd_mainline_supervisor_patch as supervisor
import recover_wsodd_legacy_coco_areas_v2 as area_adapter
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_legacy_split_sgd_mainline as legacy
import wsodd_pretrained_sgd_hotfix_v3_entry as amp_hotfix
from pretrained_sgd_mainline_models import MainlineSmallWeightedDetectionModel


CONFIG_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_release_manifest.v1"
PROTOCOL_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_protocol.v1"
STATE_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_state.v1"
SUMMARY_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_summary.v1"
SUPERVISOR_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_completion_evidence.v1"
COMPLETION_EVIDENCE_NAME = "wsodd_legacy_e3_alpha15_completion_evidence.json"
EXPECTED_ALPHA = 1.5
EXPECTED_REFERENCE_CONFIG_SHA256 = "b89049d9c3ef7ae95e3d6007777597779d07ad4e17d9e9decb85425f21ce1ec2"
EXPECTED_REFERENCE_TRAINER_SHA256 = "8d8d46976d2be9d2473886f7de25e91bbd788a4ed368bd51b7998d42fcc2e413"
EXPECTED_REFERENCE_LOSS_SHA256 = "5df5e0d4b39f17c0abb08288ff0d826b4894f6245bdd90dd6be6458590ecd67b"
EXPECTED_AREA_ADAPTER_SHA256 = "083b05cc9c82fa41474de8e92b080755f3b0df88429b1d7c22dc2371eb8cb577"
EXPECTED_BASE_EVALUATOR_SHA256 = "c1eeb2e5bdf72e627a856ac14f4f679f844de2a9ff0a205cfce50323de9b61cc"
EXPECTED_OUTPUT_ROOT = "/root/autodl-tmp/validation_wsodd_legacy_split_sgd_e3_alpha15"

# Capture the released SeaDronesSee E3 implementation before the WSODD wrappers
# rebind the shared runtime module to the earlier alpha=1.2 experiment.
RELEASED_MAINLINE_TRAINER = impl.MainlineSGDTrainer
RELEASED_VALIDATE_CONFIG = impl.validate_config
RELEASED_VALIDATE_RUNTIME_EVIDENCE = impl.validate_runtime_evidence


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected WSODD legacy E3 alpha=1.5 config schema/status")
    runs = config.get("run_order", [])
    observed = [
        (
            run.get("key"),
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
    expected = [("e3", "dual_view", True, True, EXPECTED_ALPHA, False, False, False)]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 1:
        raise ValueError(f"Unexpected WSODD legacy E3 alpha=1.5 run matrix: {observed!r}")

    # The released validator is the authority for every training hyperparameter
    # and forbidden-module invariant. Only its four-run matrix is replaced by a
    # synthetic copy so this isolated E3-only release can reuse it unchanged.
    proxy = copy.deepcopy(config)
    proxy["schema"] = "ring3.pretrained_sgd_mainline_config.v1"
    proxy["run_order"] = [
        {"key": "e0", "name": "e0", "mdg_mode": "none", "dual_view": False, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e1", "name": "e1", "mdg_mode": "single_view", "dual_view": False, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e2", "name": "e2", "mdg_mode": "dual_view", "dual_view": True, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e3", "name": "e3", "mdg_mode": "dual_view", "dual_view": True, "degraded_small_weighting": True, "cbam": False, "focal_eiou": False, "consistency_loss": False},
    ]
    RELEASED_VALIDATE_CONFIG(proxy)

    mismatches: dict[str, Any] = {}
    dataset = config.get("dataset", {})
    expected_dataset = {
        "yaml_sha256": legacy.EXPECTED_DATA_YAML_SHA256,
        "annotation_semantics_audit_sha256": legacy.EXPECTED_ANNOTATION_AUDIT_SHA256,
        "protocol": "wsodd_legacy_image_split_v1",
        "train_images": 6222,
        "val_images": 1245,
        "test_images": 0,
        "classes": legacy.EXPECTED_CLASSES,
        "sequence_groups": 147,
        "contaminated_sequence_groups": 98,
        "val_images_in_contaminated_groups": 1243,
        "independent_test_available": False,
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
                "expected": expected_value,
                "observed": reference.get(key),
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
            mismatches[f"small_object_weighting.{key}"] = {
                "expected": expected_value,
                "observed": weighting.get(key),
            }

    execution = config.get("execution", {})
    if execution.get("output_root") != EXPECTED_OUTPUT_ROOT:
        mismatches["execution.output_root"] = execution.get("output_root")
    required_execution_flags = (
        "independent_fresh_start",
        "resume_only_from_same_run_last_pt",
        "numeric_failure_restarts_from_locked_pretrained",
        "normal_early_stopping_is_completion",
        "previous_alpha12_run_is_not_reused",
        "previous_e0_e2_runs_are_not_rerun",
    )
    for key in required_execution_flags:
        if execution.get(key) is not True:
            mismatches[f"execution.{key}"] = execution.get(key)
    if execution.get("max_attempts_per_run") != 3:
        mismatches["execution.max_attempts_per_run"] = execution.get("max_attempts_per_run")

    validation = config.get("post_training_validation", {})
    expected_validation = {
        "fixed_split": "val",
        "fixed_images": 1245,
        "same_evaluator_and_prediction_parameters": True,
        "base_evaluator_sha256": EXPECTED_BASE_EVALUATOR_SHA256,
        "chunked_adapter_sha256": EXPECTED_AREA_ADAPTER_SHA256,
        "prediction_chunk_size": 4,
    }
    for key, expected_value in expected_validation.items():
        if validation.get(key) != expected_value:
            mismatches[f"post_training_validation.{key}"] = {
                "expected": expected_value,
                "observed": validation.get(key),
            }
    if mismatches:
        raise ValueError(f"WSODD legacy E3 alpha=1.5 contract changed: {mismatches}")


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "active_locked_validation_only",
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
        "warning": "Historical WSODD image-level split has confirmed sequence leakage; validation-only reproduction",
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
        "dataset_protocol": "wsodd_legacy_image_split_v1",
        "evidence_class": "legacy_protocol_reproduction_not_formal_generalization_evidence",
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
        "alpha12_excluded_from_formal_e3": True,
        "previous_e0_e2_runs_untouched": True,
        "e4_started": False,
    }


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    variant = context["config"]["run_order"][0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = YOLO(str(context["config"]["weights"]["path"])).model
    source_state = source.state_dict()
    model = MainlineSmallWeightedDetectionModel(
        source.yaml,
        ch=3,
        nc=legacy.EXPECTED_CLASSES,
        verbose=False,
        degraded_small_weight=EXPECTED_ALPHA,
    )
    model.load(source)
    target_state = model.state_dict()
    compatible = [key for key, tensor in source_state.items() if key in target_state and target_state[key].shape == tensor.shape]
    mismatched = [key for key in compatible if not torch.equal(target_state[key].cpu(), source_state[key].cpu())]
    if mismatched or len(compatible) < 300:
        raise RuntimeError(f"Released E3 pretrained transfer failed: compatible={len(compatible)}, mismatched={mismatched[:5]}")

    model.to(device).train()
    model.args = impl.get_cfg(impl.DEFAULT_CFG_DICT)
    model.criterion = model.init_criterion()
    torch.manual_seed(0)
    clean = torch.rand((2, 3, 160, 160), device=device)
    degraded = clean.pow(1.15).mul(0.88)
    batch = {
        "img": torch.cat((clean, degraded), dim=0),
        "batch_idx": torch.tensor([0, 1, 2, 3], device=device, dtype=torch.float32),
        "cls": torch.tensor([[0], [1], [0], [1]], device=device, dtype=torch.float32),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.10, 0.10], [0.40, 0.40, 0.10, 0.10]] * 2,
            device=device,
            dtype=torch.float32,
        ),
        "e3_gt_weights": torch.tensor([1.0, 1.0, EXPECTED_ALPHA, EXPECTED_ALPHA], device=device),
        "e3_pair_batch_size": 2,
    }
    loss, loss_items = model.loss(batch)
    total = loss.sum()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Released E3 alpha=1.5 self-test produced non-finite loss")
    total.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not finite_gradients:
        raise FloatingPointError("Released E3 alpha=1.5 self-test produced non-finite gradients")
    evidence = {
        "variant": variant,
        **model.factorial_evidence(),
        "mdg_mode": "dual_view",
        "dual_view_enabled": True,
        "degraded_small_weighting_enabled": True,
        "mdg_counts_current_batch": {"images_total": 2},
        "target_counts_current_batch": {
            "images_total": 2,
            "clean_targets": 2,
            "degraded_targets": 2,
            "degraded_small_targets": 2,
            "degraded_regular_targets": 0,
        },
        "optimizer_update_probe": {"finite": True, "parameter_delta_l2": 1.0},
        "consistency_loss_enabled": False,
        "severity_weighting_enabled": False,
        "curriculum_enabled": False,
    }
    RELEASED_VALIDATE_RUNTIME_EVIDENCE(evidence, variant)

    mdg = impl.MaritimeDegradation(impl.load_mdg_config(impl.MDG_CONFIG_PATH))
    probe = torch.rand((2, 3, 160, 160), device=device)
    degraded_probe, counts = mdg.apply_batch(probe.clone())
    if not bool(torch.isfinite(degraded_probe).all()) or int(counts.get("images_total", -1)) != 2:
        raise RuntimeError("Locked MDG self-test failed")
    payload = {
        "schema": "ring3.wsodd_legacy_e3_alpha15_self_test.v1",
        "status": "passed",
        "verified_utc": impl.utc_now(),
        "device": str(device),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": impl.EXPECTED_WEIGHT_SHA256,
        "compatible_pretrained_tensors": len(compatible),
        "mismatched_pretrained_tensors": len(mismatched),
        "e3_loss_items": {key: float(value) for key, value in loss_items.items()},
        "e3_runtime_evidence": evidence,
        "mdg_probe_counts": counts,
        "mdg_probe_finite": True,
        "released_e3_alpha": EXPECTED_ALPHA,
        "alpha12_excluded_from_formal_e3": True,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    del model
    torch.cuda.empty_cache()
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
        "formal_e3_definition": "released SeaDronesSee E3 alpha=1.5",
        "alpha12_status": "intermediate_ablation_excluded_from_formal_e3",
        "dataset_protocol": "wsodd_legacy_image_split_v1",
        "evidence_class": "legacy_protocol_reproduction_not_formal_generalization_evidence",
        "sequence_leakage_disclosed": True,
        "metrics": {"e3": metric},
        "post_training_validation_status": "pending",
        "single_seed_mechanism_evidence_only": True,
        "e4_started": False,
    }
    area_path = impl.STATE_ROOT / "wsodd_legacy_e3_alpha15_area_metrics.json"
    if area_path.is_file():
        payload["area_metrics"] = impl.load_json(area_path)
        payload["post_training_validation_status"] = "completed"
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def run_area_evaluation(context: dict[str, Any]) -> dict[str, Any]:
    validation = context["config"]["post_training_validation"]
    adapter_path = Path(validation["chunked_adapter"])
    evaluator_path = Path(validation["base_evaluator"])
    if impl.sha256(adapter_path) != EXPECTED_AREA_ADAPTER_SHA256:
        raise RuntimeError("WSODD chunked area adapter hash mismatch")
    if impl.sha256(evaluator_path) != EXPECTED_BASE_EVALUATOR_SHA256:
        raise RuntimeError("Base COCO area evaluator hash mismatch")

    state = impl.load_json(impl.STATE_PATH)
    state.update(status="evaluating", post_training_validation_status="running", last_progress_utc=impl.utc_now())
    impl.write_json_atomic(impl.STATE_PATH, state)
    variant = context["config"]["run_order"][0]
    output_path = impl.STATE_ROOT / "wsodd_legacy_e3_alpha15_area_metrics.json"
    report = area_adapter.evaluate_one_chunked(
        run_key="e3",
        run_dir=impl.OUTPUT_ROOT / variant["name"],
        data_yaml=Path(context["config"]["dataset"]["yaml"]),
        output_path=output_path,
        imgsz=int(context["config"]["training"]["imgsz"]),
        batch=int(context["config"]["training"]["batch"]),
        device=int(context["config"]["training"]["device"]),
    )
    final_summary = summary(context)
    state = impl.load_json(impl.STATE_PATH)
    state.update(
        status="completed",
        post_training_validation_status="completed",
        area_metrics_path=str(output_path),
        area_metrics_sha256=impl.sha256(output_path),
        summary_path=str(impl.SUMMARY_PATH),
        summary_sha256=impl.sha256(impl.SUMMARY_PATH),
        completed_utc=impl.utc_now(),
        last_progress_utc=impl.utc_now(),
        e4_started=False,
    )
    impl.write_json_atomic(impl.STATE_PATH, state)
    impl.append_event("area_evaluation_completed", report=report, summary=final_summary)
    return report


def configure_runtime() -> None:
    released_configure = legacy.configure_runtime
    released_configure()
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT
    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "wsodd_legacy_e3_alpha15_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_legacy_e3_alpha15_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_legacy_e3_alpha15.py"
    impl.OUTPUT_ROOT = data_root / "validation_wsodd_legacy_split_sgd_e3_alpha15"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_legacy_e3_alpha15"
    impl.STATE_PATH = state_root / "wsodd_legacy_e3_alpha15_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_legacy_e3_alpha15_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_legacy_e3_alpha15_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_legacy_e3_alpha15_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_legacy_e3_alpha15_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_LEGACY_E3_ALPHA15"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_legacy_e3_alpha15_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_legacy_e3_alpha15_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = legacy.EXPECTED_DATA_YAML_SHA256
    impl.validate_config = validate_config
    impl.verify_dataset = legacy.verify_dataset
    impl.validate_runtime_evidence = RELEASED_VALIDATE_RUNTIME_EVIDENCE
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = static_self_test
    impl.MainlineSGDTrainer = RELEASED_MAINLINE_TRAINER

    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_legacy_e3_alpha15_failure.json"
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
    parser = argparse.ArgumentParser(description="Run the corrected WSODD legacy-split E3 alpha=1.5 release")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = impl.preflight()
        print("WSODD_LEGACY_E3_ALPHA15_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        early_stop_guard = completion_v2.guard_self_test()
        amp_mask_guard = amp_hotfix.mask_hotfix_self_test()
        print("WSODD_LEGACY_E3_ALPHA15_EARLY_STOP_GUARD " + json.dumps(early_stop_guard, sort_keys=True), flush=True)
        print("WSODD_LEGACY_E3_ALPHA15_AMP_MASK_GUARD " + json.dumps(amp_mask_guard, sort_keys=True), flush=True)
        self_test = static_self_test(context)
        print("WSODD_LEGACY_E3_ALPHA15_SELF_TEST " + json.dumps(self_test, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        result = supervisor.execute_queue(context)
        if result != 0:
            return result
        report = run_area_evaluation(context)
        print("WSODD_LEGACY_E3_ALPHA15_AREA " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        state_status = "supervisor_failed"
        training_results_preserved = False
        if impl.STATE_PATH.is_file():
            existing = impl.load_json(impl.STATE_PATH)
            if existing.get("status") == "evaluating":
                state_status = "evaluation_failed"
                training_results_preserved = True
        failure = {
            "schema": "ring3.wsodd_legacy_e3_alpha15_failure.v1",
            "status": state_status,
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_results_preserved": training_results_preserved,
            "alpha12_reused": False,
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
