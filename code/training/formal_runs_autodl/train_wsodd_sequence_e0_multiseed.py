#!/usr/bin/env python3
"""Locked WSODD sequence-disjoint E0 replication for seeds 1 and 2.

Seed 0 is an immutable completed anchor. Each new E0 run starts from the same
locked pretrained YOLOv8s checkpoint and changes only the random seed. No
maritime degradation or other experimental module is enabled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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


CONFIG_SCHEMA = "ring3.wsodd_sequence_e0_multiseed_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e0_multiseed_release.v1"
SUPERVISOR_SCHEMA = "ring3.wsodd_sequence_e0_multiseed_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.wsodd_sequence_e0_multiseed_completion.v1"
COMPLETION_EVIDENCE_NAME = "wsodd_sequence_e0_multiseed_completion_evidence.json"

EXPECTED_WEIGHT_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_DATA_YAML_SHA256 = "180965bbca7a5416d467b0d089965db2c8fc22751843795df295d1376c62907e"
EXPECTED_DATA_MANIFEST_SHA256 = "a2edac79e4b7782d6b39d3d91bf22bfa30469811c095310e5f258714fe9869a8"
EXPECTED_ANNOTATION_AUDIT_SHA256 = "e53493e4a791595e7e7f18f53a5614f53106f09e6ad61a60b4dd23fb1b2d0efd"
EXPECTED_MDG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_E1_SUMMARY_SHA256 = "115ea8706ef652992791e9bc8cf3571e54a7c67b62f105a37cf98f363af34f6e"
EXPECTED_CLASSES = 14
EXPECTED_RUNS = [("e0_seed1", 1), ("e0_seed2", 2)]

BASE_VALIDATE_RUNTIME_EVIDENCE = impl.validate_runtime_evidence
BASE_VERIFY_RESUME = impl.verify_resume
BASE_SUPERVISOR_RUN_VARIANT = supervisor.run_variant
BASE_TRAINER = impl.MainlineSGDTrainer


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected WSODD E0 multiseed config schema/status")
    runs = config.get("run_order", [])
    observed = [(run.get("key"), run.get("seed")) for run in runs]
    if observed != EXPECTED_RUNS or len({str(run.get("name")) for run in runs}) != 2:
        raise ValueError(f"Unexpected E0 seed matrix: {observed!r}")
    for run in runs:
        expected = {
            "definition": "Single-view pretrained YOLOv8s baseline without MDG",
            "mdg_mode": "none",
            "dual_view": False,
            "degraded_small_weighting": False,
            "degraded_small_target_weight": 1.0,
            "cbam": False,
            "focal_eiou": False,
            "consistency_loss": False,
        }
        mismatches = {
            key: {"expected": value, "observed": run.get(key)}
            for key, value in expected.items()
            if run.get(key) != value
        }
        if mismatches:
            raise ValueError(f"E0 definition changed for {run.get('key')}: {mismatches}")
    policy = config.get("seed_policy", {})
    if (
        policy.get("aggregate_seeds") != [0, 1, 2]
        or policy.get("reuse_completed_seeds") != [0]
        or policy.get("train_new_seeds") != [1, 2]
        or policy.get("pair_with_E1_Mid_same_seed") is not True
    ):
        raise ValueError("Unexpected E0 three-seed policy")
    if int(config.get("training", {}).get("seed", -1)) != 0:
        raise ValueError("The source training recipe must retain seed 0 as its immutable anchor")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("E0 multiseed invariants are incomplete")


def _verified_file(path: Path, expected: str, label: str) -> dict[str, str]:
    observed = impl.sha256(path) if path.is_file() else "missing"
    if observed != expected:
        raise RuntimeError(f"Locked anchor changed for {label}: expected={expected}, observed={observed}")
    return {"path": str(path), "sha256": observed}


def _same_metrics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("mAP50", "mAP50-95", "APsmall", "APmedium", "APlarge")
    try:
        return all(abs(float(left[key]) - float(right[key])) <= 1e-12 for key in keys)
    except (KeyError, TypeError, ValueError):
        return False


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    e0 = anchors["E0-seed0"]
    e1_summary = anchors["E1-Mid-multiseed-summary"]
    checks = {
        "source_mainline_config": (
            Path(anchors["source_mainline_config"]),
            anchors["source_mainline_config_sha256"],
        ),
        "source_mainline_release_manifest": (
            Path(anchors["source_mainline_release_manifest"]),
            anchors["source_mainline_release_manifest_sha256"],
        ),
        "baseline_mdg_config_unused_by_E0": (
            Path(anchors["baseline_mdg_config"]),
            anchors["baseline_mdg_config_sha256"],
        ),
        "fixed_test_summary": (
            Path(anchors["fixed_test_summary"]),
            anchors["fixed_test_summary_sha256"],
        ),
        "E0_seed0_best": (
            Path(e0["run_dir"]) / "weights" / "best.pt",
            e0["best_weights_sha256"],
        ),
        "E1_Mid_multiseed_summary": (
            Path(e1_summary["path"]),
            e1_summary["sha256"],
        ),
    }
    verified = {
        name: _verified_file(path, expected, name)
        for name, (path, expected) in checks.items()
    }
    source = impl.load_json(Path(anchors["source_mainline_config"]))
    if config["training"] != source["training"]:
        raise RuntimeError("The non-seed training recipe differs from the completed fixed-split mainline")
    if config["dataset"]["yaml_sha256"] != source["dataset"]["yaml_sha256"]:
        raise RuntimeError("Dataset YAML differs from the completed fixed-split mainline")

    e1_payload = impl.load_json(Path(e1_summary["path"]))
    if (
        e1_payload.get("status") != "completed"
        or e1_payload.get("dataset_protocol") != "wsodd_sequence_disjoint_v1"
        or int(e1_payload.get("fixed_test_images", -1)) != 2406
        or e1_payload.get("same_evaluator_and_prediction_parameters_for_all_seeds") is not True
        or e1_payload.get("e2_e3_e4_started") is not False
        or e1_payload.get("e4_started") is not False
    ):
        raise RuntimeError("E1-Mid three-seed summary is not a valid immutable pairing anchor")
    rows = e1_payload.get("E1_Mid_per_seed", [])
    if [int(row.get("seed", -1)) for row in rows] != [0, 1, 2]:
        raise RuntimeError("E1-Mid summary does not contain seeds 0, 1, and 2 in order")
    if not _same_metrics(e1_payload.get("E0_seed0_reference", {}), e0["fixed_test"]):
        raise RuntimeError("E1-Mid summary references a different E0 seed0 fixed-test result")
    verified["E1_Mid_seed_rows"] = {
        "seeds": [0, 1, 2],
        "same_prediction_parameters": True,
        "summary_sha256": EXPECTED_E1_SUMMARY_SHA256,
    }
    return verified


def seed_set_sha256(runs: list[dict[str, Any]]) -> str:
    payload = {
        str(run["key"]): {"seed": int(run["seed"]), "mdg_mode": str(run["mdg_mode"])}
        for run in runs
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        raise RuntimeError(
            f"Locked runtime changed: ultralytics={ultralytics.__version__}, torch={torch.__version__}"
        )
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Locked RTX 4090 runtime is unavailable")
    weight_path = Path(config["weights"]["path"])
    if not weight_path.is_file() or impl.sha256(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt is missing or changed")
    if impl.sha256(impl.MDG_CONFIG_PATH) != EXPECTED_MDG_SHA256:
        raise RuntimeError("Historical MDG anchor changed; E0 does not load it during training")
    dataset_report = wsodd.verify_dataset(config)
    anchor_report = verify_reused_anchors(config)
    seed_hash = seed_set_sha256(config["run_order"])
    return {
        "config": config,
        "config_sha256": impl.sha256(impl.CONFIG_PATH),
        "release": release,
        "historical_lock_sha256": EXPECTED_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_SHA256,
        "mdg_used_by_E0": False,
        "seed_set_sha256": seed_hash,
        "dataset_report": dataset_report,
        "reused_anchors": anchor_report,
        "environment": {
            "python": impl.sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


def context_for_variant(context: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(context)
    output["config"]["training"]["seed"] = int(variant["seed"])
    output["active_seed"] = int(variant["seed"])
    return output


def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    BASE_VALIDATE_RUNTIME_EVIDENCE(evidence, variant)
    required = {
        "variant": variant,
        "mdg_mode": "none",
        "dual_view_enabled": False,
        "degraded_small_weighting_enabled": False,
        "cbam_enabled": False,
        "focal_eiou_enabled": False,
        "no_consistency_loss": True,
        "consistency_loss_enabled": False,
        "severity_weighting_enabled": False,
        "curriculum_enabled": False,
    }
    mismatches = {
        key: {"expected": value, "observed": evidence.get(key)}
        for key, value in required.items()
        if evidence.get(key) != value
    }
    if evidence.get("mdg_counts_current_batch") != {}:
        mismatches["mdg_counts_current_batch"] = {
            "expected": {},
            "observed": evidence.get("mdg_counts_current_batch"),
        }
    if mismatches:
        raise RuntimeError(f"E0 runtime wiring changed: {mismatches}")


def verify_resume(run_dir: Path, variant: dict[str, Any], context: dict[str, Any]) -> None:
    seeded_context = context_for_variant(context, variant)
    BASE_VERIFY_RESUME(run_dir, variant, seeded_context)
    metadata = impl.load_json(run_dir / impl.RUN_METADATA_NAME)
    if metadata.get("variant") != variant:
        raise RuntimeError("Refusing resume because the recorded E0 seed variant differs")
    if int(metadata.get("training_kwargs", {}).get("seed", -1)) != int(variant["seed"]):
        raise RuntimeError("Refusing resume because the recorded training seed differs")


def run_variant_seeded(
    variant: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    state.update(current_seed=int(variant["seed"]), last_progress_utc=impl.utc_now())
    impl.write_json_atomic(impl.STATE_PATH, state)
    return BASE_SUPERVISOR_RUN_VARIANT(variant, context_for_variant(context, variant), state)


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.wsodd_sequence_e0_multiseed_protocol.v1",
        "status": "active_locked",
        "created_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "mdg_used_by_E0": False,
        "seed_set_sha256": context["seed_set_sha256"],
        "dataset_report": context["dataset_report"],
        "reused_anchors": context["reused_anchors"],
        "environment": context["environment"],
        "run_order": context["config"]["run_order"],
        "training_anchor": context["config"]["training"],
        "seed_policy": context["config"]["seed_policy"],
        "invariants": context["config"]["invariants"],
    }


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.wsodd_sequence_e0_multiseed_state.v1",
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "seed_set_sha256": context["seed_set_sha256"],
        "protocol_sha256": impl.sha256(impl.PROTOCOL_PATH),
        "run_order": [run["key"] for run in context["config"]["run_order"]],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_seed": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": 4.0,
        "eta_utc": (now + timedelta(hours=4)).isoformat(),
        "post_training_validation_status": "pending",
        "reused_seed0": True,
        "mdg_used_by_E0": False,
        "e1_e2_e3_e4_started": False,
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
        compatible = [
            key
            for key, tensor in source_state.items()
            if key in target_state and target_state[key].shape == tensor.shape
        ]
        mismatched = [
            key
            for key in compatible
            if not torch.equal(target_state[key].cpu(), source_state[key].cpu())
        ]
        if mismatched or len(compatible) < 300:
            raise RuntimeError(
                f"Pretrained transfer failed for {variant['key']}: "
                f"compatible={len(compatible)}, mismatched={mismatched[:5]}"
            )
        seed = int(variant["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        model.to(device).train()
        model.args = impl.get_cfg(impl.DEFAULT_CFG_DICT)
        model.criterion = model.init_criterion()
        synthetic = {
            "variant": variant,
            **model.factorial_evidence(),
            "mdg_mode": "none",
            "dual_view_enabled": False,
            "degraded_small_weighting_enabled": False,
            "mdg_counts_current_batch": {},
            "target_counts_current_batch": {"images_total": 2, "clean_targets": 2},
            "optimizer_update_probe": {"finite": True, "parameter_delta_l2": 1.0},
            "no_consistency_loss": True,
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
        }
        validate_runtime_evidence(synthetic, variant)
        tested.append(
            {
                "variant": variant,
                "seed": seed,
                "compatible_pretrained_tensors": len(compatible),
                "mismatched_pretrained_tensors": len(mismatched),
                "mdg_mode": "none",
                "mdg_used": False,
                "model_evidence": model.factorial_evidence(),
            }
        )
        del model
        torch.cuda.empty_cache()
    payload = {
        "schema": "ring3.wsodd_sequence_e0_multiseed_self_test.v1",
        "status": "passed",
        "verified_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": EXPECTED_WEIGHT_SHA256,
        "tested_variants": tested,
        "mdg_used_by_E0": False,
        "e1_e2_e3_e4_started": False,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def validate_completed_variant(
    run_dir: Path,
    variant: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    return completion_v2.validate_completed_variant_v2(
        run_dir,
        variant,
        context_for_variant(context, variant),
    )


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
            metrics[variant["key"]] = {
                "status": "incomplete",
                "seed": int(variant["seed"]),
                "run_dir": str(run_dir),
                "completion": details,
            }
            continue
        validated = validate_completed_variant(run_dir, variant, context)
        metrics[variant["key"]] = {
            "status": "completed",
            "label": variant["label"],
            "seed": int(variant["seed"]),
            "mdg_mode": "none",
            "run_dir": str(run_dir),
            "completion_kind": validated["completion_kind"],
            "final_epoch": validated["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": validated["final_row"],
        }
    payload = {
        "schema": "ring3.wsodd_sequence_e0_multiseed_training_summary.v1",
        "created_utc": impl.utc_now(),
        "supervisor_schema": SUPERVISOR_SCHEMA,
        "metrics": metrics,
        "reused_seed0": context["config"]["reused_anchors"]["E0-seed0"],
        "post_training_validation_status": "pending",
        "three_seed_aggregation_status": "pending",
        "paired_E1_Mid_comparison_status": "pending",
        "mdg_used_by_E0": False,
        "e1_e2_e3_e4_started": False,
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
    impl.CONFIG_PATH = setup_root / "wsodd_sequence_e0_multiseed_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_sequence_e0_multiseed_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_sequence_e0_multiseed.py"
    impl.MDG_CONFIG_PATH = setup_root / "e1_mdg_config.json"
    impl.OUTPUT_ROOT = data_root / "formal_wsodd_sequence_e0_multiseed"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_sequence_e0_multiseed"
    impl.STATE_PATH = state_root / "wsodd_sequence_e0_multiseed_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_sequence_e0_multiseed_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_sequence_e0_multiseed_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_sequence_e0_multiseed_training_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_sequence_e0_multiseed_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_SEQUENCE_E0_MULTISEED"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_sequence_e0_multiseed_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_sequence_e0_multiseed_run_metadata.json"
    impl.EXPECTED_HISTORICAL_LOCK_SHA256 = EXPECTED_LOCK_SHA256
    impl.EXPECTED_AMENDMENT_SHA256 = EXPECTED_AMENDMENT_SHA256
    impl.EXPECTED_WEIGHT_SHA256 = EXPECTED_WEIGHT_SHA256
    impl.EXPECTED_DATA_YAML_SHA256 = EXPECTED_DATA_YAML_SHA256
    impl.EXPECTED_DATA_MANIFEST_SHA256 = EXPECTED_DATA_MANIFEST_SHA256
    impl.EXPECTED_MDG_SHA256 = EXPECTED_MDG_SHA256
    impl.validate_config = validate_config
    impl.preflight = preflight
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
    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_sequence_e0_multiseed_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = validate_completed_variant
    supervisor.summary = summary
    supervisor.install()
    supervisor.run_variant = run_variant_seeded
    supervisor.validate_completed_variant = validate_completed_variant
    supervisor.summary = summary
    impl.run_variant = run_variant_seeded
    impl.summary = summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run locked WSODD fixed-split E0 seeds 1 and 2"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = preflight()
        print(
            "WSODD_E0_MULTISEED_PREFLIGHT "
            + json.dumps(context, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print(
            "WSODD_E0_MULTISEED_EARLY_STOP_GUARD "
            + json.dumps(guard, sort_keys=True),
            flush=True,
        )
        result = static_self_test(context)
        print(
            "WSODD_E0_MULTISEED_SELF_TEST "
            + json.dumps(result, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.wsodd_sequence_e0_multiseed_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "mdg_used_by_E0": False,
            "e1_e2_e3_e4_started": False,
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, failure)
        if impl.STATE_PATH.is_file():
            state = impl.load_json(impl.STATE_PATH)
            state.update(
                status="supervisor_failed",
                last_progress_utc=impl.utc_now(),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            impl.write_json_atomic(impl.STATE_PATH, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
