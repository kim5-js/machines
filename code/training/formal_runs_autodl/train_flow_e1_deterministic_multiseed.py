#!/usr/bin/env python3
"""FloW source-temporal-disjoint deterministic-image-ID E1-Mid seeds 0/1/2."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pretrained_sgd_mainline_entry as mainline_entry
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_flow_e0_multiseed as flow_e0
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_sequence_e1_deterministic_multiseed as base
from image_id_deterministic_mdg import (
    ASSIGNMENT_SCHEMA,
    ImageIdDeterministicMaritimeDegradation,
    canonical_image_id,
    summarize_assignments,
)


CONFIG_SCHEMA = "ring3.flow_e1_deterministic_multiseed_config.v1"
MANIFEST_SCHEMA = "ring3.flow_e1_deterministic_multiseed_release.v1"
SUPERVISOR_SCHEMA = "ring3.flow_e1_deterministic_multiseed_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.flow_e1_deterministic_multiseed_completion.v1"
COMPLETION_EVIDENCE_NAME = "flow_e1_deterministic_multiseed_completion_evidence.json"
EXPECTED_RUNS = [
    ("e1_deterministic_seed0", 0),
    ("e1_deterministic_seed1", 1),
    ("e1_deterministic_seed2", 2),
]
EXPECTED_TRAIN_MANIFEST = "1630e342c7d4d93d676474f44b681cdf5d5707d8c210f59d5aa4b4df7f1b93db"
EXPECTED_ASSIGNMENT_MAP = "fa4c6da9ebf141858a48fb9611e09ff76f6ffb2baf685e13e3c78b40dbf03dc9"

BASE_SELF_TEST = base.static_self_test
BASE_SUMMARY = base.summary


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected FloW deterministic E1 config schema/status")
    runs = config.get("run_order", [])
    observed = [(run.get("key"), run.get("seed")) for run in runs]
    if observed != EXPECTED_RUNS or len({str(run.get("name")) for run in runs}) != 3:
        raise ValueError(f"Unexpected FloW deterministic E1 seed matrix: {observed!r}")
    assignment = config.get("deterministic_assignment", {})
    required_assignment = {
        "schema": ASSIGNMENT_SCHEMA,
        "namespace": "ring3/flow/source-temporal-disjoint-v1/e1-mid/image-id-mdg/v1",
        "image_id_source": "train.txt filename stem",
        "locked_image_shape": [960, 960],
        "expected_assignment_map_sha256": EXPECTED_ASSIGNMENT_MAP,
        "independent_of_epoch": True,
        "independent_of_batch_order": True,
        "independent_of_training_seed": True,
    }
    mismatches = {key: {"expected": value, "observed": assignment.get(key)} for key, value in required_assignment.items() if assignment.get(key) != value}
    if mismatches:
        raise ValueError(f"FloW deterministic assignment contract changed: {mismatches}")
    expected_run = {
        "mdg_scale": 1.0,
        "mdg_config": "/root/autodl-tmp/ring3_setup/autodl/e1_mdg_config.json",
        "mdg_config_sha256": base.EXPECTED_MDG_SHA256,
        "mdg_mode": "single_view",
        "mdg_assignment_mode": "deterministic_by_image_id",
        "mdg_assignment_schema": ASSIGNMENT_SCHEMA,
        "mdg_assignment_namespace": assignment["namespace"],
        "mdg_image_id_source": "filename_stem",
        "dual_view": False,
        "degraded_small_weighting": False,
        "degraded_small_target_weight": 1.0,
        "cbam": False,
        "focal_eiou": False,
        "consistency_loss": False,
    }
    for run in runs:
        run_mismatches = {key: {"expected": value, "observed": run.get(key)} for key, value in expected_run.items() if run.get(key) != value}
        if run_mismatches:
            raise ValueError(f"FloW E1 definition changed for {run.get('key')}: {run_mismatches}")
    policy = config.get("seed_policy", {})
    if policy.get("aggregate_seeds") != [0, 1, 2] or policy.get("train_new_seeds") != [0, 1, 2] or policy.get("reuse_completed_seeds") != [] or policy.get("assignment_map_is_identical_across_seeds") is not True:
        raise ValueError("Unexpected FloW deterministic E1 seed policy")
    source = impl.load_json(Path(config["reused_anchors"]["source_mainline_config"]))
    if config.get("training") != source.get("training"):
        raise ValueError("FloW E1 recipe differs from the locked pretrained-SGD mainline")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("FloW deterministic E1 invariants are incomplete")


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    checks = {
        "source_mainline_config": (Path(anchors["source_mainline_config"]), anchors["source_mainline_config_sha256"]),
        "baseline_mdg_config": (Path(anchors["baseline_mdg_config"]), anchors["baseline_mdg_config_sha256"]),
        "split_release_manifest": (Path(config["dataset"]["split_release_manifest"]), config["dataset"]["split_release_manifest_sha256"]),
        "split_audit": (Path(config["dataset"]["audit"]), config["dataset"]["audit_sha256"]),
    }
    return {name: base._verified_file(path, expected, name) for name, (path, expected) in checks.items()}


def deterministic_assignment_report(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["dataset"]["train_manifest"])
    if not manifest_path.is_file() or impl.sha256(manifest_path) != EXPECTED_TRAIN_MANIFEST:
        raise RuntimeError("Locked FloW train manifest is missing or changed")
    filenames = [line.strip() for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    image_ids = [canonical_image_id(value) for value in filenames]
    expected = config["dataset"]["image_id_uniqueness"]
    observed = {"total": len(image_ids), "unique": len(set(image_ids)), "duplicates": len(image_ids) - len(set(image_ids))}
    if any(observed[key] != int(expected[key]) for key in observed):
        raise RuntimeError(f"FloW image_id uniqueness changed: {observed}")
    transform = ImageIdDeterministicMaritimeDegradation(
        impl.load_mdg_config(impl.MDG_CONFIG_PATH),
        namespace=config["deterministic_assignment"]["namespace"],
    )
    height, width = config["deterministic_assignment"]["locked_image_shape"]
    first = summarize_assignments(transform, image_ids, height=int(height), width=int(width))
    second = summarize_assignments(transform, reversed(image_ids), height=int(height), width=int(width))
    if first != second:
        raise RuntimeError("FloW deterministic assignment map changed with manifest ordering")
    expected_assignment = config["deterministic_assignment"]
    if first["assignment_map_sha256"] != expected_assignment["expected_assignment_map_sha256"] or first["counts"] != expected_assignment["expected_counts"]:
        raise RuntimeError(f"FloW deterministic assignment changed: {first}")
    first.update(
        train_manifest_path=str(manifest_path),
        train_manifest_sha256=EXPECTED_TRAIN_MANIFEST,
        repeatability_check="passed",
        batch_order_independence_check="passed",
        same_map_for_seeds=[0, 1, 2],
    )
    return first


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SELF_TEST(context)
    payload["schema"] = "ring3.flow_e1_deterministic_multiseed_self_test.v1"
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SUMMARY(context)
    payload["schema"] = "ring3.flow_e1_deterministic_multiseed_training_summary.v1"
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def validate_completed_variant(
    run_dir: Path,
    variant: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Preserve the v2 validator's completion details for the supervisor."""

    return completion_v2.validate_completed_variant_v2(
        run_dir,
        variant,
        base.context_for_variant(context, variant),
    )

def configure_runtime() -> None:
    data_root, setup_root, state_root = impl.DATA_ROOT, impl.SETUP_ROOT, impl.STATE_ROOT
    base.CONFIG_SCHEMA = CONFIG_SCHEMA
    base.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    base.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    base.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    base.EXPECTED_DATA_YAML_SHA256 = flow_e0.EXPECTED_YAML
    base.EXPECTED_DATA_MANIFEST_SHA256 = flow_e0.EXPECTED_PROTOCOL
    base.EXPECTED_TRAIN_MANIFEST_SHA256 = EXPECTED_TRAIN_MANIFEST
    base.EXPECTED_SPLIT_COUNTS = flow_e0.EXPECTED_SPLITS
    base.EXPECTED_CLASSES = 1
    base.EXPECTED_RUNS = EXPECTED_RUNS
    base.validate_config = validate_config
    base.verify_reused_anchors = verify_reused_anchors
    base.deterministic_assignment_report = deterministic_assignment_report
    base.static_self_test = static_self_test
    base.summary = summary
    base.wsodd.verify_dataset = flow_e0.verify_dataset

    base.dose_runner.CONFIG_SCHEMA = CONFIG_SCHEMA
    base.dose_runner.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.dose_runner.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    base.dose_runner.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    base.dose_runner.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    base.dose_runner._context_for_variant = base.context_for_variant

    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "flow_e1_deterministic_multiseed_config.json"
    impl.MANIFEST_PATH = setup_root / "flow_e1_deterministic_multiseed_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_flow_e1_deterministic_multiseed.py"
    impl.MDG_CONFIG_PATH = setup_root / "e1_mdg_config.json"
    impl.OUTPUT_ROOT = data_root / "formal_flow_e1_deterministic_multiseed"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "flow_e1_deterministic_multiseed"
    impl.STATE_PATH = state_root / "flow_e1_deterministic_multiseed_state.json"
    impl.EVENTS_PATH = state_root / "flow_e1_deterministic_multiseed_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "flow_e1_deterministic_multiseed_protocol.json"
    impl.SUMMARY_PATH = state_root / "flow_e1_deterministic_multiseed_training_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "flow_e1_deterministic_multiseed_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_FLOW_E1_DETERMINISTIC_MULTISEED"
    impl.RUNTIME_EVIDENCE_NAME = "flow_e1_deterministic_multiseed_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "flow_e1_deterministic_multiseed_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = flow_e0.EXPECTED_YAML
    impl.EXPECTED_DATA_MANIFEST_SHA256 = flow_e0.EXPECTED_PROTOCOL
    impl.validate_config = validate_config
    impl.preflight = base.preflight
    impl.verify_dataset = flow_e0.verify_dataset
    impl.validate_runtime_evidence = base.validate_runtime_evidence
    impl.verify_resume = base.verify_resume
    impl.protocol_payload = base.protocol_payload
    impl.state_template = base.state_template
    impl.static_self_test = static_self_test
    impl.summary = summary
    impl.MainlineSGDTrainer = base.DeterministicWSODDTrainer
    impl.verify_args = mainline_entry.verify_args_with_serialized_device

    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "flow_e1_deterministic_multiseed_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = validate_completed_variant
    supervisor.summary = summary
    supervisor.install()
    supervisor.run_variant = base.run_variant_seeded
    supervisor.validate_completed_variant = validate_completed_variant
    base.validate_completed_variant = validate_completed_variant
    impl.run_variant = base.run_variant_seeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = base.preflight()
        print("FLOW_E1_DETERMINISTIC_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("FLOW_E1_DETERMINISTIC_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("FLOW_E1_DETERMINISTIC_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.flow_e1_deterministic_multiseed_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e2_e3_e4_started": False,
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
