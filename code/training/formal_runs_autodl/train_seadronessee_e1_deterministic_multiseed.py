#!/usr/bin/env python3
"""SeaDronesSee source-disjoint deterministic-image-id E1-Mid seeds 0/1/2."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pretrained_sgd_mainline_entry as mainline_entry
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_sequence_e1_deterministic_multiseed as base
from image_id_deterministic_mdg import (
    ASSIGNMENT_SCHEMA,
    ImageIdDeterministicMaritimeDegradation,
    canonical_image_id,
    summarize_assignments,
)


CONFIG_SCHEMA = "ring3.seadronessee_e1_deterministic_multiseed_config.v1"
MANIFEST_SCHEMA = "ring3.seadronessee_e1_deterministic_multiseed_release.v1"
SUPERVISOR_SCHEMA = "ring3.seadronessee_e1_deterministic_multiseed_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.seadronessee_e1_deterministic_multiseed_completion.v1"
COMPLETION_EVIDENCE_NAME = "seadronessee_e1_deterministic_multiseed_completion_evidence.json"
EXPECTED_RUNS = [
    ("e1_deterministic_seed0", 0),
    ("e1_deterministic_seed1", 1),
    ("e1_deterministic_seed2", 2),
]
EXPECTED_TRAIN_MANIFEST_SHA256 = "3dd90960394116002ff2a31b67921dc54cb75af90823b1f6198ca3f86e2d14cb"
EXPECTED_ASSIGNMENT_MAP_SHA256 = "46cdb900fe60c3b2c57782278f5e24489e575363f8e76f2c4d3a1abef2f63921"

SEA_VERIFY_DATASET = impl.verify_dataset
BASE_TRAINER = base.BASE_TRAINER
BASE_SELF_TEST = base.static_self_test
BASE_SUMMARY = base.summary


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected SeaDronesSee deterministic E1 config schema/status")
    runs = config.get("run_order", [])
    observed = [(run.get("key"), run.get("seed")) for run in runs]
    if observed != EXPECTED_RUNS or len({str(run.get("name")) for run in runs}) != 3:
        raise ValueError(f"Unexpected deterministic E1 seed matrix: {observed!r}")
    assignment = config.get("deterministic_assignment", {})
    required_assignment = {
        "schema": ASSIGNMENT_SCHEMA,
        "namespace": "ring3/seadronessee/source-disjoint-v1/e1-mid/image-id-mdg/v1",
        "image_id_source": "train.tsv second column filename stem",
        "locked_image_shape": [960, 960],
        "expected_assignment_map_sha256": EXPECTED_ASSIGNMENT_MAP_SHA256,
        "independent_of_epoch": True,
        "independent_of_batch_order": True,
        "independent_of_training_seed": True,
    }
    mismatches = {
        key: {"expected": value, "observed": assignment.get(key)}
        for key, value in required_assignment.items()
        if assignment.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Deterministic assignment contract changed: {mismatches}")
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
        run_mismatches = {
            key: {"expected": value, "observed": run.get(key)}
            for key, value in expected_run.items()
            if run.get(key) != value
        }
        if run_mismatches:
            raise ValueError(f"Deterministic E1 definition changed for {run.get('key')}: {run_mismatches}")
    policy = config.get("seed_policy", {})
    if (
        policy.get("aggregate_seeds") != [0, 1, 2]
        or policy.get("train_new_seeds") != [0, 1, 2]
        or policy.get("reuse_completed_seeds") != []
        or policy.get("assignment_map_is_identical_across_seeds") is not True
    ):
        raise ValueError("Unexpected deterministic E1 seed policy")
    if int(config.get("training", {}).get("seed", -1)) != 0:
        raise ValueError("The locked recipe must retain seed 0 as its anchor")
    source = impl.load_json(Path(config["reused_anchors"]["source_mainline_config"]))
    if config["training"] != source["training"]:
        raise ValueError("Deterministic E1 recipe differs from the SeaDronesSee mainline")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("Deterministic E1 invariants are incomplete")


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    checks = {
        "source_mainline_config": (
            Path(anchors["source_mainline_config"]),
            anchors["source_mainline_config_sha256"],
        ),
        "baseline_mdg_config": (
            Path(anchors["baseline_mdg_config"]),
            anchors["baseline_mdg_config_sha256"],
        ),
    }
    verified = {
        name: base._verified_file(path, expected, name)
        for name, (path, expected) in checks.items()
    }
    return verified


def deterministic_assignment_report(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["dataset"]["train_manifest"])
    if not manifest_path.is_file() or impl.sha256(manifest_path) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("Locked SeaDronesSee train manifest is missing or changed")
    filenames: list[str] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 2 or not columns[1].strip():
            raise RuntimeError(f"Invalid SeaDronesSee train manifest row {line_number}: {line!r}")
        filenames.append(columns[1].strip())
    image_ids = [canonical_image_id(value) for value in filenames]
    expected = config["dataset"]["image_id_uniqueness"]
    observed = {
        "total": len(image_ids),
        "unique": len(set(image_ids)),
        "duplicates": len(image_ids) - len(set(image_ids)),
    }
    if any(observed[key] != int(expected[key]) for key in observed):
        raise RuntimeError(f"SeaDronesSee image_id uniqueness changed: {observed}")
    transform = ImageIdDeterministicMaritimeDegradation(
        impl.load_mdg_config(impl.MDG_CONFIG_PATH),
        namespace=config["deterministic_assignment"]["namespace"],
    )
    height, width = config["deterministic_assignment"]["locked_image_shape"]
    first = summarize_assignments(transform, image_ids, height=int(height), width=int(width))
    second = summarize_assignments(transform, reversed(image_ids), height=int(height), width=int(width))
    if first != second:
        raise RuntimeError("Deterministic assignment map changed with manifest ordering")
    expected_assignment = config["deterministic_assignment"]
    if first["assignment_map_sha256"] != expected_assignment["expected_assignment_map_sha256"]:
        raise RuntimeError(f"Deterministic assignment hash changed: {first['assignment_map_sha256']}")
    if first["counts"] != expected_assignment["expected_counts"]:
        raise RuntimeError(f"Deterministic assignment counts changed: {first['counts']}")
    first.update(
        train_manifest_path=str(manifest_path),
        train_manifest_sha256=EXPECTED_TRAIN_MANIFEST_SHA256,
        manifest_filename_column=2,
        repeatability_check="passed",
        batch_order_independence_check="passed",
        same_map_for_seeds=[0, 1, 2],
    )
    return first


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SELF_TEST(context)
    payload["schema"] = "ring3.seadronessee_e1_deterministic_multiseed_self_test.v1"
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SUMMARY(context)
    payload["schema"] = "ring3.seadronessee_e1_deterministic_multiseed_training_summary.v1"
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def configure_runtime() -> None:
    data_root, setup_root, state_root = impl.DATA_ROOT, impl.SETUP_ROOT, impl.STATE_ROOT
    base.CONFIG_SCHEMA = CONFIG_SCHEMA
    base.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    base.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    base.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    base.EXPECTED_DATA_YAML_SHA256 = "66b63cf9f50952faa5abdc83979599f2de88c64178278fb5b2b4c698ed841f7c"
    base.EXPECTED_DATA_MANIFEST_SHA256 = "0030e4921a95d57722db66afb50a094c25fc3e6c00b93db5567fbbb196576360"
    base.EXPECTED_TRAIN_MANIFEST_SHA256 = EXPECTED_TRAIN_MANIFEST_SHA256
    base.EXPECTED_SPLIT_COUNTS = {"train": 8922, "val": 1555}
    base.EXPECTED_CLASSES = 5
    base.EXPECTED_RUNS = EXPECTED_RUNS
    base.validate_config = validate_config
    base.verify_reused_anchors = verify_reused_anchors
    base.deterministic_assignment_report = deterministic_assignment_report
    base.static_self_test = static_self_test
    base.summary = summary
    base.wsodd.verify_dataset = SEA_VERIFY_DATASET

    base.dose_runner.CONFIG_SCHEMA = CONFIG_SCHEMA
    base.dose_runner.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.dose_runner.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    base.dose_runner.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    base.dose_runner.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    base.dose_runner._context_for_variant = base.context_for_variant

    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "seadronessee_e1_deterministic_multiseed_config.json"
    impl.MANIFEST_PATH = setup_root / "seadronessee_e1_deterministic_multiseed_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_seadronessee_e1_deterministic_multiseed.py"
    impl.MDG_CONFIG_PATH = setup_root / "e1_mdg_config.json"
    impl.OUTPUT_ROOT = data_root / "formal_seadronessee_e1_deterministic_multiseed"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "seadronessee_e1_deterministic_multiseed"
    impl.STATE_PATH = state_root / "seadronessee_e1_deterministic_multiseed_state.json"
    impl.EVENTS_PATH = state_root / "seadronessee_e1_deterministic_multiseed_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "seadronessee_e1_deterministic_multiseed_protocol.json"
    impl.SUMMARY_PATH = state_root / "seadronessee_e1_deterministic_multiseed_training_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "seadronessee_e1_deterministic_multiseed_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_SEADRONESSEE_E1_DETERMINISTIC_MULTISEED"
    impl.RUNTIME_EVIDENCE_NAME = "seadronessee_e1_deterministic_multiseed_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "seadronessee_e1_deterministic_multiseed_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = base.EXPECTED_DATA_YAML_SHA256
    impl.EXPECTED_DATA_MANIFEST_SHA256 = base.EXPECTED_DATA_MANIFEST_SHA256
    impl.validate_config = validate_config
    impl.preflight = base.preflight
    impl.verify_dataset = SEA_VERIFY_DATASET
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
    supervisor.RECOVERY_FAILURE_PATH = state_root / "seadronessee_e1_deterministic_multiseed_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = base.validate_completed_variant
    supervisor.summary = summary
    supervisor.install()
    supervisor.run_variant = base.run_variant_seeded
    supervisor.validate_completed_variant = base.validate_completed_variant
    supervisor.summary = summary
    impl.run_variant = base.run_variant_seeded
    impl.summary = summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SeaDronesSee deterministic-image-id E1-Mid seeds 0/1/2")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = base.preflight()
        print("SEADRONESSEE_E1_DETERMINISTIC_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("SEADRONESSEE_E1_DETERMINISTIC_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("SEADRONESSEE_E1_DETERMINISTIC_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.seadronessee_e1_deterministic_multiseed_failure.v1",
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
