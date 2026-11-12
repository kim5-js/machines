#!/usr/bin/env python3
"""FloW source-temporal-disjoint E0 seeds 0/1/2, all fresh."""

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
import train_wsodd_sequence_e0_multiseed as base


CONFIG_SCHEMA = "ring3.flow_e0_multiseed_config.v1"
MANIFEST_SCHEMA = "ring3.flow_e0_multiseed_release.v1"
SUPERVISOR_SCHEMA = "ring3.flow_e0_multiseed_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.flow_e0_multiseed_completion.v1"
COMPLETION_EVIDENCE_NAME = "flow_e0_multiseed_completion_evidence.json"
EXPECTED_RUNS = [("e0_seed0", 0), ("e0_seed1", 1), ("e0_seed2", 2)]
EXPECTED_YAML = "aeb32a2ab114e906797195dff5236823f223f76544fd39b97d1d4c0143144eb2"
EXPECTED_PROTOCOL = "e6eb3e54dca983be36380581ed6fd5d3d1253d80b98f1c30c02ea4428b15f5f7"
EXPECTED_AUDIT = "bd0bac75f0c581c597a0fc83a8f352865dcc487b79993a9724dbca34673819f1"
EXPECTED_SPLIT_RELEASE = "db78a41ea440cb3c66dd7d428cfaff6077174e994311c9cd66221a5827a67761"
EXPECTED_SPLITS = {"train": 653, "val": 203, "test": 344}

FLOW_VERIFY_DATASET = impl.verify_dataset
BASE_TRAINER = impl.MainlineSGDTrainer
BASE_SELF_TEST = base.static_self_test
BASE_STATE_TEMPLATE = base.state_template
BASE_PROTOCOL_PAYLOAD = base.protocol_payload


def _verified(path: Path, expected: str, label: str) -> dict[str, str]:
    observed = impl.sha256(path) if path.is_file() else "missing"
    if observed != expected:
        raise RuntimeError(f"Locked {label} changed: expected={expected}, observed={observed}")
    return {"path": str(path), "sha256": observed}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected FloW E0 config schema/status")
    runs = config.get("run_order", [])
    observed = [(run.get("key"), run.get("seed")) for run in runs]
    if observed != EXPECTED_RUNS or len({str(run.get("name")) for run in runs}) != 3:
        raise ValueError(f"Unexpected FloW E0 seed matrix: {observed!r}")
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
    for run in runs:
        mismatches = {key: {"expected": value, "observed": run.get(key)} for key, value in expected.items() if run.get(key) != value}
        if mismatches:
            raise ValueError(f"FloW E0 definition changed for {run.get('key')}: {mismatches}")
    policy = config.get("seed_policy", {})
    if policy.get("aggregate_seeds") != [0, 1, 2] or policy.get("reuse_completed_seeds") != [] or policy.get("train_new_seeds") != [0, 1, 2]:
        raise ValueError("Unexpected FloW E0 seed policy")
    source = impl.load_json(Path(config["reused_anchors"]["source_mainline_config"]))
    if config.get("training") != source.get("training"):
        raise ValueError("FloW E0 recipe differs from the locked pretrained-SGD mainline")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("FloW E0 invariants are incomplete")


def verify_dataset(config: dict[str, Any]) -> dict[str, Any]:
    report = FLOW_VERIFY_DATASET(config)
    if report.get("counts") != EXPECTED_SPLITS or int(report.get("nc", -1)) != 1:
        raise RuntimeError(f"FloW materialized split changed: {report.get('counts')}, nc={report.get('nc')}")
    dataset = config["dataset"]
    protocol_path = Path(dataset["manifest"])
    audit_path = Path(dataset["audit"])
    release_path = Path(dataset["split_release_manifest"])
    _verified(protocol_path, EXPECTED_PROTOCOL, "FloW protocol")
    _verified(audit_path, EXPECTED_AUDIT, "FloW remote split audit")
    _verified(release_path, EXPECTED_SPLIT_RELEASE, "FloW split release")
    protocol = impl.load_json(protocol_path)
    audit = impl.load_json(audit_path)
    if protocol.get("protocol") != "flow_source_temporal_disjoint_v1" or protocol.get("split_images") != EXPECTED_SPLITS:
        raise RuntimeError("FloW protocol identity or counts changed")
    overlaps = audit.get("cross_split_overlap", {})
    overlap_counts = [entry.get("count") for category in overlaps.values() for entry in category.values()]
    if audit.get("status") != "passed" or audit.get("split_counts") != EXPECTED_SPLITS or audit.get("temporal_order", {}).get("passed") is not True or any(value != 0 for value in overlap_counts):
        raise RuntimeError("FloW strict split audit no longer passes")
    report["source_temporal_protocol"] = {
        "protocol_sha256": EXPECTED_PROTOCOL,
        "audit_sha256": EXPECTED_AUDIT,
        "split_release_sha256": EXPECTED_SPLIT_RELEASE,
        "source_group_overlap": 0,
        "temporal_order_passed": True,
    }
    return report


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    checks = {
        "source_mainline_config": (Path(anchors["source_mainline_config"]), anchors["source_mainline_config_sha256"]),
        "baseline_mdg_config_unused_by_E0": (Path(anchors["baseline_mdg_config_unused_by_E0"]), anchors["baseline_mdg_config_sha256"]),
        "split_release_manifest": (Path(config["dataset"]["split_release_manifest"]), config["dataset"]["split_release_manifest_sha256"]),
        "split_audit": (Path(config["dataset"]["audit"]), config["dataset"]["audit_sha256"]),
    }
    return {name: _verified(path, expected, name) for name, (path, expected) in checks.items()}


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    # configure_runtime() installs this wrapper back onto `base`. Keep and
    # call the pre-install implementation to avoid self-recursion.
    payload = BASE_STATE_TEMPLATE(context)
    payload.update(
        schema="ring3.flow_e0_multiseed_state.v1",
        reused_seed0=False,
        all_three_seeds_fresh=True,
        e1_e2_e3_e4_started=False,
        e4_started=False,
    )
    return payload


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    # See state_template(): base.protocol_payload points here after wiring.
    payload = BASE_PROTOCOL_PAYLOAD(context)
    payload["schema"] = "ring3.flow_e0_multiseed_protocol.v1"
    payload["all_three_seeds_fresh"] = True
    return payload


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SELF_TEST(context)
    payload["schema"] = "ring3.flow_e0_multiseed_self_test.v1"
    payload["all_three_seeds_fresh"] = True
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        details = supervisor.run_completion(run_dir, epochs=int(context["config"]["training"]["epochs"]), patience=int(context["config"]["training"]["patience"]))
        if details.get("complete") is not True:
            metrics[variant["key"]] = {"status": "incomplete", "seed": int(variant["seed"]), "run_dir": str(run_dir), "completion": details}
            continue
        validated = base.validate_completed_variant(run_dir, variant, context)
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
        "schema": "ring3.flow_e0_multiseed_training_summary.v1",
        "created_utc": impl.utc_now(),
        "supervisor_schema": SUPERVISOR_SCHEMA,
        "metrics": metrics,
        "reused_seed0": False,
        "all_three_seeds_fresh": True,
        "post_training_validation_status": "pending",
        "three_seed_aggregation_status": "pending",
        "paired_E1_comparison_status": "pending",
        "mdg_used_by_E0": False,
        "e1_e2_e3_e4_started": False,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def configure_runtime() -> None:
    data_root, setup_root, state_root = impl.DATA_ROOT, impl.SETUP_ROOT, impl.STATE_ROOT
    base.CONFIG_SCHEMA = CONFIG_SCHEMA
    base.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    base.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    base.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    base.EXPECTED_DATA_YAML_SHA256 = EXPECTED_YAML
    base.EXPECTED_DATA_MANIFEST_SHA256 = EXPECTED_PROTOCOL
    base.EXPECTED_CLASSES = 1
    base.EXPECTED_RUNS = EXPECTED_RUNS
    base.validate_config = validate_config
    base.verify_reused_anchors = verify_reused_anchors
    base.protocol_payload = protocol_payload
    base.state_template = state_template
    base.static_self_test = static_self_test
    base.summary = summary
    base.wsodd.verify_dataset = verify_dataset

    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "flow_e0_multiseed_config.json"
    impl.MANIFEST_PATH = setup_root / "flow_e0_multiseed_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_flow_e0_multiseed.py"
    impl.MDG_CONFIG_PATH = setup_root / "e1_mdg_config.json"
    impl.OUTPUT_ROOT = data_root / "formal_flow_e0_multiseed"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "flow_e0_multiseed"
    impl.STATE_PATH = state_root / "flow_e0_multiseed_state.json"
    impl.EVENTS_PATH = state_root / "flow_e0_multiseed_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "flow_e0_multiseed_protocol.json"
    impl.SUMMARY_PATH = state_root / "flow_e0_multiseed_training_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "flow_e0_multiseed_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_FLOW_E0_MULTISEED"
    impl.RUNTIME_EVIDENCE_NAME = "flow_e0_multiseed_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "flow_e0_multiseed_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = EXPECTED_YAML
    impl.EXPECTED_DATA_MANIFEST_SHA256 = EXPECTED_PROTOCOL
    impl.validate_config = validate_config
    impl.preflight = base.preflight
    impl.verify_dataset = verify_dataset
    impl.validate_runtime_evidence = base.validate_runtime_evidence
    impl.verify_resume = base.verify_resume
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
    supervisor.RECOVERY_FAILURE_PATH = state_root / "flow_e0_multiseed_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = base.validate_completed_variant
    supervisor.summary = summary
    supervisor.install()
    supervisor.run_variant = base.run_variant_seeded
    impl.run_variant = base.run_variant_seeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = base.preflight()
        print("FLOW_E0_MULTISEED_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("FLOW_E0_MULTISEED_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("FLOW_E0_MULTISEED_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.flow_e0_multiseed_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e1_e2_e3_e4_started": False,
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
