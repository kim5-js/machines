#!/usr/bin/env python3
"""SeaDronesSee source-disjoint E0 replication for seeds 1 and 2.

The completed seed-0 E0 checkpoint is reused only after exact artifact and
recipe verification.  New runs differ from that anchor only by training seed.
"""

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


CONFIG_SCHEMA = "ring3.seadronessee_e0_multiseed_config.v1"
MANIFEST_SCHEMA = "ring3.seadronessee_e0_multiseed_release.v1"
SUPERVISOR_SCHEMA = "ring3.seadronessee_e0_multiseed_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.seadronessee_e0_multiseed_completion.v1"
COMPLETION_EVIDENCE_NAME = "seadronessee_e0_multiseed_completion_evidence.json"
EXPECTED_RUNS = [("e0_seed1", 1), ("e0_seed2", 2)]

SEA_VERIFY_DATASET = impl.verify_dataset
BASE_TRAINER = impl.MainlineSGDTrainer
BASE_SUMMARY = base.summary
BASE_SELF_TEST = base.static_self_test


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected SeaDronesSee E0 multiseed config schema/status")
    runs = config.get("run_order", [])
    observed = [(run.get("key"), run.get("seed")) for run in runs]
    if observed != EXPECTED_RUNS or len({str(run.get("name")) for run in runs}) != 2:
        raise ValueError(f"Unexpected E0 seed matrix: {observed!r}")
    expected = {
        "mdg_mode": "none",
        "dual_view": False,
        "degraded_small_weighting": False,
        "degraded_small_target_weight": 1.0,
        "cbam": False,
        "focal_eiou": False,
        "consistency_loss": False,
    }
    for run in runs:
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
        or policy.get("pair_with_deterministic_E1_same_seed") is not True
    ):
        raise ValueError("Unexpected SeaDronesSee E0 seed policy")
    if int(config.get("training", {}).get("seed", -1)) != 0:
        raise ValueError("The source recipe must retain seed 0 as its anchor")
    source = impl.load_json(Path(config["reused_anchors"]["source_mainline_config"]))
    if config["training"] != source["training"]:
        raise ValueError("E0 multiseed recipe differs from the completed SeaDronesSee mainline")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("SeaDronesSee E0 invariants are incomplete")


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    anchor = anchors["E0-seed0"]
    run_dir = Path(anchor["run_dir"])
    checks = {
        "source_mainline_config": (
            Path(anchors["source_mainline_config"]),
            anchors["source_mainline_config_sha256"],
        ),
        "best_checkpoint": (run_dir / "weights" / "best.pt", anchor["best_weights_sha256"]),
        "results_csv": (run_dir / "results.csv", anchor["results_csv_sha256"]),
        "args_yaml": (run_dir / "args.yaml", anchor["args_yaml_sha256"]),
        "runtime_evidence": (
            run_dir / "pretrained_sgd_mainline_runtime_evidence.json",
            anchor["runtime_evidence_sha256"],
        ),
        "run_metadata": (
            run_dir / "pretrained_sgd_mainline_run_metadata.json",
            anchor["run_metadata_sha256"],
        ),
        "completion_evidence": (
            run_dir / "pretrained_sgd_mainline_completion_evidence.json",
            anchor["completion_evidence_sha256"],
        ),
        "area_report": (Path(anchor["area_report"]), anchor["area_report_sha256"]),
    }
    verified = {
        name: base._verified_file(path, expected, name)
        for name, (path, expected) in checks.items()
    }
    metadata = impl.load_json(run_dir / "pretrained_sgd_mainline_run_metadata.json")
    runtime = impl.load_json(run_dir / "pretrained_sgd_mainline_runtime_evidence.json")
    completion = impl.load_json(run_dir / "pretrained_sgd_mainline_completion_evidence.json")
    area = impl.load_json(Path(anchor["area_report"]))
    required_metadata = {
        "variant_key": "e0",
        "pretrained": True,
        "pretrained_source_sha256": config["weights"]["sha256"],
        "optimizer": "SGD",
        "cross_run_checkpoint_loaded": False,
    }
    required_runtime = {
        "pretrained": True,
        "pretrained_source_sha256": config["weights"]["sha256"],
        "optimizer": "SGD",
        "mdg_mode": "none",
        "dual_view_enabled": False,
        "degraded_small_weighting_enabled": False,
        "consistency_loss_enabled": False,
        "cross_run_checkpoint_loaded": False,
    }
    required_completion = {
        "status": "passed",
        "completion_kind": "early_stopping",
        "normal_training_return_confirmed": True,
        "final_epoch": 125,
        "best_epoch": int(anchor["best_epoch"]),
        "patience_gap": 100,
    }
    for label, payload, required in (
        ("metadata", metadata, required_metadata),
        ("runtime", runtime, required_runtime),
        ("completion", completion, required_completion),
    ):
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in required.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Reused E0 seed0 {label} mismatch: {mismatches}")
    if int(metadata.get("training_kwargs", {}).get("seed", -1)) != 0:
        raise RuntimeError("Reused E0 seed0 metadata has the wrong training seed")
    if area.get("status") != "completed" or int(area.get("best_epoch", -1)) != int(anchor["best_epoch"]):
        raise RuntimeError("Reused E0 seed0 area report is not complete or selects another epoch")
    verified["reuse_policy"] = {
        "seed": 0,
        "run_dir": str(run_dir),
        "training_reused": False,
        "checkpoint_reused_for_evaluation": True,
    }
    return verified


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SELF_TEST(context)
    payload["schema"] = "ring3.seadronessee_e0_multiseed_self_test.v1"
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    payload = BASE_SUMMARY(context)
    payload["schema"] = "ring3.seadronessee_e0_multiseed_training_summary.v1"
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
    base.EXPECTED_CLASSES = 5
    base.EXPECTED_RUNS = EXPECTED_RUNS
    base.validate_config = validate_config
    base.verify_reused_anchors = verify_reused_anchors
    base.static_self_test = static_self_test
    base.summary = summary
    base.wsodd.verify_dataset = SEA_VERIFY_DATASET

    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "seadronessee_e0_multiseed_config.json"
    impl.MANIFEST_PATH = setup_root / "seadronessee_e0_multiseed_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_seadronessee_e0_multiseed.py"
    impl.MDG_CONFIG_PATH = setup_root / "e1_mdg_config.json"
    impl.OUTPUT_ROOT = data_root / "formal_seadronessee_e0_multiseed"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "seadronessee_e0_multiseed"
    impl.STATE_PATH = state_root / "seadronessee_e0_multiseed_state.json"
    impl.EVENTS_PATH = state_root / "seadronessee_e0_multiseed_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "seadronessee_e0_multiseed_protocol.json"
    impl.SUMMARY_PATH = state_root / "seadronessee_e0_multiseed_training_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "seadronessee_e0_multiseed_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_SEADRONESSEE_E0_MULTISEED"
    impl.RUNTIME_EVIDENCE_NAME = "seadronessee_e0_multiseed_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "seadronessee_e0_multiseed_run_metadata.json"
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
    impl.MainlineSGDTrainer = BASE_TRAINER
    impl.verify_args = mainline_entry.verify_args_with_serialized_device

    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "seadronessee_e0_multiseed_failure.json"
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
    parser = argparse.ArgumentParser(description="Run SeaDronesSee source-disjoint E0 seeds 1 and 2")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = base.preflight()
        print("SEADRONESSEE_E0_MULTISEED_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("SEADRONESSEE_E0_MULTISEED_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("SEADRONESSEE_E0_MULTISEED_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.seadronessee_e0_multiseed_failure.v1",
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
