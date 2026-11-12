from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from ultralytics.utils.tal import TaskAlignedAssigner

import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_pretrained_sgd_mainline as wsodd
import wsodd_pretrained_sgd_hotfix_v3_entry as amp_hotfix


CONFIG_SCHEMA = "ring3.wsodd_legacy_split_sgd_mainline_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_legacy_split_sgd_mainline_release_manifest.v1"
PROTOCOL_SCHEMA = "ring3.wsodd_legacy_image_split_protocol.v1"
STATE_SCHEMA = "ring3.wsodd_legacy_split_sgd_mainline_state.v1"
SUMMARY_SCHEMA = "ring3.wsodd_legacy_split_sgd_mainline_summary.v1"
EXPECTED_DATA_YAML_SHA256 = "4284a81cf47aa7d92e945a262485ed61fffc2797afb75e9b6c2faa459aab0a34"
EXPECTED_ANNOTATION_AUDIT_SHA256 = "e53493e4a791595e7e7f18f53a5614f53106f09e6ad61a60b4dd23fb1b2d0efd"
EXPECTED_ALPHA = 1.2
EXPECTED_SPLIT_COUNTS = {"train": 6222, "val": 1245}
EXPECTED_CLASSES = 14
EXPECTED_OBJECTS = 21911
EXPECTED_SEQUENCE_AUDIT = {
    "sequence_groups": 147,
    "contaminated_sequence_groups": 98,
    "images_in_contaminated_groups": 7290,
    "train_images_in_contaminated_groups": 6047,
    "val_images_in_contaminated_groups": 1243,
}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected legacy WSODD pretrained-SGD config schema/status")
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
    expected = [
        ("e0", "none", False, False, 1.0, False, False, False),
        ("e1", "single_view", False, False, 1.0, False, False, False),
        ("e2", "dual_view", True, False, 1.0, False, False, False),
        ("e3", "dual_view", True, True, EXPECTED_ALPHA, False, False, False),
    ]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 4:
        raise ValueError(f"Unexpected legacy WSODD E0-E3 run matrix: {observed!r}")

    proxy = copy.deepcopy(config)
    proxy["schema"] = "ring3.pretrained_sgd_mainline_config.v1"
    for run in proxy["run_order"]:
        run.pop("degraded_small_target_weight", None)
    wsodd.BASE_VALIDATE_CONFIG(proxy)

    dataset = config.get("dataset", {})
    expected_dataset = {
        "yaml_sha256": EXPECTED_DATA_YAML_SHA256,
        "annotation_semantics_audit_sha256": EXPECTED_ANNOTATION_AUDIT_SHA256,
        "protocol": "wsodd_legacy_image_split_v1",
        "train_images": 6222,
        "val_images": 1245,
        "test_images": 0,
        "classes": EXPECTED_CLASSES,
        "sequence_groups": 147,
        "contaminated_sequence_groups": 98,
        "val_images_in_contaminated_groups": 1243,
        "independent_test_available": False,
    }
    mismatches = {
        key: {"expected": value, "observed": dataset.get(key)}
        for key, value in expected_dataset.items()
        if dataset.get(key) != value
    }
    weighting = config.get("small_object_weighting", {})
    if weighting.get("ordinary_target_weight") != 1.0 or weighting.get("degraded_small_target_weight") != EXPECTED_ALPHA:
        mismatches["small_object_weighting"] = weighting
    validation = config.get("post_training_validation", {})
    if validation.get("fixed_split") != "val" or validation.get("fixed_images") != 1245:
        mismatches["post_training_validation"] = validation
    execution = config.get("execution", {})
    if execution.get("output_root") != "/root/autodl-tmp/validation_wsodd_legacy_split_sgd_e0_e3":
        mismatches["output_root"] = execution.get("output_root")
    if mismatches:
        raise ValueError(f"Legacy WSODD reproduction contract changed: {mismatches}")


def _resolve_split_names(data_yaml: Path, split: str) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(data.get("path", data_yaml.parent)))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = Path(str(data[split]))
    source = value if value.is_absolute() else root / value
    if not source.is_dir():
        raise RuntimeError(f"Legacy WSODD {split} image directory is missing: {source}")
    return sorted(path.name for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"})


def verify_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    yaml_path = Path(dataset["yaml"])
    protocol_path = Path(dataset["manifest"])
    audit_path = Path(dataset["annotation_semantics_audit"])
    expected_assets = {
        yaml_path: EXPECTED_DATA_YAML_SHA256,
        audit_path: EXPECTED_ANNOTATION_AUDIT_SHA256,
    }
    for path, expected_hash in expected_assets.items():
        if not path.is_file() or impl.sha256(path) != expected_hash:
            raise RuntimeError(f"Locked legacy WSODD asset is missing or changed: {path}")
    if not protocol_path.is_file():
        raise RuntimeError(f"Legacy WSODD protocol is missing: {protocol_path}")

    process = subprocess.run(
        [sys.executable, str(impl.SETUP_ROOT / "verify_yolo_dataset.py"), str(yaml_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError("Legacy WSODD dataset verification failed:\n" + process.stdout + process.stderr)
    report = json.loads(process.stdout)
    if report.get("label_error_count") or any(report.get("missing_label_counts", {}).values()):
        raise RuntimeError(f"Legacy WSODD verification reported label errors: {report}")
    if any(report.get("counts", {}).get(split) != count for split, count in EXPECTED_SPLIT_COUNTS.items()):
        raise RuntimeError(f"Legacy WSODD split counts changed: {report.get('counts')}")

    protocol = impl.load_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "locked_validation_only":
        raise RuntimeError("Legacy WSODD protocol schema/status changed")
    if protocol.get("protocol") != "wsodd_legacy_image_split_v1":
        raise RuntimeError("Legacy WSODD protocol name changed")
    if protocol.get("split_images") != {"train": 6222, "val": 1245, "test": 0}:
        raise RuntimeError(f"Legacy WSODD protocol split counts changed: {protocol.get('split_images')}")
    sequence = protocol.get("sequence_audit", {})
    for key, expected in EXPECTED_SEQUENCE_AUDIT.items():
        if sequence.get(key) != expected:
            raise RuntimeError(f"Legacy WSODD sequence audit changed for {key}: {sequence.get(key)!r}")
    if sequence.get("exact_filename_path_fileid_hardlink_or_content_duplicates_found") is not False:
        raise RuntimeError("Legacy WSODD exact-file audit changed")

    manifests: dict[str, Any] = {}
    actual_split_names: dict[str, list[str]] = {}
    for split in ("train", "val"):
        expected = protocol["manifests"][split]
        path = protocol_path.parent / str(expected["path"])
        names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        actual = _resolve_split_names(yaml_path, split)
        if names != actual:
            raise RuntimeError(f"Legacy WSODD {split} filenames do not match the locked manifest")
        observed_hash = impl.sha256(path)
        if observed_hash != expected["sha256"] or len(names) != int(expected["lines"]):
            raise RuntimeError(f"Legacy WSODD {split} manifest changed")
        manifests[split] = {"path": str(path), "sha256": observed_hash, "lines": len(names)}
        actual_split_names[split] = actual
    overlap = sorted(set(actual_split_names["train"]) & set(actual_split_names["val"]))
    if overlap:
        raise RuntimeError(f"Legacy WSODD has exact filename overlap: {overlap[:10]}")

    audit = impl.load_json(audit_path)
    conclusion = audit.get("conclusion", {})
    if (
        audit.get("status") != "passed"
        or int(audit.get("objects", -1)) != EXPECTED_OBJECTS
        or audit.get("difficult_values") != {"0": EXPECTED_OBJECTS}
        or audit.get("ignore_like_tags") != {}
        or conclusion.get("dedicated_ignore_or_crowd_semantics_present") is not False
    ):
        raise RuntimeError(f"WSODD annotation semantics audit failed: {audit}")
    report["legacy_protocol"] = {
        "protocol_sha256": impl.sha256(protocol_path),
        "manifests": manifests,
        "sequence_audit": sequence,
        "independent_test_available": False,
        "evidence_class": protocol["evidence_class"],
    }
    report["annotation_semantics"] = audit
    return report


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.wsodd_legacy_split_sgd_mainline_runtime_protocol.v1",
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
        "post_training_validation": context["config"]["post_training_validation"],
        "invariants": context["config"]["invariants"],
        "warning": "Historical image-level split with confirmed sequence leakage; validation-only reproduction",
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
        "run_order": [run["key"] for run in context["config"]["run_order"]],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": 16.0,
        "eta_utc": (now + timedelta(hours=16)).isoformat(),
        "post_training_validation_status": "pending",
        "strict_sequence_disjoint_results_untouched": True,
        "e4_started": False,
    }


def summary(context: dict[str, Any]) -> dict[str, Any]:
    payload = wsodd.summary(context)
    payload["schema"] = SUMMARY_SCHEMA
    payload["dataset_protocol"] = "wsodd_legacy_image_split_v1"
    payload["evidence_class"] = "legacy_protocol_reproduction_not_formal_generalization_evidence"
    payload["sequence_leakage_disclosed"] = True
    payload["independent_test_available"] = False
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def run_area_evaluation(context: dict[str, Any]) -> None:
    state = impl.load_json(impl.STATE_PATH)
    state.update(
        status="evaluating",
        post_training_validation_status="running",
        last_progress_utc=impl.utc_now(),
    )
    impl.write_json_atomic(impl.STATE_PATH, state)
    area_metrics: dict[str, Any] = {}
    evaluator = Path(context["config"]["post_training_validation"]["evaluator"])
    if impl.sha256(evaluator) != context["config"]["post_training_validation"]["evaluator_sha256"]:
        raise RuntimeError("Legacy WSODD evaluator hash mismatch")
    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        output = impl.STATE_ROOT / f"wsodd_legacy_split_sgd_{variant['key']}_area_metrics.json"
        command = [
            sys.executable,
            str(evaluator),
            "--run-dir",
            str(run_dir),
            "--data",
            context["config"]["dataset"]["yaml"],
            "--output",
            str(output),
            "--imgsz",
            "960",
            "--batch",
            "8",
            "--device",
            "0",
        ]
        subprocess.run(command, check=True)
        area_metrics[variant["key"]] = impl.load_json(output)
        state = impl.load_json(impl.STATE_PATH)
        state.update(last_progress_utc=impl.utc_now(), evaluation_completed_runs=list(area_metrics))
        impl.write_json_atomic(impl.STATE_PATH, state)
    summary_payload = impl.load_json(impl.SUMMARY_PATH)
    summary_payload["area_metrics"] = area_metrics
    summary_payload["post_training_validation_status"] = "completed"
    impl.write_json_atomic(impl.SUMMARY_PATH, summary_payload)
    state = impl.load_json(impl.STATE_PATH)
    state.update(
        status="completed",
        post_training_validation_status="completed",
        last_progress_utc=impl.utc_now(),
        completed_utc=impl.utc_now(),
    )
    impl.write_json_atomic(impl.STATE_PATH, state)


def configure_runtime() -> None:
    wsodd.configure_runtime()
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT
    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "wsodd_legacy_split_sgd_mainline_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_legacy_split_sgd_mainline_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_legacy_split_sgd_mainline.py"
    impl.OUTPUT_ROOT = data_root / "validation_wsodd_legacy_split_sgd_e0_e3"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_legacy_split_sgd_mainline"
    impl.STATE_PATH = state_root / "wsodd_legacy_split_sgd_mainline_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_legacy_split_sgd_mainline_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_legacy_split_sgd_mainline_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_legacy_split_sgd_mainline_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_legacy_split_sgd_mainline_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_LEGACY_SPLIT_SGD_MAINLINE"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_legacy_split_sgd_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_legacy_split_sgd_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = EXPECTED_DATA_YAML_SHA256
    impl.validate_config = validate_config
    impl.verify_dataset = verify_dataset
    impl.validate_runtime_evidence = wsodd.validate_runtime_evidence
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = wsodd.static_self_test
    impl.MainlineSGDTrainer = wsodd.WSODDMainlineTrainer
    impl.verify_args = wsodd.mainline_entry.verify_args_with_serialized_device

    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_legacy_split_sgd_mainline_failure.json"
    supervisor.summary = summary
    impl.summary = summary
    TaskAlignedAssigner.forward = amp_hotfix.task_aligned_assigner_forward_bool_mask
    os.environ["RING3_WSODD_E3_HOTFIX"] = amp_hotfix.HOTFIX_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WSODD legacy image-split pretrained-SGD E0-E3 reproduction")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = impl.preflight()
        print("WSODD_LEGACY_SPLIT_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        early_stop_guard = completion_v2.guard_self_test()
        amp_mask_guard = amp_hotfix.mask_hotfix_self_test()
        print("WSODD_LEGACY_SPLIT_EARLY_STOP_GUARD " + json.dumps(early_stop_guard, sort_keys=True), flush=True)
        print("WSODD_LEGACY_SPLIT_AMP_MASK_GUARD " + json.dumps(amp_mask_guard, sort_keys=True), flush=True)
        self_test = wsodd.static_self_test(context)
        print("WSODD_LEGACY_SPLIT_SELF_TEST " + json.dumps(self_test, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        result = supervisor.execute_queue(context)
        if result != 0:
            return result
        run_area_evaluation(context)
        return 0
    except BaseException as exc:
        state_status = "evaluation_failed" if impl.STATE_PATH.is_file() and impl.load_json(impl.STATE_PATH).get("status") == "evaluating" else "supervisor_failed"
        failure = {
            "schema": "ring3.wsodd_legacy_split_sgd_mainline_failure.v1",
            "status": state_status,
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_results_preserved": state_status == "evaluation_failed",
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
