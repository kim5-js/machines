#!/usr/bin/env python3
"""Additive recovery entry for the WSODD E1-Mid seed1/seed2 queue.

The original locked runner completed seed1 successfully, but its adapter
discarded the completion-validation return value.  This entry leaves the
original release untouched, validates and records the completed seed1
artifacts, installs only the missing return-value adapter, and resumes the
same queue so seed2 still starts from the locked pretrained checkpoint.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import train_wsodd_sequence_e1_mid_multiseed as base


RECOVERY_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_recovery.v1"
RECOVERY_MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_mid_multiseed_recovery_release.v1"
EXPECTED_PARENT_RELEASE_SHA256 = "5212739fe08ef947260103b5d478096661a635b391752193567dfef0065fc1ef"
EXPECTED_FAILURE_ERROR = "Completed artifacts failed provenance validation; refusing to resume a completed checkpoint"
EXPECTED_FAILURE_TRACE = "TypeError: 'NoneType' object is not subscriptable"
RECOVERY_MANIFEST_NAME = "wsodd_sequence_e1_mid_multiseed_recovery_v1_manifest.json"
RECOVERY_EVIDENCE_NAME = "wsodd_sequence_e1_mid_multiseed_recovery_v1_evidence.json"
PRE_RECOVERY_STATE_NAME = "wsodd_sequence_e1_mid_multiseed_pre_recovery_state.json"


def verify_recovery_manifest() -> dict[str, Any]:
    manifest_path = base.impl.SETUP_ROOT / RECOVERY_MANIFEST_NAME
    manifest = base.impl.load_json(manifest_path)
    if manifest.get("schema") != RECOVERY_MANIFEST_SCHEMA or manifest.get("status") != "locked_additive_hotfix":
        raise RuntimeError("Unexpected recovery manifest schema/status")
    if manifest.get("parent_release_manifest_sha256") != EXPECTED_PARENT_RELEASE_SHA256:
        raise RuntimeError("Recovery manifest points to a different parent release")
    observed: list[dict[str, str]] = []
    for item in manifest.get("files", []):
        path = base.impl.SETUP_ROOT / str(item["path"])
        digest = base.impl.sha256(path) if path.is_file() else "missing"
        if digest != item.get("sha256"):
            raise RuntimeError(
                f"Recovery release file changed: {item['path']}: expected={item.get('sha256')}, observed={digest}"
            )
        observed.append({"path": str(path), "sha256": digest})
    if len(observed) != 3:
        raise RuntimeError(f"Recovery manifest must lock exactly three files, observed={len(observed)}")
    parent_path = base.impl.SETUP_ROOT / "wsodd_sequence_e1_mid_multiseed_release_manifest.json"
    if base.impl.sha256(parent_path) != EXPECTED_PARENT_RELEASE_SHA256:
        raise RuntimeError("Original E1-Mid multiseed release manifest changed")
    return {
        "path": str(manifest_path),
        "sha256": base.impl.sha256(manifest_path),
        "files": observed,
    }


def returning_validate_completed_variant(
    run_dir: Path,
    variant: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Preserve the original validator and return its completion details."""
    return base.completion_v2.validate_completed_variant_v2(
        run_dir,
        variant,
        base.context_for_variant(context, variant),
    )


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    names = {
        "results_csv": "results.csv",
        "args_yaml": "args.yaml",
        "runtime_evidence": base.impl.RUNTIME_EVIDENCE_NAME,
        "run_metadata": base.impl.RUN_METADATA_NAME,
        "completion_evidence": base.COMPLETION_EVIDENCE_NAME,
        "best_checkpoint": "weights/best.pt",
        "last_checkpoint": "weights/last.pt",
    }
    output: dict[str, str] = {}
    for label, relative in names.items():
        path = run_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Completed seed1 artifact is missing: {path}")
        output[label] = base.impl.sha256(path)
    return output


def prepare_known_failure_recovery(context: dict[str, Any], recovery_release: dict[str, Any]) -> dict[str, Any]:
    state = base.impl.load_json(base.impl.STATE_PATH)
    status = str(state.get("status", ""))
    if status == "supervisor_failed":
        if (
            state.get("current_run_key") != "e1_mid_seed1"
            or int(state.get("current_seed", -1)) != 1
            or int(state.get("current_epoch", -1)) != 150
            or state.get("error") != EXPECTED_FAILURE_ERROR
            or EXPECTED_FAILURE_TRACE not in str(state.get("traceback", ""))
        ):
            raise RuntimeError("Refusing recovery because supervisor_failed does not match the audited missing-return defect")
    elif status not in {"preflight_passed", "recovery_preflight_passed", "starting", "running", "attempt_failed", "completed"}:
        raise RuntimeError(f"Refusing recovery from unsupported queue status: {status!r}")

    if state.get("release_manifest_sha256") != EXPECTED_PARENT_RELEASE_SHA256:
        raise RuntimeError("Existing queue state belongs to a different parent release")
    if state.get("e2_e3_e4_started") is not False or state.get("e4_started") is not False:
        raise RuntimeError("Forbidden E2/E3/E4 state detected")

    seed1 = context["config"]["run_order"][0]
    if seed1.get("key") != "e1_mid_seed1" or int(seed1.get("seed", -1)) != 1:
        raise RuntimeError("Locked first variant is no longer E1-Mid seed1")
    seed1_dir = base.impl.OUTPUT_ROOT / seed1["name"]
    details = returning_validate_completed_variant(seed1_dir, seed1, context)
    if details.get("complete") is not True or details.get("completion_kind") != "max_epochs":
        raise RuntimeError(f"Seed1 is not a valid completed 150-epoch run: {details}")

    pre_state_sha = base.impl.sha256(base.impl.STATE_PATH)
    snapshot_path = base.impl.STATE_PATH.with_name(PRE_RECOVERY_STATE_NAME)
    if not snapshot_path.exists():
        base.impl.write_json_atomic(snapshot_path, state)

    evidence = {
        "schema": RECOVERY_SCHEMA,
        "status": "passed",
        "verified_utc": base.impl.utc_now(),
        "defect": "completion validator adapter omitted return",
        "code_change_scope": "orchestration return value only",
        "training_recipe_changed": False,
        "dataset_protocol_changed": False,
        "seed1_retrained": False,
        "seed2_initialization": "locked pretrained yolov8s.pt fresh initialization",
        "cross_run_checkpoint_initialization": False,
        "parent_release_manifest_sha256": EXPECTED_PARENT_RELEASE_SHA256,
        "recovery_release": recovery_release,
        "pre_recovery_state_path": str(snapshot_path),
        "pre_recovery_state_sha256": pre_state_sha,
        "seed1_run_dir": str(seed1_dir),
        "seed1_completion": details,
        "seed1_artifact_sha256": artifact_hashes(seed1_dir),
        "next_variant": context["config"]["run_order"][1],
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    evidence_path = base.impl.STATE_PATH.with_name(RECOVERY_EVIDENCE_NAME)
    base.impl.write_json_atomic(evidence_path, evidence)

    state["recovered_supervisor_failure"] = {
        "status": status,
        "error_type": state.get("error_type"),
        "error": state.get("error"),
        "traceback": state.get("traceback"),
        "recovered_utc": base.impl.utc_now(),
    }
    for key in ("error_type", "error", "traceback", "numerical_failure"):
        state.pop(key, None)
    state.update(
        status="recovery_preflight_passed",
        last_progress_utc=base.impl.utc_now(),
        recovery_release_manifest_sha256=recovery_release["sha256"],
        recovery_evidence_path=str(evidence_path),
        recovery_evidence_sha256=base.impl.sha256(evidence_path),
    )
    base.impl.write_json_atomic(base.impl.STATE_PATH, state)
    base.impl.append_event(
        "missing_return_recovery_validated",
        variant="e1_mid_seed1",
        completion_kind=details["completion_kind"],
        final_epoch=details["final_epoch"],
        recovery_release_manifest_sha256=recovery_release["sha256"],
        next_variant="e1_mid_seed2",
    )
    return evidence


def install_return_adapter() -> None:
    base.validate_completed_variant = returning_validate_completed_variant
    base.supervisor.validate_completed_variant = returning_validate_completed_variant


def main() -> int:
    base.configure_runtime()
    try:
        recovery_release = verify_recovery_manifest()
        context = base.preflight()
        if context["release"]["sha256"] != EXPECTED_PARENT_RELEASE_SHA256:
            raise RuntimeError("Preflight resolved a different parent release")
        install_return_adapter()
        guard = base.completion_v2.guard_self_test()
        print("WSODD_E1_MID_MULTISEED_RECOVERY_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        evidence = prepare_known_failure_recovery(context, recovery_release)
        print("WSODD_E1_MID_MULTISEED_RECOVERY " + json.dumps(evidence, ensure_ascii=False, sort_keys=True), flush=True)
        return base.supervisor.execute_queue(context)
    except BaseException as exc:
        payload = {
            "schema": RECOVERY_SCHEMA,
            "status": "recovery_failed",
            "failed_utc": base.impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e2_e3_e4_started": False,
            "e4_started": False,
        }
        failure_path = base.impl.STATE_PATH.with_name("wsodd_sequence_e1_mid_multiseed_recovery_v1_failure.json")
        base.impl.write_json_atomic(failure_path, payload)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
