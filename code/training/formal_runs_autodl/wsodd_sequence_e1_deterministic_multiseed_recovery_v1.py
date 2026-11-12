#!/usr/bin/env python3
"""Audited additive recovery for deterministic-image-id E1 seeds 0/1/2.

Seed 0 finished all 150 epochs under the sealed parent release.  The parent
runner then stopped because its thin completion-validator adapter discarded
the validated completion-details return value.  This entry leaves the parent
release and the completed seed-0 artifacts untouched, installs only the
missing return-value adapter, records the audited incident, and resumes the
same queue.  The supervisor will validate and register seed 0 without
retraining it, then start seeds 1 and 2 from the locked pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import train_wsodd_sequence_e1_deterministic_multiseed as base


RECOVERY_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_multiseed_recovery.v1"
RECOVERY_MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_multiseed_recovery_release.v1"
EXPECTED_PARENT_RELEASE_SHA256 = "1268eef8ea6cfb974ff806424bbea7e844440f65f2abd68de31cd4268ae2dd71"
EXPECTED_FAILURE_ERROR = "Completed artifacts failed provenance validation; refusing to resume a completed checkpoint"
EXPECTED_FAILURE_TRACE = "TypeError: 'NoneType' object is not subscriptable"
RECOVERY_MANIFEST_NAME = "wsodd_sequence_e1_deterministic_multiseed_recovery_v1_manifest.json"
RECOVERY_EVIDENCE_NAME = "wsodd_sequence_e1_deterministic_multiseed_recovery_v1_evidence.json"
PRE_RECOVERY_STATE_NAME = "wsodd_sequence_e1_deterministic_multiseed_pre_recovery_state.json"


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
    parent_path = base.impl.SETUP_ROOT / "wsodd_sequence_e1_deterministic_multiseed_release_manifest.json"
    if base.impl.sha256(parent_path) != EXPECTED_PARENT_RELEASE_SHA256:
        raise RuntimeError("Original deterministic multiseed release manifest changed")
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
    """Run the parent release validator and retain its completion details."""
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
            raise RuntimeError(f"Completed seed0 artifact is missing: {path}")
        output[label] = base.impl.sha256(path)
    return output


def inspect_known_failure(context: dict[str, Any], recovery_release: dict[str, Any]) -> dict[str, Any]:
    state = base.impl.load_json(base.impl.STATE_PATH)
    status = str(state.get("status", ""))
    if status == "supervisor_failed":
        if (
            state.get("current_run_key") != "e1_deterministic_seed0"
            or int(state.get("current_seed", -1)) != 0
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

    seed0 = context["config"]["run_order"][0]
    if seed0.get("key") != "e1_deterministic_seed0" or int(seed0.get("seed", -1)) != 0:
        raise RuntimeError("Locked first variant is no longer deterministic E1 seed0")
    seed0_dir = base.impl.OUTPUT_ROOT / seed0["name"]
    details = returning_validate_completed_variant(seed0_dir, seed0, context)
    if details.get("complete") is not True or details.get("completion_kind") != "max_epochs":
        raise RuntimeError(f"Seed0 is not a valid completed 150-epoch run: {details}")

    return {
        "state": state,
        "status": status,
        "seed0": seed0,
        "seed0_dir": seed0_dir,
        "details": details,
        "recovery_release": recovery_release,
        "artifact_hashes": artifact_hashes(seed0_dir),
    }


def prepare_known_failure_recovery(audit: dict[str, Any]) -> dict[str, Any]:
    state = audit["state"]
    seed0_dir = audit["seed0_dir"]
    details = audit["details"]
    recovery_release = audit["recovery_release"]
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
        "deterministic_assignment_changed": False,
        "seed0_retrained": False,
        "seed1_seed2_initialization": "locked pretrained yolov8s.pt fresh initialization",
        "cross_run_checkpoint_initialization": False,
        "parent_release_manifest_sha256": EXPECTED_PARENT_RELEASE_SHA256,
        "recovery_release": recovery_release,
        "pre_recovery_state_path": str(snapshot_path),
        "pre_recovery_state_sha256": pre_state_sha,
        "seed0_run_dir": str(seed0_dir),
        "seed0_completion": details,
        "seed0_best_row": base.impl.best_row(seed0_dir / "results.csv"),
        "seed0_artifact_sha256": audit["artifact_hashes"],
        "next_variant": "e1_deterministic_seed1",
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    evidence_path = base.impl.STATE_PATH.with_name(RECOVERY_EVIDENCE_NAME)
    base.impl.write_json_atomic(evidence_path, evidence)

    state["recovered_supervisor_failure"] = {
        "status": audit["status"],
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
        variant="e1_deterministic_seed0",
        completion_kind=details["completion_kind"],
        final_epoch=details["final_epoch"],
        recovery_release_manifest_sha256=recovery_release["sha256"],
        next_variant="e1_deterministic_seed1",
    )
    return evidence


def install_return_adapter() -> None:
    base.validate_completed_variant = returning_validate_completed_variant
    base.supervisor.validate_completed_variant = returning_validate_completed_variant


def audit_payload(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECOVERY_SCHEMA,
        "status": "audit_passed",
        "parent_release_manifest_sha256": EXPECTED_PARENT_RELEASE_SHA256,
        "recovery_release_manifest_sha256": audit["recovery_release"]["sha256"],
        "pre_recovery_status": audit["status"],
        "seed0_run_dir": str(audit["seed0_dir"]),
        "seed0_completion": audit["details"],
        "seed0_best_row": base.impl.best_row(audit["seed0_dir"] / "results.csv"),
        "seed0_artifact_sha256": audit["artifact_hashes"],
        "seed0_retrained": False,
        "next_variant": "e1_deterministic_seed1",
        "training_recipe_changed": False,
        "dataset_protocol_changed": False,
        "deterministic_assignment_changed": False,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and resume deterministic E1 multiseed missing-return failure")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    base.configure_runtime()
    try:
        recovery_release = verify_recovery_manifest()
        context = base.preflight()
        if context["release"]["sha256"] != EXPECTED_PARENT_RELEASE_SHA256:
            raise RuntimeError("Preflight resolved a different parent release")
        install_return_adapter()
        guard = base.completion_v2.guard_self_test()
        print("WSODD_E1_DETERMINISTIC_MULTISEED_RECOVERY_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        audit = inspect_known_failure(context, recovery_release)
        print(
            "WSODD_E1_DETERMINISTIC_MULTISEED_RECOVERY_AUDIT "
            + json.dumps(audit_payload(audit), ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        if args.audit_only:
            return 0
        evidence = prepare_known_failure_recovery(audit)
        print(
            "WSODD_E1_DETERMINISTIC_MULTISEED_RECOVERY "
            + json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
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
        failure_path = base.impl.STATE_PATH.with_name("wsodd_sequence_e1_deterministic_multiseed_recovery_v1_failure.json")
        base.impl.write_json_atomic(failure_path, payload)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
