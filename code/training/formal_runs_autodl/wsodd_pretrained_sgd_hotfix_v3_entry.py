#!/usr/bin/env python3
"""Recover WSODD E3 after the AMP empty-target mask/resume audit defect.

This is a narrow runtime hotfix.  It leaves the locked base release untouched,
normalizes the task assigner's foreground mask to bool, accepts only the
canonical run's own last.pt in resumed args, and records an audited recovery
before restoring the E3 attempt budget for this corrected code release.
"""

from __future__ import annotations

import functools
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics.utils.tal import TaskAlignedAssigner

import train_wsodd_pretrained_sgd_mainline as wsodd


HOTFIX_SCHEMA = "ring3.wsodd_e3_amp_resume_hotfix.v3"
HOTFIX_MANIFEST_SCHEMA = "ring3.wsodd_e3_amp_resume_hotfix_manifest.v3"
HOTFIX_MANIFEST_NAME = "wsodd_pretrained_sgd_hotfix_v3_manifest.json"
RECOVERY_AUDIT_NAME = "wsodd_e3_hotfix_v3_recovery_audit.json"
EXPECTED_BASE_MANIFEST_SHA256 = "fe106f29d4bbd424bb3962d8a459b151e6e5810e50cc8eefebef35e2ba0b117e"
EXPECTED_V2_MANIFEST_SHA256 = "9004d4f613164e130d3471b2c5f25d177c18b7f356f251fc8093793b80076daa"
EXPECTED_E3_SOURCE_SHA256 = "5df5e0d4b39f17c0abb08288ff0d826b4894f6245bdd90dd6be6458590ecd67b"
BASE_CONFIG_SCHEMA = "ring3.pretrained_sgd_mainline_config.v1"
EXPECTED_FAILURE_FRAGMENT = 'bitwise_and_cuda" not implemented for \'Half\''
EXPECTED_RESUME_MODEL_FRAGMENT = "found '/root/autodl-tmp/formal_wsodd_pretrained_sgd_mainline/"
EXPECTED_SELF_TEST_PREFLIGHT_ERROR = "Unexpected pretrained-SGD config schema/status"

_released_configure_runtime = wsodd.configure_runtime
_released_execute_queue = wsodd.supervisor.execute_queue
_released_validate_config = wsodd.validate_config
_original_assigner_forward = TaskAlignedAssigner.forward


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hotfix_release() -> dict[str, Any]:
    setup_root = Path(__file__).resolve().parent
    manifest_path = setup_root / HOTFIX_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != HOTFIX_MANIFEST_SCHEMA or manifest.get("status") != "locked":
        raise RuntimeError("WSODD E3 hotfix-v3 manifest is not locked or has the wrong schema")
    if manifest.get("base_release_manifest_sha256") != EXPECTED_BASE_MANIFEST_SHA256:
        raise RuntimeError("WSODD E3 hotfix-v3 references the wrong base release")
    if manifest.get("v2_hotfix_manifest_sha256") != EXPECTED_V2_MANIFEST_SHA256:
        raise RuntimeError("WSODD E3 hotfix-v3 references the wrong v2 release")
    if manifest.get("training_recipe_changed") is not False or manifest.get("dataset_protocol_changed") is not False:
        raise RuntimeError("WSODD E3 hotfix-v3 must not change the training recipe or dataset protocol")

    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 5:
        raise RuntimeError("WSODD E3 hotfix-v3 manifest must contain exactly five files")
    checked: list[dict[str, str]] = []
    for entry in entries:
        path = setup_root / str(entry["path"])
        observed = sha256(path) if path.is_file() else None
        expected = str(entry["sha256"])
        if observed != expected:
            raise RuntimeError(f"WSODD E3 hotfix-v3 file hash mismatch: {path}: {observed} != {expected}")
        checked.append({"path": str(path), "sha256": observed})
    return {
        "schema": HOTFIX_SCHEMA,
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checked_files": checked,
        "training_recipe_changed": False,
        "dataset_protocol_changed": False,
    }


def coerce_assigner_mask(output: Any) -> tuple[Any, ...]:
    if not isinstance(output, (tuple, list)) or len(output) != 5:
        raise RuntimeError("Unexpected TaskAlignedAssigner output while applying WSODD E3 hotfix-v3")
    values = list(output)
    mask = values[3]
    if not isinstance(mask, torch.Tensor):
        raise RuntimeError("TaskAlignedAssigner foreground mask is not a tensor")
    values[3] = mask.to(dtype=torch.bool)
    return tuple(values)


@functools.wraps(_original_assigner_forward)
def task_aligned_assigner_forward_bool_mask(self: TaskAlignedAssigner, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return coerce_assigner_mask(_original_assigner_forward(self, *args, **kwargs))


def mask_hotfix_self_test() -> dict[str, Any]:
    half_mask = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float16)
    sentinels = (torch.tensor(0), torch.tensor(1), torch.tensor(2), half_mask, torch.tensor(4))
    output = coerce_assigner_mask(sentinels)
    passed = output[3].dtype == torch.bool and output[3].tolist() == [[False, True, False]]
    if not passed:
        raise RuntimeError("WSODD E3 hotfix-v3 mask regression self-test failed")
    return {
        "schema": HOTFIX_SCHEMA,
        "status": "passed",
        "input_mask_dtype": str(half_mask.dtype),
        "output_mask_dtype": str(output[3].dtype),
        "mask_values_preserved": True,
    }


def validate_config_with_base_schema_isolation(config: dict[str, Any]) -> None:
    """Retain the released v2 WSODD/base-schema isolation behavior."""

    original_schema = wsodd.impl.CONFIG_SCHEMA
    try:
        wsodd.impl.CONFIG_SCHEMA = BASE_CONFIG_SCHEMA
        _released_validate_config(config)
    finally:
        wsodd.impl.CONFIG_SCHEMA = original_schema


def verify_args_allow_canonical_own_last_pt(
    run_dir: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> None:
    """Verify every locked arg and narrowly allow this run's own resume checkpoint."""

    impl = wsodd.impl
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise RuntimeError(f"Run args.yaml is missing: {run_dir}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = impl.training_kwargs(config, variant)
    expected["model"] = config["weights"]["path"]
    ignored = {"project", "name", "exist_ok", "verbose"}
    mismatches: list[str] = []
    canonical_last = (run_dir / "weights" / "last.pt").resolve()

    for key, value in expected.items():
        if key in ignored:
            continue
        observed = args.get(key)
        if key == "model":
            observed_path = Path(str(observed)).resolve() if observed else None
            equal = observed == value or (
                canonical_last.is_file() and observed_path == canonical_last and run_dir.resolve() == (impl.OUTPUT_ROOT / variant["name"]).resolve()
            )
        elif key == "device":
            equal = str(observed) == str(value)
        elif isinstance(value, float) and isinstance(observed, (float, int)):
            equal = math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == value
        if not equal:
            mismatches.append(f"{key}: expected {value!r}, found {observed!r}")
    if mismatches:
        raise RuntimeError("Run args differ from the locked SGD recipe:\n" + "\n".join(mismatches))


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copy2(source, destination / source.name)


def recover_exact_e3_failure(context: dict[str, Any]) -> dict[str, Any]:
    """Audit and reset only the exhausted attempts caused by the fixed code defect."""

    impl = wsodd.impl
    state_path = impl.STATE_PATH
    if not state_path.is_file():
        return {"schema": HOTFIX_SCHEMA, "status": "not_needed", "reason": "state_missing"}
    state = impl.load_json(state_path)
    if state.get("status") != "supervisor_failed":
        return {"schema": HOTFIX_SCHEMA, "status": "not_needed", "reason": f"state_{state.get('status')}"}

    variants = {variant["key"]: variant for variant in context["config"]["run_order"]}
    variant = variants.get("e3")
    if variant is None:
        raise RuntimeError("WSODD E3 hotfix-v3 cannot find the locked E3 variant")
    run_dir = impl.OUTPUT_ROOT / variant["name"]
    traceback_text = str(state.get("traceback", ""))
    error_text = str(state.get("error", ""))
    recognized_outer_error = (
        (
            "Run args differ from the locked SGD recipe" in error_text
            and EXPECTED_RESUME_MODEL_FRAGMENT in error_text
        )
        or error_text == EXPECTED_SELF_TEST_PREFLIGHT_ERROR
    )
    exact_failure = (
        state.get("completed_runs") == ["e0", "e1", "e2"]
        and state.get("failed_runs") == []
        and state.get("current_run_key") == "e3"
        and state.get("current_run") == variant["name"]
        and int(state.get("attempts", {}).get("e3", -1)) == int(impl.MAX_ATTEMPTS)
        and EXPECTED_FAILURE_FRAGMENT in traceback_text
        and recognized_outer_error
    )
    if not exact_failure:
        raise RuntimeError("Refusing WSODD E3 hotfix-v3 recovery because the failure signature is not exact")
    if impl.STOP_SENTINEL.exists():
        raise RuntimeError(f"Refusing WSODD E3 hotfix-v3 recovery while stop sentinel exists: {impl.STOP_SENTINEL}")

    last_pt = run_dir / "weights" / "last.pt"
    results_path = run_dir / "results.csv"
    metadata_path = run_dir / impl.RUN_METADATA_NAME
    for required in (last_pt, results_path, run_dir / "args.yaml", metadata_path):
        if not required.is_file():
            raise RuntimeError(f"WSODD E3 hotfix-v3 required recovery artifact is missing: {required}")
    rows = impl.read_rows(results_path)
    epochs = [int(float(row["epoch"])) for row in rows]
    if not rows or epochs != list(range(1, epochs[-1] + 1)):
        raise RuntimeError("WSODD E3 hotfix-v3 requires contiguous existing E3 epochs")
    if any(impl.row_has_nonfinite(row, ("train/", "val/", "metrics/", "lr/")) for row in rows):
        raise RuntimeError("WSODD E3 hotfix-v3 refuses a non-finite E3 checkpoint")
    impl.verify_resume(run_dir, variant, context)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = impl.STATE_ROOT / "wsodd_e3_hotfix_v3_recovery_archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    for source in (
        state_path,
        impl.EVENTS_PATH,
        impl.PROTOCOL_PATH,
        wsodd.supervisor.RECOVERY_FAILURE_PATH,
        results_path,
        run_dir / "args.yaml",
        metadata_path,
    ):
        _copy_if_present(source, archive_dir)

    manifest_path = Path(__file__).resolve().parent / HOTFIX_MANIFEST_NAME
    audit = {
        "schema": HOTFIX_SCHEMA,
        "status": "passed",
        "recovered_utc": impl.utc_now(),
        "reason": "amp_empty_target_foreground_mask_dtype_and_canonical_own_resume_args",
        "training_recipe_changed": False,
        "dataset_protocol_changed": False,
        "loss_formula_changed": False,
        "completed_runs_reused": ["e0", "e1", "e2"],
        "e3_resume_checkpoint": str(last_pt),
        "e3_resume_checkpoint_sha256": sha256(last_pt),
        "e3_resume_epoch": epochs[-1],
        "discarded_code_defect_attempt_count": int(state["attempts"]["e3"]),
        "self_test_preflight_side_effect_observed": error_text == EXPECTED_SELF_TEST_PREFLIGHT_ERROR,
        "new_release_attempt_budget": int(impl.MAX_ATTEMPTS),
        "cross_run_checkpoint_loaded": False,
        "previous_state_sha256": sha256(state_path),
        "results_sha256": sha256(results_path),
        "args_sha256": sha256(run_dir / "args.yaml"),
        "metadata_sha256": sha256(metadata_path),
        "hotfix_manifest_sha256": sha256(manifest_path),
        "archive_dir": str(archive_dir),
        "e4_started": False,
    }
    audit_path = impl.STATE_ROOT / RECOVERY_AUDIT_NAME
    impl.write_json_atomic(audit_path, audit)

    previous_failure = {
        "status": state.get("status"),
        "error_type": state.get("error_type"),
        "error": state.get("error"),
        "traceback_sha256": __import__("hashlib").sha256(traceback_text.encode("utf-8")).hexdigest(),
    }
    state["attempts"]["e3"] = 0
    state.update(
        status="hotfix_v3_recovery_ready",
        current_epoch=epochs[-1],
        last_progress_utc=impl.utc_now(),
        e3_hotfix_v3={
            "schema": HOTFIX_SCHEMA,
            "audit_path": str(audit_path),
            "audit_sha256": sha256(audit_path),
            "resume_from_own_last_pt": True,
            "resume_epoch": epochs[-1],
            "previous_failure": previous_failure,
        },
        error_type=None,
        error=None,
        traceback=None,
        e4_started=False,
    )
    impl.write_json_atomic(state_path, state)
    impl.append_event(
        "e3_hotfix_v3_recovery_ready",
        audit_path=str(audit_path),
        audit_sha256=sha256(audit_path),
        resume_epoch=epochs[-1],
        resume_checkpoint=str(last_pt),
        training_recipe_changed=False,
        e4_started=False,
    )
    return audit


def execute_queue_with_e3_recovery(context: dict[str, Any]) -> int:
    recovery = recover_exact_e3_failure(context)
    print("WSODD_E3_HOTFIX_V3_RECOVERY " + json.dumps(recovery, ensure_ascii=False, sort_keys=True), flush=True)
    return _released_execute_queue(context)


def configure_runtime_with_e3_hotfix() -> None:
    _released_configure_runtime()
    TaskAlignedAssigner.forward = task_aligned_assigner_forward_bool_mask
    wsodd.impl.verify_args = verify_args_allow_canonical_own_last_pt
    wsodd.supervisor.execute_queue = execute_queue_with_e3_recovery
    os.environ["RING3_WSODD_E3_HOTFIX"] = HOTFIX_SCHEMA


def main() -> int:
    release = verify_hotfix_release()
    mask_test = mask_hotfix_self_test()
    print("WSODD_E3_HOTFIX_V3_RELEASE " + json.dumps(release, ensure_ascii=False, sort_keys=True), flush=True)
    print("WSODD_E3_HOTFIX_V3_MASK_TEST " + json.dumps(mask_test, sort_keys=True), flush=True)
    wsodd.validate_config = validate_config_with_base_schema_isolation
    wsodd.configure_runtime = configure_runtime_with_e3_hotfix
    return wsodd.main()


if __name__ == "__main__":
    raise SystemExit(main())
