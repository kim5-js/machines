from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import train_pretrained_sgd_mainline as impl
from ultralytics import YOLO


PATCH_SCHEMA = "ring3.pretrained_sgd_mainline_supervisor_patch.v1"
COMPLETION_EVIDENCE_NAME = "pretrained_sgd_mainline_completion_evidence.json"
PREVIOUS_RELEASE_MANIFEST_SHA256 = "3b321b1e024713f8a952f150f5876a8f13bd29a23ed945b033fbd726c7ac9d95"
RECOVERY_FAILURE_PATH = impl.STATE_ROOT / "pretrained_sgd_mainline_recovery_failure.json"
ORIGINAL_ADD_CALLBACKS = impl.add_callbacks


def row_epoch(row: dict[str, str] | None) -> int | None:
    if row is None:
        return None
    try:
        return int(float(row["epoch"]))
    except (KeyError, TypeError, ValueError):
        return None


def run_completion(run_dir: Path, epochs: int = 150, patience: int = 100) -> dict[str, Any]:
    """Make a strict, auditable max-epoch or early-stop completion decision."""

    rows = impl.read_rows(run_dir / "results.csv")
    final = rows[-1] if rows else None
    final_epoch = row_epoch(final)
    observed_epochs = [row_epoch(row) for row in rows]
    history_contiguous = bool(rows) and observed_epochs == list(range(1, int(final_epoch or 0) + 1))
    history_finite = bool(rows) and all(
        not impl.row_has_nonfinite(row, ("train/", "val/", "metrics/", "lr/")) for row in rows
    )
    weights_present = all((run_dir / "weights" / name).is_file() for name in ("best.pt", "last.pt"))
    details: dict[str, Any] = {
        "complete": False,
        "completion_kind": None,
        "run_dir": str(run_dir),
        "final_epoch": final_epoch,
        "target_epochs": epochs,
        "patience": patience,
        "history_contiguous": history_contiguous,
        "history_finite": history_finite,
        "weights_present": weights_present,
        "final_row": final,
    }
    if not (history_contiguous and history_finite and weights_present and final_epoch is not None):
        return details
    if final_epoch >= epochs:
        details.update(complete=True, completion_kind="max_epochs")
        return details

    marker_path = run_dir / COMPLETION_EVIDENCE_NAME
    if not marker_path.is_file():
        details["completion_evidence_error"] = "missing"
        return details
    try:
        marker = impl.load_json(marker_path)
        best = impl.best_row(run_dir / "results.csv")
        best_epoch = int(float(best["epoch"]))
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        details["completion_evidence_error"] = f"unreadable: {type(exc).__name__}: {exc}"
        return details
    patience_gap = final_epoch - best_epoch
    details["completion_evidence"] = marker
    expected_marker = {
        "schema": "ring3.pretrained_sgd_mainline_completion_evidence.v1",
        "status": "passed",
        "completion_kind": "early_stopping",
        "normal_training_return_confirmed": True,
        "final_epoch": final_epoch,
        "target_epochs": epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "patience_gap": patience_gap,
    }
    mismatches = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected_marker.items()
        if marker.get(key) != value
    }
    marker_run_dir = Path(str(marker.get("run_dir", "")))
    if marker_run_dir.resolve() != run_dir.resolve():
        mismatches["run_dir"] = {"expected": str(run_dir.resolve()), "observed": str(marker_run_dir)}
    if patience_gap < patience:
        mismatches["patience_gap"] = {"expected_at_least": patience, "observed": patience_gap}
    if mismatches:
        details["completion_evidence_error"] = {"mismatches": mismatches}
        return details
    details.update(
        complete=True,
        completion_kind="early_stopping",
        best_epoch=best_epoch,
        patience_gap=patience_gap,
    )
    return details


def completed_run(run_dir: Path, epochs: int = 150) -> tuple[bool, dict[str, str] | None]:
    details = run_completion(run_dir, epochs=epochs, patience=100)
    return bool(details["complete"]), details["final_row"]


def validate_completed_variant(
    run_dir: Path,
    variant: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    details = run_completion(
        run_dir,
        epochs=int(context["config"]["training"]["epochs"]),
        patience=int(context["config"]["training"]["patience"]),
    )
    if details.get("complete") is not True:
        raise RuntimeError(f"Run is not complete under the locked completion policy: {details}")
    evidence_path = run_dir / impl.RUNTIME_EVIDENCE_NAME
    metadata_path = run_dir / impl.RUN_METADATA_NAME
    if not evidence_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Completed run lacks runtime evidence or metadata: {run_dir}")
    evidence = impl.load_json(evidence_path)
    metadata = impl.load_json(metadata_path)
    impl.validate_runtime_evidence(evidence, variant)
    impl.verify_args(run_dir, context["config"], variant)
    expected_metadata = {
        "variant_key": variant["key"],
        "config_sha256": context["config_sha256"],
        "pretrained": True,
        "optimizer": "SGD",
        "cross_run_checkpoint_loaded": False,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    allowed_releases = {context["release"]["sha256"]}
    if variant["key"] == "e0" and details.get("completion_kind") == "early_stopping":
        allowed_releases.add(PREVIOUS_RELEASE_MANIFEST_SHA256)
    if metadata.get("release_manifest_sha256") not in allowed_releases:
        mismatches["release_manifest_sha256"] = {
            "expected_one_of": sorted(allowed_releases),
            "observed": metadata.get("release_manifest_sha256"),
        }
    if Path(str(evidence.get("run_dir", ""))).resolve() != run_dir.resolve():
        mismatches["runtime_evidence.run_dir"] = {
            "expected": str(run_dir.resolve()),
            "observed": evidence.get("run_dir"),
        }
    if mismatches:
        raise RuntimeError(f"Completed run provenance mismatch: {mismatches}")
    return details


def add_callbacks(
    model: YOLO,
    *,
    variant: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    canonical_run_dir = (impl.OUTPUT_ROOT / variant["name"]).resolve()

    def enforce_canonical_save_dir(trainer: Any) -> None:
        observed = Path(trainer.save_dir).resolve()
        if observed != canonical_run_dir:
            raise RuntimeError(
                "Refusing trainer save-dir drift before any run metadata is written: "
                f"expected={canonical_run_dir}, observed={observed}"
            )

    model.add_callback("on_train_start", enforce_canonical_save_dir)
    ORIGINAL_ADD_CALLBACKS(model, variant=variant, context=context, state=state, metadata=metadata)

    def record_normal_train_end(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir).resolve()
        if save_dir != canonical_run_dir:
            raise RuntimeError(f"Training ended outside the canonical run directory: {save_dir}")
        final = impl.last_row(save_dir / "results.csv")
        final_epoch = row_epoch(final)
        best = impl.best_row(save_dir / "results.csv") if final is not None else None
        best_epoch = row_epoch(best)
        target_epochs = int(getattr(trainer.args, "epochs", context["config"]["training"]["epochs"]))
        patience = int(getattr(trainer.args, "patience", context["config"]["training"]["patience"]))
        trainer_epoch_zero_based = int(getattr(trainer, "epoch", -1))
        stopper = getattr(trainer, "stopper", None)
        stopper_best_zero_based = int(getattr(stopper, "best_epoch", -1))
        stopper_gap = trainer_epoch_zero_based - stopper_best_zero_based
        max_epochs_complete = final_epoch is not None and final_epoch >= target_epochs
        early_stop_complete = bool(
            final_epoch is not None
            and best_epoch is not None
            and final_epoch < target_epochs
            and final_epoch == trainer_epoch_zero_based + 1
            and best_epoch == stopper_best_zero_based + 1
            and stopper_gap >= patience
            and final_epoch - best_epoch >= patience
        )
        if max_epochs_complete:
            completion_kind = "max_epochs"
        elif early_stop_complete:
            completion_kind = "early_stopping"
        else:
            completion_kind = "normal_return_incomplete"
        payload = {
            "schema": "ring3.pretrained_sgd_mainline_completion_evidence.v1",
            "status": "passed" if completion_kind in {"max_epochs", "early_stopping"} else "incomplete",
            "recorded_utc": impl.utc_now(),
            "source": "normal_train_end_callback",
            "patch_schema": PATCH_SCHEMA,
            "run_dir": str(save_dir),
            "variant_key": variant["key"],
            "completion_kind": completion_kind,
            "normal_training_return_confirmed": True,
            "final_epoch": final_epoch,
            "target_epochs": target_epochs,
            "patience": patience,
            "best_epoch": best_epoch,
            "patience_gap": (final_epoch - best_epoch) if final_epoch is not None and best_epoch is not None else None,
            "trainer_epoch_zero_based": trainer_epoch_zero_based,
            "stopper_best_epoch_zero_based": stopper_best_zero_based,
            "stopper_gap": stopper_gap,
            "stopper_possible_stop": bool(getattr(stopper, "possible_stop", False)),
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
        }
        impl.write_json_atomic(save_dir / COMPLETION_EVIDENCE_NAME, payload)
        impl.append_event(
            "normal_training_end_recorded",
            variant=variant["key"],
            completion_kind=completion_kind,
            final_epoch=final_epoch,
            best_epoch=best_epoch,
            patience_gap=payload["patience_gap"],
        )

    model.add_callback("on_train_end", record_normal_train_end)


def run_variant(variant: dict[str, Any], context: dict[str, Any], state: dict[str, Any]) -> bool:
    run_dir = impl.OUTPUT_ROOT / variant["name"]
    details = run_completion(run_dir)
    if details.get("complete") is True:
        validate_completed_variant(run_dir, variant, context)
        return True
    if run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
        impl.archive_run(run_dir, "no_resumable_checkpoint")

    while int(state["attempts"].get(variant["key"], 0)) < impl.MAX_ATTEMPTS:
        details = run_completion(run_dir)
        if details.get("complete") is True:
            validate_completed_variant(run_dir, variant, context)
            return True
        if impl.STOP_SENTINEL.exists():
            state.update(status="paused", current_run=None, current_run_key=None, last_progress_utc=impl.utc_now())
            impl.write_json_atomic(impl.STATE_PATH, state)
            impl.append_event("queue_paused", reason=str(impl.STOP_SENTINEL))
            return False

        attempt = int(state["attempts"].get(variant["key"], 0)) + 1
        state["attempts"][variant["key"]] = attempt
        state.update(
            status="starting",
            current_run=variant["name"],
            current_run_key=variant["key"],
            current_epoch=0,
            target_epochs=150,
            validation_nan_streak=0,
            recent_epoch_seconds=[],
            last_cumulative_train_seconds=None,
            runtime_evidence=None,
            last_progress_utc=impl.utc_now(),
        )
        impl.write_json_atomic(impl.STATE_PATH, state)

        kwargs = impl.training_kwargs(context["config"], variant)
        resumed = run_dir.exists() and (run_dir / "weights" / "last.pt").is_file()
        if resumed:
            impl.verify_resume(run_dir, variant, context)
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            train_kwargs: dict[str, Any] = {"resume": True}
            existing = impl.last_row(run_dir / "results.csv")
            if existing and existing.get("epoch"):
                state["current_epoch"] = int(float(existing["epoch"]))
        else:
            model = YOLO(context["config"]["weights"]["path"])
            train_kwargs = kwargs

        os.environ["RING3_SGD_MAINLINE_VARIANT_JSON"] = json.dumps(variant, sort_keys=True)
        os.environ["RING3_SGD_MAINLINE_MDG_CONFIG"] = str(impl.MDG_CONFIG_PATH)
        os.environ["RING3_SGD_MAINLINE_MDG_SHA256"] = context["mdg_config_sha256"]
        metadata = {
            "schema": "ring3.pretrained_sgd_mainline_run_metadata.v1",
            "created_utc": impl.utc_now(),
            "variant_key": variant["key"],
            "variant": variant,
            "attempt": attempt,
            "resumed_from_own_last_pt": resumed,
            "parent_checkpoint": "same_run_last.pt" if resumed else None,
            "cross_run_checkpoint_loaded": False,
            "pretrained": True,
            "pretrained_source": context["config"]["weights"]["path"],
            "pretrained_source_sha256": impl.EXPECTED_WEIGHT_SHA256,
            "optimizer": "SGD",
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
            "training_kwargs": kwargs,
            "consistency_loss_enabled": False,
            "cbam_enabled": False,
            "focal_eiou_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "e4_started": False,
            "supervisor_patch_schema": PATCH_SCHEMA,
        }
        add_callbacks(model, variant=variant, context=context, state=state, metadata=metadata)
        impl.append_event(
            "attempt_started",
            variant=variant["key"],
            attempt=attempt,
            resumed=resumed,
            run_dir=str(run_dir),
            supervisor_patch_schema=PATCH_SCHEMA,
        )
        try:
            model.train(trainer=impl.MainlineSGDTrainer, **train_kwargs)
            details = validate_completed_variant(run_dir, variant, context)
            impl.append_event(
                "run_completed",
                variant=variant["key"],
                attempt=attempt,
                completion_kind=details["completion_kind"],
                final_epoch=details["final_epoch"],
                final_row=details["final_row"],
                best_row=impl.best_row(run_dir / "results.csv"),
            )
            return True
        except BaseException as exc:
            numerical = isinstance(exc, FloatingPointError) or any(
                token in (str(exc) + type(exc).__name__).lower() for token in ("non-finite", "nan", "inf")
            )
            state.update(
                status="attempt_failed",
                last_progress_utc=impl.utc_now(),
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                numerical_failure=numerical,
            )
            impl.write_json_atomic(impl.STATE_PATH, state)
            impl.append_event(
                "run_attempt_failed",
                variant=variant["key"],
                attempt=attempt,
                numerical_failure=numerical,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            traceback.print_exc()
            if run_completion(run_dir).get("complete") is True:
                raise RuntimeError(
                    "Completed artifacts failed provenance validation; refusing to resume a completed checkpoint"
                ) from exc
            if numerical and run_dir.exists():
                impl.archive_run(run_dir, "numerical_failure")
            elif run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
                impl.archive_run(run_dir, "unresumable_failure")
    impl.append_event("attempts_exhausted", variant=variant["key"], attempts=impl.MAX_ATTEMPTS)
    return False


def summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        details = run_completion(run_dir)
        if details.get("complete") is not True:
            metrics[variant["key"]] = {"status": "incomplete", "run_dir": str(run_dir), "completion": details}
            continue
        validate_completed_variant(run_dir, variant, context)
        metrics[variant["key"]] = {
            "status": "completed",
            "run_dir": str(run_dir),
            "completion_kind": details["completion_kind"],
            "final_epoch": details["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": details["final_row"],
        }
    payload: dict[str, Any] = {
        "schema": "ring3.pretrained_sgd_mainline_summary.v2",
        "created_utc": impl.utc_now(),
        "supervisor_patch_schema": PATCH_SCHEMA,
        "single_seed_mechanism_evidence_only": True,
        "metrics": metrics,
        "e4_started": False,
    }
    if all(metrics.get(key, {}).get("status") == "completed" for key in ("e0", "e1", "e2", "e3")):
        values = {key: float(metrics[key]["best_row"]["metrics/mAP50-95(B)"]) for key in metrics}
        payload["increments"] = {
            "e1_minus_e0": values["e1"] - values["e0"],
            "e2_minus_e1": values["e2"] - values["e1"],
            "e3_minus_e2": values["e3"] - values["e2"],
            "e3_minus_e0": values["e3"] - values["e0"],
        }
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def execute_queue(context: dict[str, Any]) -> int:
    impl.activate_protocol(context)
    if impl.STATE_PATH.is_file():
        state = impl.load_json(impl.STATE_PATH)
        anchors = {
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
            "historical_lock_sha256": context["historical_lock_sha256"],
            "runtime_amendment_sha256": context["runtime_amendment_sha256"],
            "mdg_config_sha256": context["mdg_config_sha256"],
        }
        if any(state.get(key) != value for key, value in anchors.items()):
            raise RuntimeError("Existing queue state belongs to a different locked release")
    else:
        state = impl.state_template(context)
        impl.write_json_atomic(impl.STATE_PATH, state)
    impl.append_event(
        "queue_started_or_resumed",
        completed_runs=state["completed_runs"],
        supervisor_patch_schema=PATCH_SCHEMA,
    )

    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        if variant["key"] in state["completed_runs"]:
            validate_completed_variant(run_dir, variant, context)
            continue
        ok = run_variant(variant, context, state)
        if str(state.get("status", "")).startswith("paused"):
            return 0
        if ok:
            if variant["key"] not in state["completed_runs"]:
                state["completed_runs"].append(variant["key"])
        elif variant["key"] not in state["failed_runs"]:
            state["failed_runs"].append(variant["key"])
        details = run_completion(run_dir)
        state.update(
            current_run=None,
            current_run_key=None,
            current_epoch=int(details.get("final_epoch") or state.get("current_epoch", 0)) if ok else state.get("current_epoch", 0),
            last_progress_utc=impl.utc_now(),
        )
        impl.write_json_atomic(impl.STATE_PATH, state)

    final_summary = summary(context)
    state.update(
        status="completed" if not state["failed_runs"] else "completed_with_failures",
        current_run=None,
        current_run_key=None,
        completed_utc=impl.utc_now(),
        last_progress_utc=impl.utc_now(),
        summary_path=str(impl.SUMMARY_PATH),
        summary_sha256=impl.sha256(impl.SUMMARY_PATH),
        e4_started=False,
    )
    impl.write_json_atomic(impl.STATE_PATH, state)
    impl.append_event(
        "queue_completed",
        status=state["status"],
        completed_runs=state["completed_runs"],
        failed_runs=state["failed_runs"],
        summary=final_summary,
    )
    return 0 if not state["failed_runs"] else 3


def preflight_allow_stop_sentinel() -> dict[str, Any]:
    if not impl.STOP_SENTINEL.is_file():
        raise RuntimeError("Recovery requires the exact stop sentinel to remain present until validation completes")
    config = impl.load_json(impl.CONFIG_PATH)
    impl.validate_config(config)
    release = impl.verify_release_manifest()
    if impl.sha256(impl.HISTORICAL_LOCK_PATH) != impl.EXPECTED_HISTORICAL_LOCK_SHA256:
        raise RuntimeError("Historical formal lock changed")
    if impl.sha256(impl.AMENDMENT_PATH) != impl.EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Runtime amendment changed")
    if impl.sha256(impl.MDG_CONFIG_PATH) != impl.EXPECTED_MDG_SHA256:
        raise RuntimeError("MDG config changed")
    impl.load_mdg_config(impl.MDG_CONFIG_PATH)
    if impl.ultralytics.__version__ != "8.4.124" or impl.torch.__version__ != "2.5.1+cu124":
        raise RuntimeError("Locked runtime versions changed")
    if not impl.torch.cuda.is_available() or "RTX 4090" not in impl.torch.cuda.get_device_name(0):
        raise RuntimeError("Locked RTX 4090 runtime is unavailable")
    weight_path = Path(config["weights"]["path"])
    if not weight_path.is_file() or impl.sha256(weight_path) != impl.EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt is missing or changed")
    dataset_report = impl.verify_dataset(config)
    return {
        "config": config,
        "config_sha256": impl.sha256(impl.CONFIG_PATH),
        "release": release,
        "historical_lock_sha256": impl.EXPECTED_HISTORICAL_LOCK_SHA256,
        "runtime_amendment_sha256": impl.EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": impl.EXPECTED_MDG_SHA256,
        "dataset_report": dataset_report,
        "environment": {
            "python": impl.sys.version.split()[0],
            "torch": impl.torch.__version__,
            "ultralytics": impl.ultralytics.__version__,
            "cuda_device": impl.torch.cuda.get_device_name(0),
        },
    }


def read_events() -> list[dict[str, Any]]:
    if not impl.EVENTS_PATH.is_file():
        return []
    events = []
    for line in impl.EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def recover_paused_e0(context: dict[str, Any]) -> dict[str, Any]:
    state = impl.load_json(impl.STATE_PATH)
    if state.get("status") != "paused_protocol_violation":
        raise RuntimeError(f"Unexpected recovery state: {state.get('status')!r}")
    expected_state = {
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "e1_started": False,
        "e4_started": False,
    }
    mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in expected_state.items()
        if state.get(key) != value
    }
    if mismatches or state.get("completed_runs") or state.get("failed_runs"):
        raise RuntimeError(f"Paused state is not the audited E0-only incident: {mismatches}")

    variants = {variant["key"]: variant for variant in context["config"]["run_order"]}
    e0_variant = variants["e0"]
    e0_dir = impl.OUTPUT_ROOT / e0_variant["name"]
    for key in ("e1", "e2", "e3"):
        if (impl.OUTPUT_ROOT / variants[key]["name"]).exists():
            raise RuntimeError(f"Refusing recovery because {key} already has a formal run directory")

    details = run_completion(e0_dir)
    if details.get("final_epoch") != 125 or details.get("history_contiguous") is not True:
        raise RuntimeError(f"Legacy E0 epoch history changed: {details}")
    if details.get("history_finite") is not True or details.get("weights_present") is not True:
        raise RuntimeError(f"Legacy E0 is not finite or lacks weights: {details}")
    best = impl.best_row(e0_dir / "results.csv")
    best_epoch = int(float(best["epoch"]))
    patience = int(context["config"]["training"]["patience"])
    patience_gap = int(details["final_epoch"]) - best_epoch
    if best_epoch != 25 or patience_gap != patience:
        raise RuntimeError(
            f"Legacy E0 does not reproduce the exact early-stop proof: best={best_epoch}, gap={patience_gap}"
        )
    impl.verify_args(e0_dir, context["config"], e0_variant)
    runtime = impl.load_json(e0_dir / impl.RUNTIME_EVIDENCE_NAME)
    metadata = impl.load_json(e0_dir / impl.RUN_METADATA_NAME)
    impl.validate_runtime_evidence(runtime, e0_variant)
    expected_metadata = {
        "variant_key": "e0",
        "attempt": 1,
        "resumed_from_own_last_pt": False,
        "parent_checkpoint": None,
        "cross_run_checkpoint_loaded": False,
        "pretrained": True,
        "optimizer": "SGD",
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
    }
    metadata_mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if metadata_mismatches or Path(str(runtime.get("run_dir", ""))).resolve() != e0_dir.resolve():
        raise RuntimeError(f"Legacy E0 provenance changed: {metadata_mismatches}")

    expected_error = f"Run ended without finite epoch-150 artifacts: {e0_dir}"
    return_event = next(
        (
            event
            for event in read_events()
            if event.get("event") == "run_attempt_failed"
            and event.get("variant") == "e0"
            and event.get("attempt") == 1
            and event.get("numerical_failure") is False
            and event.get("error") == expected_error
        ),
        None,
    )
    if return_event is None:
        raise RuntimeError("Missing the attempt-1 event proving model.train returned before the old supervisor rejected E0")

    marker = {
        "schema": "ring3.pretrained_sgd_mainline_completion_evidence.v1",
        "status": "passed",
        "recorded_utc": impl.utc_now(),
        "source": "one_time_legacy_e0_recovery",
        "patch_schema": PATCH_SCHEMA,
        "run_dir": str(e0_dir.resolve()),
        "variant_key": "e0",
        "completion_kind": "early_stopping",
        "normal_training_return_confirmed": True,
        "normal_return_proof_event": return_event,
        "final_epoch": 125,
        "target_epochs": int(context["config"]["training"]["epochs"]),
        "patience": patience,
        "best_epoch": best_epoch,
        "patience_gap": patience_gap,
        "config_sha256": context["config_sha256"],
        "origin_release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
        "recovery_release_manifest_sha256": context["release"]["sha256"],
        "invalid_attempts_reused": False,
    }
    impl.write_json_atomic(e0_dir / COMPLETION_EVIDENCE_NAME, marker)
    validated = validate_completed_variant(e0_dir, e0_variant, context)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = impl.STATE_ROOT / "pretrained_sgd_mainline_recovery_archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    old_state_sha = impl.sha256(impl.STATE_PATH)
    old_protocol_sha = impl.sha256(impl.PROTOCOL_PATH)
    shutil.copy2(impl.STATE_PATH, archive_dir / impl.STATE_PATH.name)
    shutil.copy2(impl.PROTOCOL_PATH, archive_dir / impl.PROTOCOL_PATH.name)
    shutil.copy2(impl.STOP_SENTINEL, archive_dir / impl.STOP_SENTINEL.name)

    protocol = impl.protocol_payload(context)
    impl.write_json_atomic(impl.PROTOCOL_PATH, protocol)
    new_state = impl.state_template(context)
    new_state.update(
        started_utc=state.get("started_utc", new_state["started_utc"]),
        status="recovery_ready_stop_pending",
        completed_runs=["e0"],
        failed_runs=[],
        attempts={"e0": 1},
        current_run=None,
        current_run_key=None,
        current_epoch=125,
        e1_started=False,
        e4_started=False,
        invalid_attempts=[2, 3],
        invalid_attempt_dirs=list(state.get("invalid_attempt_dirs", [])),
        recovery_audit={
            "schema": "ring3.pretrained_sgd_mainline_recovery_audit.v1",
            "recovered_utc": impl.utc_now(),
            "patch_schema": PATCH_SCHEMA,
            "archive_dir": str(archive_dir),
            "previous_state_sha256": old_state_sha,
            "previous_protocol_sha256": old_protocol_sha,
            "previous_release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
            "current_release_manifest_sha256": context["release"]["sha256"],
            "e0_completion": validated,
            "e0_best_row": best,
            "invalid_attempts_reused": False,
        },
        last_progress_utc=impl.utc_now(),
    )
    impl.write_json_atomic(impl.STATE_PATH, new_state)
    impl.append_event(
        "paused_e0_recovered",
        archive_dir=str(archive_dir),
        e0_completion=validated,
        supervisor_patch_schema=PATCH_SCHEMA,
    )
    impl.STOP_SENTINEL.unlink()
    new_state.update(status="preflight_passed", stop_sentinel_removed=True, last_progress_utc=impl.utc_now())
    impl.write_json_atomic(impl.STATE_PATH, new_state)
    return {
        "status": "recovered",
        "e0_completion": validated,
        "archive_dir": str(archive_dir),
        "new_release_manifest_sha256": context["release"]["sha256"],
        "stop_sentinel_removed": True,
        "next_run": "e1",
    }


def install() -> None:
    impl.completed_run = completed_run
    impl.add_callbacks = add_callbacks
    impl.run_variant = run_variant
    impl.summary = summary
    impl.execute_queue = execute_queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--recover-paused-e0", action="store_true")
    args = parser.parse_args()
    install()
    try:
        if args.recover_paused_e0:
            context = preflight_allow_stop_sentinel()
            result = recover_paused_e0(context)
            print("PRETRAINED_SGD_RECOVERY " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return 0
        context = impl.preflight()
        print("PRETRAINED_SGD_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        result = impl.static_self_test(context)
        print("PRETRAINED_SGD_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return execute_queue(context)
    except BaseException as exc:
        payload = {
            "schema": "ring3.pretrained_sgd_mainline_supervisor_failure.v2",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "supervisor_patch_schema": PATCH_SCHEMA,
            "e4_started": False,
        }
        if args.recover_paused_e0:
            impl.write_json_atomic(RECOVERY_FAILURE_PATH, payload)
            impl.append_event("recovery_failed", error_type=type(exc).__name__, error=str(exc))
        else:
            state = impl.load_json(impl.STATE_PATH) if impl.STATE_PATH.is_file() else {}
            state.update(payload)
            impl.write_json_atomic(impl.STATE_PATH, state)
            impl.append_event("supervisor_failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
