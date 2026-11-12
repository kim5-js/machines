from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pretrained_sgd_mainline_entry as mainline_entry
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation as alpha
import train_pretrained_sgd_mainline as impl
from pretrained_sgd_early_stop_utils import (
    classify_normal_training_return,
    is_duplicate_result_epoch,
    parse_epoch,
    proves_locked_early_stop,
)


MANIFEST_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_release_manifest.v2"
SUPERVISOR_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_supervisor.v2"
NORMAL_COMPLETION_SCHEMA = "ring3.pretrained_sgd_completion_evidence.v2"
RECOVERY_COMPLETION_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_early_stop_recovery.v2"
RECOVERY_AUDIT_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_recovery_audit.v2"
PREVIOUS_RELEASE_MANIFEST_SHA256 = "c69a8069efe925ae73206ade0d53cdfb8a21027bbaf1ffd739d2eee8f3394ace"
PREVIOUS_SUPERVISOR_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_supervisor.v1"
LEGACY_CALLBACK_ERROR = "MDG image accounting mismatch: mdg={}, targets={}"
LEGACY_DEVICE_ERROR = "Run args differ from the locked SGD recipe:\ndevice: expected 0, found '0'"
KNOWN_RECOVERY_FINAL_EPOCH = 122
KNOWN_RECOVERY_BEST_EPOCH = 22

BASE_RUN_COMPLETION = supervisor.run_completion
LEGACY_FAILURE_PATH = impl.STATE_ROOT / "pretrained_sgd_alpha_ablation_failure.json"
V2_FAILURE_PATH = impl.STATE_ROOT / "pretrained_sgd_alpha_ablation_v2_failure.json"


def row_epoch(row: dict[str, str] | None) -> int | None:
    return parse_epoch(row.get("epoch")) if row else None


def wrap_fit_epoch_callback(
    callback: Callable[[Any], None],
    *,
    state: dict[str, Any],
    variant: dict[str, Any],
    row_reader: Callable[[Path], dict[str, str] | None] = impl.last_row,
    event_writer: Callable[..., None] = impl.append_event,
) -> Callable[[Any], None]:
    """Run accounting once per persisted training epoch.

    Ultralytics 8.4.124 calls ``on_fit_epoch_end`` once more from ``final_eval``
    after a normal train stop. That validation does not append a results row and
    has no new MDG/target counters. A duplicate persisted epoch is therefore a
    post-training observer callback, not a new training epoch.
    """

    last_processed = parse_epoch(state.get("current_epoch")) or 0
    last_reported_duplicate: tuple[int, int] | None = None

    def guarded(trainer: Any) -> None:
        nonlocal last_processed, last_reported_duplicate
        result = row_reader(Path(trainer.save_dir) / "results.csv")
        result_epoch = row_epoch(result)
        recorded_epoch = max(last_processed, parse_epoch(state.get("current_epoch")) or 0)
        if result_epoch is not None and is_duplicate_result_epoch(result_epoch, recorded_epoch):
            duplicate = (result_epoch, recorded_epoch)
            if duplicate != last_reported_duplicate:
                event_writer(
                    "duplicate_fit_epoch_end_ignored",
                    variant=variant["key"],
                    persisted_epoch=result_epoch,
                    recorded_epoch=recorded_epoch,
                    trainer_epoch=int(getattr(trainer, "epoch", -1)),
                    reason="post_training_final_eval_duplicate",
                    supervisor_schema=SUPERVISOR_SCHEMA,
                )
                last_reported_duplicate = duplicate
            return

        callback(trainer)
        completed_epoch = result_epoch or parse_epoch(state.get("current_epoch"))
        if completed_epoch is not None:
            last_processed = max(last_processed, completed_epoch)

    return guarded


def guarded_original_add_callbacks(
    model: Any,
    *,
    variant: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    callbacks = model.callbacks.setdefault("on_fit_epoch_end", [])
    before = len(callbacks)
    alpha.BASE_ADD_CALLBACKS(model, variant=variant, context=context, state=state, metadata=metadata)
    added = len(callbacks) - before
    if added != 1:
        raise RuntimeError(f"Expected exactly one formal fit-epoch callback, observed {added}")
    callbacks[before] = wrap_fit_epoch_callback(callbacks[before], state=state, variant=variant)


def add_callbacks_v2(
    model: Any,
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
    guarded_original_add_callbacks(model, variant=variant, context=context, state=state, metadata=metadata)

    def record_normal_train_end(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir).resolve()
        if save_dir != canonical_run_dir:
            raise RuntimeError(f"Training ended outside the canonical run directory: {save_dir}")
        final = impl.last_row(save_dir / "results.csv")
        final_epoch = row_epoch(final)
        metric_best = impl.best_row(save_dir / "results.csv") if final is not None else None
        metric_best_epoch = row_epoch(metric_best)
        target_epochs = int(getattr(trainer.args, "epochs", context["config"]["training"]["epochs"]))
        patience = int(getattr(trainer.args, "patience", context["config"]["training"]["patience"]))
        trainer_epoch_zero_based = int(getattr(trainer, "epoch", -1))
        stopper = getattr(trainer, "stopper", None)
        # EarlyStopping receives epoch + 1, so best_epoch is already one-based.
        stopper_best_epoch = int(getattr(stopper, "best_epoch", 0))
        completion_kind = classify_normal_training_return(
            final_epoch=final_epoch,
            target_epochs=target_epochs,
            trainer_epoch_zero_based=trainer_epoch_zero_based,
            stopper_best_epoch_one_based=stopper_best_epoch,
            patience=patience,
        )
        patience_gap = (
            final_epoch - stopper_best_epoch
            if final_epoch is not None and stopper_best_epoch > 0
            else None
        )
        payload = {
            "schema": NORMAL_COMPLETION_SCHEMA,
            "status": "passed" if completion_kind in {"max_epochs", "early_stopping"} else "incomplete",
            "recorded_utc": impl.utc_now(),
            "source": "normal_train_end_callback_v2",
            "patch_schema": SUPERVISOR_SCHEMA,
            "run_dir": str(save_dir),
            "variant_key": variant["key"],
            "completion_kind": completion_kind,
            "normal_training_return_confirmed": True,
            "final_epoch": final_epoch,
            "target_epochs": target_epochs,
            "patience": patience,
            "best_epoch": stopper_best_epoch,
            "metric_best_epoch": metric_best_epoch,
            "patience_gap": patience_gap,
            "trainer_epoch_zero_based": trainer_epoch_zero_based,
            "stopper_best_epoch_one_based": stopper_best_epoch,
            "stopper_possible_stop": bool(getattr(stopper, "possible_stop", False)),
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
        }
        impl.write_json_atomic(save_dir / supervisor.COMPLETION_EVIDENCE_NAME, payload)
        impl.append_event(
            "normal_training_end_recorded",
            variant=variant["key"],
            completion_kind=completion_kind,
            final_epoch=final_epoch,
            stopper_best_epoch=stopper_best_epoch,
            metric_best_epoch=metric_best_epoch,
            patience_gap=patience_gap,
            supervisor_schema=SUPERVISOR_SCHEMA,
        )

    model.add_callback("on_train_end", record_normal_train_end)


def _marker_mismatches(marker: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }


def run_completion_v2(run_dir: Path, epochs: int = 150, patience: int = 100) -> dict[str, Any]:
    base = BASE_RUN_COMPLETION(run_dir, epochs=epochs, patience=patience)
    if base.get("complete") is True:
        return base
    if not (
        base.get("history_contiguous") is True
        and base.get("history_finite") is True
        and base.get("weights_present") is True
    ):
        return base

    marker_path = run_dir / supervisor.COMPLETION_EVIDENCE_NAME
    if not marker_path.is_file():
        return base
    try:
        marker = impl.load_json(marker_path)
        metric_best_epoch = row_epoch(impl.best_row(run_dir / "results.csv"))
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        base["completion_evidence_error"] = f"unreadable: {type(exc).__name__}: {exc}"
        return base

    final_epoch = parse_epoch(base.get("final_epoch"))
    stopper_best_epoch = parse_epoch(marker.get("best_epoch"))
    patience_gap = (
        final_epoch - stopper_best_epoch
        if final_epoch is not None and stopper_best_epoch is not None
        else None
    )
    expected_common = {
        "status": "passed",
        "completion_kind": "early_stopping",
        "final_epoch": final_epoch,
        "target_epochs": int(epochs),
        "patience": int(patience),
        "best_epoch": stopper_best_epoch,
        "patience_gap": patience_gap,
    }
    mismatches = _marker_mismatches(marker, expected_common)
    if Path(str(marker.get("run_dir", ""))).resolve() != run_dir.resolve():
        mismatches["run_dir"] = {"expected": str(run_dir.resolve()), "observed": marker.get("run_dir")}

    if marker.get("schema") == NORMAL_COMPLETION_SCHEMA:
        expected_normal = {
            "source": "normal_train_end_callback_v2",
            "patch_schema": SUPERVISOR_SCHEMA,
            "normal_training_return_confirmed": True,
            "metric_best_epoch": metric_best_epoch,
            "stopper_best_epoch_one_based": stopper_best_epoch,
        }
        mismatches.update(_marker_mismatches(marker, expected_normal))
        classified = classify_normal_training_return(
            final_epoch=final_epoch,
            target_epochs=epochs,
            trainer_epoch_zero_based=marker.get("trainer_epoch_zero_based"),
            stopper_best_epoch_one_based=stopper_best_epoch,
            patience=patience,
        )
        if classified != "early_stopping":
            mismatches["classification"] = {"expected": "early_stopping", "observed": classified}
        recovered = False
    elif marker.get("schema") == RECOVERY_COMPLETION_SCHEMA:
        expected_recovery = {
            "source": "known_final_eval_callback_failure_recovery",
            "variant_key": "e3_alpha12",
            "normal_training_return_confirmed": False,
            "training_loop_early_stop_proven": True,
            "final_evaluation_completed_before_callback_failure": True,
            "observer_failure_after_completion": True,
            "observer_error_type": "RuntimeError",
            "observer_error": LEGACY_CALLBACK_ERROR,
            "origin_release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
            "invalid_training_attempt_reused": False,
            "final_epoch": KNOWN_RECOVERY_FINAL_EPOCH,
            "best_epoch": KNOWN_RECOVERY_BEST_EPOCH,
        }
        mismatches.update(_marker_mismatches(marker, expected_recovery))
        recovered = True
    else:
        return base

    if not proves_locked_early_stop(
        final_epoch=final_epoch,
        best_epoch=stopper_best_epoch,
        target_epochs=epochs,
        patience=patience,
        history_contiguous=bool(base["history_contiguous"]),
        history_finite=bool(base["history_finite"]),
        weights_present=bool(base["weights_present"]),
    ):
        mismatches["early_stop_proof"] = "finite contiguous history, checkpoints, and patience gap are required"
    if mismatches:
        base["completion_evidence_error"] = {"mismatches": mismatches}
        return base

    base.pop("completion_evidence_error", None)
    base.update(
        complete=True,
        completion_kind="early_stopping",
        best_epoch=stopper_best_epoch,
        metric_best_epoch=metric_best_epoch,
        patience_gap=patience_gap,
        completion_evidence=marker,
        recovery_audited=recovered,
    )
    return base


def completed_run_v2(run_dir: Path, epochs: int = 150) -> tuple[bool, dict[str, str] | None]:
    details = run_completion_v2(run_dir, epochs=epochs, patience=100)
    return bool(details.get("complete")), details.get("final_row")


def validate_completed_variant_v2(
    run_dir: Path,
    variant: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    details = run_completion_v2(
        run_dir,
        epochs=int(context["config"]["training"]["epochs"]),
        patience=int(context["config"]["training"]["patience"]),
    )
    if details.get("complete") is not True:
        raise RuntimeError(f"Run is not complete under the v2 completion policy: {details}")
    evidence_path = run_dir / impl.RUNTIME_EVIDENCE_NAME
    metadata_path = run_dir / impl.RUN_METADATA_NAME
    if not evidence_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Completed run lacks runtime evidence or metadata: {run_dir}")
    evidence = impl.load_json(evidence_path)
    metadata = impl.load_json(metadata_path)
    impl.validate_runtime_evidence(evidence, variant)
    impl.verify_args(run_dir, context["config"], variant)

    marker = details.get("completion_evidence", {})
    recovered = bool(details.get("recovery_audited"))
    expected_release = PREVIOUS_RELEASE_MANIFEST_SHA256 if recovered else context["release"]["sha256"]
    expected_metadata = {
        "variant_key": variant["key"],
        "config_sha256": context["config_sha256"],
        "pretrained": True,
        "optimizer": "SGD",
        "cross_run_checkpoint_loaded": False,
        "release_manifest_sha256": expected_release,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if recovered:
        if variant["key"] != "e3_alpha12":
            mismatches["recovered_variant"] = {"expected": "e3_alpha12", "observed": variant["key"]}
        expected_marker = {
            "schema": RECOVERY_COMPLETION_SCHEMA,
            "origin_release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
            "recovery_release_manifest_sha256": context["release"]["sha256"],
        }
        mismatches.update(
            {
                f"completion_evidence.{key}": {"expected": value, "observed": marker.get(key)}
                for key, value in expected_marker.items()
                if marker.get(key) != value
            }
        )
    if Path(str(evidence.get("run_dir", ""))).resolve() != run_dir.resolve():
        mismatches["runtime_evidence.run_dir"] = {
            "expected": str(run_dir.resolve()),
            "observed": evidence.get("run_dir"),
        }
    if mismatches:
        raise RuntimeError(f"Completed run provenance mismatch: {mismatches}")
    return details


def read_events() -> list[dict[str, Any]]:
    if not impl.EVENTS_PATH.is_file():
        return []
    return [json.loads(line) for line in impl.EVENTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def guard_self_test() -> dict[str, Any]:
    state: dict[str, Any] = {"current_epoch": 0}
    current_row: dict[str, str] = {"epoch": "1"}
    calls: list[int] = []
    events: list[dict[str, Any]] = []

    class FakeTrainer:
        save_dir = "/tmp/ring3_guard_self_test"
        epoch = 0

    def original(_trainer: Any) -> None:
        epoch = int(current_row["epoch"])
        calls.append(epoch)
        state["current_epoch"] = epoch

    def row_reader(_path: Path) -> dict[str, str]:
        return dict(current_row)

    def event_writer(event: str, **details: Any) -> None:
        events.append({"event": event, **details})

    guarded = wrap_fit_epoch_callback(
        original,
        state=state,
        variant={"key": "self_test"},
        row_reader=row_reader,
        event_writer=event_writer,
    )
    trainer = FakeTrainer()
    guarded(trainer)
    trainer.epoch = 1
    guarded(trainer)
    current_row["epoch"] = "2"
    guarded(trainer)
    if calls != [1, 2] or len(events) != 1 or events[0].get("reason") != "post_training_final_eval_duplicate":
        raise RuntimeError(f"Duplicate callback guard self-test failed: calls={calls}, events={events}")
    classification = classify_normal_training_return(
        final_epoch=122,
        target_epochs=150,
        trainer_epoch_zero_based=121,
        stopper_best_epoch_one_based=22,
        patience=100,
    )
    if classification != "early_stopping":
        raise RuntimeError(f"Early-stop epoch convention self-test failed: {classification}")
    return {
        "schema": "ring3.pretrained_sgd_early_stop_guard_self_test.v2",
        "status": "passed",
        "real_epoch_callbacks": calls,
        "duplicate_callbacks_ignored": len(events),
        "early_stop_classification": classification,
    }


def recover_known_alpha12_if_needed(context: dict[str, Any]) -> dict[str, Any]:
    if not impl.STATE_PATH.is_file():
        return {"status": "not_needed", "reason": "no_existing_state"}
    state = impl.load_json(impl.STATE_PATH)
    if state.get("release_manifest_sha256") == context["release"]["sha256"]:
        if "e3_alpha12" not in state.get("completed_runs", []):
            raise RuntimeError("The v2 queue state exists but alpha1.2 is not marked complete")
        variant = next(item for item in context["config"]["run_order"] if item["key"] == "e3_alpha12")
        details = validate_completed_variant_v2(impl.OUTPUT_ROOT / variant["name"], variant, context)
        return {"status": "already_recovered", "completion": details}

    expected_state = {
        "schema": "ring3.pretrained_sgd_alpha_ablation_state.v1",
        "status": "starting",
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "current_run_key": "e3_alpha12",
        "e4_started": False,
    }
    mismatches = _marker_mismatches(state, expected_state)
    if mismatches or state.get("completed_runs") or state.get("failed_runs"):
        raise RuntimeError(f"Existing state is not the audited alpha1.2 incident: {mismatches}")
    if state.get("attempts") != {"e3_alpha12": 2}:
        raise RuntimeError(f"Unexpected alpha attempt history: {state.get('attempts')!r}")

    old_manifest = impl.SETUP_ROOT / "pretrained_sgd_alpha_ablation_release_manifest.json"
    if not old_manifest.is_file() or impl.sha256(old_manifest) != PREVIOUS_RELEASE_MANIFEST_SHA256:
        raise RuntimeError("The predecessor alpha release manifest changed")
    protocol = impl.load_json(impl.PROTOCOL_PATH)
    if protocol.get("release_manifest", {}).get("sha256") != PREVIOUS_RELEASE_MANIFEST_SHA256:
        raise RuntimeError("The predecessor alpha protocol release anchor changed")
    failure = impl.load_json(LEGACY_FAILURE_PATH)
    expected_failure = {
        "status": "supervisor_failed",
        "error_type": "RuntimeError",
        "error": LEGACY_DEVICE_ERROR,
        "supervisor_schema": PREVIOUS_SUPERVISOR_SCHEMA,
        "e4_started": False,
    }
    failure_mismatches = _marker_mismatches(failure, expected_failure)
    if failure_mismatches:
        raise RuntimeError(f"The predecessor supervisor failure changed: {failure_mismatches}")

    variants = {variant["key"]: variant for variant in context["config"]["run_order"]}
    alpha12 = variants["e3_alpha12"]
    alpha135 = variants["e3_alpha135"]
    run_dir = impl.OUTPUT_ROOT / alpha12["name"]
    alpha135_dir = impl.OUTPUT_ROOT / alpha135["name"]
    if alpha135_dir.exists():
        raise RuntimeError("Refusing recovery because alpha1.35 already has a run directory")
    unexpected_dirs = [
        path.name
        for path in impl.OUTPUT_ROOT.iterdir()
        if path.is_dir() and path.resolve() != run_dir.resolve()
    ]
    if unexpected_dirs:
        raise RuntimeError(f"Unexpected alpha run directories exist: {unexpected_dirs}")

    details = BASE_RUN_COMPLETION(run_dir, epochs=150, patience=100)
    best = impl.best_row(run_dir / "results.csv")
    best_epoch = row_epoch(best)
    if details.get("final_epoch") != KNOWN_RECOVERY_FINAL_EPOCH or best_epoch != KNOWN_RECOVERY_BEST_EPOCH:
        raise RuntimeError(f"The exact alpha1.2 early-stop epochs changed: completion={details}, best={best_epoch}")
    if not proves_locked_early_stop(
        final_epoch=details.get("final_epoch"),
        best_epoch=best_epoch,
        target_epochs=150,
        patience=100,
        history_contiguous=bool(details.get("history_contiguous")),
        history_finite=bool(details.get("history_finite")),
        weights_present=bool(details.get("weights_present")),
    ):
        raise RuntimeError(f"Alpha1.2 does not prove a finite locked early stop: {details}")

    impl.verify_args(run_dir, context["config"], alpha12)
    runtime = impl.load_json(run_dir / impl.RUNTIME_EVIDENCE_NAME)
    metadata = impl.load_json(run_dir / impl.RUN_METADATA_NAME)
    impl.validate_runtime_evidence(runtime, alpha12)
    expected_metadata = {
        "variant_key": "e3_alpha12",
        "attempt": 1,
        "resumed_from_own_last_pt": False,
        "parent_checkpoint": None,
        "cross_run_checkpoint_loaded": False,
        "pretrained": True,
        "pretrained_source_sha256": impl.EXPECTED_WEIGHT_SHA256,
        "optimizer": "SGD",
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
        "supervisor_patch_schema": PREVIOUS_SUPERVISOR_SCHEMA,
    }
    metadata_mismatches = _marker_mismatches(metadata, expected_metadata)
    if metadata_mismatches or Path(str(runtime.get("run_dir", ""))).resolve() != run_dir.resolve():
        raise RuntimeError(f"Alpha1.2 provenance changed: {metadata_mismatches}")

    mdg_counts = state.get("mdg_last_epoch_counts", {})
    target_counts = state.get("target_last_epoch_counts", {})
    if int(mdg_counts.get("images_total", -1)) != 8922 or int(target_counts.get("images_total", -2)) != 8922:
        raise RuntimeError("The final real alpha1.2 MDG/target image accounting changed")
    if int(target_counts.get("clean_targets", -1)) <= 0 or target_counts.get("clean_targets") != target_counts.get("degraded_targets"):
        raise RuntimeError("The final real alpha1.2 clean/degraded target accounting changed")

    events = read_events()
    callback_failure = next(
        (
            event
            for event in events
            if event.get("event") == "run_attempt_failed"
            and event.get("variant") == "e3_alpha12"
            and event.get("attempt") == 1
            and event.get("numerical_failure") is False
            and event.get("error") == LEGACY_CALLBACK_ERROR
        ),
        None,
    )
    device_failure = next(
        (
            event
            for event in events
            if event.get("event") == "supervisor_failed" and event.get("error") == LEGACY_DEVICE_ERROR
        ),
        None,
    )
    attempt2_training = any(
        event.get("event") == "training_started" and event.get("attempt") == 2 for event in events
    )
    if callback_failure is None or device_failure is None or attempt2_training:
        raise RuntimeError("The exact observer-failure and no-second-training event proof is missing")

    log_path = impl.DATA_ROOT / "logs" / "pretrained_sgd_alpha_ablation.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    required_log_fragments = (
        "EarlyStopping:",
        "Best results observed at epoch 22",
        "122 epochs completed",
        "Results saved to",
        LEGACY_CALLBACK_ERROR,
    )
    if any(fragment not in log_text for fragment in required_log_fragments):
        raise RuntimeError("The alpha1.2 normal early-stop/final-validation log proof is incomplete")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = impl.STATE_ROOT / "pretrained_sgd_alpha_ablation_v2_recovery_archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived: dict[str, str] = {}
    for path in (impl.STATE_PATH, impl.EVENTS_PATH, impl.PROTOCOL_PATH, LEGACY_FAILURE_PATH):
        if path.is_file():
            archived[path.name] = impl.sha256(path)
            shutil.copy2(path, archive_dir / path.name)

    marker = {
        "schema": RECOVERY_COMPLETION_SCHEMA,
        "status": "passed",
        "recorded_utc": impl.utc_now(),
        "source": "known_final_eval_callback_failure_recovery",
        "patch_schema": SUPERVISOR_SCHEMA,
        "run_dir": str(run_dir.resolve()),
        "variant_key": "e3_alpha12",
        "completion_kind": "early_stopping",
        "normal_training_return_confirmed": False,
        "training_loop_early_stop_proven": True,
        "final_evaluation_completed_before_callback_failure": True,
        "observer_failure_after_completion": True,
        "observer_error_type": "RuntimeError",
        "observer_error": LEGACY_CALLBACK_ERROR,
        "observer_failure_event": callback_failure,
        "resume_validation_failure_event": device_failure,
        "final_epoch": KNOWN_RECOVERY_FINAL_EPOCH,
        "target_epochs": 150,
        "patience": 100,
        "best_epoch": KNOWN_RECOVERY_BEST_EPOCH,
        "metric_best_epoch": KNOWN_RECOVERY_BEST_EPOCH,
        "patience_gap": 100,
        "config_sha256": context["config_sha256"],
        "origin_release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
        "recovery_release_manifest_sha256": context["release"]["sha256"],
        "invalid_training_attempt_reused": False,
        "attempt2_started_training": False,
    }
    impl.write_json_atomic(run_dir / supervisor.COMPLETION_EVIDENCE_NAME, marker)
    validated = validate_completed_variant_v2(run_dir, alpha12, context)

    new_protocol = impl.protocol_payload(context)
    impl.write_json_atomic(impl.PROTOCOL_PATH, new_protocol)
    new_state = impl.state_template(context)
    new_state.update(
        started_utc=state.get("started_utc", new_state["started_utc"]),
        status="preflight_passed",
        completed_runs=["e3_alpha12"],
        failed_runs=[],
        attempts={"e3_alpha12": 2},
        current_run=None,
        current_run_key=None,
        current_epoch=KNOWN_RECOVERY_FINAL_EPOCH,
        target_epochs=150,
        protocol_sha256=impl.sha256(impl.PROTOCOL_PATH),
        runtime_evidence=runtime,
        mdg_last_epoch_counts=mdg_counts,
        target_last_epoch_counts=target_counts,
        recovery_audit={
            "schema": RECOVERY_AUDIT_SCHEMA,
            "recovered_utc": impl.utc_now(),
            "archive_dir": str(archive_dir),
            "archived_sha256": archived,
            "previous_release_manifest_sha256": PREVIOUS_RELEASE_MANIFEST_SHA256,
            "current_release_manifest_sha256": context["release"]["sha256"],
            "alpha12_completion": validated,
            "alpha12_best_row": best,
            "alpha12_rerun": False,
            "attempt2_started_training": False,
        },
        last_progress_utc=impl.utc_now(),
        e4_started=False,
    )
    impl.write_json_atomic(impl.STATE_PATH, new_state)
    resolved_failure = {
        "schema": "ring3.pretrained_sgd_alpha_ablation_failure_resolution.v2",
        "status": "resolved_by_audited_recovery",
        "resolved_utc": impl.utc_now(),
        "previous_failure_sha256": archived.get(LEGACY_FAILURE_PATH.name),
        "archive_dir": str(archive_dir),
        "alpha12_rerun": False,
        "next_run": "e3_alpha135",
        "e4_started": False,
    }
    impl.write_json_atomic(LEGACY_FAILURE_PATH, resolved_failure)
    impl.append_event(
        "normal_early_stop_recovered",
        variant="e3_alpha12",
        final_epoch=KNOWN_RECOVERY_FINAL_EPOCH,
        best_epoch=KNOWN_RECOVERY_BEST_EPOCH,
        archive_dir=str(archive_dir),
        alpha12_rerun=False,
        next_run="e3_alpha135",
        supervisor_schema=SUPERVISOR_SCHEMA,
    )
    return {
        "status": "recovered",
        "completion": validated,
        "archive_dir": str(archive_dir),
        "alpha12_rerun": False,
        "next_run": "e3_alpha135",
    }


def configure_runtime_v2() -> None:
    alpha.configure_runtime()
    impl.MANIFEST_PATH = impl.SETUP_ROOT / "pretrained_sgd_alpha_ablation_release_manifest_v2.json"
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.MODEL_MODULE_PATH = impl.SETUP_ROOT / "train_pretrained_sgd_alpha_ablation_v2.py"
    impl.verify_args = mainline_entry.verify_args_with_serialized_device

    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.RECOVERY_FAILURE_PATH = V2_FAILURE_PATH
    supervisor.ORIGINAL_ADD_CALLBACKS = guarded_original_add_callbacks
    supervisor.add_callbacks = add_callbacks_v2
    supervisor.run_completion = run_completion_v2
    supervisor.completed_run = completed_run_v2
    supervisor.validate_completed_variant = validate_completed_variant_v2
    supervisor.install()
    supervisor.summary = alpha.alpha_summary
    impl.summary = alpha.alpha_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args()
    configure_runtime_v2()
    try:
        context = impl.preflight()
        print("PRETRAINED_SGD_ALPHA_V2_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard_result = guard_self_test()
        print("PRETRAINED_SGD_ALPHA_V2_GUARD_SELF_TEST " + json.dumps(guard_result, sort_keys=True), flush=True)
        static_result = alpha.static_self_test(context)
        print("PRETRAINED_SGD_ALPHA_V2_STATIC_SELF_TEST " + json.dumps(static_result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        recovery = recover_known_alpha12_if_needed(context)
        print("PRETRAINED_SGD_ALPHA_V2_RECOVERY " + json.dumps(recovery, ensure_ascii=False, sort_keys=True), flush=True)
        if args.recover_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        payload = {
            "schema": "ring3.pretrained_sgd_alpha_ablation_supervisor_failure.v2",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "supervisor_schema": SUPERVISOR_SCHEMA,
            "e4_started": False,
        }
        impl.write_json_atomic(V2_FAILURE_PATH, payload)
        impl.append_event("supervisor_failed_v2", error_type=type(exc).__name__, error=str(exc))
        print("PRETRAINED_SGD_ALPHA_V2_FAILURE " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
