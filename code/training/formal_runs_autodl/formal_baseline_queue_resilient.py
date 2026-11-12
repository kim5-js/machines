from __future__ import annotations

import json
import math
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

import formal_baseline_queue as base


AMENDMENT_PATH = base.SETUP_ROOT / "formal_runtime_amendment.json"
AMENDMENT_HASH_PATH = base.STATE_ROOT / "formal_runtime_amendment.sha256"
FAILED_OUTPUT_ROOT = base.DATA_ROOT / "formal_failed_runs"
RTDETR_FP32_ETA_FACTOR = 1.8


def load_amendment() -> dict[str, Any]:
    if not AMENDMENT_PATH.is_file():
        raise RuntimeError(f"Runtime amendment is missing: {AMENDMENT_PATH}")
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment.get("schema") != "ring3.formal_runtime_amendment.v1":
        raise RuntimeError(
            f"Unexpected runtime amendment schema: {amendment.get('schema')!r}"
        )
    if amendment.get("status") != "authorized_active":
        raise RuntimeError(
            "Runtime amendment requires status='authorized_active'; "
            f"found {amendment.get('status')!r}"
        )
    if amendment.get("preserved_formal_lock_sha256") != base.sha256(base.LOCK_PATH):
        raise RuntimeError("Runtime amendment does not reference the active formal lock")
    if amendment.get("model_overrides") != {"rtdetr_l": {"amp": False}}:
        raise RuntimeError("Only the authorized RT-DETR amp=false override is permitted")
    policy = amendment.get("failure_policy", {})
    if policy != {
        "max_attempts_per_run": 3,
        "archive_terminal_nonfinite": True,
        "continue_after_exhausted": True,
    }:
        raise RuntimeError(f"Unexpected failure policy: {policy!r}")
    return amendment


def verify_amendment_immutability() -> str:
    current = base.sha256(AMENDMENT_PATH)
    base.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if AMENDMENT_HASH_PATH.exists():
        recorded = AMENDMENT_HASH_PATH.read_text(encoding="ascii").strip()
        if recorded != current:
            raise RuntimeError(
                "Runtime amendment changed after activation: "
                f"recorded={recorded}, current={current}"
            )
    else:
        AMENDMENT_HASH_PATH.write_text(current + "\n", encoding="ascii")
    return current


def effective_amp(
    lock: dict[str, Any], amendment: dict[str, Any], model_key: str
) -> bool:
    override = amendment["model_overrides"].get(model_key, {})
    return bool(override.get("amp", lock["common_train"]["amp"]))


def effective_seconds_per_epoch(model_key: str, dataset_key: str) -> float:
    calibrated = base.SECONDS_PER_EPOCH[(model_key, dataset_key)]
    return calibrated * (RTDETR_FP32_ETA_FACTOR if model_key == "rtdetr_l" else 1.0)


def row_epoch(row: dict[str, str] | None) -> int:
    if not row:
        return 0
    try:
        return int(float(row["epoch"]))
    except (KeyError, TypeError, ValueError):
        return 0


def row_has_nonfinite(row: dict[str, str] | None, prefixes: tuple[str, ...]) -> bool:
    if not row:
        return False
    for key, value in row.items():
        if not key.startswith(prefixes) or not value:
            continue
        try:
            if not math.isfinite(float(value)):
                return True
        except ValueError:
            continue
    return False


def archive_run(
    run_dir: Path,
    *,
    reason: str,
    lock_hash: str,
    amendment_hash: str,
    final_row: dict[str, str] | None,
) -> Path:
    FAILED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = FAILED_OUTPUT_ROOT / f"{run_dir.name}__{reason}__{stamp}"
    run_dir.rename(destination)
    base.write_json_atomic(
        destination / "archive_record.json",
        {
            "schema": "ring3.formal_failed_run_archive.v1",
            "archived_utc": base.utc_now(),
            "reason": reason,
            "source_run": run_dir.name,
            "formal_lock_sha256": lock_hash,
            "runtime_amendment_sha256": amendment_hash,
            "final_row": final_row,
        },
    )
    return destination


def verify_effective_resume_args(
    run_dir: Path,
    lock: dict[str, Any],
    amendment: dict[str, Any],
    model_key: str,
    dataset_key: str,
) -> None:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise RuntimeError(f"Cannot resume without args.yaml: {run_dir}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    common = lock["common_train"]
    expected = {
        "data": lock["dataset_protocols"][dataset_key]["yaml"],
        "epochs": common["epochs"],
        "imgsz": common["imgsz"],
        "batch": common["batch"],
        "optimizer": common["optimizer"],
        "lr0": lock["dataset_protocols"][dataset_key]["lr0"],
        "lrf": common["lrf"],
        "momentum": common["momentum"],
        "weight_decay": common["weight_decay"],
        "seed": common["seed"],
        "amp": effective_amp(lock, amendment, model_key),
    }
    mismatches = [
        f"{key}: expected {value!r}, found {args.get(key)!r}"
        for key, value in expected.items()
        if args.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Refusing to resume mismatched run {run_dir}:\n" + "\n".join(mismatches)
        )


def is_numerical_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, FloatingPointError) or "nan" in message or "non-finite" in message


def run_attempt(
    *,
    lock: dict[str, Any],
    amendment: dict[str, Any],
    state: dict[str, Any],
    model_key: str,
    dataset_key: str,
    name: str,
    run_dir: Path,
) -> tuple[bool, dict[str, str] | None]:
    if run_dir.exists():
        verify_effective_resume_args(run_dir, lock, amendment, model_key, dataset_key)
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Incomplete run lacks last.pt: {run_dir}")
        model = base.YOLO(str(checkpoint))
        kwargs: dict[str, Any] = {"resume": True}
        resumed = True
    else:
        model = base.YOLO(lock["weights"][model_key]["path"])
        kwargs = base.training_kwargs(lock, model_key, dataset_key, name)
        kwargs["amp"] = effective_amp(lock, amendment, model_key)
        resumed = False

    base.append_event(
        "run_process_started",
        run=name,
        resumed=resumed,
        effective_amp=effective_amp(lock, amendment, model_key),
    )

    def on_train_batch_end(trainer: Any) -> None:
        loss = trainer.loss
        if loss is not None and not bool(loss.detach().isfinite().all()):
            raise FloatingPointError(
                f"Non-finite training loss in {name} at epoch {int(trainer.epoch) + 1}"
            )

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        latest = base.read_last_row(run_dir / "results.csv")
        train_bad = row_has_nonfinite(latest, ("train/",))
        val_bad = row_has_nonfinite(latest, ("val/",))
        streaks = state.setdefault("validation_nan_streaks", {})
        streaks[name] = int(streaks.get(name, 0)) + 1 if val_bad else 0
        state.update(current_epoch=epoch, last_progress_utc=base.utc_now())
        base.write_json_atomic(base.STATE_PATH, state)
        if train_bad:
            raise FloatingPointError(f"Non-finite epoch training loss in {name} at epoch {epoch}")
        if streaks[name] >= 3:
            raise FloatingPointError(
                f"Validation loss remained non-finite for {streaks[name]} epochs in {name}"
            )

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(**kwargs)
    return base.completed_run(run_dir, int(lock["common_train"]["epochs"]))


def execute_queue(
    lock: dict[str, Any],
    amendment: dict[str, Any],
    state: dict[str, Any],
    lock_hash: str,
    amendment_hash: str,
) -> None:
    epochs = int(lock["common_train"]["epochs"])
    max_attempts = int(amendment["failure_policy"]["max_attempts_per_run"])
    state.setdefault("completed_runs", [])
    state.setdefault("failed_runs", {})
    state.setdefault("run_attempts", {})
    state.setdefault("validation_nan_streaks", {})

    for model_key in base.MODEL_ORDER:
        for dataset_key in base.DATASET_ORDER:
            name = base.run_name(model_key, dataset_key, lock)
            run_dir = base.OUTPUT_ROOT / name
            complete, final_row = base.completed_run(run_dir, epochs)
            if complete:
                if name not in state["completed_runs"]:
                    state["completed_runs"].append(name)
                state["failed_runs"].pop(name, None)
                base.append_event("run_skipped_complete", run=name, final_row=final_row)
                base.write_json_atomic(base.STATE_PATH, state)
                continue

            if run_dir.exists() and row_epoch(final_row) >= epochs:
                archived = archive_run(
                    run_dir,
                    reason="terminal_nonfinite",
                    lock_hash=lock_hash,
                    amendment_hash=amendment_hash,
                    final_row=final_row,
                )
                base.append_event(
                    "run_archived_for_retry",
                    run=name,
                    reason="terminal_nonfinite",
                    archived_to=str(archived),
                )
                final_row = None

            attempts_used = int(state["run_attempts"].get(name, 0))
            succeeded = False
            for attempt in range(attempts_used + 1, max_attempts + 1):
                if base.STOP_SENTINEL.exists():
                    state.update(
                        status="paused_by_explicit_sentinel",
                        current_run=None,
                        last_progress_utc=base.utc_now(),
                    )
                    base.write_json_atomic(base.STATE_PATH, state)
                    base.append_event("queue_paused", reason=str(base.STOP_SENTINEL))
                    return

                current_row = base.read_last_row(run_dir / "results.csv")
                current_epoch = row_epoch(current_row)
                expected_per_epoch = effective_seconds_per_epoch(model_key, dataset_key)
                now = datetime.now(timezone.utc)
                state["run_attempts"][name] = attempt
                state["validation_nan_streaks"][name] = 0 if not run_dir.exists() else int(
                    state["validation_nan_streaks"].get(name, 0)
                )
                state.update(
                    status="running",
                    current_run=name,
                    current_model=model_key,
                    current_dataset=dataset_key,
                    current_epoch=current_epoch,
                    target_epochs=epochs,
                    expected_seconds_per_epoch=round(expected_per_epoch, 3),
                    expected_run_end_utc=(
                        now + timedelta(seconds=expected_per_epoch * (epochs - current_epoch))
                    ).isoformat(),
                    effective_amp=effective_amp(lock, amendment, model_key),
                    last_progress_utc=base.utc_now(),
                )
                base.write_json_atomic(base.STATE_PATH, state)
                base.append_event(
                    "run_attempt_started",
                    run=name,
                    model=model_key,
                    dataset=dataset_key,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    resumed=run_dir.exists(),
                    effective_amp=effective_amp(lock, amendment, model_key),
                )

                try:
                    complete, final_row = run_attempt(
                        lock=lock,
                        amendment=amendment,
                        state=state,
                        model_key=model_key,
                        dataset_key=dataset_key,
                        name=name,
                        run_dir=run_dir,
                    )
                    if not complete:
                        raise RuntimeError(
                            f"Run ended without a finite epoch-{epochs} result: "
                            f"{run_dir}; last_row={final_row}"
                        )
                except Exception as exc:
                    failure = {
                        "attempt": attempt,
                        "failed_utc": base.utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    latest = base.read_last_row(run_dir / "results.csv")
                    should_archive = run_dir.exists() and (
                        is_numerical_error(exc) or row_epoch(latest) >= epochs
                    )
                    if should_archive:
                        reason = "numerical" if is_numerical_error(exc) else "terminal_nonfinite"
                        archived = archive_run(
                            run_dir,
                            reason=reason,
                            lock_hash=lock_hash,
                            amendment_hash=amendment_hash,
                            final_row=latest,
                        )
                        failure["archived_to"] = str(archived)
                    state["last_run_failure"] = {"run": name, **failure}
                    state["last_progress_utc"] = base.utc_now()
                    base.write_json_atomic(base.STATE_PATH, state)
                    base.append_event("run_attempt_failed", run=name, **failure)
                    if attempt < max_attempts:
                        state["status"] = "retrying"
                        state["validation_nan_streaks"][name] = 0
                        base.write_json_atomic(base.STATE_PATH, state)
                        base.append_event(
                            "run_retry_scheduled",
                            run=name,
                            next_attempt=attempt + 1,
                        )
                        continue
                    state["failed_runs"][name] = failure
                    state.update(
                        status="continuing_after_exhausted_run",
                        current_run=None,
                        current_epoch=0,
                        last_progress_utc=base.utc_now(),
                    )
                    base.write_json_atomic(base.STATE_PATH, state)
                    base.append_event(
                        "run_attempts_exhausted_continuing",
                        run=name,
                        attempts=max_attempts,
                        error=str(exc),
                    )
                    break
                else:
                    if name not in state["completed_runs"]:
                        state["completed_runs"].append(name)
                    state["failed_runs"].pop(name, None)
                    state["validation_nan_streaks"].pop(name, None)
                    state.update(
                        current_run=None,
                        current_epoch=epochs,
                        last_progress_utc=base.utc_now(),
                    )
                    base.write_json_atomic(base.STATE_PATH, state)
                    base.append_event(
                        "run_completed",
                        run=name,
                        attempt=attempt,
                        effective_amp=effective_amp(lock, amendment, model_key),
                        final_row=final_row,
                    )
                    succeeded = True
                    break

            if not succeeded and name not in state["failed_runs"]:
                state["failed_runs"][name] = {
                    "failed_utc": base.utc_now(),
                    "error_type": "AttemptBudgetExhausted",
                    "error": "No run attempts remained after process restart",
                }

    segment_status = "completed_with_failures" if state["failed_runs"] else "completed"
    state.update(
        status=segment_status,
        current_run=None,
        current_epoch=epochs,
        segment_completed_utc=base.utc_now(),
        last_progress_utc=base.utc_now(),
    )
    base.write_json_atomic(base.STATE_PATH, state)
    base.append_event(
        "queue_segment_completed",
        status=segment_status,
        completed_runs=state["completed_runs"],
        failed_runs=state["failed_runs"],
    )


def main() -> None:
    lock = base.load_lock()
    amendment = load_amendment()
    lock_hash = base.preflight(lock)
    amendment_hash = verify_amendment_immutability()

    if base.STATE_PATH.exists():
        state = json.loads(base.STATE_PATH.read_text(encoding="utf-8"))
        if state.get("lock_sha256") != lock_hash:
            raise RuntimeError("Queue state belongs to a different formal lock")
    else:
        state = base.initial_state(lock, lock_hash)

    existing_amendment_hash = state.get("runtime_amendment_sha256")
    if existing_amendment_hash and existing_amendment_hash != amendment_hash:
        raise RuntimeError("Queue state belongs to a different runtime amendment")
    if state.get("status") == "failed":
        state.setdefault("prior_queue_failures", []).append(
            {
                key: state.get(key)
                for key in ("failed_utc", "error_type", "error", "traceback", "current_run")
            }
        )
    for key in ("failed_utc", "error_type", "error", "traceback"):
        state.pop(key, None)
    state.update(
        runtime_amendment_path=str(AMENDMENT_PATH),
        runtime_amendment_sha256=amendment_hash,
        failure_policy=amendment["failure_policy"],
        last_progress_utc=base.utc_now(),
    )
    base.write_json_atomic(base.STATE_PATH, state)
    base.append_event(
        "resilient_queue_started_or_resumed",
        lock_sha256=lock_hash,
        runtime_amendment_sha256=amendment_hash,
    )

    try:
        execute_queue(lock, amendment, state, lock_hash, amendment_hash)
    except BaseException as exc:
        state.update(
            status="supervisor_failed",
            failed_utc=base.utc_now(),
            last_progress_utc=base.utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        base.write_json_atomic(base.STATE_PATH, state)
        base.append_event(
            "resilient_queue_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            current_run=state.get("current_run"),
        )
        raise


if __name__ == "__main__":
    main()
