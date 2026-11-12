from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

import formal_baseline_queue as base
from maritime_degradation import MaritimeDegradation, load_config


EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
MODEL_KEY = "yolov8s"
DATASET_KEY = "seadronessee"
STAGE = "E1"
FORMAL_NAME = "seed0_yolov8s_seadronessee_e1_mdg_img960_e150"
MAX_FORMAL_ATTEMPTS = 3

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
CONFIG_PATH = SETUP_ROOT / "e1_mdg_config.json"
AMENDMENT_PATH = SETUP_ROOT / "formal_runtime_amendment.json"
FORMAL_ROOT = DATA_ROOT / "formal_e1"
DIAGNOSTICS_ROOT = DATA_ROOT / "diagnostics"
FAILED_ROOT = DATA_ROOT / "formal_failed_runs"
STATE_ROOT = DATA_ROOT / "formal_state"
FORMAL_STATE_PATH = STATE_ROOT / "e1_mdg_state.json"
FORMAL_EVENTS_PATH = STATE_ROOT / "e1_mdg_events.jsonl"
FORMAL_PROTOCOL_PATH = STATE_ROOT / "e1_mdg_protocol.json"
SMOKE_STATE_PATH = DIAGNOSTICS_ROOT / "e1_mdg_smoke_state.json"
SMOKE_EVENTS_PATH = DIAGNOSTICS_ROOT / "e1_mdg_smoke_events.jsonl"
BASELINE_STATE_PATH = STATE_ROOT / "baseline_queue_state.json"
BASELINE_RUN = DATA_ROOT / "formal_baselines" / "seed0_yolov8s_seadronessee_img960_e150"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_event(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": utc_now(), "event": event, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_target_dataset(lock: dict[str, Any]) -> dict[str, Any]:
    protocol = lock["dataset_protocols"][DATASET_KEY]
    yaml_path = Path(protocol["yaml"])
    if not yaml_path.is_file() or sha256(yaml_path) != protocol["yaml_sha256"]:
        raise RuntimeError(f"Locked SeaDronesSee YAML is missing or changed: {yaml_path}")
    manifest_path = Path(protocol["manifest_path"])
    if not manifest_path.is_file() or sha256(manifest_path) != protocol["manifest_sha256"]:
        raise RuntimeError(f"Locked SeaDronesSee manifest is missing or changed: {manifest_path}")

    verifier = SETUP_ROOT / "verify_yolo_dataset.py"
    result = subprocess.run(
        [sys.executable, str(verifier), str(yaml_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("SeaDronesSee verification failed:\n" + result.stdout + result.stderr)
    report = json.loads(result.stdout)
    if report.get("label_error_count") or any(report.get("missing_label_counts", {}).values()):
        raise RuntimeError("SeaDronesSee verification reported label errors")
    expected_counts = protocol["split_counts"]
    observed_counts = report.get("counts", {})
    if any(observed_counts.get(split) != count for split, count in expected_counts.items()):
        raise RuntimeError(
            "SeaDronesSee split counts changed: "
            + json.dumps({"expected": expected_counts, "observed": observed_counts}, ensure_ascii=False)
        )
    return report


def read_best_metric_row(results_path: Path, metric: str = "metrics/mAP50-95(B)") -> dict[str, str]:
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    finite = [row for row in rows if row.get(metric) and math.isfinite(float(row[metric]))]
    if not finite:
        raise RuntimeError(f"No finite {metric} row in {results_path}")
    return max(finite, key=lambda row: float(row[metric]))


def verify_baseline_complete(lock: dict[str, Any]) -> dict[str, Any]:
    if not BASELINE_STATE_PATH.is_file():
        raise RuntimeError(f"Baseline state is missing: {BASELINE_STATE_PATH}")
    queue_state = load_json(BASELINE_STATE_PATH)
    if queue_state.get("status") != "completed" or queue_state.get("lock_sha256") != EXPECTED_LOCK_SHA256:
        raise RuntimeError(f"Baseline segment is not locked-complete: {queue_state.get('status')!r}")
    complete, final_row = base.completed_run(BASELINE_RUN, int(lock["common_train"]["epochs"]))
    if not complete or final_row is None:
        raise RuntimeError(f"E0 baseline is incomplete or non-finite: {BASELINE_RUN}")
    args_path = BASELINE_RUN / "args.yaml"
    base.verify_resume_args(BASELINE_RUN, lock, DATASET_KEY)
    best_row = read_best_metric_row(BASELINE_RUN / "results.csv")
    mean_seconds = float(final_row["time"]) / float(final_row["epoch"])
    return {
        "run_dir": str(BASELINE_RUN),
        "args_sha256": sha256(args_path),
        "final_row": final_row,
        "best_row": best_row,
        "measured_mean_seconds_per_epoch": mean_seconds,
    }


def code_hashes() -> dict[str, str]:
    paths = {
        "train_e1_mdg.py": Path(__file__).resolve(),
        "maritime_degradation.py": SETUP_ROOT / "maritime_degradation.py",
        "e1_mdg_config.json": CONFIG_PATH,
    }
    return {name: sha256(path) for name, path in paths.items()}


def preflight() -> dict[str, Any]:
    lock = base.load_lock()
    lock_hash = base.verify_lock_immutability()
    if lock_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError(f"Formal lock hash mismatch: {lock_hash}")
    if not AMENDMENT_PATH.is_file() or sha256(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Formal runtime amendment is missing or changed")
    base.verify_environment(lock)
    base.validate_training_lock(lock)

    weight = lock["weights"][MODEL_KEY]
    weight_path = Path(weight["path"])
    if not weight_path.is_file() or sha256(weight_path) != weight["sha256"]:
        raise RuntimeError(f"Locked YOLOv8s weight is missing or changed: {weight_path}")
    dataset_report = verify_target_dataset(lock)
    baseline = verify_baseline_complete(lock)
    config = load_config(CONFIG_PATH)

    return {
        "lock": lock,
        "lock_sha256": lock_hash,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config": config,
        "mdg_config_sha256": sha256(CONFIG_PATH),
        "code_sha256": code_hashes(),
        "dataset_report": dataset_report,
        "baseline": baseline,
    }


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    lock = context["lock"]
    return {
        "schema": "ring3.e1_mdg_protocol.v1",
        "stage": STAGE,
        "model": MODEL_KEY,
        "dataset": DATASET_KEY,
        "formal_name": FORMAL_NAME,
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "code_sha256": context["code_sha256"],
        "weight_sha256": lock["weights"][MODEL_KEY]["sha256"],
        "dataset_yaml_sha256": lock["dataset_protocols"][DATASET_KEY]["yaml_sha256"],
        "dataset_manifest_sha256": lock["dataset_protocols"][DATASET_KEY]["manifest_sha256"],
        "e0_args_sha256": context["baseline"]["args_sha256"],
        "invariants": context["mdg_config"]["invariants"],
    }


def verify_smoke_evidence(context: dict[str, Any]) -> dict[str, Any]:
    if not SMOKE_STATE_PATH.is_file():
        raise RuntimeError("E1 formal run requires a completed MDG smoke test")
    smoke = load_json(SMOKE_STATE_PATH)
    expected = protocol_payload(context)
    if smoke.get("status") != "passed":
        raise RuntimeError(f"Latest MDG smoke test did not pass: {smoke.get('status')!r}")
    for key in ("formal_lock_sha256", "runtime_amendment_sha256", "mdg_config_sha256", "code_sha256"):
        if smoke.get(key) != expected[key]:
            raise RuntimeError(f"MDG smoke evidence is stale for {key}")
    return smoke


def activate_protocol(context: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(context)
    if FORMAL_PROTOCOL_PATH.exists():
        recorded = load_json(FORMAL_PROTOCOL_PATH)
        if recorded != payload:
            raise RuntimeError("E1 protocol changed after formal activation")
    else:
        write_json_atomic(FORMAL_PROTOCOL_PATH, payload)
    return payload


class MDGDetectionTrainer(DetectionTrainer):
    """Ultralytics detection trainer with E1 train-only MDG in preprocess_batch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config_path = Path(os.environ["RING3_E1_MDG_CONFIG"])
        expected_hash = os.environ["RING3_E1_MDG_CONFIG_SHA256"]
        if sha256(config_path) != expected_hash:
            raise RuntimeError("MDG config hash changed before trainer construction")
        self.mdg = MaritimeDegradation(load_config(config_path))
        self._mdg_epoch_counts: Counter[str] = Counter()
        self._mdg_total_counts: Counter[str] = Counter()
        super().__init__(*args, **kwargs)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        batch["img"], counts = self.mdg.apply_batch(batch["img"])
        self._mdg_epoch_counts.update(counts)
        self._mdg_total_counts.update(counts)
        return batch

    def consume_mdg_epoch_counts(self) -> dict[str, int]:
        counts = dict(self._mdg_epoch_counts)
        self._mdg_epoch_counts.clear()
        return counts

    def mdg_total_counts(self) -> dict[str, int]:
        return dict(self._mdg_total_counts)


def row_has_nonfinite(row: dict[str, str] | None, prefixes: tuple[str, ...]) -> bool:
    if row is None:
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


def initial_state(mode: str, run_dir: Path, context: dict[str, Any], target_epochs: int) -> dict[str, Any]:
    expected_seconds = context["baseline"]["measured_mean_seconds_per_epoch"] * 1.1
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.e1_mdg_state.v1",
        "status": "preflight_passed",
        "mode": mode,
        "stage": STAGE,
        "current_run": run_dir.name,
        "run_dir": str(run_dir),
        "current_epoch": 0,
        "target_epochs": target_epochs,
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "expected_seconds_per_epoch": expected_seconds,
        "eta_utc": (now + timedelta(seconds=expected_seconds * target_epochs)).isoformat(),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "code_sha256": context["code_sha256"],
        "baseline": context["baseline"],
        "recent_epoch_seconds": [],
        "validation_nan_streak": 0,
        "attempts": 0,
    }


def archive_stale_run(run_dir: Path, reason: str, events_path: Path) -> Path:
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = FAILED_ROOT / f"{run_dir.name}__{reason}__{stamp}"
    if destination.exists():
        raise RuntimeError(f"Failed-run archive already exists: {destination}")
    os.replace(run_dir, destination)
    append_event(events_path, "run_archived", reason=reason, source=str(run_dir), destination=str(destination))
    return destination


def training_metadata(mode: str, context: dict[str, Any], kwargs: dict[str, Any], attempt: int) -> dict[str, Any]:
    payload = protocol_payload(context)
    payload.update(
        {
            "schema": "ring3.e1_mdg_run_metadata.v1",
            "mode": mode,
            "attempt": attempt,
            "created_utc": utc_now(),
            "training_kwargs": kwargs,
            "diagnostic_excluded_from_paper_results": mode == "smoke",
            "e0_baseline": context["baseline"],
        }
    )
    return payload


def add_callbacks(
    model: YOLO,
    *,
    mode: str,
    state: dict[str, Any],
    state_path: Path,
    events_path: Path,
    metadata: dict[str, Any],
) -> None:
    def on_train_start(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir)
        state.update(status="running", run_dir=str(save_dir), last_progress_utc=utc_now())
        write_json_atomic(state_path, state)
        write_json_atomic(save_dir / "e1_mdg_metadata.json", metadata)
        shutil.copyfile(CONFIG_PATH, save_dir / "e1_mdg_config.json")
        append_event(events_path, "training_started", mode=mode, run_dir=str(save_dir), attempt=state.get("attempts", 1))

    def on_train_batch_end(trainer: Any) -> None:
        loss = trainer.loss
        if loss is not None and not bool(loss.detach().isfinite().all()):
            raise FloatingPointError(f"Non-finite E1 training loss at epoch {int(trainer.epoch) + 1}")

    def on_fit_epoch_end(trainer: Any) -> None:
        raw_epoch = int(trainer.epoch) + 1
        save_dir = Path(trainer.save_dir)
        row = base.read_last_row(save_dir / "results.csv")
        if row_has_nonfinite(row, ("train/",)):
            raise FloatingPointError(f"Non-finite E1 epoch training loss at epoch {raw_epoch}")
        val_bad = row_has_nonfinite(row, ("val/",))
        streak = int(state.get("validation_nan_streak", 0)) + 1 if val_bad else 0
        if streak >= 3:
            raise FloatingPointError(f"E1 validation loss remained non-finite for {streak} epochs")

        epoch_counts = trainer.consume_mdg_epoch_counts()
        target_epochs = int(state["target_epochs"])
        if not epoch_counts and int(state.get("current_epoch", 0)) >= target_epochs:
            return
        epoch = min(raw_epoch, target_epochs)
        recent = list(state.get("recent_epoch_seconds", []))
        cumulative = None
        if row and row.get("time"):
            cumulative = float(row["time"])
            previous = state.get("last_cumulative_train_seconds")
            measured = cumulative - float(previous) if previous is not None else cumulative
            if math.isfinite(measured) and measured > 0.0:
                recent.append(measured)
                recent = recent[-10:]
        expected = sum(recent) / len(recent) if recent else float(state["expected_seconds_per_epoch"])
        remaining = max(0, target_epochs - epoch)
        now = datetime.now(timezone.utc)
        state.update(
            status="running",
            current_epoch=epoch,
            last_progress_utc=now.isoformat(),
            validation_nan_streak=streak,
            mdg_last_epoch_counts=epoch_counts,
            mdg_total_counts=trainer.mdg_total_counts(),
            recent_epoch_seconds=recent,
            expected_seconds_per_epoch=expected,
            eta_utc=(now + timedelta(seconds=remaining * expected)).isoformat(),
        )
        if cumulative is not None:
            state["last_cumulative_train_seconds"] = cumulative
        write_json_atomic(state_path, state)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(state["target_epochs"]):
            append_event(
                events_path,
                "epoch_completed",
                epoch=epoch,
                expected_seconds_per_epoch=expected,
                eta_utc=state["eta_utc"],
                mdg_counts=epoch_counts,
            )
        print(
            "E1_STATE "
            + json.dumps(
                {
                    "epoch": epoch,
                    "target": state["target_epochs"],
                    "seconds_per_epoch": round(expected, 3),
                    "eta_utc": state["eta_utc"],
                    "mdg_counts": epoch_counts,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def run_once(mode: str, context: dict[str, Any]) -> int:
    lock = context["lock"]
    target_epochs = 1 if mode == "smoke" else int(lock["common_train"]["epochs"])
    if mode == "smoke":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"diagnostic_e1_mdg_yolov8s_seadronessee_img960_e1_{stamp}"
        output_root = DIAGNOSTICS_ROOT
        state_path, events_path = SMOKE_STATE_PATH, SMOKE_EVENTS_PATH
    else:
        name = FORMAL_NAME
        output_root = FORMAL_ROOT
        state_path, events_path = FORMAL_STATE_PATH, FORMAL_EVENTS_PATH
    run_dir = output_root / name

    existing_state = load_json(state_path) if state_path.exists() else None
    if mode == "formal" and run_dir.exists():
        complete, final_row = base.completed_run(run_dir, target_epochs)
        if complete:
            state = existing_state or initial_state(mode, run_dir, context, target_epochs)
            state.update(status="completed", current_epoch=target_epochs, final_row=final_row, completed_utc=utc_now())
            write_json_atomic(state_path, state)
            return 0
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            archive_stale_run(run_dir, "no_resumable_checkpoint", events_path)

    state = existing_state or initial_state(mode, run_dir, context, target_epochs)
    if mode == "smoke":
        state = initial_state(mode, run_dir, context, target_epochs)
        attempt = 1
    else:
        attempt = int(state.get("attempts", 0)) + 1
        if attempt > MAX_FORMAL_ATTEMPTS:
            state.update(status="failed_exhausted", last_progress_utc=utc_now())
            write_json_atomic(state_path, state)
            append_event(events_path, "attempts_exhausted", attempts=state.get("attempts", 0))
            return 3
        state["attempts"] = attempt
        state["status"] = "starting"
        state["last_progress_utc"] = utc_now()
        write_json_atomic(state_path, state)

    kwargs = base.training_kwargs(lock, MODEL_KEY, DATASET_KEY, name)
    kwargs["project"] = str(output_root)
    kwargs["epochs"] = target_epochs

    if mode == "formal" and run_dir.exists():
        base.verify_resume_args(run_dir, lock, DATASET_KEY)
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        train_kwargs: dict[str, Any] = {"resume": True}
        resumed = True
    else:
        model = YOLO(lock["weights"][MODEL_KEY]["path"])
        train_kwargs = kwargs
        resumed = False

    os.environ["RING3_E1_MDG_CONFIG"] = str(CONFIG_PATH)
    os.environ["RING3_E1_MDG_CONFIG_SHA256"] = context["mdg_config_sha256"]
    metadata = training_metadata(mode, context, kwargs, attempt)
    metadata["resumed"] = resumed
    add_callbacks(
        model,
        mode=mode,
        state=state,
        state_path=state_path,
        events_path=events_path,
        metadata=metadata,
    )
    append_event(events_path, "attempt_started", mode=mode, attempt=attempt, resumed=resumed, run_dir=str(run_dir))

    try:
        model.train(trainer=MDGDetectionTrainer, **train_kwargs)
        actual_dir = Path(state.get("run_dir", run_dir))
        complete, final_row = base.completed_run(actual_dir, target_epochs)
        if not complete or final_row is None:
            raise RuntimeError(f"E1 run ended without a finite epoch-{target_epochs} result: {actual_dir}")
        state.update(
            status="passed" if mode == "smoke" else "completed",
            current_epoch=target_epochs,
            final_row=final_row,
            completed_utc=utc_now(),
            last_progress_utc=utc_now(),
        )
        write_json_atomic(state_path, state)
        append_event(events_path, "run_completed", mode=mode, attempt=attempt, final_row=final_row)
        return 0
    except BaseException as exc:
        state.update(
            status="failed" if mode == "smoke" else "attempt_failed",
            last_progress_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_json_atomic(state_path, state)
        append_event(
            events_path,
            "run_attempt_failed",
            mode=mode,
            attempt=attempt,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        traceback.print_exc()
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked SeaDronesSee E1 (YOLOv8s + online MDG only)")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()

    try:
        context = preflight()
        if args.mode == "formal":
            verify_smoke_evidence(context)
            activate_protocol(context)
        return run_once(args.mode, context)
    except BaseException as exc:
        state_path = SMOKE_STATE_PATH if args.mode == "smoke" else FORMAL_STATE_PATH
        events_path = SMOKE_EVENTS_PATH if args.mode == "smoke" else FORMAL_EVENTS_PATH
        previous = load_json(state_path) if state_path.exists() else {}
        previous.update(
            status="preflight_failed",
            mode=args.mode,
            stage=STAGE,
            last_progress_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_json_atomic(state_path, previous)
        append_event(events_path, "preflight_failed", mode=args.mode, error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
