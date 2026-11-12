from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(
    os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl"))
)
LOCK_PATH = SETUP_ROOT / "formal_experiment_lock.json"
OUTPUT_ROOT = DATA_ROOT / "formal_baselines"
STATE_ROOT = DATA_ROOT / "formal_state"
STATE_PATH = STATE_ROOT / "baseline_queue_state.json"
EVENTS_PATH = STATE_ROOT / "baseline_queue_events.jsonl"
LOCK_HASH_PATH = STATE_ROOT / "formal_experiment_lock.sha256"
STOP_SENTINEL = STATE_ROOT / "STOP_AFTER_CURRENT"

os.environ.setdefault("PIP_CACHE_DIR", str(DATA_ROOT / ".cache" / "pip"))
os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_ROOT / ".config" / "Ultralytics"))

import torch
import torchvision
import ultralytics
import yaml
from ultralytics import YOLO


MODEL_ORDER = ("yolov8s", "yolo11s", "yolo26s", "rtdetr_l")
DATASET_ORDER = ("flow", "wsodd", "seadronessee")
SECONDS_PER_EPOCH = {
    ("yolov8s", "flow"): 25.453,
    ("yolov8s", "wsodd"): 62.229,
    ("yolov8s", "seadronessee"): 162.346,
    ("yolo11s", "flow"): 25.388,
    ("yolo11s", "wsodd"): 65.255,
    ("yolo11s", "seadronessee"): 158.398,
    ("yolo26s", "flow"): 27.241,
    ("yolo26s", "wsodd"): 82.827,
    ("yolo26s", "seadronessee"): 174.558,
    ("rtdetr_l", "flow"): 64.024,
    ("rtdetr_l", "wsodd"): 212.583,
    ("rtdetr_l", "seadronessee"): 436.697,
}


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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def append_event(event: str, **details: Any) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": utc_now(), "event": event, **details}
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise RuntimeError(f"Formal lock is missing: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("schema") != "ring3.formal_experiment_lock.v1":
        raise RuntimeError(f"Unexpected lock schema: {lock.get('schema')!r}")
    if lock.get("status") != "locked_authorized":
        raise RuntimeError(
            "Formal training requires status='locked_authorized'; "
            f"found {lock.get('status')!r}"
        )
    return lock


def verify_lock_immutability() -> str:
    current = sha256(LOCK_PATH)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_HASH_PATH.exists():
        recorded = LOCK_HASH_PATH.read_text(encoding="ascii").strip()
        if recorded != current:
            raise RuntimeError(
                "Formal lock changed after activation: "
                f"recorded={recorded}, current={current}"
            )
    else:
        LOCK_HASH_PATH.write_text(current + "\n", encoding="ascii")
    return current


def verify_environment(lock: dict[str, Any]) -> None:
    expected = lock["environment"]
    observed = {
        "prefix": sys.prefix,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    failures = [
        f"{key}: expected {expected[key]!r}, observed {observed[key]!r}"
        for key in observed
        if observed[key] != expected[key]
    ]
    if failures:
        raise RuntimeError("Environment lock mismatch:\n" + "\n".join(failures))


def verify_weights(lock: dict[str, Any]) -> None:
    for model_key in MODEL_ORDER:
        spec = lock["weights"][model_key]
        path = Path(spec["path"])
        if not path.is_file():
            raise RuntimeError(f"Missing locked weight for {model_key}: {path}")
        observed = sha256(path)
        if observed != spec["sha256"]:
            raise RuntimeError(
                f"Weight hash mismatch for {model_key}: "
                f"expected={spec['sha256']}, observed={observed}"
            )


def verify_datasets(lock: dict[str, Any]) -> None:
    verifier = SETUP_ROOT / "verify_yolo_dataset.py"
    if not verifier.is_file():
        raise RuntimeError(f"Dataset verifier is missing: {verifier}")
    for dataset_key in DATASET_ORDER:
        protocol = lock["dataset_protocols"][dataset_key]
        yaml_path = Path(protocol["yaml"])
        if not yaml_path.is_file():
            raise RuntimeError(f"Missing dataset YAML for {dataset_key}: {yaml_path}")
        expected_yaml_hash = protocol.get("yaml_sha256")
        if expected_yaml_hash and sha256(yaml_path) != expected_yaml_hash:
            raise RuntimeError(f"Dataset YAML hash mismatch for {dataset_key}: {yaml_path}")
        manifest_path_value = protocol.get("manifest_path")
        if manifest_path_value:
            manifest_path = Path(manifest_path_value)
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Protocol manifest is missing for {dataset_key}: {manifest_path}"
                )
            if sha256(manifest_path) != protocol["manifest_sha256"]:
                raise RuntimeError(
                    f"Protocol manifest hash mismatch for {dataset_key}: {manifest_path}"
                )
        result = subprocess.run(
            [sys.executable, str(verifier), str(yaml_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Dataset verification failed for {dataset_key}:\n"
                + result.stdout
                + result.stderr
            )
        report = json.loads(result.stdout)
        if report.get("label_error_count") or any(
            report.get("missing_label_counts", {}).values()
        ):
            raise RuntimeError(
                f"Dataset verification reported label errors for {dataset_key}: "
                + json.dumps(report, ensure_ascii=False)
            )
        expected_counts = protocol.get("split_counts", {})
        observed_counts = report.get("counts", {})
        count_mismatches = [
            f"{split}: expected {count}, observed {observed_counts.get(split)}"
            for split, count in expected_counts.items()
            if observed_counts.get(split) != count
        ]
        if count_mismatches:
            raise RuntimeError(
                f"Dataset split-count mismatch for {dataset_key}:\n"
                + "\n".join(count_mismatches)
            )


def validate_training_lock(lock: dict[str, Any]) -> None:
    common = lock["common_train"]
    required = {
        "epochs": 150,
        "imgsz": 960,
        "batch": 8,
        "workers": 8,
        "optimizer": "AdamW",
        "momentum": 0.9,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "patience": 150,
        "nbs": 64,
        "seed": 0,
    }
    mismatches = [
        f"{key}: expected {value!r}, found {common.get(key)!r}"
        for key, value in required.items()
        if common.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("Training lock mismatch:\n" + "\n".join(mismatches))
    if str(common.get("optimizer", "")).lower() == "auto":
        raise RuntimeError("optimizer=auto is forbidden for formal training")


def preflight(lock: dict[str, Any]) -> str:
    validate_training_lock(lock)
    verify_environment(lock)
    verify_weights(lock)
    verify_datasets(lock)
    return verify_lock_immutability()


def run_name(model_key: str, dataset_key: str, lock: dict[str, Any]) -> str:
    common = lock["common_train"]
    return (
        f"seed{common['seed']}_{model_key}_{dataset_key}_"
        f"img{common['imgsz']}_e{common['epochs']}"
    )


def read_last_row(results_path: Path) -> dict[str, str] | None:
    if not results_path.is_file():
        return None
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {
        str(key).strip(): value.strip() if isinstance(value, str) else value
        for key, value in rows[-1].items()
    }


def row_is_finite(row: dict[str, str]) -> bool:
    checked = 0
    for key, value in row.items():
        if key == "epoch" or not value:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        checked += 1
        if not math.isfinite(number):
            return False
    return checked > 0


def completed_run(run_dir: Path, epochs: int) -> tuple[bool, dict[str, str] | None]:
    row = read_last_row(run_dir / "results.csv")
    if row is None:
        return False, None
    try:
        final_epoch = int(float(row["epoch"]))
    except (KeyError, TypeError, ValueError):
        return False, row
    weights_ok = all((run_dir / "weights" / name).is_file() for name in ("best.pt", "last.pt"))
    return final_epoch >= epochs and weights_ok and row_is_finite(row), row


def verify_resume_args(run_dir: Path, lock: dict[str, Any], dataset_key: str) -> None:
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


def training_kwargs(
    lock: dict[str, Any], model_key: str, dataset_key: str, name: str
) -> dict[str, Any]:
    common = lock["common_train"]
    protocol = lock["dataset_protocols"][dataset_key]
    return {
        "data": protocol["yaml"],
        "epochs": common["epochs"],
        "imgsz": common["imgsz"],
        "batch": common["batch"],
        "device": 0,
        "workers": common["workers"],
        "optimizer": common["optimizer"],
        "lr0": protocol["lr0"],
        "momentum": common["momentum"],
        "lrf": common["lrf"],
        "weight_decay": common["weight_decay"],
        "warmup_epochs": common["warmup_epochs"],
        "warmup_momentum": common["warmup_momentum"],
        "warmup_bias_lr": common["warmup_bias_lr"],
        "patience": common["patience"],
        "nbs": common["nbs"],
        "amp": common["amp"],
        "deterministic": common["deterministic"],
        "pretrained": common["pretrained"],
        "cache": common["cache"],
        "close_mosaic": common["close_mosaic"],
        "save": common["save"],
        "save_period": common["save_period"],
        "plots": common["plots"],
        "seed": common["seed"],
        "project": str(OUTPUT_ROOT),
        "name": name,
        "exist_ok": False,
        "verbose": True,
    }


def initial_state(lock: dict[str, Any], lock_hash: str) -> dict[str, Any]:
    epochs = int(lock["common_train"]["epochs"])
    expected_seconds = sum(
        SECONDS_PER_EPOCH[(model, dataset)] * epochs
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
    )
    started = datetime.now(timezone.utc)
    return {
        "schema": "ring3.formal_baseline_queue_state.v1",
        "status": "preflight_passed",
        "started_utc": started.isoformat(),
        "last_progress_utc": started.isoformat(),
        "lock_path": str(LOCK_PATH),
        "lock_sha256": lock_hash,
        "run_count": len(MODEL_ORDER) * len(DATASET_ORDER),
        "completed_runs": [],
        "current_run": None,
        "current_epoch": 0,
        "target_epochs": epochs,
        "expected_total_seconds": round(expected_seconds, 3),
        "calibration_based_eta_utc": (started + timedelta(seconds=expected_seconds)).isoformat(),
    }


def execute_queue(lock: dict[str, Any], state: dict[str, Any]) -> None:
    common = lock["common_train"]
    epochs = int(common["epochs"])
    for model_key in MODEL_ORDER:
        for dataset_key in DATASET_ORDER:
            name = run_name(model_key, dataset_key, lock)
            run_dir = OUTPUT_ROOT / name
            complete, final_row = completed_run(run_dir, epochs)
            if complete:
                if name not in state["completed_runs"]:
                    state["completed_runs"].append(name)
                append_event("run_skipped_complete", run=name, final_row=final_row)
                write_json_atomic(STATE_PATH, state)
                continue

            if STOP_SENTINEL.exists():
                state.update(
                    status="paused_before_next_run",
                    current_run=None,
                    last_progress_utc=utc_now(),
                )
                write_json_atomic(STATE_PATH, state)
                append_event("queue_paused", reason=str(STOP_SENTINEL))
                return

            expected_per_epoch = SECONDS_PER_EPOCH[(model_key, dataset_key)]
            now = datetime.now(timezone.utc)
            current_epoch = int(float(final_row["epoch"])) if final_row else 0
            state.update(
                status="running",
                current_run=name,
                current_model=model_key,
                current_dataset=dataset_key,
                current_epoch=current_epoch,
                target_epochs=epochs,
                expected_seconds_per_epoch=expected_per_epoch,
                expected_run_end_utc=(
                    now
                    + timedelta(
                        seconds=expected_per_epoch
                        * (epochs - current_epoch)
                    )
                ).isoformat(),
                last_progress_utc=utc_now(),
            )
            write_json_atomic(STATE_PATH, state)
            append_event(
                "run_started",
                run=name,
                model=model_key,
                dataset=dataset_key,
                resumed=run_dir.exists(),
            )

            if run_dir.exists():
                verify_resume_args(run_dir, lock, dataset_key)
                checkpoint = run_dir / "weights" / "last.pt"
                if not checkpoint.is_file():
                    raise RuntimeError(f"Incomplete run lacks last.pt: {run_dir}")
                model = YOLO(str(checkpoint))
                kwargs: dict[str, Any] = {"resume": True}
            else:
                model = YOLO(lock["weights"][model_key]["path"])
                kwargs = training_kwargs(lock, model_key, dataset_key, name)

            def on_fit_epoch_end(trainer: Any) -> None:
                state.update(
                    current_epoch=int(trainer.epoch) + 1,
                    last_progress_utc=utc_now(),
                )
                write_json_atomic(STATE_PATH, state)

            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.train(**kwargs)

            complete, final_row = completed_run(run_dir, epochs)
            if not complete:
                raise RuntimeError(
                    f"Run ended without a finite epoch-{epochs} result: {run_dir}; "
                    f"last_row={final_row}"
                )
            state["completed_runs"].append(name)
            state.update(
                current_run=None,
                current_epoch=epochs,
                last_progress_utc=utc_now(),
            )
            write_json_atomic(STATE_PATH, state)
            append_event("run_completed", run=name, final_row=final_row)

    state.update(
        status="completed",
        current_run=None,
        current_epoch=epochs,
        completed_utc=utc_now(),
        last_progress_utc=utc_now(),
    )
    write_json_atomic(STATE_PATH, state)
    append_event("queue_completed", completed_runs=state["completed_runs"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the immutable seed-0 modern-baseline matrix on AutoDL"
    )
    parser.add_argument(
        "--preflight-only", action="store_true", help="Verify the lock without training"
    )
    parser.add_argument(
        "--print-matrix", action="store_true", help="Print the fixed run order and exit"
    )
    args = parser.parse_args()

    lock = load_lock()
    lock_hash = preflight(lock)
    if args.print_matrix:
        for index, (model, dataset) in enumerate(
            ((m, d) for m in MODEL_ORDER for d in DATASET_ORDER), start=1
        ):
            print(f"{index:02d} {run_name(model, dataset, lock)}")
        return
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "lock_sha256": lock_hash,
                    "checked_utc": utc_now(),
                },
                indent=2,
            )
        )
        return

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("lock_sha256") != lock_hash:
            raise RuntimeError("Queue state belongs to a different formal lock")
    else:
        state = initial_state(lock, lock_hash)
    write_json_atomic(STATE_PATH, state)
    append_event("queue_started_or_resumed", lock_sha256=lock_hash)

    try:
        execute_queue(lock, state)
    except BaseException as exc:
        state.update(
            status="failed",
            failed_utc=utc_now(),
            last_progress_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_json_atomic(STATE_PATH, state)
        append_event(
            "queue_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            current_run=state.get("current_run"),
        )
        raise


if __name__ == "__main__":
    main()
