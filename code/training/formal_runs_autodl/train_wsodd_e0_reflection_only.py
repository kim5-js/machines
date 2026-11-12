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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from wsodd_reflection_only_dataset import build_or_validate_overlay, resolve_images, sha256


DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
CONFIG_PATH = SETUP_ROOT / "wsodd_e0_reflection_only_config.json"
MANIFEST_PATH = SETUP_ROOT / "wsodd_e0_reflection_only_release_manifest.json"
STATE_PATH = DATA_ROOT / "formal_state" / "wsodd_e0_reflection_only_state.json"
EVENTS_PATH = DATA_ROOT / "formal_state" / "wsodd_e0_reflection_only_events.jsonl"
PROTOCOL_PATH = DATA_ROOT / "formal_state" / "wsodd_e0_reflection_only_protocol.json"
SUMMARY_PATH = DATA_ROOT / "formal_state" / "wsodd_e0_reflection_only_summary.json"
EVALUATION_PATH = DATA_ROOT / "formal_state" / "wsodd_e0_reflection_only_fixed_test_metrics.json"
STOP_SENTINEL = DATA_ROOT / "formal_state" / "STOP_WSODD_E0_REFLECTION_ONLY"
FAILED_ROOT = DATA_ROOT / "formal_failed_runs" / "wsodd_e0_reflection_only"
RUNTIME_EVIDENCE = "wsodd_e0_reflection_only_runtime_evidence.json"
RUN_METADATA = "wsodd_e0_reflection_only_run_metadata.json"
COMPLETION_EVIDENCE = "wsodd_e0_reflection_only_completion_evidence.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def devices_equivalent(expected: Any, observed: Any) -> bool:
    if isinstance(expected, (list, tuple)) or isinstance(observed, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(observed, (list, tuple)):
            return False
        return [str(item).strip() for item in expected] == [str(item).strip() for item in observed]
    return str(expected).strip() == str(observed).strip()


def classify_normal_training_return(
    *,
    final_epoch: int | None,
    target_epochs: int,
    trainer_epoch_zero_based: int,
    stopper_best_epoch_one_based: int,
    patience: int,
) -> str:
    """Classify only a normal Ultralytics return; interruption remains incomplete."""
    if final_epoch is None or final_epoch <= 0:
        return "normal_return_incomplete"
    if final_epoch >= target_epochs and trainer_epoch_zero_based + 1 >= target_epochs:
        return "max_epochs"
    if (
        0 < stopper_best_epoch_one_based <= final_epoch < target_epochs
        and final_epoch - stopper_best_epoch_one_based >= patience
        and trainer_epoch_zero_based + 1 >= final_epoch
    ):
        return "early_stopping"
    return "normal_return_incomplete"


def early_stop_classifier_self_test() -> dict[str, Any]:
    cases = {
        "max_epochs": classify_normal_training_return(
            final_epoch=150,
            target_epochs=150,
            trainer_epoch_zero_based=149,
            stopper_best_epoch_one_based=90,
            patience=100,
        ),
        "early_stopping": classify_normal_training_return(
            final_epoch=125,
            target_epochs=150,
            trainer_epoch_zero_based=124,
            stopper_best_epoch_one_based=25,
            patience=100,
        ),
        "interruption": classify_normal_training_return(
            final_epoch=80,
            target_epochs=150,
            trainer_epoch_zero_based=79,
            stopper_best_epoch_one_based=25,
            patience=100,
        ),
    }
    expected = {
        "max_epochs": "max_epochs",
        "early_stopping": "early_stopping",
        "interruption": "normal_return_incomplete",
    }
    if cases != expected:
        raise RuntimeError(f"Normal-completion classifier self-test failed: {cases}")
    return {"status": "passed", "cases": cases}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_event(event: str, **details: Any) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), "event": event, **details}, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def finite_rows(path: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in read_rows(path):
        finite = True
        for key, value in row.items():
            if not key.startswith(("train/", "val/", "metrics/", "lr/")) or value in (None, ""):
                continue
            try:
                finite = finite and math.isfinite(float(value))
            except ValueError:
                continue
        if finite:
            output.append(row)
    return output


def best_row(path: Path) -> dict[str, str]:
    rows = [row for row in finite_rows(path) if row.get("metrics/mAP50-95(B)")]
    if not rows:
        raise RuntimeError(f"No finite validation rows: {path}")
    return max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "ring3.wsodd_e0_reflection_only_config.v1" or config.get("status") != "locked_requested":
        raise RuntimeError("Unexpected reflection-only config schema or status")
    variant = config.get("variant", {})
    expected_variant = {
        "key": "e0_reflection_only",
        "mdg_mode": "none",
        "dual_view": False,
        "cbam": False,
        "focal_eiou": False,
        "consistency_loss": False,
        "small_object_weighting": False,
    }
    for key, expected in expected_variant.items():
        if variant.get(key) != expected:
            raise RuntimeError(f"Forbidden variant drift for {key}: {variant.get(key)!r}")
    expected_training = {
        "model": "/root/autodl-tmp/weights/yolov8s.pt", "pretrained": True, "epochs": 150,
        "patience": 100, "batch": 8, "imgsz": 960, "workers": 8, "optimizer": "SGD",
        "lr0": 0.006, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0000625,
        "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "nbs": 8, "hsv_h": 0.015,
        "hsv_s": 0.6, "hsv_v": 0.2, "degrees": 0.0, "translate": 0.1,
        "scale": 0.45, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
        "fliplr": 0.5, "bgr": 0.0, "mosaic": 0.7, "mixup": 0.05, "cutmix": 0.0,
        "copy_paste": 0.0, "copy_paste_mode": "flip", "auto_augment": "randaugment",
        "erasing": 0.2, "close_mosaic": 10, "save_period": -1, "amp": True,
        "deterministic": True, "seed": 0, "device": 0, "cache": False, "rect": False,
        "cos_lr": False, "single_cls": False, "fraction": 1.0, "multi_scale": False,
        "dropout": 0.0, "iou": 0.7, "max_det": 300, "val": True, "split": "val",
        "save": True, "plots": True,
    }
    if config.get("training") != expected_training:
        changed = {
            key: {"expected": value, "observed": config.get("training", {}).get(key)}
            for key, value in expected_training.items()
            if config.get("training", {}).get(key) != value
        }
        raise RuntimeError(f"Formal E0 training recipe changed: {changed}")
    reflection = config.get("reflection_only", {})
    expected_reflection = {
        "sample_ratio": 0.1, "sample_seed": 0, "low_light_probability": 0.0,
        "reflection_probability": 1.0, "allow_combined": False,
        "mask": "soft_elliptical_gaussian", "area_preset": "medium",
        "reflection_max_area_ratio": 0.035, "reflection_min_area_factor": 0.35,
        "reflection_aspect_min": 1.4, "reflection_aspect_max": 3.2,
        "reflection_intensity_min": 16.0, "reflection_intensity_max": 34.0,
        "jpeg_quality": 95, "source_train_images": 3977, "augmented_train_copies": 397,
        "effective_train_images_per_epoch": 4374, "validation_images_unchanged": 1084,
        "test_images_unchanged": 2406,
    }
    for key, expected in expected_reflection.items():
        if reflection.get(key) != expected:
            raise RuntimeError(f"Reflection-only recipe changed for {key}: {reflection.get(key)!r}")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise RuntimeError("Reflection-only invariants are not all locked true")


def verify_release_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("schema") != "ring3.wsodd_e0_reflection_only_release_manifest.v1" or manifest.get("status") != "locked":
        raise RuntimeError("Unexpected reflection-only release manifest")
    verified = []
    for item in manifest.get("files", []):
        path = SETUP_ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Release file missing or changed: {path}")
        verified.append({"path": item["path"], "sha256": item["sha256"]})
    if len(verified) < 10:
        raise RuntimeError("Reflection-only release manifest is incomplete")
    return {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH), "verified_files": verified}


def verify_assets(config: dict[str, Any]) -> dict[str, Any]:
    assets = {
        "dataset_yaml": (config["dataset"]["yaml"], config["dataset"]["yaml_sha256"]),
        "protocol_manifest": (config["dataset"]["protocol_manifest"], config["dataset"]["protocol_manifest_sha256"]),
        "train_manifest": (config["dataset"]["train_manifest"], config["dataset"]["train_manifest_sha256"]),
        "val_manifest": (config["dataset"]["val_manifest"], config["dataset"]["val_manifest_sha256"]),
        "test_manifest": (config["dataset"]["test_manifest"], config["dataset"]["test_manifest_sha256"]),
        "annotation_semantics_audit": (config["dataset"]["annotation_semantics_audit"], config["dataset"]["annotation_semantics_audit_sha256"]),
        "weights": (config["weights"]["path"], config["weights"]["sha256"]),
        "historical_formal_lock": (config["locks"]["historical_formal_lock"], config["locks"]["historical_formal_lock_sha256"]),
        "runtime_amendment": (config["locks"]["runtime_amendment"], config["locks"]["runtime_amendment_sha256"]),
        "evaluator": (config["evaluation"]["evaluator"], config["evaluation"]["evaluator_sha256"]),
        "robust_helper": (config["evaluation"]["robust_helper"], config["evaluation"]["robust_helper_sha256"]),
        "baseline_summary": (config["evaluation"]["baseline_summary"], config["evaluation"]["baseline_summary_sha256"]),
    }
    verified = {}
    for key, (raw_path, expected_hash) in assets.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Locked asset missing or changed: {key}={path}")
        verified[key] = {"path": str(path), "sha256": expected_hash}
    split_counts = {
        split: len(resolve_images(Path(config["dataset"]["yaml"]), split)) for split in ("train", "val", "test")
    }
    expected = {"train": 3977, "val": 1084, "test": 2406}
    if split_counts != expected:
        raise RuntimeError(f"WSODD split counts changed: {split_counts}")
    verified["split_counts"] = split_counts
    return verified


def preflight() -> dict[str, Any]:
    if STOP_SENTINEL.exists():
        raise RuntimeError(f"Stop sentinel exists: {STOP_SENTINEL}")
    config = load_json(CONFIG_PATH)
    validate_config(config)
    release = verify_release_manifest()
    assets = verify_assets(config)
    if ultralytics.__version__ != "8.4.124" or torch.__version__ != "2.5.1+cu124":
        raise RuntimeError(f"Environment drift: ultralytics={ultralytics.__version__}, torch={torch.__version__}")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Unexpected GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    completion_self_test = early_stop_classifier_self_test()
    overlay_yaml, overlay = build_or_validate_overlay(config)
    context = {
        "config": config,
        "config_sha256": sha256(CONFIG_PATH),
        "release": release,
        "assets": assets,
        "overlay_yaml": str(overlay_yaml),
        "overlay_metadata": overlay,
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "cuda_device": torch.cuda.get_device_name(0),
        },
        "normal_completion_classifier_self_test": completion_self_test,
    }
    protocol = {
        "schema": "ring3.wsodd_e0_reflection_only_protocol.v1",
        "status": "active_locked",
        "created_utc": utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": release["sha256"],
        "assets": assets,
        "environment": context["environment"],
        "normal_completion_classifier_self_test": context["normal_completion_classifier_self_test"],
        "variant": config["variant"],
        "training": config["training"],
        "reflection_only": config["reflection_only"],
        "overlay": {key: value for key, value in overlay.items() if key != "augmented_records"},
        "evaluation": config["evaluation"],
        "invariants": config["invariants"],
    }
    if PROTOCOL_PATH.is_file():
        existing = load_json(PROTOCOL_PATH)
        left, right = dict(existing), dict(protocol)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        if left != right:
            raise RuntimeError("Activated reflection-only protocol changed")
    else:
        write_json_atomic(PROTOCOL_PATH, protocol)
        append_event("protocol_activated", protocol_sha256=sha256(PROTOCOL_PATH))
    return context


def training_kwargs(context: dict[str, Any]) -> dict[str, Any]:
    training = context["config"]["training"]
    kwargs = dict(training)
    kwargs.update(
        data=context["overlay_yaml"],
        project=context["config"]["execution"]["output_root"],
        name=context["config"]["variant"]["name"],
        exist_ok=False,
        verbose=True,
    )
    return kwargs


def verify_args(run_dir: Path, context: dict[str, Any]) -> None:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise RuntimeError(f"Missing args.yaml: {run_dir}")
    observed = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = training_kwargs(context)
    expected["model"] = context["config"]["weights"]["path"]
    ignored = {"project", "name", "exist_ok", "verbose"}
    mismatches = {}
    for key, value in expected.items():
        if key in ignored:
            continue
        actual = observed.get(key)
        if key == "device":
            equal = devices_equivalent(value, actual)
        elif isinstance(value, float) and isinstance(actual, (float, int)):
            equal = math.isclose(value, float(actual), rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = value == actual
        if not equal:
            mismatches[key] = {"expected": value, "observed": actual}
    if mismatches:
        raise RuntimeError(f"Run args differ from locked E0 recipe: {mismatches}")


def history_completion(run_dir: Path, context: dict[str, Any]) -> dict[str, Any]:
    results = run_dir / "results.csv"
    rows = read_rows(results)
    finite = finite_rows(results)
    epochs = []
    for row in rows:
        try:
            epochs.append(int(float(row["epoch"])))
        except (KeyError, TypeError, ValueError):
            pass
    contiguous = bool(epochs) and epochs == list(range(1, epochs[-1] + 1))
    weights_present = all((run_dir / "weights" / name).is_file() for name in ("best.pt", "last.pt"))
    marker_path = run_dir / COMPLETION_EVIDENCE
    marker = load_json(marker_path) if marker_path.is_file() else {}
    return {
        "complete": bool(
            rows and len(rows) == len(finite) and contiguous and weights_present
            and marker.get("status") == "passed"
            and marker.get("completion_kind") in {"max_epochs", "early_stopping"}
            and marker.get("config_sha256") == context["config_sha256"]
            and marker.get("release_manifest_sha256") == context["release"]["sha256"]
        ),
        "history_rows": len(rows),
        "history_finite": len(rows) == len(finite),
        "history_contiguous": contiguous,
        "weights_present": weights_present,
        "final_epoch": epochs[-1] if epochs else None,
        "completion_evidence": marker,
    }


def archive_run(run_dir: Path, reason: str) -> Path:
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = FAILED_ROOT / f"{run_dir.name}__{reason}__{stamp}"
    shutil.move(str(run_dir), str(destination))
    append_event("run_archived", source=str(run_dir), destination=str(destination), reason=reason)
    return destination


def add_callbacks(model: YOLO, context: dict[str, Any], state: dict[str, Any], metadata: dict[str, Any]) -> None:
    run_dir = Path(context["config"]["execution"]["output_root"]) / context["config"]["variant"]["name"]
    evidence_written = False
    last_processed_epoch = int(state.get("current_epoch", 0) or 0)

    def on_train_start(trainer: Any) -> None:
        observed = Path(trainer.save_dir).resolve()
        if observed != run_dir.resolve():
            raise RuntimeError(f"Noncanonical save directory: {observed} != {run_dir.resolve()}")
        state.update(status="running", run_dir=str(observed), last_progress_utc=utc_now())
        write_json_atomic(STATE_PATH, state)
        write_json_atomic(observed / RUN_METADATA, metadata)
        shutil.copyfile(CONFIG_PATH, observed / CONFIG_PATH.name)
        append_event("training_started", attempt=metadata["attempt"], resumed=metadata["resumed_from_own_last_pt"])

    def on_train_batch_end(trainer: Any) -> None:
        nonlocal evidence_written
        if trainer.loss is not None and not bool(torch.isfinite(trainer.loss.detach()).all()):
            raise FloatingPointError(f"Non-finite training loss at epoch {int(trainer.epoch) + 1}")
        if evidence_written:
            return
        optimizer_name = trainer.optimizer.__class__.__name__
        dataset_size = len(trainer.train_loader.dataset)
        model_types = sorted({module.__class__.__name__ for module in trainer.model.modules()})
        forbidden = [name for name in model_types if "CBAM" in name or "Focal" in name or "Consistency" in name]
        if optimizer_name != "SGD" or dataset_size != 4374 or forbidden:
            raise RuntimeError(
                f"Runtime wiring mismatch: optimizer={optimizer_name}, dataset={dataset_size}, forbidden={forbidden}"
            )
        evidence = {
            "schema": "ring3.wsodd_e0_reflection_only_runtime_evidence.v1",
            "status": "passed",
            "verified_utc": utc_now(),
            "run_dir": str(trainer.save_dir),
            "pretrained": True,
            "pretrained_source": context["config"]["weights"]["path"],
            "pretrained_source_sha256": context["config"]["weights"]["sha256"],
            "resumed_from_own_last_pt": metadata["resumed_from_own_last_pt"],
            "parent_checkpoint": "same_run_last.pt" if metadata["resumed_from_own_last_pt"] else None,
            "cross_run_checkpoint_loaded": False,
            "optimizer_class": optimizer_name,
            "optimizer_requested": "SGD",
            "train_dataset_images": dataset_size,
            "source_train_images": 3977,
            "reflection_augmented_copies": 397,
            "single_view": True,
            "mdg_enabled": False,
            "dual_view_enabled": False,
            "cbam_enabled": False,
            "focal_eiou_enabled": False,
            "consistency_loss_enabled": False,
            "small_object_weighting_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "forbidden_runtime_module_types": forbidden,
            "first_batch_loss_finite": True,
            "overlay_metadata_sha256": sha256(Path(context["config"]["execution"]["overlay_root"]) / "metadata.json"),
        }
        write_json_atomic(Path(trainer.save_dir) / RUNTIME_EVIDENCE, evidence)
        state["runtime_evidence"] = evidence
        write_json_atomic(STATE_PATH, state)
        append_event("runtime_wiring_verified", evidence=evidence)
        evidence_written = True

    def on_fit_epoch_end(trainer: Any) -> None:
        nonlocal last_processed_epoch
        results_path = Path(trainer.save_dir) / "results.csv"
        rows = read_rows(results_path)
        if not rows:
            return
        row = rows[-1]
        epoch = int(float(row["epoch"]))
        if epoch <= last_processed_epoch:
            return
        if len(finite_rows(results_path)) != len(rows):
            raise FloatingPointError(f"Non-finite persisted training or validation row at epoch {epoch}")
        cumulative = float(row.get("time", 0.0) or 0.0)
        previous = state.get("last_cumulative_train_seconds")
        measured = cumulative - float(previous) if previous is not None else cumulative
        recent = list(state.get("recent_epoch_seconds", []))
        if math.isfinite(measured) and measured > 0.0:
            recent = (recent + [measured])[-10:]
        expected = sum(recent) / len(recent) if recent else 150.0
        best = best_row(results_path)
        now = datetime.now(timezone.utc)
        state.update(
            status="running", current_epoch=epoch, last_progress_utc=now.isoformat(),
            current_row=row, best_row=best, recent_epoch_seconds=recent,
            expected_seconds_per_epoch=expected,
            eta_utc=(now + timedelta(seconds=max(0, 150 - epoch) * expected)).isoformat(),
            last_cumulative_train_seconds=cumulative,
        )
        write_json_atomic(STATE_PATH, state)
        if epoch == 1 or epoch % 10 == 0 or epoch >= 147:
            append_event("epoch_completed", epoch=epoch, current_row=row, best_row=best, eta_utc=state["eta_utc"])
        print(
            "WSODD_E0_REFLECTION_ONLY_STATE "
            + json.dumps({"epoch": epoch, "target": 150, "best_epoch": best["epoch"], "eta_utc": state["eta_utc"]}),
            flush=True,
        )
        last_processed_epoch = epoch
        if STOP_SENTINEL.exists():
            state.update(status="pausing", last_progress_utc=utc_now())
            write_json_atomic(STATE_PATH, state)
            trainer.stop = True

    def on_train_end(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir).resolve()
        rows = read_rows(save_dir / "results.csv")
        final_epoch = int(float(rows[-1]["epoch"])) if rows else None
        best = best_row(save_dir / "results.csv") if rows else None
        stopper = getattr(trainer, "stopper", None)
        stopper_best = int(getattr(stopper, "best_epoch", 0))
        completion_kind = classify_normal_training_return(
            final_epoch=final_epoch,
            target_epochs=150,
            trainer_epoch_zero_based=int(getattr(trainer, "epoch", -1)),
            stopper_best_epoch_one_based=stopper_best,
            patience=100,
        )
        payload = {
            "schema": "ring3.wsodd_e0_reflection_only_completion_evidence.v1",
            "status": "passed" if completion_kind in {"max_epochs", "early_stopping"} else "incomplete",
            "recorded_utc": utc_now(),
            "source": "normal_train_end_callback",
            "run_dir": str(save_dir),
            "completion_kind": completion_kind,
            "normal_training_return_confirmed": True,
            "final_epoch": final_epoch,
            "target_epochs": 150,
            "patience": 100,
            "best_epoch": stopper_best,
            "metric_best_epoch": int(float(best["epoch"])) if best else None,
            "patience_gap": final_epoch - stopper_best if final_epoch is not None and stopper_best > 0 else None,
            "trainer_epoch_zero_based": int(getattr(trainer, "epoch", -1)),
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
        }
        write_json_atomic(save_dir / COMPLETION_EVIDENCE, payload)
        append_event("normal_training_end_recorded", **{key: payload[key] for key in ("completion_kind", "final_epoch", "best_epoch", "patience_gap")})

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.wsodd_e0_reflection_only_state.v1",
        "status": "preflight_passed", "started_utc": now.isoformat(), "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"], "release_manifest_sha256": context["release"]["sha256"],
        "current_run": context["config"]["variant"]["name"], "current_epoch": 0, "target_epochs": 150,
        "attempts": 0, "recent_epoch_seconds": [], "evaluation_status": "pending",
        "eta_utc": (now + timedelta(hours=5)).isoformat(), "e4_started": False,
    }


def train_until_complete(context: dict[str, Any], state: dict[str, Any]) -> Path | None:
    config = context["config"]
    run_dir = Path(config["execution"]["output_root"]) / config["variant"]["name"]
    completion = history_completion(run_dir, context)
    if completion["complete"]:
        verify_args(run_dir, context)
        append_event("completed_training_reused", completion=completion)
        return run_dir
    if run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
        archive_run(run_dir, "unresumable_partial_run")

    maximum = int(config["execution"]["max_attempts"])
    while int(state.get("attempts", 0)) < maximum:
        if STOP_SENTINEL.exists():
            state.update(status="paused", last_progress_utc=utc_now())
            write_json_atomic(STATE_PATH, state)
            append_event("run_paused", sentinel=str(STOP_SENTINEL))
            return None
        state["attempts"] = int(state.get("attempts", 0)) + 1
        state.update(status="starting", current_epoch=0, last_progress_utc=utc_now(), error=None)
        write_json_atomic(STATE_PATH, state)
        resumed = run_dir.exists() and (run_dir / "weights" / "last.pt").is_file()
        if resumed:
            metadata_path = run_dir / RUN_METADATA
            if not metadata_path.is_file():
                raise RuntimeError("Refusing same-run resume without metadata")
            existing_metadata = load_json(metadata_path)
            if existing_metadata.get("config_sha256") != context["config_sha256"]:
                raise RuntimeError("Refusing same-run resume after config drift")
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            kwargs: dict[str, Any] = {"resume": True}
        else:
            model = YOLO(config["weights"]["path"])
            kwargs = training_kwargs(context)
        metadata = {
            "schema": "ring3.wsodd_e0_reflection_only_run_metadata.v1", "created_utc": utc_now(),
            "variant": config["variant"], "attempt": state["attempts"],
            "resumed_from_own_last_pt": resumed, "parent_checkpoint": "same_run_last.pt" if resumed else None,
            "cross_run_checkpoint_loaded": False, "pretrained": True,
            "pretrained_source": config["weights"]["path"], "pretrained_source_sha256": config["weights"]["sha256"],
            "optimizer": "SGD", "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
            "overlay_metadata_sha256": sha256(Path(config["execution"]["overlay_root"]) / "metadata.json"),
            "no_mdg": True, "no_dual_view": True, "no_cbam": True, "no_focal_eiou": True,
            "no_consistency": True, "no_small_object_weighting": True, "no_e4": True,
        }
        add_callbacks(model, context, state, metadata)
        append_event("attempt_started", attempt=state["attempts"], resumed=resumed, run_dir=str(run_dir))
        try:
            model.train(**kwargs)
            if STOP_SENTINEL.exists():
                state.update(status="paused", last_progress_utc=utc_now())
                write_json_atomic(STATE_PATH, state)
                return None
            completion = history_completion(run_dir, context)
            if not completion["complete"]:
                raise RuntimeError(f"Training returned without a proved normal completion: {completion}")
            verify_args(run_dir, context)
            if not (run_dir / RUNTIME_EVIDENCE).is_file():
                raise RuntimeError("Training completed without runtime wiring evidence")
            state.update(
                status="training_completed", current_epoch=completion["final_epoch"],
                completion=completion, evaluation_status="starting", last_progress_utc=utc_now(),
            )
            write_json_atomic(STATE_PATH, state)
            append_event("training_completed", completion=completion, best_row=best_row(run_dir / "results.csv"))
            return run_dir
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            numerical = isinstance(exc, FloatingPointError) or any(token in message.lower() for token in ("non-finite", "nan", "inf"))
            state.update(
                status="attempt_failed", last_progress_utc=utc_now(), error=message,
                traceback=traceback.format_exc(), numerical_failure=numerical,
            )
            write_json_atomic(STATE_PATH, state)
            append_event("attempt_failed", attempt=state["attempts"], numerical_failure=numerical, error=message)
            traceback.print_exc()
            if numerical and run_dir.exists():
                archive_run(run_dir, "numerical_failure")
            elif run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
                archive_run(run_dir, "unresumable_failure")
            del model
            torch.cuda.empty_cache()
    state.update(status="supervisor_failed", last_progress_utc=utc_now(), error="attempts_exhausted")
    write_json_atomic(STATE_PATH, state)
    append_event("attempts_exhausted", attempts=state["attempts"])
    return None


def evaluate_and_finish(context: dict[str, Any], state: dict[str, Any], run_dir: Path) -> None:
    command = [
        sys.executable, str(SETUP_ROOT / "evaluate_wsodd_e0_reflection_only.py"),
        "--config", str(CONFIG_PATH), "--run-dir", str(run_dir), "--output", str(EVALUATION_PATH),
    ]
    state.update(status="evaluating", evaluation_status="running", last_progress_utc=utc_now())
    write_json_atomic(STATE_PATH, state)
    append_event("fixed_test_evaluation_started", command=command)
    subprocess.run(command, check=True)
    evaluation = load_json(EVALUATION_PATH)
    if evaluation.get("status") != "completed":
        raise RuntimeError("Fixed-test evaluator did not produce completed evidence")
    completion = history_completion(run_dir, context)
    payload = {
        "schema": "ring3.wsodd_e0_reflection_only_summary.v1", "status": "completed",
        "completed_utc": utc_now(), "variant": context["config"]["variant"],
        "training_completion": completion, "training_best": best_row(run_dir / "results.csv"),
        "fixed_test": evaluation, "formal_e0_e3_untouched": True, "e4_started": False,
        "single_seed_mechanism_evidence_only": True,
    }
    write_json_atomic(SUMMARY_PATH, payload)
    state.update(
        status="completed", evaluation_status="completed", last_progress_utc=utc_now(),
        summary=str(SUMMARY_PATH), summary_sha256=sha256(SUMMARY_PATH),
        evaluation=str(EVALUATION_PATH), evaluation_sha256=sha256(EVALUATION_PATH), e4_started=False,
    )
    write_json_atomic(STATE_PATH, state)
    append_event("experiment_completed", fixed_test=evaluation["reflection_only_fixed_test"], delta=evaluation["reflection_only_minus_formal_e0"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated formal WSODD E0 plus reflection-only")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        context = preflight()
        print("WSODD_E0_REFLECTION_ONLY_PREFLIGHT " + json.dumps({
            "config_sha256": context["config_sha256"], "release": context["release"],
            "environment": context["environment"], "overlay_yaml": context["overlay_yaml"],
            "overlay_counts": context["overlay_metadata"]["counts"],
        }, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        if STATE_PATH.is_file():
            state = load_json(STATE_PATH)
            if state.get("config_sha256") != context["config_sha256"] or state.get("release_manifest_sha256") != context["release"]["sha256"]:
                raise RuntimeError("Existing state belongs to a different reflection-only release")
        else:
            state = state_template(context)
            write_json_atomic(STATE_PATH, state)
            append_event("experiment_started", state=state)
        run_dir = train_until_complete(context, state)
        if run_dir is None:
            return 4 if state.get("status") == "supervisor_failed" else 0
        evaluate_and_finish(context, state, run_dir)
        return 0
    except BaseException as exc:
        failure = {
            "schema": "ring3.wsodd_e0_reflection_only_failure.v1", "status": "supervisor_failed",
            "failed_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "e4_started": False,
        }
        write_json_atomic(DATA_ROOT / "formal_state" / "wsodd_e0_reflection_only_failure.json", failure)
        if STATE_PATH.is_file():
            state = load_json(STATE_PATH)
            state.update(status="supervisor_failed", last_progress_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
            write_json_atomic(STATE_PATH, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
