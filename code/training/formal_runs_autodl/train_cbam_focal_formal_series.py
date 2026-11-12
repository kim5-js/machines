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
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

import formal_baseline_queue as base
from cbam_focal_formal_models import (
    CBAMFocalSmallWeightedDetectionModel,
    FactorialDetectionModel,
    SmallObjectWeightedFocalEIoULoss,
)
from cbam_focal_sdc import FocalEIoULoss
from maritime_degradation import MaritimeDegradation, load_config as load_mdg_config


EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_MDG_CONFIG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_WEIGHT_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPECTED_DATA_YAML_SHA256 = "66b63cf9f50952faa5abdc83979599f2de88c64178278fb5b2b4c698ed841f7c"
EXPECTED_DATA_MANIFEST_SHA256 = "0030e4921a95d57722db66afb50a094c25fc3e6c00b93db5567fbbb196576360"
MODEL_KEY = "yolov8s"
DATASET_KEY = "seadronessee"
CONFIG_SCHEMA = "ring3.cbam_focal_formal_series_config.v1"
MAX_ATTEMPTS = 3

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
STATE_ROOT = DATA_ROOT / "formal_state"
DIAGNOSTICS_ROOT = DATA_ROOT / "diagnostics"
FORMAL_ROOT = DATA_ROOT / "formal_cbam_focal_series"
FAILED_ROOT = DATA_ROOT / "formal_failed_runs" / "cbam_focal_series"
CONFIG_PATH = SETUP_ROOT / "cbam_focal_formal_series_config.json"
MANIFEST_PATH = SETUP_ROOT / "cbam_focal_formal_series_release_manifest.json"
MDG_CONFIG_PATH = SETUP_ROOT / "e1_mdg_config.json"
AMENDMENT_PATH = SETUP_ROOT / "formal_runtime_amendment.json"
STATE_PATH = STATE_ROOT / "cbam_focal_formal_queue_state.json"
EVENTS_PATH = STATE_ROOT / "cbam_focal_formal_queue_events.jsonl"
PROTOCOL_PATH = STATE_ROOT / "cbam_focal_formal_queue_protocol.json"
SUMMARY_PATH = STATE_ROOT / "cbam_focal_formal_queue_summary.json"
SELF_TEST_PATH = DIAGNOSTICS_ROOT / "cbam_focal_formal_series_self_test.json"
STOP_SENTINEL = STATE_ROOT / "STOP_CBAM_FOCAL_FORMAL_SERIES"
E0_RUN = DATA_ROOT / "formal_baselines" / "seed0_yolov8s_seadronessee_img960_e150"

os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_ROOT / ".config" / "Ultralytics"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_event(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), "event": event, **details}, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def best_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    finite = [
        row
        for row in rows
        if row.get("metrics/mAP50-95(B)") and math.isfinite(float(row["metrics/mAP50-95(B)"]))
    ]
    if not finite:
        raise RuntimeError(f"No finite mAP50-95 row in {path}")
    return max(finite, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def row_has_nonfinite(row: dict[str, str] | None, prefixes: tuple[str, ...]) -> bool:
    if not row:
        return False
    for key, value in row.items():
        if not key.startswith(prefixes) or value in (None, ""):
            continue
        try:
            if not math.isfinite(float(value)):
                return True
        except ValueError:
            continue
    return False


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected formal-series config schema/status")
    runs = config.get("runs", [])
    expected = [
        ("cbam_only", True, False, False, False),
        ("focal_eiou_only", False, True, False, False),
        ("cbam_focal", True, True, False, False),
        ("cbam_focal_e2", True, True, True, False),
        ("cbam_focal_e3", True, True, True, True),
    ]
    observed = [
        (
            run.get("key"),
            run.get("cbam"),
            run.get("focal_eiou"),
            run.get("dual_view"),
            run.get("degraded_small_weighting"),
        )
        for run in runs
    ]
    if observed != expected or len({str(run.get("name")) for run in runs}) != len(runs):
        raise ValueError(f"Unexpected run matrix: {observed!r}")
    training = config.get("training", {})
    locked = {
        "pretrained": True,
        "model": "/root/autodl-tmp/weights/yolov8s.pt",
        "epochs": 150,
        "imgsz": 960,
        "batch": 8,
        "optimizer": "AdamW",
        "lr0": 0.001111,
        "seed": 0,
        "independent_fresh_start_for_every_run": True,
        "resume_only_from_same_run_last_pt": True,
        "max_attempts_per_run": 3,
    }
    for key, expected_value in locked.items():
        if training.get(key) != expected_value:
            raise ValueError(f"Formal-series training field changed: {key}")
    invariants = config.get("invariants", {})
    if any(value is not True for value in invariants.values()):
        raise ValueError("Formal-series invariants are incomplete")


def verify_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("schema") != "ring3.cbam_focal_formal_series_release_manifest.v1":
        raise RuntimeError("Unexpected formal-series release manifest")
    if manifest.get("status") != "locked":
        raise RuntimeError("Formal-series release is not locked")
    verified = []
    for item in manifest.get("files", []):
        path = SETUP_ROOT / str(item["path"])
        if not path.is_file():
            raise RuntimeError(f"Release file is missing: {path}")
        observed = sha256(path)
        if observed != item.get("sha256"):
            raise RuntimeError(
                f"Release hash mismatch for {item.get('path')}: expected={item.get('sha256')}, observed={observed}"
            )
        verified.append({"path": str(item["path"]), "sha256": observed})
    if len(verified) < 8:
        raise RuntimeError("Formal-series release manifest is incomplete")
    return {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH), "verified_files": verified}


def verify_dataset(lock: dict[str, Any]) -> dict[str, Any]:
    protocol = lock["dataset_protocols"][DATASET_KEY]
    yaml_path = Path(protocol["yaml"])
    manifest_path = Path(protocol["manifest_path"])
    if sha256(yaml_path) != EXPECTED_DATA_YAML_SHA256:
        raise RuntimeError("Locked SeaDronesSee YAML changed")
    if sha256(manifest_path) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("Locked SeaDronesSee manifest changed")
    result = subprocess.run(
        [sys.executable, str(SETUP_ROOT / "verify_yolo_dataset.py"), str(yaml_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("SeaDronesSee verification failed:\n" + result.stdout + result.stderr)
    report = json.loads(result.stdout)
    expected_counts = protocol["split_counts"]
    if report.get("label_error_count") or any(report.get("missing_label_counts", {}).values()):
        raise RuntimeError("SeaDronesSee verification reported label errors")
    if any(report.get("counts", {}).get(split) != count for split, count in expected_counts.items()):
        raise RuntimeError("SeaDronesSee split counts changed")
    return report


def preflight() -> dict[str, Any]:
    if STOP_SENTINEL.exists():
        raise RuntimeError(f"Stop sentinel exists: {STOP_SENTINEL}")
    config = load_json(CONFIG_PATH)
    validate_config(config)
    release = verify_manifest()
    lock = base.load_lock()
    lock_hash = base.verify_lock_immutability()
    if lock_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Formal lock changed")
    if not AMENDMENT_PATH.is_file() or sha256(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Runtime amendment changed")
    if not MDG_CONFIG_PATH.is_file() or sha256(MDG_CONFIG_PATH) != EXPECTED_MDG_CONFIG_SHA256:
        raise RuntimeError("MDG config changed")
    load_mdg_config(MDG_CONFIG_PATH)
    base.verify_environment(lock)
    base.validate_training_lock(lock)
    weight_path = Path(lock["weights"][MODEL_KEY]["path"])
    if not weight_path.is_file() or sha256(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt changed")
    dataset_report = verify_dataset(lock)
    complete, e0_final = base.completed_run(E0_RUN, 150)
    if not complete or e0_final is None:
        raise RuntimeError("Completed formal E0 anchor is unavailable")
    base.verify_resume_args(E0_RUN, lock, DATASET_KEY)
    e0_best = best_row(E0_RUN / "results.csv")
    return {
        "config": config,
        "config_sha256": sha256(CONFIG_PATH),
        "release": release,
        "lock": lock,
        "lock_sha256": lock_hash,
        "runtime_amendment_sha256": sha256(AMENDMENT_PATH),
        "mdg_config_sha256": sha256(MDG_CONFIG_PATH),
        "dataset_report": dataset_report,
        "e0_anchor": {
            "run_dir": str(E0_RUN),
            "results_sha256": sha256(E0_RUN / "results.csv"),
            "args_sha256": sha256(E0_RUN / "args.yaml"),
            "best_row": e0_best,
            "final_row": e0_final,
        },
    }


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.cbam_focal_formal_queue_protocol.v1",
        "status": "active_locked",
        "created_utc": utc_now(),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "e0_anchor": context["e0_anchor"],
        "run_order": context["config"]["runs"],
        "training": context["config"]["training"],
        "architecture": context["config"]["architecture"],
        "focal_eiou": context["config"]["focal_eiou"],
        "small_object_weighting": context["config"]["small_object_weighting"],
        "invariants": context["config"]["invariants"],
    }


def activate_protocol(context: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(context)
    if PROTOCOL_PATH.exists():
        existing = load_json(PROTOCOL_PATH)
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_utc", None)
        comparable_payload.pop("created_utc", None)
        if comparable_existing != comparable_payload:
            raise RuntimeError("Formal-series protocol changed after activation")
        return existing
    write_json_atomic(PROTOCOL_PATH, payload)
    append_event(EVENTS_PATH, "protocol_activated", protocol_sha256=sha256(PROTOCOL_PATH))
    return payload


class FactorialTrainer(DetectionTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.variant = json.loads(os.environ["RING3_FACTORIAL_VARIANT_JSON"])
        self.dual_view = bool(self.variant["dual_view"])
        self.small_weighting = bool(self.variant["degraded_small_weighting"])
        self.mdg = None
        if self.dual_view:
            mdg_path = Path(os.environ["RING3_FACTORIAL_MDG_CONFIG"])
            if sha256(mdg_path) != os.environ["RING3_FACTORIAL_MDG_CONFIG_SHA256"]:
                raise RuntimeError("MDG config changed before trainer construction")
            self.mdg = MaritimeDegradation(load_mdg_config(mdg_path))
        self._last_mdg_counts: dict[str, int] = {}
        self._last_target_counts: dict[str, int] = {}
        self._mdg_epoch_counts: Counter[str] = Counter()
        self._target_epoch_counts: Counter[str] = Counter()
        self._mdg_total_counts: Counter[str] = Counter()
        self._target_total_counts: Counter[str] = Counter()
        self._optimizer_update_probe: dict[str, Any] = {}
        self._optimizer_step_calls = 0
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        if self.small_weighting:
            model = CBAMFocalSmallWeightedDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                degraded_small_weight=1.5,
            )
        else:
            model = FactorialDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                use_cbam=bool(self.variant["cbam"]),
                use_focal_eiou=bool(self.variant["focal_eiou"]),
            )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        clean = batch["img"]
        pair_batch_size = int(clean.shape[0])
        source_batch_idx = batch["batch_idx"]
        source_cls = batch["cls"]
        source_bboxes = batch["bboxes"]
        if not self.dual_view:
            self._last_target_counts = {
                "images_total": pair_batch_size,
                "clean_targets": int(source_bboxes.shape[0]),
            }
            self._target_epoch_counts.update(self._last_target_counts)
            self._target_total_counts.update(self._last_target_counts)
            return batch

        if self.mdg is None:
            raise RuntimeError("Dual-view run has no MDG instance")
        degraded, mdg_counts = self.mdg.apply_batch(clean.clone())
        batch["img"] = torch.cat((clean, degraded), dim=0)
        batch["batch_idx"] = torch.cat((source_batch_idx, source_batch_idx + pair_batch_size), dim=0)
        batch["cls"] = torch.cat((source_cls, source_cls.clone()), dim=0)
        batch["bboxes"] = torch.cat((source_bboxes, source_bboxes.clone()), dim=0)
        batch["factorial_pair_batch_size"] = pair_batch_size
        if self.small_weighting:
            area_px2 = (
                source_bboxes[:, 2] * float(clean.shape[-1])
                * source_bboxes[:, 3] * float(clean.shape[-2])
            )
            small_mask = area_px2 < 1024.0
            clean_weights = torch.ones(
                source_bboxes.shape[0], device=source_bboxes.device, dtype=source_bboxes.dtype
            )
            degraded_weights = torch.where(
                small_mask, torch.full_like(clean_weights, 1.5), clean_weights
            )
            batch["factorial_gt_weights"] = torch.cat((clean_weights, degraded_weights), dim=0)
        else:
            small_mask = torch.zeros(source_bboxes.shape[0], device=source_bboxes.device, dtype=torch.bool)
        for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad"):
            value = batch.get(key)
            if isinstance(value, list):
                batch[key] = value + list(value)
            elif isinstance(value, tuple) and len(value) == pair_batch_size:
                batch[key] = value + value
        target_counts = {
            "images_total": pair_batch_size,
            "clean_targets": int(source_bboxes.shape[0]),
            "degraded_targets": int(source_bboxes.shape[0]),
            "degraded_small_targets": int(small_mask.sum()),
            "degraded_regular_targets": int((~small_mask).sum()),
        }
        self._last_mdg_counts = dict(mdg_counts)
        self._last_target_counts = target_counts
        self._mdg_epoch_counts.update(mdg_counts)
        self._target_epoch_counts.update(target_counts)
        self._mdg_total_counts.update(mdg_counts)
        self._target_total_counts.update(target_counts)
        return batch

    def optimizer_step(self) -> None:
        self._optimizer_step_calls += 1
        model = unwrap_model(self.model)
        parameter_before = None
        target_parameter = None
        pre_step_gradient: dict[str, Any] = {}
        if not self._optimizer_update_probe and isinstance(model, FactorialDetectionModel):
            parameters = model.p3_target_parameters()
            if parameters:
                target_parameter = parameters[0]
                parameter_before = target_parameter.detach().float().clone()
                if target_parameter.grad is not None:
                    gradient = target_parameter.grad.detach().float()
                    pre_step_gradient = {
                        "finite": bool(torch.isfinite(gradient).all()),
                        "nonzero": bool((gradient != 0).any()),
                        "l2_norm_scaled": float(gradient.square().sum().sqrt()),
                    }
        super().optimizer_step()
        if parameter_before is not None and target_parameter is not None:
            delta = target_parameter.detach().float() - parameter_before
            candidate = {
                "optimizer_step_index": self._optimizer_step_calls,
                "target_feature_layer_index": model.factorial_p3_layer_index,
                "target_parameter_shape": [int(value) for value in target_parameter.shape],
                "pre_step_gradient": pre_step_gradient,
                "finite": bool(torch.isfinite(delta).all()),
                "parameter_delta_l2": float(delta.square().sum().sqrt()),
                "parameter_delta_max_abs": float(delta.abs().max()),
            }
            if candidate["finite"] is not True:
                raise FloatingPointError(f"P3 optimizer update became non-finite: {candidate}")
            if math.isfinite(candidate["parameter_delta_l2"]) and candidate["parameter_delta_l2"] > 0.0:
                self._optimizer_update_probe = candidate

    def runtime_evidence(self) -> dict[str, Any]:
        model = unwrap_model(self.model)
        if not isinstance(model, FactorialDetectionModel):
            raise TypeError(f"Unexpected factorial model type: {type(model).__name__}")
        return {
            "variant": self.variant,
            **model.factorial_evidence(),
            "dual_view_enabled": self.dual_view,
            "degraded_small_weighting_enabled": self.small_weighting,
            "mdg_counts_current_batch": dict(self._last_mdg_counts),
            "target_counts_current_batch": dict(self._last_target_counts),
            "optimizer_update_probe": dict(self._optimizer_update_probe),
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
        }

    def consume_epoch_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        mdg, targets = dict(self._mdg_epoch_counts), dict(self._target_epoch_counts)
        self._mdg_epoch_counts.clear()
        self._target_epoch_counts.clear()
        return mdg, targets

    def total_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        return dict(self._mdg_total_counts), dict(self._target_total_counts)


def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    if evidence.get("variant", {}).get("key") != variant["key"]:
        raise RuntimeError("Runtime evidence belongs to a different run")
    if evidence.get("cbam_enabled") is not bool(variant["cbam"]):
        raise RuntimeError("CBAM enablement evidence is wrong")
    if evidence.get("focal_eiou_enabled") is not bool(variant["focal_eiou"]):
        raise RuntimeError("Focal-EIoU enablement evidence is wrong")
    expected_cbam_types = ["C2f_CBAM", "C2f_CBAM"] if variant["cbam"] else ["C2f", "C2f"]
    if evidence.get("cbam_layer_types") != expected_cbam_types:
        raise RuntimeError(f"CBAM topology evidence failed: {evidence.get('cbam_layer_types')}")
    if variant["cbam"] and int(evidence.get("cbam_attention_parameter_count", 0)) <= 0:
        raise RuntimeError("CBAM attention parameters are absent")
    if not variant["cbam"] and int(evidence.get("cbam_attention_parameter_count", -1)) != 0:
        raise RuntimeError("CBAM attention unexpectedly exists in a control")
    criterion_type = str(evidence.get("criterion_type"))
    if variant["degraded_small_weighting"]:
        if criterion_type != "SmallObjectWeightedFocalEIoULoss":
            raise RuntimeError(f"Combined E3 criterion is wrong: {criterion_type}")
    elif variant["focal_eiou"]:
        if criterion_type != "v8DetectionLossWithFocalEIoU":
            raise RuntimeError(f"Focal-EIoU criterion is wrong: {criterion_type}")
    elif criterion_type != "v8DetectionLoss":
        raise RuntimeError(f"Control criterion is wrong: {criterion_type}")
    if variant["focal_eiou"]:
        focal = evidence.get("focal_eiou") or {}
        if focal.get("dfl_preserved") is not True:
            raise RuntimeError("DFL preservation evidence is missing")
        for key, expected in (("alpha", 0.25), ("gamma", 2.0), ("eps", 1e-7)):
            if not math.isclose(float(focal.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Focal-EIoU {key} changed")
    if evidence.get("detect_input_indices") != [15, 18, 21]:
        raise RuntimeError("Detect inputs changed")
    if evidence.get("target_feature_layer_index") != 15 or evidence.get("target_feature_layer_type") != "C2f":
        raise RuntimeError("Head P3 topology changed")
    if evidence.get("no_consistency_loss") is not True or evidence.get("consistency_loss_enabled") is not False:
        raise RuntimeError("A consistency loss was unexpectedly enabled")
    if evidence.get("severity_weighting_enabled") is not False or evidence.get("curriculum_enabled") is not False:
        raise RuntimeError("Severity weighting or curriculum was unexpectedly enabled")
    update = evidence.get("optimizer_update_probe", {})
    if update.get("finite") is not True or float(update.get("parameter_delta_l2", 0.0)) <= 0.0:
        raise RuntimeError("Finite nonzero head-P3 optimizer update is not yet verified")
    targets = evidence.get("target_counts_current_batch", {})
    if variant["dual_view"]:
        if int(targets.get("clean_targets", -1)) != int(targets.get("degraded_targets", -2)):
            raise RuntimeError("Clean/degraded target counts differ")
        mdg = evidence.get("mdg_counts_current_batch", {})
        if int(mdg.get("images_total", -1)) != int(targets.get("images_total", -2)):
            raise RuntimeError("MDG/pair image counts differ")
    if variant["degraded_small_weighting"]:
        required = {
            "ordinary_target_weight": 1.0,
            "degraded_small_target_weight": 1.5,
            "normalization_denominator_unchanged": True,
            "background_anchor_weight_unchanged": True,
            "clean_weighted_target_violations": 0,
            "negative_anchor_weight_violations": 0,
        }
        for key, expected in required.items():
            if evidence.get(key) != expected:
                raise RuntimeError(f"Combined E3 weighting evidence not ready for {key}: {evidence.get(key)!r}")
        if int(evidence.get("weighted_positive_anchor_count", 0)) <= 0:
            raise RuntimeError("No degraded-small positive anchor has been weighted yet")
        if float(evidence.get("weighting_delta_scaled", 0.0)) <= 0.0:
            raise RuntimeError("Degraded-small weighting did not increase the detection objective")


def initial_state(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    runs = context["config"]["runs"]
    expected_hours = 3 * 5.1 + 2 * 7.2
    return {
        "schema": "ring3.cbam_focal_formal_queue_state.v1",
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "run_order": [run["key"] for run in runs],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": expected_hours,
        "eta_utc": (now + timedelta(hours=expected_hours)).isoformat(),
        "e0_anchor": context["e0_anchor"],
        "old_pilot_reused_as_formal_result": False,
        "e4_started": False,
    }


def training_kwargs(lock: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    kwargs = base.training_kwargs(lock, MODEL_KEY, DATASET_KEY, variant["name"])
    kwargs["project"] = str(FORMAL_ROOT)
    kwargs["epochs"] = 150
    kwargs["exist_ok"] = False
    return kwargs


def verify_resume(run_dir: Path, variant: dict[str, Any], context: dict[str, Any]) -> None:
    base.verify_resume_args(run_dir, context["lock"], DATASET_KEY)
    metadata_path = run_dir / "cbam_focal_formal_run_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Refusing resume without formal-series metadata")
    metadata = load_json(metadata_path)
    expected = {
        "variant_key": variant["key"],
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "formal_lock_sha256": context["lock_sha256"],
        "pretrained": True,
        "cross_run_checkpoint_loaded": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Refusing resume with mismatched run metadata")


def archive_run(run_dir: Path, reason: str) -> Path:
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = FAILED_ROOT / f"{run_dir.name}__{reason}__{stamp}"
    shutil.move(str(run_dir), str(destination))
    append_event(EVENTS_PATH, "run_archived", source=str(run_dir), destination=str(destination), reason=reason)
    return destination


def add_callbacks(
    model: YOLO,
    *,
    variant: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    evidence_written = False

    def on_train_start(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir)
        state.update(status="running", run_dir=str(save_dir), last_progress_utc=utc_now())
        write_json_atomic(STATE_PATH, state)
        write_json_atomic(save_dir / "cbam_focal_formal_run_metadata.json", metadata)
        shutil.copyfile(CONFIG_PATH, save_dir / CONFIG_PATH.name)
        if variant["dual_view"]:
            shutil.copyfile(MDG_CONFIG_PATH, save_dir / MDG_CONFIG_PATH.name)
        append_event(
            EVENTS_PATH,
            "training_started",
            variant=variant["key"],
            run_dir=str(save_dir),
            attempt=metadata["attempt"],
            resumed=metadata["resumed_from_own_last_pt"],
        )

    def on_train_batch_end(trainer: Any) -> None:
        nonlocal evidence_written
        if trainer.loss is not None and not bool(trainer.loss.detach().isfinite().all()):
            raise FloatingPointError(
                f"Non-finite training loss in {variant['key']} at epoch {int(trainer.epoch) + 1}"
            )
        if evidence_written:
            return
        evidence = trainer.runtime_evidence()
        try:
            validate_runtime_evidence(evidence, variant)
        except RuntimeError:
            return
        evidence_written = True
        evidence.update(
            {
                "schema": "ring3.cbam_focal_formal_runtime_evidence.v1",
                "verified_utc": utc_now(),
                "run_dir": str(trainer.save_dir),
                "pretrained": True,
                "parent_checkpoint": None if not metadata["resumed_from_own_last_pt"] else "same_run_last.pt",
                "cross_run_checkpoint_loaded": False,
            }
        )
        evidence_path = Path(trainer.save_dir) / "cbam_focal_formal_runtime_evidence.json"
        write_json_atomic(evidence_path, evidence)
        state["runtime_evidence"] = evidence
        write_json_atomic(STATE_PATH, state)
        append_event(
            EVENTS_PATH,
            "runtime_wiring_verified",
            variant=variant["key"],
            cbam=variant["cbam"],
            focal_eiou=variant["focal_eiou"],
            dual_view=variant["dual_view"],
            degraded_small_weighting=variant["degraded_small_weighting"],
            p3_parameter_delta_l2=evidence["optimizer_update_probe"]["parameter_delta_l2"],
        )

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = min(int(trainer.epoch) + 1, 150)
        save_dir = Path(trainer.save_dir)
        row = base.read_last_row(save_dir / "results.csv")
        if row_has_nonfinite(row, ("train/",)):
            raise FloatingPointError(f"Non-finite training epoch in {variant['key']} at epoch {epoch}")
        val_bad = row_has_nonfinite(row, ("val/",))
        streak = int(state.get("validation_nan_streak", 0)) + 1 if val_bad else 0
        if streak >= 3:
            raise FloatingPointError(
                f"Validation loss stayed non-finite for {streak} epochs in {variant['key']}"
            )
        mdg_counts, target_counts = trainer.consume_epoch_counts()
        if variant["dual_view"]:
            if int(mdg_counts.get("images_total", -1)) != int(target_counts.get("images_total", -2)):
                raise RuntimeError(f"MDG/pair accounting mismatch: mdg={mdg_counts}, targets={target_counts}")
            if int(target_counts.get("clean_targets", -1)) != int(target_counts.get("degraded_targets", -2)):
                raise RuntimeError(f"Clean/degraded target mismatch: {target_counts}")
        recent = list(state.get("recent_epoch_seconds", []))
        cumulative = float(row["time"]) if row and row.get("time") else None
        if cumulative is not None:
            previous = state.get("last_cumulative_train_seconds")
            measured = cumulative - float(previous) if previous is not None else cumulative
            if math.isfinite(measured) and measured > 0.0:
                recent.append(measured)
                recent = recent[-10:]
        expected = sum(recent) / len(recent) if recent else float(state.get("expected_seconds_per_epoch", 130.0))
        now = datetime.now(timezone.utc)
        current_best = best_row(save_dir / "results.csv") if row else None
        state.update(
            status="running",
            current_epoch=epoch,
            last_progress_utc=now.isoformat(),
            validation_nan_streak=streak,
            recent_epoch_seconds=recent,
            expected_seconds_per_epoch=expected,
            eta_utc=(now + timedelta(seconds=max(0, 150 - epoch) * expected)).isoformat(),
            current_row=row,
            best_row=current_best,
            mdg_last_epoch_counts=mdg_counts,
            target_last_epoch_counts=target_counts,
        )
        if cumulative is not None:
            state["last_cumulative_train_seconds"] = cumulative
        write_json_atomic(STATE_PATH, state)
        if epoch == 1 or epoch % 10 == 0 or epoch == 150:
            append_event(
                EVENTS_PATH,
                "epoch_completed",
                variant=variant["key"],
                epoch=epoch,
                current_row=row,
                best_row=current_best,
                expected_seconds_per_epoch=expected,
                eta_utc=state["eta_utc"],
            )
        print(
            "CBAM_FOCAL_FORMAL_STATE "
            + json.dumps(
                {
                    "variant": variant["key"],
                    "epoch": epoch,
                    "target": 150,
                    "best_epoch": current_best.get("epoch") if current_best else None,
                    "eta_utc": state["eta_utc"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def run_variant(
    variant: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    run_dir = FORMAL_ROOT / variant["name"]
    complete, final_row = base.completed_run(run_dir, 150)
    if complete and final_row is not None:
        evidence_path = run_dir / "cbam_focal_formal_runtime_evidence.json"
        if not evidence_path.is_file():
            raise RuntimeError(f"Completed run lacks runtime evidence: {run_dir}")
        validate_runtime_evidence(load_json(evidence_path), variant)
        return True
    if run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
        archive_run(run_dir, "no_resumable_checkpoint")

    while int(state["attempts"].get(variant["key"], 0)) < MAX_ATTEMPTS:
        if STOP_SENTINEL.exists():
            state.update(status="paused", current_run=None, current_run_key=None, last_progress_utc=utc_now())
            write_json_atomic(STATE_PATH, state)
            append_event(EVENTS_PATH, "queue_paused", reason=str(STOP_SENTINEL))
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
            last_progress_utc=utc_now(),
        )
        write_json_atomic(STATE_PATH, state)

        kwargs = training_kwargs(context["lock"], variant)
        resumed = run_dir.exists() and (run_dir / "weights" / "last.pt").is_file()
        if resumed:
            verify_resume(run_dir, variant, context)
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            train_kwargs: dict[str, Any] = {"resume": True}
            existing = base.read_last_row(run_dir / "results.csv")
            if existing and existing.get("epoch"):
                state["current_epoch"] = int(float(existing["epoch"]))
        else:
            model = YOLO(context["lock"]["weights"][MODEL_KEY]["path"])
            train_kwargs = kwargs
        os.environ["RING3_FACTORIAL_VARIANT_JSON"] = json.dumps(variant, sort_keys=True)
        os.environ["RING3_FACTORIAL_MDG_CONFIG"] = str(MDG_CONFIG_PATH)
        os.environ["RING3_FACTORIAL_MDG_CONFIG_SHA256"] = context["mdg_config_sha256"]
        metadata = {
            "schema": "ring3.cbam_focal_formal_run_metadata.v1",
            "created_utc": utc_now(),
            "variant_key": variant["key"],
            "variant": variant,
            "attempt": attempt,
            "resumed_from_own_last_pt": resumed,
            "parent_checkpoint": "same_run_last.pt" if resumed else None,
            "cross_run_checkpoint_loaded": False,
            "pretrained": True,
            "independent_initialization_from": context["lock"]["weights"][MODEL_KEY]["path"],
            "formal_lock_sha256": context["lock_sha256"],
            "runtime_amendment_sha256": context["runtime_amendment_sha256"],
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
            "training_kwargs": kwargs,
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "e4_started": False,
        }
        add_callbacks(model, variant=variant, state=state, metadata=metadata)
        append_event(
            EVENTS_PATH,
            "attempt_started",
            variant=variant["key"],
            attempt=attempt,
            resumed=resumed,
            run_dir=str(run_dir),
        )
        try:
            model.train(trainer=FactorialTrainer, **train_kwargs)
            actual_dir = Path(state.get("run_dir", run_dir))
            complete, final_row = base.completed_run(actual_dir, 150)
            if not complete or final_row is None:
                raise RuntimeError(f"Run ended without finite epoch-150 artifacts: {actual_dir}")
            evidence_path = actual_dir / "cbam_focal_formal_runtime_evidence.json"
            if not evidence_path.is_file():
                raise RuntimeError("Run ended without runtime wiring evidence")
            evidence = load_json(evidence_path)
            validate_runtime_evidence(evidence, variant)
            append_event(
                EVENTS_PATH,
                "run_completed",
                variant=variant["key"],
                attempt=attempt,
                final_row=final_row,
                best_row=best_row(actual_dir / "results.csv"),
            )
            return True
        except BaseException as exc:
            numerical = isinstance(exc, FloatingPointError) or any(
                token in (str(exc) + type(exc).__name__).lower()
                for token in ("non-finite", "nan", "inf")
            )
            state.update(
                status="attempt_failed",
                last_progress_utc=utc_now(),
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                numerical_failure=numerical,
            )
            write_json_atomic(STATE_PATH, state)
            append_event(
                EVENTS_PATH,
                "run_attempt_failed",
                variant=variant["key"],
                attempt=attempt,
                numerical_failure=numerical,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            traceback.print_exc()
            if numerical and run_dir.exists():
                archive_run(run_dir, "numerical_failure")
            elif run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
                archive_run(run_dir, "unresumable_failure")
    append_event(EVENTS_PATH, "attempts_exhausted", variant=variant["key"], attempts=MAX_ATTEMPTS)
    return False


def summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "e0": {
            "run_dir": str(E0_RUN),
            "best_row": context["e0_anchor"]["best_row"],
            "final_row": context["e0_anchor"]["final_row"],
        }
    }
    e0_map = float(context["e0_anchor"]["best_row"]["metrics/mAP50-95(B)"])
    for variant in context["config"]["runs"]:
        run_dir = FORMAL_ROOT / variant["name"]
        complete, final = base.completed_run(run_dir, 150)
        if not complete or final is None:
            metrics[variant["key"]] = {"status": "incomplete", "run_dir": str(run_dir)}
            continue
        best = best_row(run_dir / "results.csv")
        metrics[variant["key"]] = {
            "status": "completed",
            "run_dir": str(run_dir),
            "best_row": best,
            "final_row": final,
            "best_map50_95_delta_vs_e0": float(best["metrics/mAP50-95(B)"]) - e0_map,
        }
    payload = {
        "schema": "ring3.cbam_focal_formal_queue_summary.v1",
        "created_utc": utc_now(),
        "single_seed_mechanism_evidence_only": True,
        "metrics": metrics,
        "e4_started": False,
    }
    if all(metrics.get(key, {}).get("status") == "completed" for key in ("cbam_focal", "cbam_focal_e2", "cbam_focal_e3")):
        combo = float(metrics["cbam_focal"]["best_row"]["metrics/mAP50-95(B)"])
        e2 = float(metrics["cbam_focal_e2"]["best_row"]["metrics/mAP50-95(B)"])
        e3 = float(metrics["cbam_focal_e3"]["best_row"]["metrics/mAP50-95(B)"])
        payload["increments"] = {
            "combined_e2_minus_combined_base": e2 - combo,
            "combined_e3_minus_combined_e2": e3 - e2,
            "combined_e3_minus_e0": e3 - e0_map,
        }
    write_json_atomic(SUMMARY_PATH, payload)
    return payload


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_path = Path(context["lock"]["weights"][MODEL_KEY]["path"])
    source = YOLO(str(source_path)).model
    tested = []
    for variant in context["config"]["runs"]:
        if variant["degraded_small_weighting"]:
            model = CBAMFocalSmallWeightedDetectionModel(
                source.yaml, ch=3, nc=int(source.nc), verbose=False, degraded_small_weight=1.5
            )
        else:
            model = FactorialDetectionModel(
                source.yaml,
                ch=3,
                nc=int(source.nc),
                verbose=False,
                use_cbam=bool(variant["cbam"]),
                use_focal_eiou=bool(variant["focal_eiou"]),
            )
        model.load(source)
        if variant["cbam"]:
            transferred = {
                str(index): bool(
                    torch.equal(
                        model.model[index].cv1.conv.weight.detach().cpu(),
                        source.model[index].cv1.conv.weight.detach().cpu(),
                    )
                )
                for index in (4, 6)
            }
            if not all(transferred.values()):
                raise RuntimeError(f"Pretrained C2f weights did not transfer into CBAM: {transferred}")
        else:
            transferred = {}
        model.to(device).train()
        model.args = get_cfg(DEFAULT_CFG_DICT)
        model.criterion = model.init_criterion()
        evidence = model.factorial_evidence()
        synthetic_evidence = {
            "variant": variant,
            **evidence,
            "dual_view_enabled": bool(variant["dual_view"]),
            "degraded_small_weighting_enabled": bool(variant["degraded_small_weighting"]),
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "pretrained_base_weight_matches": transferred,
        }
        if variant["degraded_small_weighting"] and not isinstance(model.criterion, SmallObjectWeightedFocalEIoULoss):
            raise RuntimeError("Static self-test did not install the combined E3 loss")
        if variant["degraded_small_weighting"]:
            torch.manual_seed(0)
            clean = torch.rand((2, 3, 160, 160), device=device)
            degraded = clean.pow(1.15).mul(0.88)
            batch = {
                "img": torch.cat((clean, degraded), dim=0),
                "batch_idx": torch.tensor([0, 1, 2, 3], device=device, dtype=torch.float32),
                "cls": torch.tensor([[0], [1], [0], [1]], device=device, dtype=torch.float32),
                "bboxes": torch.tensor(
                    [[0.50, 0.50, 0.10, 0.10], [0.40, 0.40, 0.10, 0.10]] * 2,
                    device=device,
                    dtype=torch.float32,
                ),
                "factorial_gt_weights": torch.tensor([1.0, 1.0, 1.5, 1.5], device=device),
                "factorial_pair_batch_size": 2,
            }
            combined_loss, combined_items = model.loss(batch)
            combined_total = combined_loss.sum()
            if not bool(torch.isfinite(combined_total)):
                raise FloatingPointError("Combined E3 self-test produced a non-finite loss")
            combined_total.backward()
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            combined_evidence = model.factorial_evidence()
            if not finite_gradients:
                raise FloatingPointError("Combined E3 self-test produced a non-finite gradient")
            if int(combined_evidence.get("clean_weighted_target_violations", -1)) != 0:
                raise RuntimeError("Combined E3 self-test leaked weighting into the clean branch")
            if int(combined_evidence.get("weighted_positive_anchor_count", 0)) <= 0:
                raise RuntimeError("Combined E3 self-test did not weight a degraded-small positive anchor")
            if float(combined_evidence.get("weighting_delta_scaled", 0.0)) <= 0.0:
                raise RuntimeError("Combined E3 self-test did not increase the detection objective")
            synthetic_evidence["combined_e3_loss_items"] = {
                key: float(value) for key, value in combined_items.items()
            }
            synthetic_evidence["combined_e3_total_loss_scaled"] = float(combined_total.detach())
            synthetic_evidence["combined_e3_finite_gradients"] = finite_gradients
            synthetic_evidence["combined_e3_weighting_evidence"] = combined_evidence
        tested.append(synthetic_evidence)
        del model

    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0]], device=device, requires_grad=True)
    target = torch.tensor([[0.2, 0.1, 2.2, 2.1]], device=device)
    focal_loss = FocalEIoULoss()(pred, target).mean()
    focal_loss.backward()
    if not bool(torch.isfinite(focal_loss)) or pred.grad is None or not bool(torch.isfinite(pred.grad).all()):
        raise FloatingPointError("Standalone Focal-EIoU forward/backward failed")
    payload = {
        "schema": "ring3.cbam_focal_formal_series_self_test.v1",
        "status": "passed",
        "verified_utc": utc_now(),
        "device": str(device),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": sha256(source_path),
        "tested_variants": tested,
        "standalone_focal_eiou_loss": float(focal_loss.detach()),
        "standalone_focal_eiou_gradient_finite": True,
        "e4_started": False,
    }
    write_json_atomic(SELF_TEST_PATH, payload)
    return payload


def execute_queue(context: dict[str, Any]) -> int:
    activate_protocol(context)
    if STATE_PATH.exists():
        state = load_json(STATE_PATH)
        anchors = {
            "formal_lock_sha256": context["lock_sha256"],
            "runtime_amendment_sha256": context["runtime_amendment_sha256"],
            "mdg_config_sha256": context["mdg_config_sha256"],
            "config_sha256": context["config_sha256"],
            "release_manifest_sha256": context["release"]["sha256"],
        }
        if any(state.get(key) != value for key, value in anchors.items()):
            raise RuntimeError("Existing queue state belongs to a different locked release")
    else:
        state = initial_state(context)
        write_json_atomic(STATE_PATH, state)
    append_event(EVENTS_PATH, "queue_started_or_resumed", completed_runs=state["completed_runs"])

    for variant in context["config"]["runs"]:
        if variant["key"] in state["completed_runs"]:
            run_dir = FORMAL_ROOT / variant["name"]
            complete, _ = base.completed_run(run_dir, 150)
            if not complete:
                raise RuntimeError(f"State marks an incomplete run complete: {variant['key']}")
            continue
        ok = run_variant(variant, context, state)
        if state.get("status") == "paused":
            return 0
        if ok:
            if variant["key"] not in state["completed_runs"]:
                state["completed_runs"].append(variant["key"])
        else:
            if variant["key"] not in state["failed_runs"]:
                state["failed_runs"].append(variant["key"])
        state.update(current_run=None, current_run_key=None, current_epoch=150 if ok else state.get("current_epoch", 0), last_progress_utc=utc_now())
        write_json_atomic(STATE_PATH, state)

    final_summary = summary(context)
    state.update(
        status="completed" if not state["failed_runs"] else "completed_with_failures",
        current_run=None,
        current_run_key=None,
        completed_utc=utc_now(),
        last_progress_utc=utc_now(),
        summary_path=str(SUMMARY_PATH),
        summary_sha256=sha256(SUMMARY_PATH),
        e4_started=False,
    )
    write_json_atomic(STATE_PATH, state)
    append_event(
        EVENTS_PATH,
        "queue_completed",
        status=state["status"],
        completed_runs=state["completed_runs"],
        failed_runs=state["failed_runs"],
        summary=final_summary,
    )
    return 0 if not state["failed_runs"] else 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked CBAM/Focal-EIoU factorial + E2/E3 series")
    parser.add_argument("--mode", choices=("preflight", "self-test", "queue"), default="queue")
    args = parser.parse_args()
    try:
        context = preflight()
        if args.mode == "preflight":
            print(json.dumps({"status": "preflight_passed", "context": context}, ensure_ascii=False, indent=2))
            return 0
        evidence = static_self_test(context)
        if args.mode == "self-test":
            print(json.dumps(evidence, ensure_ascii=False, indent=2))
            return 0
        return execute_queue(context)
    except BaseException as exc:
        previous = load_json(STATE_PATH) if STATE_PATH.exists() else {}
        previous.update(
            status="supervisor_failed",
            last_progress_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            e4_started=False,
        )
        write_json_atomic(STATE_PATH, previous)
        append_event(EVENTS_PATH, "supervisor_failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
