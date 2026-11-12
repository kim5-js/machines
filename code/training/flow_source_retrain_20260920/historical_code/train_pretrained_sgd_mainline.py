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
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from cbam_focal_formal_models import FactorialDetectionModel
from maritime_degradation import MaritimeDegradation, load_config as load_mdg_config
from pretrained_sgd_mainline_models import MainlineSmallWeightedDetectionModel


CONFIG_SCHEMA = "ring3.pretrained_sgd_mainline_config.v1"
MANIFEST_SCHEMA = "ring3.pretrained_sgd_mainline_release_manifest.v1"
EXPECTED_HISTORICAL_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_MDG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_DATA_YAML_SHA256 = "66b63cf9f50952faa5abdc83979599f2de88c64178278fb5b2b4c698ed841f7c"
EXPECTED_DATA_MANIFEST_SHA256 = "0030e4921a95d57722db66afb50a094c25fc3e6c00b93db5567fbbb196576360"
EXPECTED_WEIGHT_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
MAX_ATTEMPTS = 3

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
CONFIG_PATH = SETUP_ROOT / "pretrained_sgd_mainline_config.json"
MANIFEST_PATH = SETUP_ROOT / "pretrained_sgd_mainline_release_manifest.json"
MODEL_MODULE_PATH = SETUP_ROOT / "pretrained_sgd_mainline_models.py"
HISTORICAL_LOCK_PATH = SETUP_ROOT / "formal_experiment_lock.json"
AMENDMENT_PATH = SETUP_ROOT / "formal_runtime_amendment.json"
MDG_CONFIG_PATH = SETUP_ROOT / "e1_mdg_config.json"
OUTPUT_ROOT = DATA_ROOT / "formal_pretrained_sgd_mainline"
FAILED_ROOT = DATA_ROOT / "formal_failed_runs" / "pretrained_sgd_mainline"
STATE_ROOT = DATA_ROOT / "formal_state"
STATE_PATH = STATE_ROOT / "pretrained_sgd_mainline_state.json"
EVENTS_PATH = STATE_ROOT / "pretrained_sgd_mainline_events.jsonl"
PROTOCOL_PATH = STATE_ROOT / "pretrained_sgd_mainline_protocol.json"
SUMMARY_PATH = STATE_ROOT / "pretrained_sgd_mainline_summary.json"
SELF_TEST_PATH = DATA_ROOT / "diagnostics" / "pretrained_sgd_mainline_self_test.json"
STOP_SENTINEL = STATE_ROOT / "STOP_PRETRAINED_SGD_MAINLINE"
RUNTIME_EVIDENCE_NAME = "pretrained_sgd_mainline_runtime_evidence.json"
RUN_METADATA_NAME = "pretrained_sgd_mainline_run_metadata.json"

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


def last_row(path: Path) -> dict[str, str] | None:
    rows = read_rows(path)
    return rows[-1] if rows else None


def best_row(path: Path) -> dict[str, str]:
    metric = "metrics/mAP50-95(B)"
    rows = [row for row in read_rows(path) if row.get(metric) and math.isfinite(float(row[metric]))]
    if not rows:
        raise RuntimeError(f"No finite {metric} row in {path}")
    return max(rows, key=lambda row: float(row[metric]))


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


def completed_run(run_dir: Path, epochs: int = 150) -> tuple[bool, dict[str, str] | None]:
    row = last_row(run_dir / "results.csv")
    if row is None:
        return False, None
    try:
        epoch = int(float(row["epoch"]))
    except (KeyError, TypeError, ValueError):
        return False, row
    finite = not row_has_nonfinite(row, ("train/", "val/", "metrics/", "lr/"))
    weights = all((run_dir / "weights" / name).is_file() for name in ("best.pt", "last.pt"))
    return epoch >= epochs and finite and weights, row


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected pretrained-SGD config schema/status")
    runs = config.get("run_order", [])
    observed = [
        (
            run.get("key"),
            run.get("mdg_mode"),
            run.get("dual_view"),
            run.get("degraded_small_weighting"),
            run.get("cbam"),
            run.get("focal_eiou"),
            run.get("consistency_loss"),
        )
        for run in runs
    ]
    expected = [
        ("e0", "none", False, False, False, False, False),
        ("e1", "single_view", False, False, False, False, False),
        ("e2", "dual_view", True, False, False, False, False),
        ("e3", "dual_view", True, True, False, False, False),
    ]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 4:
        raise ValueError(f"Unexpected E0-E3 run matrix: {observed!r}")

    training = config.get("training", {})
    locked = {
        "model": "/root/autodl-tmp/weights/yolov8s.pt",
        "pretrained": True,
        "epochs": 150,
        "patience": 100,
        "batch": 8,
        "imgsz": 960,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.006,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0000625,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "nbs": 8,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.2,
        "translate": 0.1,
        "scale": 0.45,
        "fliplr": 0.5,
        "mosaic": 0.7,
        "mixup": 0.05,
        "erasing": 0.2,
        "close_mosaic": 10,
        "save_period": -1,
        "seed": 0,
    }
    mismatches = [
        f"{key}: expected {expected_value!r}, found {training.get(key)!r}"
        for key, expected_value in locked.items()
        if training.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError("Pretrained-SGD training recipe changed:\n" + "\n".join(mismatches))
    if str(training.get("optimizer", "")).lower() == "auto":
        raise ValueError("optimizer=auto is forbidden")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("Pretrained-SGD invariants are incomplete")


def verify_release_manifest() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "locked":
        raise RuntimeError("Unexpected pretrained-SGD release manifest")
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
    if len(verified) < 10:
        raise RuntimeError("Pretrained-SGD release manifest is incomplete")
    return {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH), "verified_files": verified}


def verify_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    yaml_path = Path(dataset["yaml"])
    manifest_path = Path(dataset["manifest"])
    if not yaml_path.is_file() or sha256(yaml_path) != EXPECTED_DATA_YAML_SHA256:
        raise RuntimeError("Audited SeaDronesSee YAML is missing or changed")
    if not manifest_path.is_file() or sha256(manifest_path) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("Audited SeaDronesSee manifest is missing or changed")
    process = subprocess.run(
        [sys.executable, str(SETUP_ROOT / "verify_yolo_dataset.py"), str(yaml_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError("SeaDronesSee verification failed:\n" + process.stdout + process.stderr)
    report = json.loads(process.stdout)
    if report.get("label_error_count") or any(report.get("missing_label_counts", {}).values()):
        raise RuntimeError("SeaDronesSee verification reported label errors")
    expected_counts = {"train": int(dataset["train_images"]), "val": int(dataset["val_images"])}
    if any(report.get("counts", {}).get(split) != count for split, count in expected_counts.items()):
        raise RuntimeError(f"SeaDronesSee split counts changed: {report.get('counts')}")
    return report


def preflight() -> dict[str, Any]:
    if STOP_SENTINEL.exists():
        raise RuntimeError(f"Stop sentinel exists: {STOP_SENTINEL}")
    config = load_json(CONFIG_PATH)
    validate_config(config)
    release = verify_release_manifest()
    if sha256(HISTORICAL_LOCK_PATH) != EXPECTED_HISTORICAL_LOCK_SHA256:
        raise RuntimeError("Historical formal lock changed")
    if sha256(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Runtime amendment changed")
    if sha256(MDG_CONFIG_PATH) != EXPECTED_MDG_SHA256:
        raise RuntimeError("MDG config changed")
    load_mdg_config(MDG_CONFIG_PATH)
    if ultralytics.__version__ != "8.4.124":
        raise RuntimeError(f"Unexpected Ultralytics version: {ultralytics.__version__}")
    if torch.__version__ != "2.5.1+cu124":
        raise RuntimeError(f"Unexpected Torch version: {torch.__version__}")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Unexpected CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    weight_path = Path(config["weights"]["path"])
    if not weight_path.is_file() or sha256(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt is missing or changed")
    dataset_report = verify_dataset(config)
    return {
        "config": config,
        "config_sha256": sha256(CONFIG_PATH),
        "release": release,
        "historical_lock_sha256": EXPECTED_HISTORICAL_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_SHA256,
        "dataset_report": dataset_report,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


class MainlineSGDTrainer(DetectionTrainer):
    """Plain YOLOv8s trainer for E0, E1, E2, and degraded-small E3."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.variant = json.loads(os.environ["RING3_SGD_MAINLINE_VARIANT_JSON"])
        self.mdg_mode = str(self.variant["mdg_mode"])
        self.dual_view = bool(self.variant["dual_view"])
        self.small_weighting = bool(self.variant["degraded_small_weighting"])
        self.mdg = None
        if self.mdg_mode != "none":
            mdg_path = Path(os.environ["RING3_SGD_MAINLINE_MDG_CONFIG"])
            if sha256(mdg_path) != os.environ["RING3_SGD_MAINLINE_MDG_SHA256"]:
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
            model = MainlineSmallWeightedDetectionModel(
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
                use_cbam=False,
                use_focal_eiou=False,
            )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        clean = batch["img"]
        image_count = int(clean.shape[0])
        source_batch_idx = batch["batch_idx"]
        source_cls = batch["cls"]
        source_bboxes = batch["bboxes"]

        if self.mdg_mode == "none":
            targets = {"images_total": image_count, "clean_targets": int(source_bboxes.shape[0])}
            self._last_target_counts = targets
            self._target_epoch_counts.update(targets)
            self._target_total_counts.update(targets)
            return batch

        if self.mdg is None:
            raise RuntimeError(f"{self.mdg_mode} run has no MDG instance")
        degraded, mdg_counts = self.mdg.apply_batch(clean.clone())
        self._last_mdg_counts = dict(mdg_counts)
        self._mdg_epoch_counts.update(mdg_counts)
        self._mdg_total_counts.update(mdg_counts)

        if self.mdg_mode == "single_view":
            batch["img"] = degraded
            targets = {"images_total": image_count, "degraded_targets": int(source_bboxes.shape[0])}
            self._last_target_counts = targets
            self._target_epoch_counts.update(targets)
            self._target_total_counts.update(targets)
            return batch

        if self.mdg_mode != "dual_view" or not self.dual_view:
            raise RuntimeError(f"Inconsistent MDG/dual-view mode: {self.variant}")
        batch["img"] = torch.cat((clean, degraded), dim=0)
        batch["batch_idx"] = torch.cat((source_batch_idx, source_batch_idx + image_count), dim=0)
        batch["cls"] = torch.cat((source_cls, source_cls.clone()), dim=0)
        batch["bboxes"] = torch.cat((source_bboxes, source_bboxes.clone()), dim=0)
        batch["e3_pair_batch_size"] = image_count

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
                small_mask,
                torch.full_like(clean_weights, 1.5),
                clean_weights,
            )
            batch["e3_gt_weights"] = torch.cat((clean_weights, degraded_weights), dim=0)
        else:
            small_mask = torch.zeros(source_bboxes.shape[0], device=source_bboxes.device, dtype=torch.bool)
            batch.pop("e3_pair_batch_size", None)

        for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad"):
            value = batch.get(key)
            if isinstance(value, list):
                batch[key] = value + list(value)
            elif isinstance(value, tuple) and len(value) == image_count:
                batch[key] = value + value

        targets = {
            "images_total": image_count,
            "clean_targets": int(source_bboxes.shape[0]),
            "degraded_targets": int(source_bboxes.shape[0]),
            "degraded_small_targets": int(small_mask.sum()),
            "degraded_regular_targets": int((~small_mask).sum()),
        }
        self._last_target_counts = targets
        self._target_epoch_counts.update(targets)
        self._target_total_counts.update(targets)
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
                "target_feature_layer_index": int(model.factorial_p3_layer_index),
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
            raise TypeError(f"Unexpected mainline model type: {type(model).__name__}")
        return {
            "variant": self.variant,
            **model.factorial_evidence(),
            "mdg_mode": self.mdg_mode,
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
    if evidence.get("cbam_enabled") is not False or evidence.get("focal_eiou_enabled") is not False:
        raise RuntimeError("CBAM or Focal-EIoU leaked into the SGD mainline")
    if evidence.get("cbam_layer_types") != ["C2f", "C2f"]:
        raise RuntimeError("Unexpected CBAM topology evidence")
    if int(evidence.get("cbam_attention_parameter_count", -1)) != 0:
        raise RuntimeError("Attention parameters unexpectedly exist")
    expected_criterion = "SmallObjectWeightedDetectionLoss" if variant["degraded_small_weighting"] else "v8DetectionLoss"
    if evidence.get("criterion_type") != expected_criterion:
        raise RuntimeError(f"Wrong detection criterion: {evidence.get('criterion_type')!r}")
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
    mdg = evidence.get("mdg_counts_current_batch", {})
    if variant["mdg_mode"] == "none":
        if mdg:
            raise RuntimeError("E0 unexpectedly applied MDG")
    elif int(mdg.get("images_total", -1)) != int(targets.get("images_total", -2)):
        raise RuntimeError("MDG image accounting does not close")
    if variant["dual_view"]:
        if int(targets.get("clean_targets", -1)) != int(targets.get("degraded_targets", -2)):
            raise RuntimeError("Clean/degraded target counts differ")
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
                raise RuntimeError(f"E3 weighting evidence not ready for {key}: {evidence.get(key)!r}")
        if int(evidence.get("weighted_positive_anchor_count", 0)) <= 0:
            raise RuntimeError("No degraded-small positive anchor has been weighted yet")
        if float(evidence.get("weighting_delta_scaled", 0.0)) <= 0.0:
            raise RuntimeError("E3 weighting did not increase the detection objective")


def training_kwargs(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    keys = (
        "epochs", "patience", "batch", "imgsz", "workers", "optimizer", "lr0", "lrf", "momentum",
        "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl",
        "nbs", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
        "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "copy_paste_mode",
        "auto_augment", "erasing", "close_mosaic", "save_period", "amp", "deterministic", "seed",
        "device", "cache", "rect", "cos_lr", "single_cls", "fraction", "multi_scale", "dropout", "iou",
        "max_det", "val", "split", "save", "plots", "pretrained",
    )
    kwargs = {key: training[key] for key in keys}
    kwargs.update(
        data=config["dataset"]["yaml"],
        project=str(OUTPUT_ROOT),
        name=variant["name"],
        exist_ok=False,
        verbose=True,
    )
    return kwargs


def verify_args(run_dir: Path, config: dict[str, Any], variant: dict[str, Any]) -> None:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise RuntimeError(f"Run args.yaml is missing: {run_dir}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = training_kwargs(config, variant)
    expected["model"] = config["weights"]["path"]
    ignored = {"project", "name", "exist_ok", "verbose"}
    mismatches = []
    for key, value in expected.items():
        if key in ignored:
            continue
        observed = args.get(key)
        if isinstance(value, float) and isinstance(observed, (float, int)):
            equal = math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == value
        if not equal:
            mismatches.append(f"{key}: expected {value!r}, found {observed!r}")
    if mismatches:
        raise RuntimeError("Run args differ from the locked SGD recipe:\n" + "\n".join(mismatches))


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.pretrained_sgd_mainline_protocol.v1",
        "status": "active_locked",
        "created_utc": utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "dataset_report": context["dataset_report"],
        "environment": context["environment"],
        "run_order": context["config"]["run_order"],
        "training": context["config"]["training"],
        "intentional_differences_from_historical_local": context["config"]["intentional_differences_from_historical_local"],
        "invariants": context["config"]["invariants"],
    }


def activate_protocol(context: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(context)
    if PROTOCOL_PATH.is_file():
        existing = load_json(PROTOCOL_PATH)
        left, right = dict(existing), dict(payload)
        left.pop("created_utc", None)
        right.pop("created_utc", None)
        if left != right:
            raise RuntimeError("Pretrained-SGD protocol changed after activation")
        return existing
    write_json_atomic(PROTOCOL_PATH, payload)
    append_event("protocol_activated", protocol_sha256=sha256(PROTOCOL_PATH))
    return payload


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.pretrained_sgd_mainline_state.v1",
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "run_order": [run["key"] for run in context["config"]["run_order"]],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": 25.0,
        "eta_utc": (now + timedelta(hours=25)).isoformat(),
        "old_adamw_runs_reused": False,
        "old_scratch_runs_reused": False,
        "e4_started": False,
    }


def add_callbacks(
    model: YOLO,
    *,
    variant: dict[str, Any],
    context: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    evidence_written = False

    def on_train_start(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir)
        state.update(status="running", run_dir=str(save_dir), last_progress_utc=utc_now())
        write_json_atomic(STATE_PATH, state)
        write_json_atomic(save_dir / RUN_METADATA_NAME, metadata)
        shutil.copyfile(CONFIG_PATH, save_dir / CONFIG_PATH.name)
        if variant["mdg_mode"] != "none":
            shutil.copyfile(MDG_CONFIG_PATH, save_dir / MDG_CONFIG_PATH.name)
        append_event(
            "training_started",
            variant=variant["key"],
            run_dir=str(save_dir),
            attempt=metadata["attempt"],
            resumed=metadata["resumed_from_own_last_pt"],
        )

    def on_train_batch_end(trainer: MainlineSGDTrainer) -> None:
        nonlocal evidence_written
        if trainer.loss is not None and not bool(trainer.loss.detach().isfinite().all()):
            raise FloatingPointError(f"Non-finite training loss in {variant['key']} at epoch {int(trainer.epoch) + 1}")
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
                "schema": "ring3.pretrained_sgd_mainline_runtime_evidence.v1",
                "verified_utc": utc_now(),
                "run_dir": str(trainer.save_dir),
                "pretrained": True,
                "pretrained_source": context["config"]["weights"]["path"],
                "pretrained_source_sha256": EXPECTED_WEIGHT_SHA256,
                "parent_checkpoint": None if not metadata["resumed_from_own_last_pt"] else "same_run_last.pt",
                "cross_run_checkpoint_loaded": False,
                "optimizer": "SGD",
            }
        )
        write_json_atomic(Path(trainer.save_dir) / RUNTIME_EVIDENCE_NAME, evidence)
        state["runtime_evidence"] = evidence
        write_json_atomic(STATE_PATH, state)
        append_event(
            "runtime_wiring_verified",
            variant=variant["key"],
            mdg_mode=variant["mdg_mode"],
            dual_view=variant["dual_view"],
            degraded_small_weighting=variant["degraded_small_weighting"],
            p3_parameter_delta_l2=evidence["optimizer_update_probe"]["parameter_delta_l2"],
        )

    def on_fit_epoch_end(trainer: MainlineSGDTrainer) -> None:
        epoch = min(int(trainer.epoch) + 1, 150)
        save_dir = Path(trainer.save_dir)
        row = last_row(save_dir / "results.csv")
        if row_has_nonfinite(row, ("train/",)):
            raise FloatingPointError(f"Non-finite training epoch in {variant['key']} at epoch {epoch}")
        validation_bad = row_has_nonfinite(row, ("val/",))
        streak = int(state.get("validation_nan_streak", 0)) + 1 if validation_bad else 0
        if streak >= 3:
            raise FloatingPointError(f"Validation loss stayed non-finite for {streak} epochs in {variant['key']}")
        mdg_counts, target_counts = trainer.consume_epoch_counts()
        if variant["mdg_mode"] != "none":
            if int(mdg_counts.get("images_total", -1)) != int(target_counts.get("images_total", -2)):
                raise RuntimeError(f"MDG image accounting mismatch: mdg={mdg_counts}, targets={target_counts}")
        if variant["dual_view"] and int(target_counts.get("clean_targets", -1)) != int(target_counts.get("degraded_targets", -2)):
            raise RuntimeError(f"Clean/degraded target mismatch: {target_counts}")

        recent = list(state.get("recent_epoch_seconds", []))
        cumulative = float(row["time"]) if row and row.get("time") else None
        if cumulative is not None:
            previous = state.get("last_cumulative_train_seconds")
            measured = cumulative - float(previous) if previous is not None else cumulative
            if math.isfinite(measured) and measured > 0.0:
                recent = (recent + [measured])[-10:]
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
                "epoch_completed",
                variant=variant["key"],
                epoch=epoch,
                current_row=row,
                best_row=current_best,
                expected_seconds_per_epoch=expected,
                eta_utc=state["eta_utc"],
            )
        print(
            "PRETRAINED_SGD_MAINLINE_STATE "
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


def archive_run(run_dir: Path, reason: str) -> Path:
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = FAILED_ROOT / f"{run_dir.name}__{reason}__{stamp}"
    shutil.move(str(run_dir), str(destination))
    append_event("run_archived", source=str(run_dir), destination=str(destination), reason=reason)
    return destination


def verify_resume(run_dir: Path, variant: dict[str, Any], context: dict[str, Any]) -> None:
    verify_args(run_dir, context["config"], variant)
    metadata_path = run_dir / RUN_METADATA_NAME
    if not metadata_path.is_file():
        raise RuntimeError("Refusing resume without mainline metadata")
    metadata = load_json(metadata_path)
    expected = {
        "variant_key": variant["key"],
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained": True,
        "optimizer": "SGD",
        "cross_run_checkpoint_loaded": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Refusing resume with mismatched run metadata")


def run_variant(variant: dict[str, Any], context: dict[str, Any], state: dict[str, Any]) -> bool:
    run_dir = OUTPUT_ROOT / variant["name"]
    complete, final = completed_run(run_dir)
    if complete and final is not None:
        evidence_path = run_dir / RUNTIME_EVIDENCE_NAME
        if not evidence_path.is_file():
            raise RuntimeError(f"Completed run lacks runtime evidence: {run_dir}")
        validate_runtime_evidence(load_json(evidence_path), variant)
        verify_args(run_dir, context["config"], variant)
        return True
    if run_dir.exists() and not (run_dir / "weights" / "last.pt").is_file():
        archive_run(run_dir, "no_resumable_checkpoint")

    while int(state["attempts"].get(variant["key"], 0)) < MAX_ATTEMPTS:
        if STOP_SENTINEL.exists():
            state.update(status="paused", current_run=None, current_run_key=None, last_progress_utc=utc_now())
            write_json_atomic(STATE_PATH, state)
            append_event("queue_paused", reason=str(STOP_SENTINEL))
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

        kwargs = training_kwargs(context["config"], variant)
        resumed = run_dir.exists() and (run_dir / "weights" / "last.pt").is_file()
        if resumed:
            verify_resume(run_dir, variant, context)
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            train_kwargs: dict[str, Any] = {"resume": True}
            existing = last_row(run_dir / "results.csv")
            if existing and existing.get("epoch"):
                state["current_epoch"] = int(float(existing["epoch"]))
        else:
            model = YOLO(context["config"]["weights"]["path"])
            train_kwargs = kwargs

        os.environ["RING3_SGD_MAINLINE_VARIANT_JSON"] = json.dumps(variant, sort_keys=True)
        os.environ["RING3_SGD_MAINLINE_MDG_CONFIG"] = str(MDG_CONFIG_PATH)
        os.environ["RING3_SGD_MAINLINE_MDG_SHA256"] = context["mdg_config_sha256"]
        metadata = {
            "schema": "ring3.pretrained_sgd_mainline_run_metadata.v1",
            "created_utc": utc_now(),
            "variant_key": variant["key"],
            "variant": variant,
            "attempt": attempt,
            "resumed_from_own_last_pt": resumed,
            "parent_checkpoint": "same_run_last.pt" if resumed else None,
            "cross_run_checkpoint_loaded": False,
            "pretrained": True,
            "pretrained_source": context["config"]["weights"]["path"],
            "pretrained_source_sha256": EXPECTED_WEIGHT_SHA256,
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
        }
        add_callbacks(model, variant=variant, context=context, state=state, metadata=metadata)
        append_event(
            "attempt_started",
            variant=variant["key"],
            attempt=attempt,
            resumed=resumed,
            run_dir=str(run_dir),
        )
        try:
            model.train(trainer=MainlineSGDTrainer, **train_kwargs)
            actual_dir = Path(state.get("run_dir", run_dir))
            complete, final = completed_run(actual_dir)
            if not complete or final is None:
                raise RuntimeError(f"Run ended without finite epoch-150 artifacts: {actual_dir}")
            evidence_path = actual_dir / RUNTIME_EVIDENCE_NAME
            if not evidence_path.is_file():
                raise RuntimeError("Run ended without runtime wiring evidence")
            evidence = load_json(evidence_path)
            validate_runtime_evidence(evidence, variant)
            verify_args(actual_dir, context["config"], variant)
            append_event(
                "run_completed",
                variant=variant["key"],
                attempt=attempt,
                final_row=final,
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
    append_event("attempts_exhausted", variant=variant["key"], attempts=MAX_ATTEMPTS)
    return False


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_path = Path(context["config"]["weights"]["path"])
    source = YOLO(str(source_path)).model
    source_state = source.state_dict()
    tested = []
    for variant in context["config"]["run_order"]:
        if variant["degraded_small_weighting"]:
            model = MainlineSmallWeightedDetectionModel(
                source.yaml, ch=3, nc=5, verbose=False, degraded_small_weight=1.5
            )
        else:
            model = FactorialDetectionModel(
                source.yaml, ch=3, nc=5, verbose=False, use_cbam=False, use_focal_eiou=False
            )
        model.load(source)
        target_state = model.state_dict()
        compatible = [key for key, tensor in source_state.items() if key in target_state and target_state[key].shape == tensor.shape]
        mismatched = [key for key in compatible if not torch.equal(target_state[key].cpu(), source_state[key].cpu())]
        if mismatched or len(compatible) < 300:
            raise RuntimeError(f"Pretrained transfer failed for {variant['key']}: compatible={len(compatible)}, mismatched={mismatched[:5]}")
        model.to(device).train()
        model.args = get_cfg(DEFAULT_CFG_DICT)
        model.criterion = model.init_criterion()
        evidence = model.factorial_evidence()
        if evidence.get("cbam_enabled") is not False or evidence.get("focal_eiou_enabled") is not False:
            raise RuntimeError("Static self-test detected forbidden CBAM/Focal-EIoU")
        item = {
            "variant": variant,
            "compatible_pretrained_tensors": len(compatible),
            "mismatched_pretrained_tensors": len(mismatched),
            "model_evidence": evidence,
        }
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
                "e3_gt_weights": torch.tensor([1.0, 1.0, 1.5, 1.5], device=device),
                "e3_pair_batch_size": 2,
            }
            loss, loss_items = model.loss(batch)
            total = loss.sum()
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Static E3 self-test produced a non-finite loss")
            total.backward()
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            weighting = model.factorial_evidence()
            if not finite_gradients or int(weighting.get("weighted_positive_anchor_count", 0)) <= 0:
                raise RuntimeError("Static E3 self-test did not activate finite degraded-small weighting")
            item["e3_loss_items"] = {key: float(value) for key, value in loss_items.items()}
            item["e3_weighting_evidence"] = weighting
        tested.append(item)
        del model
        torch.cuda.empty_cache()

    mdg = MaritimeDegradation(load_mdg_config(MDG_CONFIG_PATH))
    probe = torch.rand((2, 3, 160, 160), device=device)
    degraded, counts = mdg.apply_batch(probe.clone())
    if not bool(torch.isfinite(degraded).all()) or int(counts.get("images_total", -1)) != 2:
        raise RuntimeError("Static MDG self-test failed")
    payload = {
        "schema": "ring3.pretrained_sgd_mainline_self_test.v1",
        "status": "passed",
        "verified_utc": utc_now(),
        "device": str(device),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": EXPECTED_WEIGHT_SHA256,
        "tested_variants": tested,
        "mdg_probe_counts": counts,
        "mdg_probe_finite": True,
        "e4_started": False,
    }
    write_json_atomic(SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in context["config"]["run_order"]:
        run_dir = OUTPUT_ROOT / variant["name"]
        complete, final = completed_run(run_dir)
        if not complete or final is None:
            metrics[variant["key"]] = {"status": "incomplete", "run_dir": str(run_dir)}
            continue
        metrics[variant["key"]] = {
            "status": "completed",
            "run_dir": str(run_dir),
            "best_row": best_row(run_dir / "results.csv"),
            "final_row": final,
        }
    payload: dict[str, Any] = {
        "schema": "ring3.pretrained_sgd_mainline_summary.v1",
        "created_utc": utc_now(),
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
    write_json_atomic(SUMMARY_PATH, payload)
    return payload


def execute_queue(context: dict[str, Any]) -> int:
    activate_protocol(context)
    if STATE_PATH.is_file():
        state = load_json(STATE_PATH)
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
        state = state_template(context)
        write_json_atomic(STATE_PATH, state)
    append_event("queue_started_or_resumed", completed_runs=state["completed_runs"])

    for variant in context["config"]["run_order"]:
        if variant["key"] in state["completed_runs"]:
            complete, _ = completed_run(OUTPUT_ROOT / variant["name"])
            if not complete:
                raise RuntimeError(f"State marks an incomplete run complete: {variant['key']}")
            continue
        ok = run_variant(variant, context, state)
        if state.get("status") == "paused":
            return 0
        if ok:
            if variant["key"] not in state["completed_runs"]:
                state["completed_runs"].append(variant["key"])
        elif variant["key"] not in state["failed_runs"]:
            state["failed_runs"].append(variant["key"])
        state.update(
            current_run=None,
            current_run_key=None,
            current_epoch=150 if ok else state.get("current_epoch", 0),
            last_progress_utc=utc_now(),
        )
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
        "queue_completed",
        status=state["status"],
        completed_runs=state["completed_runs"],
        failed_runs=state["failed_runs"],
        summary=final_summary,
    )
    return 0 if not state["failed_runs"] else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    try:
        context = preflight()
        print("PRETRAINED_SGD_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        result = static_self_test(context)
        print("PRETRAINED_SGD_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return execute_queue(context)
    except BaseException as exc:
        payload = {
            "schema": "ring3.pretrained_sgd_mainline_preflight_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e4_started": False,
        }
        write_json_atomic(STATE_PATH, payload)
        append_event("supervisor_failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
