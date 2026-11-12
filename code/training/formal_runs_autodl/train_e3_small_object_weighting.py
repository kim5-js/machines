from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

import formal_baseline_queue as base
from e3_small_object_weighting import (
    SmallObjectWeightedDualViewDetectionModel,
    load_e3_config,
    self_test as e3_model_self_test,
)
from maritime_degradation import MaritimeDegradation, load_config as load_mdg_config


EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_MDG_CONFIG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_WEIGHT_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPECTED_DATA_YAML_SHA256 = "66b63cf9f50952faa5abdc83979599f2de88c64178278fb5b2b4c698ed841f7c"
EXPECTED_DATA_MANIFEST_SHA256 = "0030e4921a95d57722db66afb50a094c25fc3e6c00b93db5567fbbb196576360"
EXPECTED_E2_STATE_SHA256 = "249e0cde222f0520f6c00406ddc22c79b7596f369c00f789043141b73e1f9326"
EXPECTED_E2_PROTOCOL_SHA256 = "47036a666c03c78ea027bb6f07ef9dd8eaac9b145968047df5b118f98e9813ec"
EXPECTED_E2_RESULTS_SHA256 = "3a272ecac8c235479223d158068585b24b088e92800762606b04b0a4e51cfe8a"
EXPECTED_E2_ARGS_SHA256 = "ce815854eb855d65134d8f6c57422ed41f2cfaa52318705849c2816245254525"
EXPECTED_E2_RUNTIME_SHA256 = "4a66ef5c1bb34ee9c9149939467b1542672928a151e6a0ef478c297e053a1dd7"
MODEL_KEY = "yolov8s"
DATASET_KEY = "seadronessee"
FORMAL_NAME = "seed0_yolov8s_seadronessee_e3_dualview_smallobj_alpha15_img960_e150"
MAX_FORMAL_ATTEMPTS = 3

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
STATE_ROOT = DATA_ROOT / "formal_state"
DIAGNOSTICS_ROOT = DATA_ROOT / "diagnostics"
FORMAL_ROOT = DATA_ROOT / "formal_e3_small_object_weighting"
FAILED_ROOT = DATA_ROOT / "formal_failed_runs"
CONFIG_PATH = SETUP_ROOT / "e3_small_object_weighting_config.json"
MDG_CONFIG_PATH = SETUP_ROOT / "e1_mdg_config.json"
AMENDMENT_PATH = SETUP_ROOT / "formal_runtime_amendment.json"
DESIGNATION_PATH = SETUP_ROOT / "formal_e2_dualview_mainline_designation.json"
RELEASE_MANIFEST_PATH = SETUP_ROOT / "e3_small_object_weighting_release_manifest.json"
FORMAL_STATE_PATH = STATE_ROOT / "e3_small_object_weighting_state.json"
FORMAL_EVENTS_PATH = STATE_ROOT / "e3_small_object_weighting_events.jsonl"
FORMAL_PROTOCOL_PATH = STATE_ROOT / "e3_small_object_weighting_protocol.json"
SMOKE_STATE_PATH = DIAGNOSTICS_ROOT / "e3_small_object_weighting_smoke_state.json"
SMOKE_EVENTS_PATH = DIAGNOSTICS_ROOT / "e3_small_object_weighting_smoke_events.jsonl"
SELF_TEST_PATH = DIAGNOSTICS_ROOT / "e3_small_object_weighting_self_test.json"
STOP_SENTINEL = STATE_ROOT / "STOP_E3_SMALL_OBJECT_WEIGHTING"
E2_STATE_PATH = STATE_ROOT / "e2_lambda0_state.json"
E2_PROTOCOL_PATH = STATE_ROOT / "e2_lambda0_protocol.json"
E2_RUN = DATA_ROOT / "formal_e2_lambda_ablation" / "seed0_yolov8s_seadronessee_e2_mdg_p3cons_lambda0_img960_e150"

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


def read_best_metric_row(results_path: Path) -> dict[str, str]:
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    valid = [row for row in rows if row.get("metrics/mAP50-95(B)") and math.isfinite(float(row["metrics/mAP50-95(B)"]))]
    if not valid:
        raise RuntimeError(f"No finite mAP50-95 row in {results_path}")
    return max(valid, key=lambda row: float(row["metrics/mAP50-95(B)"]))


def verify_release_manifest() -> dict[str, Any]:
    manifest = load_json(RELEASE_MANIFEST_PATH)
    if manifest.get("schema") != "ring3.e3_small_object_weighting_release_manifest.v1" or manifest.get("status") != "locked":
        raise RuntimeError("E3 release manifest is missing its locked schema/status")
    verified = []
    for item in manifest.get("files", []):
        path = SETUP_ROOT / str(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"E3 release file is missing or changed: {path}")
        verified.append({"path": str(item["path"]), "sha256": item["sha256"]})
    if len(verified) < 10:
        raise RuntimeError("E3 release manifest is incomplete")
    return {"path": str(RELEASE_MANIFEST_PATH), "sha256": sha256(RELEASE_MANIFEST_PATH), "verified_files": verified}


def verify_parent_e2(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    expected_paths = {
        E2_STATE_PATH: EXPECTED_E2_STATE_SHA256,
        E2_PROTOCOL_PATH: EXPECTED_E2_PROTOCOL_SHA256,
        E2_RUN / "results.csv": EXPECTED_E2_RESULTS_SHA256,
        E2_RUN / "args.yaml": EXPECTED_E2_ARGS_SHA256,
        E2_RUN / "e2_sdc_runtime_evidence.json": EXPECTED_E2_RUNTIME_SHA256,
    }
    for path, expected in expected_paths.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"New formal E2 anchor is missing or changed: {path}")
    state = load_json(E2_STATE_PATH)
    runtime = load_json(E2_RUN / "e2_sdc_runtime_evidence.json")
    if state.get("status") != "completed" or int(state.get("current_epoch", 0)) != 150:
        raise RuntimeError("New formal E2 is not completed at epoch 150")
    if float(state.get("consistency_lambda", float("nan"))) != 0.0:
        raise RuntimeError("New formal E2 is not the lambda-zero run")
    if float(runtime.get("consistency_loss_weighted_per_pair", -1.0)) != 0.0:
        raise RuntimeError("New formal E2 has a weighted consistency contribution")
    if float(runtime.get("consistency_loss_scaled_contribution", -1.0)) != 0.0:
        raise RuntimeError("New formal E2 has a scaled consistency contribution")
    probe = runtime.get("consistency_only_gradient_probe", {})
    if int(probe.get("nonzero_gradient_tensors", -1)) != 0 or float(probe.get("l2_norm", -1.0)) != 0.0:
        raise RuntimeError("New formal E2 has a consistency-only gradient")
    if int(runtime.get("clean_detection_targets", -1)) != int(runtime.get("degraded_detection_targets", -2)):
        raise RuntimeError("New formal E2 clean/degraded targets do not match")
    if runtime.get("detect_input_indices") != [15, 18, 21] or runtime.get("target_feature_layer_type") != "C2f":
        raise RuntimeError("New formal E2 topology evidence changed")
    complete, final_row = base.completed_run(E2_RUN, 150)
    if not complete or final_row is None:
        raise RuntimeError("New formal E2 artifacts are incomplete")
    base.verify_resume_args(E2_RUN, lock, DATASET_KEY)
    args = yaml.safe_load((E2_RUN / "args.yaml").read_text(encoding="utf-8"))
    if args.get("pretrained") is not True or args.get("model") != lock["weights"][MODEL_KEY]["path"] or args.get("resume") is not False:
        raise RuntimeError("New formal E2 was not an independent locked-pretrained start")
    designation = load_json(DESIGNATION_PATH)
    if designation.get("status") != "locked_designation" or designation.get("objective", {}).get("consistency_loss") is not False:
        raise RuntimeError("New formal E2 designation is invalid")
    expected_parent = {
        "state_sha256": EXPECTED_E2_STATE_SHA256,
        "protocol_sha256": EXPECTED_E2_PROTOCOL_SHA256,
        "results_sha256": EXPECTED_E2_RESULTS_SHA256,
        "args_sha256": EXPECTED_E2_ARGS_SHA256,
        "runtime_evidence_sha256": EXPECTED_E2_RUNTIME_SHA256,
    }
    if any(config["parent_e2"].get(key) != value for key, value in expected_parent.items()):
        raise RuntimeError("E3 parent-E2 hashes differ from the designation")
    return {
        "designation_path": str(DESIGNATION_PATH),
        "designation_sha256": sha256(DESIGNATION_PATH),
        "run_dir": str(E2_RUN),
        **expected_parent,
        "best_row": read_best_metric_row(E2_RUN / "results.csv"),
        "final_row": final_row,
        "formal_definition": "L_det(clean) + L_det(degraded), no consistency loss",
    }


def preflight() -> dict[str, Any]:
    if STOP_SENTINEL.exists():
        raise RuntimeError(f"Explicit E3 stop sentinel exists: {STOP_SENTINEL}")
    release = verify_release_manifest()
    if sha256(AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256 or sha256(MDG_CONFIG_PATH) != EXPECTED_MDG_CONFIG_SHA256:
        raise RuntimeError("Formal runtime amendment or MDG config changed")
    config = load_e3_config(CONFIG_PATH)
    lock = base.load_lock()
    lock_hash = base.preflight(lock)
    if lock_hash != EXPECTED_LOCK_SHA256:
        raise RuntimeError(f"Formal lock hash mismatch: {lock_hash}")
    if sha256(Path(lock["weights"][MODEL_KEY]["path"])) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt changed")
    dataset = lock["dataset_protocols"][DATASET_KEY]
    if sha256(Path(dataset["yaml"])) != EXPECTED_DATA_YAML_SHA256 or sha256(Path(dataset["manifest_path"])) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("Locked SeaDronesSee protocol changed")
    common = lock["common_train"]
    expected = {
        "pretrained": common["pretrained"], "model": lock["weights"][MODEL_KEY]["path"],
        "epochs": common["epochs"], "imgsz": common["imgsz"], "batch": common["batch"],
        "workers": common["workers"], "optimizer": common["optimizer"], "lr0": dataset["lr0"],
        "lrf": common["lrf"], "momentum": common["momentum"], "weight_decay": common["weight_decay"],
        "seed": common["seed"], "device": 0,
    }
    mismatches = [f"{key}: expected {value!r}, found {config['training'].get(key)!r}" for key, value in expected.items() if config["training"].get(key) != value]
    if mismatches:
        raise RuntimeError("E3 differs from locked E2 training conditions:\n" + "\n".join(mismatches))
    return {
        "lock": lock,
        "lock_sha256": lock_hash,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_CONFIG_SHA256,
        "e3_config_sha256": sha256(CONFIG_PATH),
        "release_manifest": release,
        "parent_e2": verify_parent_e2(config, lock),
    }


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.e3_small_object_weighting_protocol.v1",
        "status": "locked_active",
        "stage": "E3",
        "run_name": FORMAL_NAME,
        "run_dir": str(FORMAL_ROOT / FORMAL_NAME),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "e3_config_sha256": context["e3_config_sha256"],
        "release_manifest": context["release_manifest"],
        "parent_e2": context["parent_e2"],
        "objective": {
            "clean_detection_loss": True, "degraded_detection_loss": True,
            "degraded_small_weight": 1.5, "ordinary_weight": 1.0,
            "small_area_lt_px2": 1024, "consistency_loss": False,
            "severity_weighting": False, "curriculum": False, "cbam": False, "focal_eiou": False,
        },
        "training": context["lock"]["common_train"],
        "model": context["lock"]["weights"][MODEL_KEY],
        "dataset": context["lock"]["dataset_protocols"][DATASET_KEY],
        "max_attempts": MAX_FORMAL_ATTEMPTS,
        "resume_policy": "same_run_last_pt_only_for_non_numeric_interruptions",
        "numeric_failure_policy": "archive_then_restart_from_same_locked_pretrained",
    }


def activate_protocol(context: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(context)
    if FORMAL_PROTOCOL_PATH.is_file() and load_json(FORMAL_PROTOCOL_PATH) != payload:
        raise RuntimeError("E3 protocol changed after activation")
    if not FORMAL_PROTOCOL_PATH.is_file():
        write_json_atomic(FORMAL_PROTOCOL_PATH, payload)
    return payload


def validate_runtime_evidence(evidence: dict[str, Any]) -> None:
    expected = {
        "ordinary_target_weight": 1.0, "degraded_small_target_weight": 1.5,
        "weighting_scope": "degraded_small_assigned_positive_anchors_only",
        "criterion_type": "SmallObjectWeightedDetectionLoss",
        "detect_input_indices": [15, 18, 21], "p3_layer_index": 15, "p3_layer_type": "C2f",
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E3 runtime wiring identity mismatch")
    if any(evidence.get(key) is not True for key in ("normalization_denominator_unchanged", "background_anchor_weight_unchanged", "no_consistency_loss")):
        raise RuntimeError("E3 runtime invariants failed")
    checks = (
        int(evidence.get("clean_detection_targets", -1)) == int(evidence.get("degraded_detection_targets", -2)),
        int(evidence.get("degraded_small_targets", 0)) > 0,
        int(evidence.get("clean_weighted_target_violations", -1)) == 0,
        int(evidence.get("weighted_positive_anchor_count", 0)) > 0,
        int(evidence.get("negative_anchor_weight_violations", -1)) == 0,
        float(evidence.get("weighting_delta_scaled", 0.0)) > 0.0,
        int(evidence.get("mdg_counts_current_batch", {}).get("images_total", -1)) == int(evidence.get("pair_batch_size", -2)),
    )
    if not all(checks):
        raise RuntimeError(f"E3 runtime weighting evidence failed: {evidence}")
    update = evidence.get("optimizer_update_probe", {})
    if update.get("finite") is not True or float(update.get("parameter_delta_l2", 0.0)) <= 0.0:
        raise RuntimeError("E3 P3 detection-path update was not verified")
    for mapping in (
        evidence.get("weighted_loss_per_component", {}),
        evidence.get("unit_weight_control_loss_per_component", {}),
        evidence.get("weighting_delta_per_component", {}),
    ):
        if not mapping or any(not math.isfinite(float(value)) for value in mapping.values()):
            raise FloatingPointError(f"E3 loss evidence is non-finite: {mapping}")


class E3SmallObjectWeightingTrainer(DetectionTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config_path = Path(os.environ["RING3_E3_CONFIG"])
        mdg_path = Path(os.environ["RING3_E3_MDG_CONFIG"])
        if sha256(config_path) != os.environ["RING3_E3_CONFIG_SHA256"]:
            raise RuntimeError("E3 config changed before trainer construction")
        if sha256(mdg_path) != os.environ["RING3_E3_MDG_CONFIG_SHA256"]:
            raise RuntimeError("E3 MDG config changed before trainer construction")
        self.e3_config = load_e3_config(config_path)
        self.mdg = MaritimeDegradation(load_mdg_config(mdg_path))
        self.small_area_threshold = float(self.e3_config["small_object_definition"]["area_lt_px2"])
        self.small_weight = float(self.e3_config["target_weights"]["degraded_small"])
        self._mdg_epoch_counts: Counter[str] = Counter()
        self._mdg_total_counts: Counter[str] = Counter()
        self._target_epoch_counts: Counter[str] = Counter()
        self._target_total_counts: Counter[str] = Counter()
        self._last_mdg_counts: dict[str, int] = {}
        self._last_target_counts: dict[str, int] = {}
        self._optimizer_update_probe: dict[str, Any] = {}
        self._optimizer_step_calls = 0
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        model = self.set_model_names_for_load(
            SmallObjectWeightedDualViewDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                degraded_small_weight=self.small_weight,
            )
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        clean = batch["img"]
        pair_batch_size = int(clean.shape[0])
        source_batch_idx, source_cls, source_bboxes = batch["batch_idx"], batch["cls"], batch["bboxes"]
        area_px2 = (
            source_bboxes[:, 2] * float(clean.shape[-1])
            * source_bboxes[:, 3] * float(clean.shape[-2])
        )
        small_mask = area_px2 < self.small_area_threshold
        clean_weights = torch.ones(source_bboxes.shape[0], device=source_bboxes.device, dtype=source_bboxes.dtype)
        degraded_weights = torch.where(
            small_mask, torch.full_like(clean_weights, self.small_weight), clean_weights
        )
        degraded, mdg_counts = self.mdg.apply_batch(clean.clone())
        batch["img"] = torch.cat((clean, degraded), dim=0)
        batch["batch_idx"] = torch.cat((source_batch_idx, source_batch_idx + pair_batch_size), dim=0)
        batch["cls"] = torch.cat((source_cls, source_cls.clone()), dim=0)
        batch["bboxes"] = torch.cat((source_bboxes, source_bboxes.clone()), dim=0)
        batch["e3_gt_weights"] = torch.cat((clean_weights, degraded_weights), dim=0)
        batch["e3_pair_batch_size"] = pair_batch_size
        batch["e3_mdg_counts"] = mdg_counts
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
        self._mdg_total_counts.update(mdg_counts)
        self._target_epoch_counts.update(target_counts)
        self._target_total_counts.update(target_counts)
        return batch

    def optimizer_step(self) -> None:
        self._optimizer_step_calls += 1
        model = unwrap_model(self.model)
        parameter_before = None
        target_parameter = None
        pre_step_gradient: dict[str, Any] = {}
        if not self._optimizer_update_probe and isinstance(model, SmallObjectWeightedDualViewDetectionModel):
            parameters = model.e3_target_parameters()
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
                "target_feature_layer_index": model.e3_p3_layer_index,
                "target_parameter_shape": [int(value) for value in target_parameter.shape],
                "pre_step_gradient": pre_step_gradient,
                "finite": bool(torch.isfinite(delta).all()),
                "parameter_delta_l2": float(delta.square().sum().sqrt()),
                "parameter_delta_max_abs": float(delta.abs().max()),
            }
            if candidate["finite"] is not True:
                raise FloatingPointError(f"E3 optimizer update became non-finite: {candidate}")
            if math.isfinite(candidate["parameter_delta_l2"]) and candidate["parameter_delta_l2"] > 0.0:
                self._optimizer_update_probe = candidate

    def runtime_evidence(self) -> dict[str, Any]:
        model = unwrap_model(self.model)
        if not isinstance(model, SmallObjectWeightedDualViewDetectionModel):
            raise TypeError(f"Unexpected E3 model type: {type(model).__name__}")
        return {
            **model.e3_loss_evidence(),
            "small_object_area_lt_px2": self.small_area_threshold,
            "small_object_coordinate_space": "post_augmentation_training_pixels",
            "mdg_counts_current_batch": dict(self._last_mdg_counts),
            "target_counts_current_batch": dict(self._last_target_counts),
            "optimizer_update_probe": dict(self._optimizer_update_probe),
        }

    def consume_epoch_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        mdg, targets = dict(self._mdg_epoch_counts), dict(self._target_epoch_counts)
        self._mdg_epoch_counts.clear()
        self._target_epoch_counts.clear()
        return mdg, targets

    def total_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        return dict(self._mdg_total_counts), dict(self._target_total_counts)


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
    now = datetime.now(timezone.utc)
    expected_seconds = 160.0
    return {
        "schema": "ring3.e3_small_object_weighting_state.v1",
        "status": "preflight_passed",
        "mode": mode,
        "stage": "E3",
        "current_run": run_dir.name,
        "run_dir": str(run_dir),
        "current_epoch": 0,
        "target_epochs": target_epochs,
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "expected_seconds_per_epoch": expected_seconds,
        "eta_utc": (now + timedelta(seconds=target_epochs * expected_seconds)).isoformat(),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "e3_config_sha256": context["e3_config_sha256"],
        "release_manifest_sha256": context["release_manifest"]["sha256"],
        "parent_e2": context["parent_e2"],
        "attempts": 0,
        "validation_nan_streak": 0,
        "paper_result_eligible": mode == "formal",
        "independent_initialization_from": context["lock"]["weights"][MODEL_KEY]["path"],
        "pretrained": True,
        "consistency_loss_enabled": False,
        "severity_weighting_enabled": False,
        "curriculum_enabled": False,
        "cbam_enabled": False,
        "focal_eiou_enabled": False,
        "ordinary_target_weight": 1.0,
        "degraded_small_target_weight": 1.5,
        "small_object_area_lt_px2": 1024,
    }


def archive_run(run_dir: Path, reason: str, events_path: Path) -> Path:
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = FAILED_ROOT / f"{run_dir.name}_{stamp}_{reason}"
    shutil.move(str(run_dir), str(destination))
    append_event(events_path, "run_archived", source=str(run_dir), destination=str(destination), reason=reason)
    return destination


def verify_resume_metadata(run_dir: Path, context: dict[str, Any]) -> None:
    base.verify_resume_args(run_dir, context["lock"], DATASET_KEY)
    metadata_path = run_dir / "e3_small_object_weighting_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Refusing E3 resume without its own metadata")
    metadata = load_json(metadata_path)
    expected = {
        "e3_config_sha256": context["e3_config_sha256"],
        "release_manifest_sha256": context["release_manifest"]["sha256"],
        "parent_e2_state_sha256": EXPECTED_E2_STATE_SHA256,
        "pretrained": True,
        "consistency_loss_enabled": False,
        "degraded_small_target_weight": 1.5,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Refusing to resume E3 with mismatched evidence")


def training_metadata(
    mode: str,
    context: dict[str, Any],
    kwargs: dict[str, Any],
    attempt: int,
    *,
    resumed: bool,
) -> dict[str, Any]:
    payload = protocol_payload(context)
    payload.update(
        {
            "schema": "ring3.e3_small_object_weighting_run_metadata.v1",
            "mode": mode,
            "attempt": attempt,
            "created_utc": utc_now(),
            "training_kwargs": kwargs,
            "diagnostic_excluded_from_paper_results": mode == "smoke",
            "pretrained": True,
            "independent_initialization_from": context["lock"]["weights"][MODEL_KEY]["path"],
            "parent_checkpoint": None if not resumed else "same_run_last.pt",
            "cross_run_checkpoint_loaded": False,
            "resumed_from_own_last_pt": resumed,
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
            "cbam_enabled": False,
            "focal_eiou_enabled": False,
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
    evidence_written = False

    def on_train_start(trainer: Any) -> None:
        save_dir = Path(trainer.save_dir)
        state.update(status="running", run_dir=str(save_dir), last_progress_utc=utc_now())
        write_json_atomic(state_path, state)
        write_json_atomic(save_dir / "e3_small_object_weighting_metadata.json", metadata)
        shutil.copyfile(CONFIG_PATH, save_dir / "e3_small_object_weighting_config.json")
        shutil.copyfile(MDG_CONFIG_PATH, save_dir / "e1_mdg_config.json")
        shutil.copyfile(DESIGNATION_PATH, save_dir / "formal_e2_dualview_mainline_designation.json")
        append_event(
            events_path,
            "training_started",
            mode=mode,
            run_dir=str(save_dir),
            attempt=state.get("attempts", 1),
            resumed=metadata["resumed_from_own_last_pt"],
        )

    def on_train_batch_end(trainer: Any) -> None:
        nonlocal evidence_written
        if trainer.loss is not None and not bool(trainer.loss.detach().isfinite().all()):
            raise FloatingPointError(f"Non-finite E3 training loss at epoch {int(trainer.epoch) + 1}")
        if evidence_written:
            return
        evidence = trainer.runtime_evidence()
        try:
            validate_runtime_evidence(evidence)
        except RuntimeError:
            # The first few optimizer steps may not yet contain a small-object
            # positive anchor or a completed P3 update probe. Keep sampling.
            return
        evidence_written = True
        evidence.update(
            {
                "schema": "ring3.e3_small_object_weighting_runtime_evidence.v1",
                "verified_utc": utc_now(),
                "mode": mode,
                "run_dir": str(trainer.save_dir),
                "mdg_counts_first_verified_batch": dict(evidence.get("mdg_counts_current_batch", {})),
                "target_counts_first_verified_batch": dict(evidence.get("target_counts_current_batch", {})),
            }
        )
        evidence_path = Path(trainer.save_dir) / "e3_small_object_weighting_runtime_evidence.json"
        write_json_atomic(evidence_path, evidence)
        state["runtime_evidence"] = evidence
        write_json_atomic(state_path, state)
        append_event(
            events_path,
            "runtime_wiring_verified",
            mode=mode,
            weighted_positive_anchor_count=evidence["weighted_positive_anchor_count"],
            degraded_small_targets=evidence["degraded_small_targets"],
            weighting_delta_scaled=evidence["weighting_delta_scaled"],
            p3_parameter_delta_l2=evidence["optimizer_update_probe"]["parameter_delta_l2"],
        )

    def on_fit_epoch_end(trainer: Any) -> None:
        raw_epoch = int(trainer.epoch) + 1
        save_dir = Path(trainer.save_dir)
        row = base.read_last_row(save_dir / "results.csv")
        if row_has_nonfinite(row, ("train/",)):
            raise FloatingPointError(f"Non-finite E3 epoch training loss at epoch {raw_epoch}")
        validation_bad = row_has_nonfinite(row, ("val/",))
        streak = int(state.get("validation_nan_streak", 0)) + 1 if validation_bad else 0
        if streak >= 3:
            raise FloatingPointError(f"E3 validation loss remained non-finite for {streak} epochs")

        mdg_counts, target_counts = trainer.consume_epoch_counts()
        target_epochs = int(state["target_epochs"])
        if not mdg_counts and int(state.get("current_epoch", 0)) >= target_epochs:
            return
        if int(mdg_counts.get("images_total", -1)) != int(target_counts.get("images_total", -2)):
            raise RuntimeError(f"E3 MDG/image accounting mismatch: mdg={mdg_counts}, targets={target_counts}")
        if int(target_counts.get("clean_targets", -1)) != int(target_counts.get("degraded_targets", -2)):
            raise RuntimeError(f"E3 clean/degraded target accounting mismatch: {target_counts}")

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
        now = datetime.now(timezone.utc)
        evidence_path = save_dir / "e3_small_object_weighting_runtime_evidence.json"
        evidence = load_json(evidence_path) if evidence_path.is_file() else state.get("runtime_evidence", {})
        validate_runtime_evidence(evidence)
        best_row = read_best_metric_row(save_dir / "results.csv") if row else None
        mdg_total, target_total = trainer.total_counts()
        state.update(
            status="running",
            current_epoch=epoch,
            current_row=row,
            best_row=best_row,
            last_progress_utc=now.isoformat(),
            validation_nan_streak=streak,
            mdg_last_epoch_counts=mdg_counts,
            mdg_total_counts=mdg_total,
            target_last_epoch_counts=target_counts,
            target_total_counts=target_total,
            runtime_evidence=evidence,
            recent_epoch_seconds=recent,
            expected_seconds_per_epoch=expected,
            eta_utc=(now + timedelta(seconds=max(0, target_epochs - epoch) * expected)).isoformat(),
        )
        if cumulative is not None:
            state["last_cumulative_train_seconds"] = cumulative
        write_json_atomic(state_path, state)
        append_event(
            save_dir / "e3_small_object_weighting_epoch_metrics.jsonl",
            "epoch_completed",
            epoch=epoch,
            current_row=row,
            best_row=best_row,
            mdg_counts=mdg_counts,
            target_counts=target_counts,
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == target_epochs:
            append_event(
                events_path,
                "epoch_completed",
                epoch=epoch,
                current_row=row,
                best_row=best_row,
                expected_seconds_per_epoch=expected,
                eta_utc=state["eta_utc"],
                mdg_counts=mdg_counts,
                target_counts=target_counts,
            )
        print(
            "E3_STATE "
            + json.dumps(
                {
                    "epoch": epoch,
                    "target": target_epochs,
                    "seconds_per_epoch": round(expected, 3),
                    "eta_utc": state["eta_utc"],
                    "mdg_counts": mdg_counts,
                    "target_counts": target_counts,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def verify_smoke_evidence(context: dict[str, Any]) -> dict[str, Any]:
    if not SMOKE_STATE_PATH.is_file():
        raise RuntimeError("E3 formal run requires a completed E3 smoke test")
    smoke = load_json(SMOKE_STATE_PATH)
    if smoke.get("status") != "passed":
        raise RuntimeError(f"Latest E3 smoke test did not pass: {smoke.get('status')!r}")
    expected = {
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "e3_config_sha256": context["e3_config_sha256"],
        "release_manifest_sha256": context["release_manifest"]["sha256"],
    }
    if any(smoke.get(key) != value for key, value in expected.items()):
        raise RuntimeError("E3 smoke evidence is stale for the current locked release")
    validate_runtime_evidence(smoke.get("runtime_evidence", {}))
    run_dir = Path(str(smoke["run_dir"]))
    complete, _ = base.completed_run(run_dir, 1)
    if not complete:
        raise RuntimeError("E3 smoke artifacts are incomplete")
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    if (
        args.get("pretrained") is not True
        or args.get("model") != context["lock"]["weights"][MODEL_KEY]["path"]
        or args.get("resume") is not False
    ):
        raise RuntimeError("E3 smoke did not independently start from locked pretrained weights")
    return smoke


def run_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evidence = e3_model_self_test(
        CONFIG_PATH,
        Path(context["lock"]["weights"][MODEL_KEY]["path"]),
        device,
    )
    evidence.update(
        {
            "verified_utc": utc_now(),
            "formal_lock_sha256": context["lock_sha256"],
            "runtime_amendment_sha256": context["runtime_amendment_sha256"],
            "mdg_config_sha256": context["mdg_config_sha256"],
            "e3_config_sha256": context["e3_config_sha256"],
            "release_manifest_sha256": context["release_manifest"]["sha256"],
            "pretrained_source_sha256": sha256(Path(context["lock"]["weights"][MODEL_KEY]["path"])),
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
        }
    )
    write_json_atomic(SELF_TEST_PATH, evidence)
    return evidence


def run_once(mode: str, context: dict[str, Any]) -> int:
    lock = context["lock"]
    target_epochs = 1 if mode == "smoke" else int(lock["common_train"]["epochs"])
    if mode == "smoke":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"diagnostic_e3_dualview_smallobj_alpha15_img960_e1_{stamp}"
        output_root = DIAGNOSTICS_ROOT
        state_path, events_path = SMOKE_STATE_PATH, SMOKE_EVENTS_PATH
    else:
        name = FORMAL_NAME
        output_root = FORMAL_ROOT
        state_path, events_path = FORMAL_STATE_PATH, FORMAL_EVENTS_PATH
    run_dir = output_root / name

    existing_state = load_json(state_path) if state_path.is_file() else None
    if mode == "formal" and run_dir.exists():
        complete, final_row = base.completed_run(run_dir, target_epochs)
        if complete and final_row is not None:
            evidence_path = run_dir / "e3_small_object_weighting_runtime_evidence.json"
            if not evidence_path.is_file():
                raise RuntimeError("Completed E3 run is missing runtime evidence")
            evidence = load_json(evidence_path)
            validate_runtime_evidence(evidence)
            state = existing_state or initial_state(mode, run_dir, context, target_epochs)
            state.update(
                status="completed",
                current_epoch=target_epochs,
                final_row=final_row,
                best_row=read_best_metric_row(run_dir / "results.csv"),
                runtime_evidence=evidence,
                completed_utc=utc_now(),
                last_progress_utc=utc_now(),
            )
            write_json_atomic(state_path, state)
            return 0
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            archive_run(run_dir, "no_resumable_checkpoint", events_path)

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
        state.update(attempts=attempt, status="starting", last_progress_utc=utc_now())
        write_json_atomic(state_path, state)

    kwargs = base.training_kwargs(lock, MODEL_KEY, DATASET_KEY, name)
    kwargs.update(project=str(output_root), epochs=target_epochs)
    resumed = mode == "formal" and run_dir.exists()
    if resumed:
        verify_resume_metadata(run_dir, context)
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        train_kwargs: dict[str, Any] = {"resume": True}
    else:
        model = YOLO(lock["weights"][MODEL_KEY]["path"])
        train_kwargs = kwargs

    os.environ["RING3_E3_CONFIG"] = str(CONFIG_PATH)
    os.environ["RING3_E3_CONFIG_SHA256"] = context["e3_config_sha256"]
    os.environ["RING3_E3_MDG_CONFIG"] = str(MDG_CONFIG_PATH)
    os.environ["RING3_E3_MDG_CONFIG_SHA256"] = context["mdg_config_sha256"]
    metadata = training_metadata(mode, context, kwargs, attempt, resumed=resumed)
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
        model.train(trainer=E3SmallObjectWeightingTrainer, **train_kwargs)
        actual_dir = Path(state.get("run_dir", run_dir))
        complete, final_row = base.completed_run(actual_dir, target_epochs)
        if not complete or final_row is None:
            raise RuntimeError(f"E3 run ended without a finite epoch-{target_epochs} result: {actual_dir}")
        evidence_path = actual_dir / "e3_small_object_weighting_runtime_evidence.json"
        if not evidence_path.is_file():
            raise RuntimeError("E3 run ended without runtime weighting evidence")
        evidence = load_json(evidence_path)
        validate_runtime_evidence(evidence)
        args = yaml.safe_load((actual_dir / "args.yaml").read_text(encoding="utf-8"))
        if mode == "formal":
            base.verify_resume_args(actual_dir, lock, DATASET_KEY)
        else:
            smoke_expected = {
                "data": lock["dataset_protocols"][DATASET_KEY]["yaml"],
                "epochs": 1,
                "imgsz": lock["common_train"]["imgsz"],
                "batch": lock["common_train"]["batch"],
                "optimizer": lock["common_train"]["optimizer"],
                "lr0": lock["dataset_protocols"][DATASET_KEY]["lr0"],
                "seed": lock["common_train"]["seed"],
            }
            if any(args.get(key) != value for key, value in smoke_expected.items()):
                raise RuntimeError("E3 smoke args.yaml differs from the locked one-epoch diagnostic conditions")
        if args.get("pretrained") is not True:
            raise RuntimeError("E3 args.yaml does not preserve pretrained=True")
        state.update(
            status="passed" if mode == "smoke" else "completed",
            current_epoch=target_epochs,
            final_row=final_row,
            best_row=read_best_metric_row(actual_dir / "results.csv"),
            runtime_evidence=evidence,
            completed_utc=utc_now(),
            last_progress_utc=utc_now(),
        )
        write_json_atomic(state_path, state)
        append_event(
            events_path,
            "run_completed",
            mode=mode,
            attempt=attempt,
            final_row=final_row,
            best_row=state["best_row"],
        )
        return 0
    except BaseException as exc:
        numeric = isinstance(exc, FloatingPointError)
        state.update(
            status=("failed" if mode == "smoke" else ("numeric_attempt_failed" if numeric else "attempt_failed")),
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
            numeric=numeric,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if mode == "formal" and numeric and run_dir.exists():
            destination = archive_run(run_dir, "numeric_failure", events_path)
            state["numeric_failure_archive"] = str(destination)
            write_json_atomic(state_path, state)
        traceback.print_exc()
        return 2


def run_queue(context: dict[str, Any]) -> int:
    run_self_test(context)
    try:
        verify_smoke_evidence(context)
    except BaseException:
        smoke_result = run_once("smoke", context)
        if smoke_result != 0:
            return smoke_result
        context = preflight()
        verify_smoke_evidence(context)
    activate_protocol(context)
    return run_once("formal", context)


def record_preflight_failure(mode: str, exc: BaseException) -> None:
    if mode == "self-test":
        state_path, events_path = SELF_TEST_PATH, SMOKE_EVENTS_PATH
    elif mode == "smoke":
        state_path, events_path = SMOKE_STATE_PATH, SMOKE_EVENTS_PATH
    else:
        state_path, events_path = FORMAL_STATE_PATH, FORMAL_EVENTS_PATH
    previous = load_json(state_path) if state_path.is_file() else {}
    previous.update(
        status="preflight_failed",
        mode=mode,
        stage="E3",
        last_progress_utc=utc_now(),
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
    write_json_atomic(state_path, previous)
    append_event(events_path, "preflight_failed", mode=mode, error_type=type(exc).__name__, error=str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked E3 dual-view degraded-small-object weighting")
    parser.add_argument("--mode", choices=("self-test", "smoke", "formal", "queue"), default="queue")
    args = parser.parse_args()
    try:
        context = preflight()
        if args.mode == "self-test":
            print(json.dumps(run_self_test(context), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.mode == "smoke":
            run_self_test(context)
            return run_once("smoke", context)
        if args.mode == "formal":
            run_self_test(context)
            verify_smoke_evidence(context)
            activate_protocol(context)
            return run_once("formal", context)
        return run_queue(context)
    except BaseException as exc:
        record_preflight_failure(args.mode, exc)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
