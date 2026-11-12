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
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

import formal_baseline_queue as base
from maritime_degradation import MaritimeDegradation, load_config as load_mdg_config
from sdc_consistency import P3ConsistencyDetectionModel, load_sdc_config


EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_MDG_CONFIG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
MODEL_KEY = "yolov8s"
DATASET_KEY = "seadronessee"
STAGE = "E2"
FORMAL_NAME = "seed0_yolov8s_seadronessee_e2_mdg_p3cons_img960_e150"
MAX_FORMAL_ATTEMPTS = 3

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
SDC_CONFIG_PATH = SETUP_ROOT / "e2_p3_consistency_config.json"
MDG_CONFIG_PATH = SETUP_ROOT / "e1_mdg_config.json"
AMENDMENT_PATH = SETUP_ROOT / "formal_runtime_amendment.json"
FORMAL_ROOT = DATA_ROOT / "formal_e2"
DIAGNOSTICS_ROOT = DATA_ROOT / "diagnostics"
FAILED_ROOT = DATA_ROOT / "formal_failed_runs"
STATE_ROOT = DATA_ROOT / "formal_state"
FORMAL_STATE_PATH = STATE_ROOT / "e2_sdc_state.json"
FORMAL_EVENTS_PATH = STATE_ROOT / "e2_sdc_events.jsonl"
FORMAL_PROTOCOL_PATH = STATE_ROOT / "e2_sdc_protocol.json"
SMOKE_STATE_PATH = DIAGNOSTICS_ROOT / "e2_sdc_smoke_state.json"
SMOKE_EVENTS_PATH = DIAGNOSTICS_ROOT / "e2_sdc_smoke_events.jsonl"
BASELINE_STATE_PATH = STATE_ROOT / "baseline_queue_state.json"
BASELINE_RUN = DATA_ROOT / "formal_baselines" / "seed0_yolov8s_seadronessee_img960_e150"
E1_STATE_PATH = STATE_ROOT / "e1_mdg_state.json"
E1_PROTOCOL_PATH = STATE_ROOT / "e1_mdg_protocol.json"
E1_RUN = DATA_ROOT / "formal_e1" / "seed0_yolov8s_seadronessee_e1_mdg_img960_e150"


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


def verify_target_dataset(lock: dict[str, Any]) -> dict[str, Any]:
    protocol = lock["dataset_protocols"][DATASET_KEY]
    yaml_path = Path(protocol["yaml"])
    if not yaml_path.is_file() or sha256(yaml_path) != protocol["yaml_sha256"]:
        raise RuntimeError(f"Locked SeaDronesSee YAML is missing or changed: {yaml_path}")
    manifest_path = Path(protocol["manifest_path"])
    if not manifest_path.is_file() or sha256(manifest_path) != protocol["manifest_sha256"]:
        raise RuntimeError(f"Locked SeaDronesSee manifest is missing or changed: {manifest_path}")
    result = subprocess.run(
        [sys.executable, str(SETUP_ROOT / "verify_yolo_dataset.py"), str(yaml_path)],
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


def verify_baseline_complete(lock: dict[str, Any]) -> dict[str, Any]:
    if not BASELINE_STATE_PATH.is_file():
        raise RuntimeError(f"Baseline state is missing: {BASELINE_STATE_PATH}")
    queue_state = load_json(BASELINE_STATE_PATH)
    if queue_state.get("status") != "completed" or queue_state.get("lock_sha256") != EXPECTED_LOCK_SHA256:
        raise RuntimeError(f"Baseline segment is not locked-complete: {queue_state.get('status')!r}")
    complete, final_row = base.completed_run(BASELINE_RUN, int(lock["common_train"]["epochs"]))
    if not complete or final_row is None:
        raise RuntimeError(f"E0 baseline is incomplete or non-finite: {BASELINE_RUN}")
    base.verify_resume_args(BASELINE_RUN, lock, DATASET_KEY)
    return {
        "run_dir": str(BASELINE_RUN),
        "args_sha256": sha256(BASELINE_RUN / "args.yaml"),
        "final_row": final_row,
        "best_row": read_best_metric_row(BASELINE_RUN / "results.csv"),
        "measured_mean_seconds_per_epoch": float(final_row["time"]) / float(final_row["epoch"]),
    }


def verify_e1_complete(lock: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    if not E1_STATE_PATH.is_file() or not E1_PROTOCOL_PATH.is_file():
        raise RuntimeError("E1 state or protocol is missing")
    state = load_json(E1_STATE_PATH)
    protocol = load_json(E1_PROTOCOL_PATH)
    if state.get("status") != "completed" or int(state.get("current_epoch", 0)) != int(lock["common_train"]["epochs"]):
        raise RuntimeError(f"E1 is not complete: {state.get('status')!r}, epoch={state.get('current_epoch')!r}")
    expected_hashes = {
        "formal_lock_sha256": EXPECTED_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_CONFIG_SHA256,
    }
    for key, expected in expected_hashes.items():
        if state.get(key) != expected or protocol.get(key) != expected:
            raise RuntimeError(f"E1 evidence mismatch for {key}")
    for name, expected in protocol.get("code_sha256", {}).items():
        path = SETUP_ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"E1 recorded code changed after completion: {name}")
    complete, final_row = base.completed_run(E1_RUN, int(lock["common_train"]["epochs"]))
    if not complete or final_row is None:
        raise RuntimeError(f"E1 run is incomplete or non-finite: {E1_RUN}")
    base.verify_resume_args(E1_RUN, lock, DATASET_KEY)
    best_row = read_best_metric_row(E1_RUN / "results.csv")
    e0_best = float(baseline["best_row"]["metrics/mAP50-95(B)"])
    e1_best = float(best_row["metrics/mAP50-95(B)"])
    return {
        "run_dir": str(E1_RUN),
        "state_sha256": sha256(E1_STATE_PATH),
        "protocol_sha256": sha256(E1_PROTOCOL_PATH),
        "args_sha256": sha256(E1_RUN / "args.yaml"),
        "final_row": final_row,
        "best_row": best_row,
        "best_map50_95_delta_vs_e0": e1_best - e0_best,
        "original_validation_gate": "did_not_improve_e0" if e1_best <= e0_best else "improved_e0",
        "measured_mean_seconds_per_epoch": float(final_row["time"]) / float(final_row["epoch"]),
    }


def code_hashes() -> dict[str, str]:
    paths = {
        "train_e2_sdc.py": Path(__file__).resolve(),
        "sdc_consistency.py": SETUP_ROOT / "sdc_consistency.py",
        "maritime_degradation.py": SETUP_ROOT / "maritime_degradation.py",
        "e2_p3_consistency_config.json": SDC_CONFIG_PATH,
        "e1_mdg_config.json": MDG_CONFIG_PATH,
        "run_e2_sdc.sh": SETUP_ROOT / "run_e2_sdc.sh",
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
    e1 = verify_e1_complete(lock, baseline)

    if not MDG_CONFIG_PATH.is_file() or sha256(MDG_CONFIG_PATH) != EXPECTED_MDG_CONFIG_SHA256:
        raise RuntimeError("E1 MDG config is missing or changed")
    mdg_config = load_mdg_config(MDG_CONFIG_PATH)
    sdc_config = load_sdc_config(SDC_CONFIG_PATH)
    if sdc_config["mdg"]["config_sha256"] != EXPECTED_MDG_CONFIG_SHA256:
        raise RuntimeError("E2 does not point to the locked E1 MDG config")
    if e1["original_validation_gate"] != "did_not_improve_e0":
        raise RuntimeError("Recorded E2 authorization no longer matches the observed E1 gate status")

    return {
        "lock": lock,
        "lock_sha256": lock_hash,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config": mdg_config,
        "mdg_config_sha256": EXPECTED_MDG_CONFIG_SHA256,
        "sdc_config": sdc_config,
        "sdc_config_sha256": sha256(SDC_CONFIG_PATH),
        "code_sha256": code_hashes(),
        "dataset_report": dataset_report,
        "baseline": baseline,
        "e1": e1,
    }


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    lock = context["lock"]
    return {
        "schema": "ring3.e2_sdc_protocol.v1",
        "stage": STAGE,
        "model": MODEL_KEY,
        "dataset": DATASET_KEY,
        "formal_name": FORMAL_NAME,
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "sdc_config_sha256": context["sdc_config_sha256"],
        "code_sha256": context["code_sha256"],
        "weight_sha256": lock["weights"][MODEL_KEY]["sha256"],
        "dataset_yaml_sha256": lock["dataset_protocols"][DATASET_KEY]["yaml_sha256"],
        "dataset_manifest_sha256": lock["dataset_protocols"][DATASET_KEY]["manifest_sha256"],
        "e0_args_sha256": context["baseline"]["args_sha256"],
        "e1_protocol_sha256": context["e1"]["protocol_sha256"],
        "e1_original_validation_gate": context["e1"]["original_validation_gate"],
        "activation": context["sdc_config"]["activation"],
        "dual_view": context["sdc_config"]["dual_view"],
        "consistency": context["sdc_config"]["consistency"],
        "invariants": context["sdc_config"]["invariants"],
    }


def validate_runtime_evidence(evidence: dict[str, Any]) -> None:
    detect_inputs = evidence.get("detect_input_indices")
    target_index = evidence.get("target_feature_layer_index")
    feature_shape = evidence.get("feature_shape")
    pair_batch_size = int(evidence.get("pair_batch_size", 0))
    raw_loss = float(evidence.get("consistency_loss_raw", float("nan")))
    gradient = evidence.get("consistency_only_gradient_probe", {})
    update = evidence.get("optimizer_update_probe", {})
    if not isinstance(detect_inputs, list) or len(detect_inputs) < 3 or target_index != detect_inputs[0]:
        raise RuntimeError(f"Invalid Detect/P3 runtime evidence: {evidence}")
    if not isinstance(feature_shape, list) or len(feature_shape) != 4 or feature_shape[0] != pair_batch_size * 2:
        raise RuntimeError(f"Invalid paired P3 feature shape evidence: {evidence}")
    if pair_batch_size <= 0 or not math.isfinite(raw_loss) or raw_loss <= 0.0:
        raise RuntimeError(f"Consistency loss did not become finite and nonzero: {evidence}")
    if gradient.get("finite") is not True or int(gradient.get("nonzero_gradient_tensors", 0)) <= 0:
        raise RuntimeError(f"Consistency-only gradient evidence failed: {gradient}")
    if not math.isfinite(float(gradient.get("l2_norm", float("nan")))) or float(gradient["l2_norm"]) <= 0.0:
        raise RuntimeError(f"Consistency-only gradient norm failed: {gradient}")
    if update.get("finite") is not True or float(update.get("parameter_delta_l2", 0.0)) <= 0.0:
        raise RuntimeError(f"P3 optimizer update evidence failed: {update}")


def verify_smoke_evidence(context: dict[str, Any]) -> dict[str, Any]:
    if not SMOKE_STATE_PATH.is_file():
        raise RuntimeError("E2 formal run requires a completed P3-consistency smoke test")
    smoke = load_json(SMOKE_STATE_PATH)
    expected = protocol_payload(context)
    if smoke.get("status") != "passed":
        raise RuntimeError(f"Latest E2 smoke test did not pass: {smoke.get('status')!r}")
    for key in (
        "formal_lock_sha256",
        "runtime_amendment_sha256",
        "mdg_config_sha256",
        "sdc_config_sha256",
        "code_sha256",
    ):
        if smoke.get(key) != expected[key]:
            raise RuntimeError(f"E2 smoke evidence is stale for {key}")
    validate_runtime_evidence(smoke.get("runtime_evidence", {}))
    return smoke


def activate_protocol(context: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(context)
    if FORMAL_PROTOCOL_PATH.exists():
        recorded = load_json(FORMAL_PROTOCOL_PATH)
        if recorded != payload:
            raise RuntimeError("E2 protocol changed after formal activation")
    else:
        write_json_atomic(FORMAL_PROTOCOL_PATH, payload)
    return payload


class P3ConsistencyTrainer(DetectionTrainer):
    """Ultralytics trainer for paired clean/degraded E2 training with P3 consistency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        sdc_path = Path(os.environ["RING3_E2_SDC_CONFIG"])
        mdg_path = Path(os.environ["RING3_E2_MDG_CONFIG"])
        if sha256(sdc_path) != os.environ["RING3_E2_SDC_CONFIG_SHA256"]:
            raise RuntimeError("E2 SDC config hash changed before trainer construction")
        if sha256(mdg_path) != os.environ["RING3_E2_MDG_CONFIG_SHA256"]:
            raise RuntimeError("E2 MDG config hash changed before trainer construction")
        self.sdc_config = load_sdc_config(sdc_path)
        self.mdg = MaritimeDegradation(load_mdg_config(mdg_path))
        self._mdg_epoch_counts: Counter[str] = Counter()
        self._mdg_total_counts: Counter[str] = Counter()
        self._sdc_epoch_batches = 0
        self._sdc_epoch_pairs = 0
        self._sdc_epoch_weighted: Counter[str] = Counter()
        self._optimizer_update_probe: dict[str, Any] = {}
        self._optimizer_step_calls = 0
        self._last_mdg_counts: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        model = self.set_model_names_for_load(
            P3ConsistencyDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                consistency_weight=float(self.sdc_config["consistency"]["lambda"]),
            )
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        clean = batch["img"]
        pair_batch_size = int(clean.shape[0])
        degraded, counts = self.mdg.apply_batch(clean.clone())
        batch["img"] = torch.cat((clean, degraded), dim=0)
        batch["batch_idx"] = torch.cat((batch["batch_idx"], batch["batch_idx"] + pair_batch_size), dim=0)
        batch["cls"] = torch.cat((batch["cls"], batch["cls"].clone()), dim=0)
        batch["bboxes"] = torch.cat((batch["bboxes"], batch["bboxes"].clone()), dim=0)
        for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad"):
            value = batch.get(key)
            if isinstance(value, list):
                batch[key] = value + list(value)
            elif isinstance(value, tuple) and len(value) == pair_batch_size:
                batch[key] = value + value
        batch["sdc_pair_batch_size"] = pair_batch_size
        batch["sdc_mdg_counts"] = counts
        self._last_mdg_counts = dict(counts)
        self._mdg_epoch_counts.update(counts)
        self._mdg_total_counts.update(counts)
        return batch

    def optimizer_step(self) -> None:
        self._optimizer_step_calls += 1
        model = unwrap_model(self.model)
        parameter_before: torch.Tensor | None = None
        target_parameter: torch.nn.Parameter | None = None
        pre_step_gradient: dict[str, Any] = {}
        if not self._optimizer_update_probe and isinstance(model, P3ConsistencyDetectionModel):
            parameters = model.sdc_target_parameters()
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
            finite = bool(torch.isfinite(delta).all())
            delta_l2 = float(delta.square().sum().sqrt())
            candidate = {
                "optimizer_step_index": self._optimizer_step_calls,
                "target_feature_layer_index": model.sdc_feature_layer_index,
                "target_parameter_shape": [int(value) for value in target_parameter.shape],
                "pre_step_gradient": pre_step_gradient,
                "finite": finite,
                "parameter_delta_l2": delta_l2,
                "parameter_delta_max_abs": float(delta.abs().max()),
            }
            if not finite:
                raise FloatingPointError(f"P3 optimizer update became non-finite: {candidate}")
            if math.isfinite(delta_l2) and delta_l2 > 0.0:
                self._optimizer_update_probe = candidate

    def record_sdc_batch(self, metrics: dict[str, Any]) -> None:
        pairs = int(metrics.get("pair_batch_size", 0))
        if pairs <= 0:
            raise RuntimeError(f"Missing paired-batch metrics: {metrics}")
        self._sdc_epoch_batches += 1
        self._sdc_epoch_pairs += pairs
        for key in (
            "consistency_loss_raw",
            "consistency_loss_weighted_per_pair",
            "cosine_mean",
            "feature_mean_abs_delta",
        ):
            value = float(metrics[key])
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite E2 metric {key}: {value}")
            self._sdc_epoch_weighted[key] += value * pairs

    def consume_epoch_metrics(self) -> tuple[dict[str, int], dict[str, Any]]:
        mdg_counts = dict(self._mdg_epoch_counts)
        self._mdg_epoch_counts.clear()
        pairs = self._sdc_epoch_pairs
        sdc = {
            "batches": self._sdc_epoch_batches,
            "pairs": pairs,
            **{
                key: (float(value) / pairs if pairs else float("nan"))
                for key, value in self._sdc_epoch_weighted.items()
            },
        }
        self._sdc_epoch_batches = 0
        self._sdc_epoch_pairs = 0
        self._sdc_epoch_weighted.clear()
        return mdg_counts, sdc

    def mdg_total_counts(self) -> dict[str, int]:
        return dict(self._mdg_total_counts)

    def runtime_evidence(self) -> dict[str, Any]:
        model = unwrap_model(self.model)
        if not isinstance(model, P3ConsistencyDetectionModel):
            raise TypeError(f"Unexpected E2 model type: {type(model).__name__}")
        return {
            **model.sdc_last_metrics(),
            "consistency_only_gradient_probe": model.sdc_gradient_probe(),
            "optimizer_update_probe": dict(self._optimizer_update_probe),
            "mdg_counts_current_batch": dict(self._last_mdg_counts),
        }


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
    expected_seconds = context["e1"]["measured_mean_seconds_per_epoch"] * 1.75
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.e2_sdc_state.v1",
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
        "sdc_config_sha256": context["sdc_config_sha256"],
        "code_sha256": context["code_sha256"],
        "baseline": context["baseline"],
        "e1": context["e1"],
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
            "schema": "ring3.e2_sdc_run_metadata.v1",
            "mode": mode,
            "attempt": attempt,
            "created_utc": utc_now(),
            "training_kwargs": kwargs,
            "diagnostic_excluded_from_paper_results": mode == "smoke",
            "e0_baseline": context["baseline"],
            "e1_component_baseline": context["e1"],
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
        write_json_atomic(save_dir / "e2_sdc_metadata.json", metadata)
        shutil.copyfile(SDC_CONFIG_PATH, save_dir / "e2_p3_consistency_config.json")
        shutil.copyfile(MDG_CONFIG_PATH, save_dir / "e1_mdg_config.json")
        append_event(events_path, "training_started", mode=mode, run_dir=str(save_dir), attempt=state.get("attempts", 1))

    def on_train_batch_end(trainer: Any) -> None:
        nonlocal evidence_written
        if trainer.loss is not None and not bool(trainer.loss.detach().isfinite().all()):
            raise FloatingPointError(f"Non-finite E2 training loss at epoch {int(trainer.epoch) + 1}")
        evidence = trainer.runtime_evidence()
        trainer.record_sdc_batch(evidence)
        if not evidence_written:
            try:
                validate_runtime_evidence(evidence)
            except RuntimeError:
                return
            evidence_written = True
            evidence.update(
                {
                    "schema": "ring3.e2_sdc_runtime_evidence.v1",
                    "verified_utc": utc_now(),
                    "mode": mode,
                    "run_dir": str(trainer.save_dir),
                    "mdg_counts_first_verified_batch": dict(evidence.get("mdg_counts_current_batch", {})),
                }
            )
            write_json_atomic(Path(trainer.save_dir) / "e2_sdc_runtime_evidence.json", evidence)
            state["runtime_evidence"] = evidence
            write_json_atomic(state_path, state)
            append_event(
                events_path,
                "runtime_wiring_verified",
                mode=mode,
                target_feature_layer_index=evidence["target_feature_layer_index"],
                feature_shape=evidence["feature_shape"],
                consistency_loss_raw=evidence["consistency_loss_raw"],
                consistency_gradient_l2=evidence["consistency_only_gradient_probe"]["l2_norm"],
                p3_parameter_delta_l2=evidence["optimizer_update_probe"]["parameter_delta_l2"],
            )

    def on_fit_epoch_end(trainer: Any) -> None:
        raw_epoch = int(trainer.epoch) + 1
        save_dir = Path(trainer.save_dir)
        row = base.read_last_row(save_dir / "results.csv")
        if row_has_nonfinite(row, ("train/",)):
            raise FloatingPointError(f"Non-finite E2 epoch training loss at epoch {raw_epoch}")
        val_bad = row_has_nonfinite(row, ("val/",))
        streak = int(state.get("validation_nan_streak", 0)) + 1 if val_bad else 0
        if streak >= 3:
            raise FloatingPointError(f"E2 validation loss remained non-finite for {streak} epochs")

        mdg_counts, sdc_metrics = trainer.consume_epoch_metrics()
        target_epochs = int(state["target_epochs"])
        if not mdg_counts and int(state.get("current_epoch", 0)) >= target_epochs:
            return
        if int(mdg_counts.get("images_total", 0)) != int(sdc_metrics.get("pairs", -1)):
            raise RuntimeError(f"E2 MDG/paired-view accounting mismatch: mdg={mdg_counts}, sdc={sdc_metrics}")
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
        runtime_evidence_path = save_dir / "e2_sdc_runtime_evidence.json"
        runtime_evidence = load_json(runtime_evidence_path) if runtime_evidence_path.is_file() else state.get("runtime_evidence", {})
        validate_runtime_evidence(runtime_evidence)
        state.update(
            status="running",
            current_epoch=epoch,
            last_progress_utc=now.isoformat(),
            validation_nan_streak=streak,
            mdg_last_epoch_counts=mdg_counts,
            mdg_total_counts=trainer.mdg_total_counts(),
            sdc_last_epoch_metrics=sdc_metrics,
            runtime_evidence=runtime_evidence,
            recent_epoch_seconds=recent,
            expected_seconds_per_epoch=expected,
            eta_utc=(now + timedelta(seconds=remaining * expected)).isoformat(),
        )
        if cumulative is not None:
            state["last_cumulative_train_seconds"] = cumulative
        write_json_atomic(state_path, state)
        append_event(
            save_dir / "e2_sdc_epoch_metrics.jsonl",
            "epoch_completed",
            epoch=epoch,
            mdg_counts=mdg_counts,
            sdc_metrics=sdc_metrics,
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == target_epochs:
            append_event(
                events_path,
                "epoch_completed",
                epoch=epoch,
                expected_seconds_per_epoch=expected,
                eta_utc=state["eta_utc"],
                mdg_counts=mdg_counts,
                sdc_metrics=sdc_metrics,
            )
        print(
            "E2_STATE "
            + json.dumps(
                {
                    "epoch": epoch,
                    "target": target_epochs,
                    "seconds_per_epoch": round(expected, 3),
                    "eta_utc": state["eta_utc"],
                    "mdg_counts": mdg_counts,
                    "sdc_metrics": sdc_metrics,
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
        name = f"diagnostic_e2_sdc_p3_yolov8s_seadronessee_img960_e1_{stamp}"
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

    os.environ["RING3_E2_SDC_CONFIG"] = str(SDC_CONFIG_PATH)
    os.environ["RING3_E2_SDC_CONFIG_SHA256"] = context["sdc_config_sha256"]
    os.environ["RING3_E2_MDG_CONFIG"] = str(MDG_CONFIG_PATH)
    os.environ["RING3_E2_MDG_CONFIG_SHA256"] = context["mdg_config_sha256"]
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
        model.train(trainer=P3ConsistencyTrainer, **train_kwargs)
        actual_dir = Path(state.get("run_dir", run_dir))
        complete, final_row = base.completed_run(actual_dir, target_epochs)
        if not complete or final_row is None:
            raise RuntimeError(f"E2 run ended without a finite epoch-{target_epochs} result: {actual_dir}")
        evidence_path = actual_dir / "e2_sdc_runtime_evidence.json"
        if not evidence_path.is_file():
            raise RuntimeError("E2 run ended without runtime wiring evidence")
        runtime_evidence = load_json(evidence_path)
        validate_runtime_evidence(runtime_evidence)
        state.update(
            status="passed" if mode == "smoke" else "completed",
            current_epoch=target_epochs,
            final_row=final_row,
            runtime_evidence=runtime_evidence,
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
    parser = argparse.ArgumentParser(description="Run locked SeaDronesSee E2 (YOLOv8s + MDG + P3 consistency)")
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
