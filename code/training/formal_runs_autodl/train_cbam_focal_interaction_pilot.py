from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg

import train_e2_sdc as e2
from cbam_focal_sdc import (
    CBAMFocalP3DetectionModel,
    EXPECTED_CBAM_LAYERS,
    EXPECTED_DETECT_INPUTS,
    FocalEIoULoss,
    FocalEIoUBboxLoss,
    load_pilot_config,
    v8DetectionLossWithFocalEIoU,
)


EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_MDG_CONFIG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_E2_TRAIN_SHA256 = "2c5d415a614b5f9bf53d5deb0934548390ef14e8409808167857d11b844dc721"
EXPECTED_E2_SDC_SHA256 = "87979f6b2f7c6d66103d2883c843b8b796866e1db638119337b1b1cf2b62fd38"
EXPECTED_MDG_CODE_SHA256 = "b0d60602f77db9bbb935f5e48f44a5766f361ef35bc07e90fd06b23c3f8b85cc"

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
PILOT_CONFIG_PATH = SETUP_ROOT / "cbam_focal_interaction_pilot_config.json"
RELEASE_MANIFEST_PATH = SETUP_ROOT / "cbam_focal_interaction_release_manifest.json"
PILOT_ROOT = DATA_ROOT / "pilots" / "cbam_focal_interaction"
FAILED_ROOT = DATA_ROOT / "pilot_failed_runs" / "cbam_focal_interaction"
STATE_ROOT = DATA_ROOT / "formal_state"
PILOT_STATE_PATH = STATE_ROOT / "cbam_focal_interaction_pilot_state.json"
PILOT_EVENTS_PATH = STATE_ROOT / "cbam_focal_interaction_pilot_events.jsonl"
PILOT_PROTOCOL_PATH = STATE_ROOT / "cbam_focal_interaction_pilot_protocol.json"
SMOKE_STATE_PATH = DATA_ROOT / "diagnostics" / "cbam_focal_interaction_smoke_state.json"
SMOKE_EVENTS_PATH = DATA_ROOT / "diagnostics" / "cbam_focal_interaction_smoke_events.jsonl"
SELF_TEST_PATH = DATA_ROOT / "diagnostics" / "cbam_focal_interaction_self_test.json"
STOP_SENTINEL = STATE_ROOT / "STOP_CBAM_FOCAL_INTERACTION_PILOT"
SCRATCH_QUEUE_STATE = STATE_ROOT / "scratch_independent_queue_state.json"
PILOT_NAME = "seed0_yolov8s_seadronessee_cbam_focal_dualview_lambda0_img960_e30_pilot"

_ORIGINAL_PREFLIGHT = e2.preflight
_ORIGINAL_INITIAL_STATE = e2.initial_state
_ORIGINAL_TRAINING_METADATA = e2.training_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_manifest() -> dict[str, Any]:
    if not RELEASE_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Pilot release manifest is missing: {RELEASE_MANIFEST_PATH}")
    manifest = e2.load_json(RELEASE_MANIFEST_PATH)
    if manifest.get("schema") != "ring3.cbam_focal_interaction_release_manifest.v1":
        raise RuntimeError("Unexpected pilot release manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Pilot release manifest has no file entries")
    verified: list[dict[str, Any]] = []
    for entry in entries:
        relative = str(entry["path"])
        path = SETUP_ROOT / relative
        expected = str(entry["sha256"])
        if not path.is_file():
            raise RuntimeError(f"Pilot release file is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Pilot release hash mismatch for {relative}: expected={expected}, observed={observed}"
            )
        verified.append({"path": relative, "sha256": observed})
    return {
        "path": str(RELEASE_MANIFEST_PATH),
        "sha256": sha256(RELEASE_MANIFEST_PATH),
        "verified_files": verified,
    }


def code_hashes() -> dict[str, str]:
    paths = {
        "train_cbam_focal_interaction_pilot.py": Path(__file__).resolve(),
        "cbam_focal_sdc.py": SETUP_ROOT / "cbam_focal_sdc.py",
        "cbam_focal_interaction_pilot_config.json": PILOT_CONFIG_PATH,
        "cbam_focal_interaction_release_manifest.json": RELEASE_MANIFEST_PATH,
        "run_cbam_focal_interaction_pilot.sh": SETUP_ROOT / "run_cbam_focal_interaction_pilot.sh",
        "train_e2_sdc.py": SETUP_ROOT / "train_e2_sdc.py",
        "sdc_consistency.py": SETUP_ROOT / "sdc_consistency.py",
        "maritime_degradation.py": SETUP_ROOT / "maritime_degradation.py",
        "e1_mdg_config.json": e2.MDG_CONFIG_PATH,
    }
    return {name: sha256(path) for name, path in paths.items()}


def configure() -> None:
    e2.STAGE = "CBAM-Focal-EIoU-interaction-pilot"
    e2.FORMAL_NAME = PILOT_NAME
    e2.SDC_CONFIG_PATH = PILOT_CONFIG_PATH
    e2.FORMAL_ROOT = PILOT_ROOT
    e2.FAILED_ROOT = FAILED_ROOT
    e2.FORMAL_STATE_PATH = PILOT_STATE_PATH
    e2.FORMAL_EVENTS_PATH = PILOT_EVENTS_PATH
    e2.FORMAL_PROTOCOL_PATH = PILOT_PROTOCOL_PATH
    e2.SMOKE_STATE_PATH = SMOKE_STATE_PATH
    e2.SMOKE_EVENTS_PATH = SMOKE_EVENTS_PATH
    e2.P3ConsistencyDetectionModel = CBAMFocalP3DetectionModel
    e2.load_sdc_config = load_pilot_config
    e2.code_hashes = code_hashes
    e2.protocol_payload = protocol_payload
    e2.validate_runtime_evidence = validate_runtime_evidence
    e2.initial_state = initial_state
    e2.training_metadata = training_metadata


def verify_anchor_files() -> None:
    expected = {
        SETUP_ROOT / "train_e2_sdc.py": EXPECTED_E2_TRAIN_SHA256,
        SETUP_ROOT / "sdc_consistency.py": EXPECTED_E2_SDC_SHA256,
        SETUP_ROOT / "maritime_degradation.py": EXPECTED_MDG_CODE_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"Completed E2 anchor code changed: {path.name}")


def verify_scratch_queue_terminal() -> dict[str, Any]:
    if not SCRATCH_QUEUE_STATE.is_file():
        raise RuntimeError("Completed scratch queue state is missing")
    state = e2.load_json(SCRATCH_QUEUE_STATE)
    if state.get("status") not in {"completed", "completed_with_failures"}:
        raise RuntimeError(f"Scratch queue is not terminal: {state.get('status')!r}")
    if state.get("current_run") is not None:
        raise RuntimeError("Scratch queue still records an active run")
    return {
        "path": str(SCRATCH_QUEUE_STATE),
        "sha256": sha256(SCRATCH_QUEUE_STATE),
        "status": state.get("status"),
        "completed_runs": state.get("completed_runs", []),
        "failed_runs": state.get("failed_runs", []),
    }


def preflight_current() -> dict[str, Any]:
    if STOP_SENTINEL.exists():
        raise RuntimeError(f"Explicit pilot stop sentinel exists: {STOP_SENTINEL}")
    release = verify_release_manifest()
    verify_anchor_files()
    context = _ORIGINAL_PREFLIGHT()
    if context["lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Unexpected formal lock in interaction-pilot preflight")
    if context["runtime_amendment_sha256"] != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Unexpected runtime amendment in interaction-pilot preflight")
    if context["mdg_config_sha256"] != EXPECTED_MDG_CONFIG_SHA256:
        raise RuntimeError("Unexpected MDG config in interaction-pilot preflight")

    config = context["sdc_config"]
    pilot_epochs = int(config["pilot_epochs"])
    if str(config["initialization"]["source_sha256"]) != context["lock"]["weights"][e2.MODEL_KEY]["sha256"]:
        raise RuntimeError("Pilot initialization no longer points to the locked YOLOv8s pretrained weight")
    original_common_train = copy.deepcopy(context["lock"]["common_train"])
    if int(original_common_train["epochs"]) != 150:
        raise RuntimeError("The active formal lock no longer records 150 epochs")
    pilot_lock = copy.deepcopy(context["lock"])
    pilot_lock["common_train"]["epochs"] = pilot_epochs
    context["lock"] = pilot_lock
    context["formal_common_train"] = original_common_train
    context["pilot_epochs"] = pilot_epochs
    context["release_manifest"] = release
    context["scratch_queue_anchor"] = verify_scratch_queue_terminal()
    return context


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    lock = context["lock"]
    config = context["sdc_config"]
    return {
        "schema": "ring3.cbam_focal_interaction_pilot_protocol.v1",
        "stage": "interaction-pilot",
        "paper_result_eligible": False,
        "pilot_name": PILOT_NAME,
        "pilot_epochs": int(context["pilot_epochs"]),
        "formal_lock_epochs_unchanged_on_disk": int(context["formal_common_train"]["epochs"]),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        # Keep the base E2 smoke verifier's field name as an explicit alias.
        # The value is the pilot config hash; no formal E2 config is changed.
        "sdc_config_sha256": context["sdc_config_sha256"],
        "pilot_config_sha256": context["sdc_config_sha256"],
        "code_sha256": context["code_sha256"],
        "release_manifest": context["release_manifest"],
        "scratch_queue_anchor": context["scratch_queue_anchor"],
        "weight_sha256": lock["weights"][e2.MODEL_KEY]["sha256"],
        "dataset_yaml_sha256": lock["dataset_protocols"][e2.DATASET_KEY]["yaml_sha256"],
        "dataset_manifest_sha256": lock["dataset_protocols"][e2.DATASET_KEY]["manifest_sha256"],
        "e0_anchor": context["baseline"],
        "e1_anchor": context["e1"],
        "architecture": config["architecture"],
        "bbox_regression": config["bbox_regression"],
        "dual_view": config["dual_view"],
        "consistency": config["consistency"],
        "diagnostic": config["diagnostic"],
        "invariants": config["invariants"],
        "activation": config["activation"],
    }


def initial_state(mode: str, run_dir: Path, context: dict[str, Any], target_epochs: int) -> dict[str, Any]:
    state = _ORIGINAL_INITIAL_STATE(mode, run_dir, context, target_epochs)
    state.update(
        schema="ring3.cbam_focal_interaction_pilot_state.v1",
        stage="interaction-pilot",
        paper_result_eligible=False,
        pilot_epochs=int(context["pilot_epochs"]),
        formal_lock_epochs_unchanged_on_disk=int(context["formal_common_train"]["epochs"]),
        independent_initialization_from=context["lock"]["weights"][e2.MODEL_KEY]["path"],
        object_aware_consistency_started=False,
        e3_formal_started=False,
        e4_formal_started=False,
        release_manifest=context["release_manifest"],
        scratch_queue_anchor=context["scratch_queue_anchor"],
    )
    return state


def training_metadata(mode: str, context: dict[str, Any], kwargs: dict[str, Any], attempt: int) -> dict[str, Any]:
    metadata = _ORIGINAL_TRAINING_METADATA(mode, context, kwargs, attempt)
    metadata.update(
        schema="ring3.cbam_focal_interaction_pilot_run_metadata.v1",
        paper_result_eligible=False,
        diagnostic_or_pilot_excluded_from_paper_results=True,
        pilot_epochs=int(context["pilot_epochs"]),
        formal_lock_epochs_unchanged_on_disk=int(context["formal_common_train"]["epochs"]),
        architecture=context["sdc_config"]["architecture"],
        bbox_regression=context["sdc_config"]["bbox_regression"],
        object_aware_consistency_started=False,
    )
    return metadata


def validate_runtime_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("detect_input_indices") != EXPECTED_DETECT_INPUTS:
        raise RuntimeError(f"Detect inputs changed: {evidence}")
    if evidence.get("target_feature_layer_index") != 15 or evidence.get("target_feature_layer_type") != "C2f":
        raise RuntimeError(f"Head P3 target changed: {evidence}")
    if evidence.get("cbam_layer_indices") != EXPECTED_CBAM_LAYERS:
        raise RuntimeError(f"CBAM layer indices changed: {evidence}")
    if evidence.get("cbam_layer_types") != ["C2f_CBAM", "C2f_CBAM"]:
        raise RuntimeError(f"CBAM modules are not active: {evidence}")
    if int(evidence.get("cbam_attention_parameter_count", 0)) <= 0:
        raise RuntimeError(f"CBAM attention parameters are missing: {evidence}")
    if evidence.get("criterion_type") != "v8DetectionLossWithFocalEIoU":
        raise RuntimeError(f"Focal-EIoU criterion is not active: {evidence}")
    if evidence.get("bbox_loss_type") != "FocalEIoUBboxLoss":
        raise RuntimeError(f"Focal-EIoU bbox loss is not active: {evidence}")
    focal = evidence.get("focal_eiou", {})
    if focal.get("dfl_preserved") is not True:
        raise RuntimeError(f"DFL preservation evidence is missing: {evidence}")
    for key, expected in (("alpha", 0.25), ("gamma", 2.0), ("eps", 1e-7)):
        if not math.isclose(float(focal.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Focal-EIoU {key} changed: {evidence}")

    feature_shape = evidence.get("feature_shape")
    pair_batch_size = int(evidence.get("pair_batch_size", 0))
    concatenated = int(evidence.get("concatenated_batch_size", 0))
    detection_batch = int(evidence.get("detection_batch_size", 0))
    if (
        not isinstance(feature_shape, list)
        or len(feature_shape) != 4
        or pair_batch_size <= 0
        or feature_shape[0] != pair_batch_size * 2
        or concatenated != pair_batch_size * 2
        or detection_batch != pair_batch_size * 2
    ):
        raise RuntimeError(f"Paired feature or detection batch evidence failed: {evidence}")
    if int(evidence.get("clean_detection_targets", -1)) != int(evidence.get("degraded_detection_targets", -2)):
        raise RuntimeError(f"Clean/degraded target accounting failed: {evidence}")

    raw = float(evidence.get("consistency_loss_raw", float("nan")))
    weighted = float(evidence.get("consistency_loss_weighted_per_pair", float("nan")))
    scaled = float(evidence.get("consistency_loss_scaled_contribution", float("nan")))
    detection_loss = float(evidence.get("dual_view_detection_loss_scaled", float("nan")))
    if not math.isfinite(raw) or raw <= 0.0:
        raise RuntimeError(f"Raw P3 discrepancy is not finite and positive: {evidence}")
    if weighted != 0.0 or scaled != 0.0 or float(evidence.get("consistency_lambda", float("nan"))) != 0.0:
        raise RuntimeError(f"Lambda-zero consistency contribution is not exactly zero: {evidence}")
    if not math.isfinite(detection_loss) or detection_loss <= 0.0:
        raise RuntimeError(f"Focal-EIoU dual-view detection loss is invalid: {evidence}")

    gradient = evidence.get("consistency_only_gradient_probe", {})
    if (
        int(gradient.get("parameter_tensors", 0)) != 12
        or int(gradient.get("gradient_tensors", 0)) != 12
        or int(gradient.get("nonzero_gradient_tensors", -1)) != 0
        or gradient.get("finite") is not True
        or float(gradient.get("l2_norm", float("nan"))) != 0.0
        or gradient.get("expected_behavior") != "zero_by_lambda_control"
    ):
        raise RuntimeError(f"Lambda-zero gradient evidence failed: {gradient}")
    update = evidence.get("optimizer_update_probe", {})
    if update.get("finite") is not True or float(update.get("parameter_delta_l2", 0.0)) <= 0.0:
        raise RuntimeError(f"Head P3 parameters did not update from detection losses: {update}")


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_path = Path(context["lock"]["weights"][e2.MODEL_KEY]["path"])
    source = YOLO(str(source_path)).model
    model = CBAMFocalP3DetectionModel(
        source.yaml,
        ch=3,
        nc=int(source.nc),
        verbose=False,
        consistency_weight=0.0,
    )
    model.load(source)
    base_weight_matches = {
        str(index): bool(
            torch.equal(
                model.model[index].cv1.conv.weight.detach().cpu(),
                source.model[index].cv1.conv.weight.detach().cpu(),
            )
        )
        for index in EXPECTED_CBAM_LAYERS
    }
    if not all(base_weight_matches.values()):
        raise RuntimeError(f"Pretrained backbone weights did not transfer into CBAM modules: {base_weight_matches}")

    model.to(device).train()
    model.args = get_cfg(DEFAULT_CFG_DICT)
    model.criterion = model.init_criterion()
    if not isinstance(model.criterion, v8DetectionLossWithFocalEIoU):
        raise RuntimeError("Static self-test did not construct the Focal-EIoU criterion")
    if not isinstance(model.criterion.bbox_loss, FocalEIoUBboxLoss):
        raise RuntimeError("Static self-test did not install FocalEIoUBboxLoss")

    torch.manual_seed(0)
    clean = torch.rand((2, 3, 160, 160), device=device)
    degraded = clean.pow(1.18).mul(0.86)
    model._sdc_captured_feature = None
    model._sdc_capture_enabled = True
    try:
        model.forward(torch.cat((clean, degraded), dim=0))
    finally:
        model._sdc_capture_enabled = False
    feature = model._sdc_captured_feature
    if feature is None:
        raise RuntimeError("Static self-test did not capture P3")
    raw, metrics = model._pair_consistency(feature, 2)
    model._probe_consistency_gradient(raw * 0.0)

    pred_box = torch.tensor([[0.0, 0.0, 2.0, 2.0]], device=device, requires_grad=True)
    target_box = torch.tensor([[0.2, 0.1, 2.2, 2.1]], device=device)
    focal_loss = FocalEIoULoss()(pred_box, target_box).mean()
    focal_loss.backward()
    if not bool(torch.isfinite(focal_loss)) or pred_box.grad is None or not bool(torch.isfinite(pred_box.grad).all()):
        raise FloatingPointError("Standalone Focal-EIoU forward/backward self-test failed")

    evidence = {
        "schema": "ring3.cbam_focal_interaction_self_test.v1",
        "status": "passed",
        "verified_utc": e2.utc_now(),
        "device": str(device),
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "pilot_config_sha256": context["sdc_config_sha256"],
        "code_sha256": context["code_sha256"],
        "pretrained_source_sha256": sha256(source_path),
        "pretrained_base_weight_matches": base_weight_matches,
        **model.cbam_focal_evidence(),
        **metrics,
        "consistency_lambda": 0.0,
        "consistency_only_gradient_probe": model.sdc_gradient_probe(),
        "standalone_focal_eiou_loss": float(focal_loss.detach()),
        "standalone_focal_eiou_gradient_finite": True,
    }
    e2.write_json_atomic(SELF_TEST_PATH, evidence)
    return evidence


def run_queue(context: dict[str, Any]) -> int:
    try:
        e2.verify_smoke_evidence(context)
    except BaseException:
        smoke_result = e2.run_once("smoke", context)
        if smoke_result != 0:
            return smoke_result
        context = preflight_current()
        e2.verify_smoke_evidence(context)
    e2.activate_protocol(context)
    return e2.run_once("formal", context)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gated CBAM plus Focal-EIoU lambda-zero interaction pilot")
    parser.add_argument("--mode", choices=("self-test", "queue", "smoke", "pilot"), default="queue")
    args = parser.parse_args()
    configure()
    try:
        context = preflight_current()
        evidence = static_self_test(context)
        if args.mode == "self-test":
            print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.mode == "smoke":
            return e2.run_once("smoke", context)
        if args.mode == "pilot":
            e2.verify_smoke_evidence(context)
            e2.activate_protocol(context)
            return e2.run_once("formal", context)
        return run_queue(context)
    except BaseException as exc:
        state_path = SELF_TEST_PATH if args.mode == "self-test" else (
            SMOKE_STATE_PATH if args.mode == "smoke" else PILOT_STATE_PATH
        )
        events_path = SMOKE_EVENTS_PATH if args.mode in {"self-test", "smoke"} else PILOT_EVENTS_PATH
        previous = e2.load_json(state_path) if state_path.exists() else {}
        previous.update(
            status="preflight_failed",
            stage="interaction-pilot",
            paper_result_eligible=False,
            last_progress_utc=e2.utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        e2.write_json_atomic(state_path, previous)
        e2.append_event(events_path, "preflight_failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
