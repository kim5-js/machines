from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import train_e2_sdc as e2
from sdc_lambda_ablation import LambdaAblationDetectionModel, load_sdc_ablation_config, self_test


EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_MDG_CONFIG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_E2_CONFIG_SHA256 = "fd95632a444fbb33eba462531a5aabc9eeadf755f09f8581a1c5c6cbd5bababc"
EXPECTED_E2_TRAIN_SHA256 = "2c5d415a614b5f9bf53d5deb0934548390ef14e8409808167857d11b844dc721"
EXPECTED_E2_SDC_SHA256 = "87979f6b2f7c6d66103d2883c843b8b796866e1db638119337b1b1cf2b62fd38"
EXPECTED_MDG_CODE_SHA256 = "b0d60602f77db9bbb935f5e48f44a5766f361ef35bc07e90fd06b23c3f8b85cc"
EXPECTED_E2_RUNNER_SHA256 = "be9fb1e63286778a267770cda9023a6dd5548f36b2f1e8b1cd216f3c1c1055bc"

DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", str(DATA_ROOT / "ring3_setup" / "autodl")))
FORMAL_ROOT = DATA_ROOT / "formal_e2_lambda_ablation"
STATE_ROOT = DATA_ROOT / "formal_state"
DIAGNOSTICS_ROOT = DATA_ROOT / "diagnostics"
QUEUE_STATE_PATH = STATE_ROOT / "e2_lambda_ablation_queue_state.json"
QUEUE_EVENTS_PATH = STATE_ROOT / "e2_lambda_ablation_queue_events.jsonl"
QUEUE_PROTOCOL_PATH = STATE_ROOT / "e2_lambda_ablation_queue_protocol.json"
E2_STATE_PATH = STATE_ROOT / "e2_sdc_state.json"
E2_PROTOCOL_PATH = STATE_ROOT / "e2_sdc_protocol.json"
E2_RUN = DATA_ROOT / "formal_e2" / "seed0_yolov8s_seadronessee_e2_mdg_p3cons_img960_e150"

RUN_SPECS: dict[str, dict[str, Any]] = {
    "lambda0": {
        "ablation_id": "lambda0",
        "lambda": 0.0,
        "config_name": "e2_p3_consistency_lambda0_config.json",
        "formal_name": "seed0_yolov8s_seadronessee_e2_mdg_p3cons_lambda0_img960_e150",
        "state_name": "e2_lambda0_state.json",
        "events_name": "e2_lambda0_events.jsonl",
        "protocol_name": "e2_lambda0_protocol.json",
        "smoke_state_name": "e2_lambda0_smoke_state.json",
        "smoke_events_name": "e2_lambda0_smoke_events.jsonl",
        "self_test_name": "e2_lambda0_self_test.json",
    },
    "lambda003": {
        "ablation_id": "lambda003",
        "lambda": 0.03,
        "config_name": "e2_p3_consistency_lambda003_config.json",
        "formal_name": "seed0_yolov8s_seadronessee_e2_mdg_p3cons_lambda003_img960_e150",
        "state_name": "e2_lambda003_state.json",
        "events_name": "e2_lambda003_events.jsonl",
        "protocol_name": "e2_lambda003_protocol.json",
        "smoke_state_name": "e2_lambda003_smoke_state.json",
        "smoke_events_name": "e2_lambda003_smoke_events.jsonl",
        "self_test_name": "e2_lambda003_self_test.json",
    },
}
RUN_ORDER = ("lambda0", "lambda003")

_ORIGINAL_PREFLIGHT = e2.preflight
_ORIGINAL_INITIAL_STATE = e2.initial_state
CURRENT_SPEC: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_spec() -> dict[str, Any]:
    if CURRENT_SPEC is None:
        raise RuntimeError("Lambda-ablation run was not configured")
    return CURRENT_SPEC


def configure(ablation_id: str) -> dict[str, Any]:
    global CURRENT_SPEC
    if ablation_id not in RUN_SPECS:
        raise ValueError(f"Unknown lambda-ablation run: {ablation_id!r}")
    spec = dict(RUN_SPECS[ablation_id])
    spec["config_path"] = SETUP_ROOT / spec["config_name"]
    spec["formal_state_path"] = STATE_ROOT / spec["state_name"]
    spec["formal_events_path"] = STATE_ROOT / spec["events_name"]
    spec["formal_protocol_path"] = STATE_ROOT / spec["protocol_name"]
    spec["smoke_state_path"] = DIAGNOSTICS_ROOT / spec["smoke_state_name"]
    spec["smoke_events_path"] = DIAGNOSTICS_ROOT / spec["smoke_events_name"]
    spec["self_test_path"] = DIAGNOSTICS_ROOT / spec["self_test_name"]
    CURRENT_SPEC = spec

    e2.STAGE = "E2-lambda-ablation"
    e2.FORMAL_NAME = str(spec["formal_name"])
    e2.SDC_CONFIG_PATH = Path(spec["config_path"])
    e2.FORMAL_ROOT = FORMAL_ROOT
    e2.FAILED_ROOT = DATA_ROOT / "formal_failed_runs" / "e2_lambda_ablation"
    e2.FORMAL_STATE_PATH = Path(spec["formal_state_path"])
    e2.FORMAL_EVENTS_PATH = Path(spec["formal_events_path"])
    e2.FORMAL_PROTOCOL_PATH = Path(spec["formal_protocol_path"])
    e2.SMOKE_STATE_PATH = Path(spec["smoke_state_path"])
    e2.SMOKE_EVENTS_PATH = Path(spec["smoke_events_path"])
    e2.P3ConsistencyDetectionModel = LambdaAblationDetectionModel
    e2.load_sdc_config = load_sdc_ablation_config
    e2.code_hashes = code_hashes
    e2.protocol_payload = protocol_payload
    e2.validate_runtime_evidence = validate_runtime_evidence
    e2.initial_state = initial_state
    return spec


def code_hashes() -> dict[str, str]:
    spec = current_spec()
    paths = {
        "train_e2_lambda_ablation.py": Path(__file__).resolve(),
        "sdc_lambda_ablation.py": SETUP_ROOT / "sdc_lambda_ablation.py",
        "train_e2_sdc.py": SETUP_ROOT / "train_e2_sdc.py",
        "sdc_consistency.py": SETUP_ROOT / "sdc_consistency.py",
        "maritime_degradation.py": SETUP_ROOT / "maritime_degradation.py",
        str(spec["config_name"]): Path(spec["config_path"]),
        "e1_mdg_config.json": e2.MDG_CONFIG_PATH,
        "run_e2_lambda_ablation_queue.sh": SETUP_ROOT / "run_e2_lambda_ablation_queue.sh",
    }
    return {name: sha256(path) for name, path in paths.items()}


def verify_completed_e2_anchor(lock: dict[str, Any]) -> dict[str, Any]:
    required_files = {
        "e2_p3_consistency_config.json": (SETUP_ROOT / "e2_p3_consistency_config.json", EXPECTED_E2_CONFIG_SHA256),
        "train_e2_sdc.py": (SETUP_ROOT / "train_e2_sdc.py", EXPECTED_E2_TRAIN_SHA256),
        "sdc_consistency.py": (SETUP_ROOT / "sdc_consistency.py", EXPECTED_E2_SDC_SHA256),
        "maritime_degradation.py": (SETUP_ROOT / "maritime_degradation.py", EXPECTED_MDG_CODE_SHA256),
        "run_e2_sdc.sh": (SETUP_ROOT / "run_e2_sdc.sh", EXPECTED_E2_RUNNER_SHA256),
    }
    for name, (path, expected) in required_files.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Completed E2 anchor file changed: {name}")
    if not E2_STATE_PATH.is_file() or not E2_PROTOCOL_PATH.is_file():
        raise RuntimeError("Completed E2 state or protocol is missing")
    state = e2.load_json(E2_STATE_PATH)
    protocol = e2.load_json(E2_PROTOCOL_PATH)
    if state.get("status") != "completed" or int(state.get("current_epoch", 0)) != int(lock["common_train"]["epochs"]):
        raise RuntimeError(f"E2 lambda=0.1 anchor is not complete: {state.get('status')!r}")
    expected_hashes = {
        "formal_lock_sha256": EXPECTED_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_CONFIG_SHA256,
        "sdc_config_sha256": EXPECTED_E2_CONFIG_SHA256,
    }
    for key, expected in expected_hashes.items():
        if state.get(key) != expected or protocol.get(key) != expected:
            raise RuntimeError(f"E2 lambda=0.1 anchor evidence mismatch for {key}")
    complete, final_row = e2.base.completed_run(E2_RUN, int(lock["common_train"]["epochs"]))
    if not complete or final_row is None:
        raise RuntimeError("E2 lambda=0.1 anchor results are incomplete or non-finite")
    e2.base.verify_resume_args(E2_RUN, lock, e2.DATASET_KEY)
    evidence_path = E2_RUN / "e2_sdc_runtime_evidence.json"
    evidence = e2.load_json(evidence_path)
    if not math.isclose(float(evidence.get("consistency_lambda", -1.0)), 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("E2 anchor runtime evidence no longer records lambda=0.1")
    return {
        "run_dir": str(E2_RUN),
        "state_sha256": sha256(E2_STATE_PATH),
        "protocol_sha256": sha256(E2_PROTOCOL_PATH),
        "args_sha256": sha256(E2_RUN / "args.yaml"),
        "runtime_evidence_sha256": sha256(evidence_path),
        "final_row": final_row,
        "best_row": e2.read_best_metric_row(E2_RUN / "results.csv"),
        "consistency_lambda": 0.1,
    }


def preflight_current() -> dict[str, Any]:
    context = _ORIGINAL_PREFLIGHT()
    if context["lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Unexpected formal lock in lambda-ablation preflight")
    config = context["sdc_config"]
    spec = current_spec()
    if config.get("ablation_id") != spec["ablation_id"]:
        raise RuntimeError("Configured lambda-ablation id does not match the selected run")
    if not math.isclose(float(config["consistency"]["lambda"]), float(spec["lambda"]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Configured lambda does not match the selected run")
    context["e2_lambda_0_1_anchor"] = verify_completed_e2_anchor(context["lock"])
    return context


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    spec = current_spec()
    lock = context["lock"]
    config = context["sdc_config"]
    return {
        "schema": "ring3.e2_lambda_ablation_protocol.v1",
        "stage": "E2-lambda-ablation",
        "ablation_id": spec["ablation_id"],
        "consistency_lambda": float(spec["lambda"]),
        "model": e2.MODEL_KEY,
        "dataset": e2.DATASET_KEY,
        "formal_name": spec["formal_name"],
        "formal_lock_sha256": context["lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "sdc_config_sha256": context["sdc_config_sha256"],
        "code_sha256": context["code_sha256"],
        "weight_sha256": lock["weights"][e2.MODEL_KEY]["sha256"],
        "dataset_yaml_sha256": lock["dataset_protocols"][e2.DATASET_KEY]["yaml_sha256"],
        "dataset_manifest_sha256": lock["dataset_protocols"][e2.DATASET_KEY]["manifest_sha256"],
        "e0_args_sha256": context["baseline"]["args_sha256"],
        "e1_protocol_sha256": context["e1"]["protocol_sha256"],
        "e2_lambda_0_1_anchor": context["e2_lambda_0_1_anchor"],
        "activation": config["activation"],
        "dual_view": config["dual_view"],
        "consistency": config["consistency"],
        "required_runtime_evidence": config["required_runtime_evidence"],
        "invariants": config["invariants"],
    }


def initial_state(mode: str, run_dir: Path, context: dict[str, Any], target_epochs: int) -> dict[str, Any]:
    state = _ORIGINAL_INITIAL_STATE(mode, run_dir, context, target_epochs)
    spec = current_spec()
    state.update(
        schema="ring3.e2_lambda_ablation_state.v1",
        ablation_id=spec["ablation_id"],
        consistency_lambda=float(spec["lambda"]),
        independent_initialization_from=context["lock"]["weights"][e2.MODEL_KEY]["path"],
        e2_lambda_0_1_anchor=context["e2_lambda_0_1_anchor"],
    )
    return state


def validate_runtime_evidence(evidence: dict[str, Any]) -> None:
    spec = current_spec()
    expected_lambda = float(spec["lambda"])
    detect_inputs = evidence.get("detect_input_indices")
    target_index = evidence.get("target_feature_layer_index")
    target_type = evidence.get("target_feature_layer_type")
    feature_shape = evidence.get("feature_shape")
    pair_batch_size = int(evidence.get("pair_batch_size", 0))
    concatenated = int(evidence.get("concatenated_batch_size", 0))
    detection_batch = int(evidence.get("detection_batch_size", 0))
    raw_loss = float(evidence.get("consistency_loss_raw", float("nan")))
    weighted = float(evidence.get("consistency_loss_weighted_per_pair", float("nan")))
    scaled = float(evidence.get("consistency_loss_scaled_contribution", float("nan")))
    detection_loss = float(evidence.get("dual_view_detection_loss_scaled", float("nan")))
    feature_delta = float(evidence.get("feature_mean_abs_delta", float("nan")))
    clean_targets = int(evidence.get("clean_detection_targets", -1))
    degraded_targets = int(evidence.get("degraded_detection_targets", -2))
    gradient = evidence.get("consistency_only_gradient_probe", {})
    update = evidence.get("optimizer_update_probe", {})

    if detect_inputs != [15, 18, 21] or target_index != 15 or target_type != "C2f":
        raise RuntimeError(f"Invalid locked Detect/P3 evidence: {evidence}")
    if (
        not isinstance(feature_shape, list)
        or len(feature_shape) != 4
        or pair_batch_size <= 0
        or feature_shape[0] != pair_batch_size * 2
        or concatenated != pair_batch_size * 2
        or detection_batch != pair_batch_size * 2
    ):
        raise RuntimeError(f"Invalid paired feature/detection batch evidence: {evidence}")
    if clean_targets < 0 or clean_targets != degraded_targets:
        raise RuntimeError(f"Clean/degraded target accounting failed: {evidence}")
    if not math.isfinite(detection_loss) or detection_loss <= 0.0:
        raise RuntimeError(f"Dual-view detection loss did not become finite and nonzero: {evidence}")
    if not math.isfinite(raw_loss) or raw_loss <= 0.0 or not math.isfinite(feature_delta) or feature_delta <= 0.0:
        raise RuntimeError(f"Raw P3 discrepancy did not become finite and nonzero: {evidence}")
    if not math.isclose(float(evidence.get("consistency_lambda", float("nan"))), expected_lambda, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Runtime lambda mismatch: {evidence}")
    if not math.isfinite(weighted) or not math.isfinite(scaled):
        raise RuntimeError(f"Weighted P3 contribution is non-finite: {evidence}")
    if not math.isclose(weighted, raw_loss * expected_lambda, rel_tol=1e-5, abs_tol=1e-10):
        raise RuntimeError(f"Per-pair lambda scaling mismatch: {evidence}")
    if not math.isclose(scaled, weighted * pair_batch_size, rel_tol=1e-5, abs_tol=1e-10):
        raise RuntimeError(f"Ultralytics batch scaling mismatch: {evidence}")
    if int(gradient.get("parameter_tensors", 0)) != 12 or int(gradient.get("gradient_tensors", 0)) != 12:
        raise RuntimeError(f"Expected exactly 12 P3 consistency-gradient tensors: {gradient}")
    if gradient.get("finite") is not True:
        raise RuntimeError(f"Consistency-gradient evidence is non-finite: {gradient}")

    nonzero_gradients = int(gradient.get("nonzero_gradient_tensors", -1))
    gradient_norm = float(gradient.get("l2_norm", float("nan")))
    if expected_lambda == 0.0:
        if weighted != 0.0 or scaled != 0.0:
            raise RuntimeError(f"Lambda-zero control contributed nonzero consistency loss: {evidence}")
        if nonzero_gradients != 0 or gradient_norm != 0.0:
            raise RuntimeError(f"Lambda-zero control leaked a consistency gradient: {gradient}")
        if gradient.get("expected_behavior") != "zero_by_lambda_control":
            raise RuntimeError(f"Lambda-zero gradient behavior was not explicitly recorded: {gradient}")
    else:
        if weighted <= 0.0 or scaled <= 0.0 or nonzero_gradients != 12 or not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"Enabled consistency did not produce 12 finite nonzero gradients: {gradient}")
        if gradient.get("expected_behavior") != "finite_nonzero_for_enabled_consistency":
            raise RuntimeError(f"Enabled consistency gradient behavior was not explicitly recorded: {gradient}")
    if update.get("finite") is not True or not math.isfinite(float(update.get("parameter_delta_l2", float("nan")))):
        raise RuntimeError(f"P3 optimizer-update evidence is non-finite: {update}")
    if float(update.get("parameter_delta_l2", 0.0)) <= 0.0:
        raise RuntimeError(f"P3 parameters did not update: {update}")


def write_preflight_failure(mode: str, exc: BaseException) -> None:
    spec = current_spec()
    state_path = Path(spec["smoke_state_path"] if mode == "smoke" else spec["formal_state_path"])
    events_path = Path(spec["smoke_events_path"] if mode == "smoke" else spec["formal_events_path"])
    previous = e2.load_json(state_path) if state_path.exists() else {}
    previous.update(
        status="preflight_failed",
        mode=mode,
        stage="E2-lambda-ablation",
        ablation_id=spec["ablation_id"],
        consistency_lambda=float(spec["lambda"]),
        last_progress_utc=utc_now(),
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
    e2.write_json_atomic(state_path, previous)
    e2.append_event(
        events_path,
        "preflight_failed",
        mode=mode,
        ablation_id=spec["ablation_id"],
        error_type=type(exc).__name__,
        error=str(exc),
    )


def run_one(ablation_id: str, mode: str) -> int:
    spec = configure(ablation_id)
    try:
        context = preflight_current()
        if mode == "smoke":
            weight_path = Path(context["lock"]["weights"][e2.MODEL_KEY]["path"])
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            static_evidence = self_test(Path(spec["config_path"]), weight_path, device)
            static_evidence.update(
                verified_utc=utc_now(),
                formal_lock_sha256=context["lock_sha256"],
                runtime_amendment_sha256=context["runtime_amendment_sha256"],
                mdg_config_sha256=context["mdg_config_sha256"],
                sdc_config_sha256=context["sdc_config_sha256"],
                code_sha256=context["code_sha256"],
            )
            e2.write_json_atomic(Path(spec["self_test_path"]), static_evidence)
        else:
            e2.verify_smoke_evidence(context)
            e2.activate_protocol(context)
        return e2.run_once(mode, context)
    except BaseException as exc:
        write_preflight_failure(mode, exc)
        traceback.print_exc()
        return 4


def smoke_is_current(ablation_id: str) -> bool:
    configure(ablation_id)
    try:
        context = preflight_current()
        e2.verify_smoke_evidence(context)
        return True
    except BaseException:
        return False


def queue_protocol_payload() -> dict[str, Any]:
    common_paths = {
        "train_e2_lambda_ablation.py": Path(__file__).resolve(),
        "sdc_lambda_ablation.py": SETUP_ROOT / "sdc_lambda_ablation.py",
        "train_e2_sdc.py": SETUP_ROOT / "train_e2_sdc.py",
        "sdc_consistency.py": SETUP_ROOT / "sdc_consistency.py",
        "maritime_degradation.py": SETUP_ROOT / "maritime_degradation.py",
        "run_e2_lambda_ablation_queue.sh": SETUP_ROOT / "run_e2_lambda_ablation_queue.sh",
    }
    configs = {run_id: sha256(SETUP_ROOT / RUN_SPECS[run_id]["config_name"]) for run_id in RUN_ORDER}
    return {
        "schema": "ring3.e2_lambda_ablation_queue_protocol.v1",
        "stage": "E2-lambda-ablation",
        "run_order": list(RUN_ORDER),
        "lambda_values": {run_id: float(RUN_SPECS[run_id]["lambda"]) for run_id in RUN_ORDER},
        "independent_pretrained_initialization_for_each_run": True,
        "formal_lock_sha256": EXPECTED_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_CONFIG_SHA256,
        "completed_e2_lambda_0_1_config_sha256": EXPECTED_E2_CONFIG_SHA256,
        "config_sha256": configs,
        "code_sha256": {name: sha256(path) for name, path in common_paths.items()},
    }


def activate_queue_protocol() -> dict[str, Any]:
    payload = queue_protocol_payload()
    if QUEUE_PROTOCOL_PATH.exists():
        recorded = e2.load_json(QUEUE_PROTOCOL_PATH)
        if recorded != payload:
            raise RuntimeError("Lambda-ablation queue protocol changed after activation")
    else:
        e2.write_json_atomic(QUEUE_PROTOCOL_PATH, payload)
    return payload


def load_queue_state(protocol: dict[str, Any]) -> dict[str, Any]:
    if QUEUE_STATE_PATH.exists():
        state = e2.load_json(QUEUE_STATE_PATH)
        if state.get("protocol_sha256") != sha256(QUEUE_PROTOCOL_PATH):
            raise RuntimeError("Lambda-ablation queue state points to a different protocol")
        return state
    now = utc_now()
    return {
        "schema": "ring3.e2_lambda_ablation_queue_state.v1",
        "status": "starting",
        "started_utc": now,
        "last_progress_utc": now,
        "current_run": None,
        "current_lambda": None,
        "run_order": list(RUN_ORDER),
        "completed_runs": [],
        "failed_runs": [],
        "results": {},
        "protocol_sha256": sha256(QUEUE_PROTOCOL_PATH),
        "formal_lock_sha256": protocol["formal_lock_sha256"],
        "runtime_amendment_sha256": protocol["runtime_amendment_sha256"],
        "mdg_config_sha256": protocol["mdg_config_sha256"],
        "config_sha256": protocol["config_sha256"],
        "code_sha256": protocol["code_sha256"],
    }


def collect_result(ablation_id: str) -> dict[str, Any]:
    spec = RUN_SPECS[ablation_id]
    run_dir = FORMAL_ROOT / spec["formal_name"]
    final_row = e2.base.read_last_row(run_dir / "results.csv")
    if final_row is None:
        raise RuntimeError(f"Missing final row for {ablation_id}")
    best_row = e2.read_best_metric_row(run_dir / "results.csv")
    anchor_best = e2.read_best_metric_row(E2_RUN / "results.csv")
    metrics = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    return {
        "run_dir": str(run_dir),
        "consistency_lambda": float(spec["lambda"]),
        "args_sha256": sha256(run_dir / "args.yaml"),
        "best_row": best_row,
        "final_row": final_row,
        "best_delta_vs_lambda_0_1": {
            metric: float(best_row[metric]) - float(anchor_best[metric]) for metric in metrics
        },
    }


def update_queue_state(state: dict[str, Any], **updates: Any) -> None:
    state.update(updates)
    state["last_progress_utc"] = utc_now()
    e2.write_json_atomic(QUEUE_STATE_PATH, state)


def run_queue() -> int:
    try:
        protocol = activate_queue_protocol()
        state = load_queue_state(protocol)
    except BaseException as exc:
        failure = {
            "schema": "ring3.e2_lambda_ablation_queue_state.v1",
            "status": "preflight_failed",
            "last_progress_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        e2.write_json_atomic(QUEUE_STATE_PATH, failure)
        e2.append_event(QUEUE_EVENTS_PATH, "queue_preflight_failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 4

    update_queue_state(state, status="running")
    e2.append_event(QUEUE_EVENTS_PATH, "queue_started_or_resumed", run_order=list(RUN_ORDER))
    completed = list(state.get("completed_runs", []))
    failed = list(state.get("failed_runs", []))

    for ablation_id in RUN_ORDER:
        spec = RUN_SPECS[ablation_id]
        update_queue_state(
            state,
            status="running",
            current_run=ablation_id,
            current_lambda=float(spec["lambda"]),
            current_phase="smoke_check",
        )
        if not smoke_is_current(ablation_id):
            e2.append_event(
                QUEUE_EVENTS_PATH,
                "smoke_started",
                ablation_id=ablation_id,
                consistency_lambda=float(spec["lambda"]),
            )
            rc = run_one(ablation_id, "smoke")
            if rc != 0:
                update_queue_state(
                    state,
                    status="preflight_failed",
                    current_phase="smoke_failed",
                    last_exit_code=rc,
                )
                e2.append_event(QUEUE_EVENTS_PATH, "smoke_failed", ablation_id=ablation_id, exit_code=rc)
                return rc
            e2.append_event(QUEUE_EVENTS_PATH, "smoke_completed", ablation_id=ablation_id)

        update_queue_state(state, current_phase="formal")
        e2.append_event(
            QUEUE_EVENTS_PATH,
            "formal_run_entered",
            ablation_id=ablation_id,
            consistency_lambda=float(spec["lambda"]),
        )
        while True:
            rc = run_one(ablation_id, "formal")
            update_queue_state(state, last_exit_code=rc)
            if rc == 0:
                if ablation_id not in completed:
                    completed.append(ablation_id)
                if ablation_id in failed:
                    failed.remove(ablation_id)
                result = collect_result(ablation_id)
                state.setdefault("results", {})[ablation_id] = result
                update_queue_state(
                    state,
                    completed_runs=completed,
                    failed_runs=failed,
                    current_phase="formal_completed",
                )
                e2.append_event(QUEUE_EVENTS_PATH, "formal_run_completed", ablation_id=ablation_id, result=result)
                break
            if rc == 3:
                if ablation_id not in failed:
                    failed.append(ablation_id)
                update_queue_state(
                    state,
                    completed_runs=completed,
                    failed_runs=failed,
                    current_phase="attempts_exhausted",
                )
                e2.append_event(QUEUE_EVENTS_PATH, "formal_attempts_exhausted", ablation_id=ablation_id)
                break
            if rc == 4:
                update_queue_state(state, status="preflight_failed", current_phase="formal_preflight_failed")
                e2.append_event(QUEUE_EVENTS_PATH, "formal_preflight_failed", ablation_id=ablation_id)
                return rc
            e2.append_event(QUEUE_EVENTS_PATH, "formal_attempt_will_retry", ablation_id=ablation_id, exit_code=rc)

    terminal = "completed_with_failures" if failed else "completed"
    update_queue_state(
        state,
        status=terminal,
        current_run=None,
        current_lambda=None,
        current_phase="done",
        completed_runs=completed,
        failed_runs=failed,
        completed_utc=utc_now(),
    )
    e2.append_event(QUEUE_EVENTS_PATH, "queue_completed", status=terminal, completed_runs=completed, failed_runs=failed)
    return 3 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked E2 P3 lambda=0 and lambda=0.03 ablation queue")
    parser.add_argument("--mode", choices=("smoke", "formal", "check-smoke", "queue"), required=True)
    parser.add_argument("--run", choices=tuple(RUN_SPECS))
    args = parser.parse_args()
    if args.mode == "queue":
        if args.run is not None:
            parser.error("--run is not used with --mode queue")
        return run_queue()
    if args.run is None:
        parser.error("--run is required for smoke, formal, and check-smoke modes")
    if args.mode == "check-smoke":
        return 0 if smoke_is_current(args.run) else 1
    return run_one(args.run, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
