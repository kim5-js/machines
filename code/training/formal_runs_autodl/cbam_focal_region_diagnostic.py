from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules import C2f

import cbam_focal_sdc
import formal_baseline_queue as base
import p3_region_cosine_diagnostic as diagnostic
import train_cbam_focal_interaction_pilot as pilot
from cbam_focal_sdc import (
    CBAMFocalP3DetectionModel,
    C2f_CBAM,
    EXPECTED_CBAM_LAYERS,
    EXPECTED_DETECT_INPUTS,
    FocalEIoUBboxLoss,
    v8DetectionLossWithFocalEIoU,
)


EXPECTED_EPOCHS = 30
RUN_NAME = "cbam_focal_lambda0_best_200b"
DATA_ROOT = diagnostic.DATA_ROOT
SETUP_ROOT = diagnostic.SETUP_ROOT
TEMPLATE_PATH = SETUP_ROOT / "cbam_focal_region_diagnostic_template.json"
PILOT_STATE_PATH = DATA_ROOT / "formal_state" / "cbam_focal_interaction_pilot_state.json"
PILOT_PROTOCOL_PATH = DATA_ROOT / "formal_state" / "cbam_focal_interaction_pilot_protocol.json"
EFFECTIVE_CONFIG_PATH = DATA_ROOT / "formal_state" / "cbam_focal_region_diagnostic_effective_config.json"
DIAGNOSTIC_STATE_PATH = DATA_ROOT / "formal_state" / "cbam_focal_region_diagnostic_state.json"
STOP_SENTINEL = DATA_ROOT / "formal_state" / "STOP_CBAM_FOCAL_INTERACTION_PILOT"

_PREFLIGHT_EVIDENCE: dict[str, Any] = {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    observed = diagnostic.sha256(path)
    if observed != expected:
        raise RuntimeError(f"Hash mismatch for {path.name}: expected={expected}, observed={observed}")


def verify_checkpoint_architecture(checkpoint: Path) -> dict[str, Any]:
    loaded = YOLO(str(checkpoint))
    model = loaded.model
    if not isinstance(model, CBAMFocalP3DetectionModel):
        raise RuntimeError(f"Pilot checkpoint model type changed: {type(model).__name__}")

    cbam_types = [type(model.model[index]).__name__ for index in EXPECTED_CBAM_LAYERS]
    if cbam_types != ["C2f_CBAM", "C2f_CBAM"]:
        raise RuntimeError(f"CBAM layers are missing from the pilot checkpoint: {cbam_types}")
    if any(not isinstance(model.model[index], C2f_CBAM) for index in EXPECTED_CBAM_LAYERS):
        raise RuntimeError("Pilot checkpoint CBAM modules failed the concrete type check")

    detect = model.model[-1]
    detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
    if detect_inputs != EXPECTED_DETECT_INPUTS:
        raise RuntimeError(f"Detect inputs changed: {detect_inputs}")
    p3_index = int(detect_inputs[0])
    if p3_index != 15 or type(model.model[p3_index]).__name__ != "C2f":
        raise RuntimeError(f"Head P3 target changed: index={p3_index}, type={type(model.model[p3_index]).__name__}")
    if not isinstance(model.model[p3_index], C2f):
        raise RuntimeError("Head P3 feature is no longer a C2f module")

    criterion = model.init_criterion()
    if not isinstance(criterion, v8DetectionLossWithFocalEIoU):
        raise RuntimeError(f"Focal-EIoU criterion is missing: {type(criterion).__name__}")
    if not isinstance(criterion.bbox_loss, FocalEIoUBboxLoss):
        raise RuntimeError(f"Focal-EIoU bbox loss is missing: {type(criterion.bbox_loss).__name__}")

    attention_parameters = sum(
        parameter.numel()
        for index in EXPECTED_CBAM_LAYERS
        for parameter in model.model[index].attention.parameters()
    )
    if attention_parameters <= 0:
        raise RuntimeError("Pilot checkpoint contains no CBAM attention parameters")
    return {
        "checkpoint_model_type": type(model).__name__,
        "cbam_layer_indices": EXPECTED_CBAM_LAYERS,
        "cbam_layer_types": cbam_types,
        "cbam_attention_parameter_count": attention_parameters,
        "detect_input_indices": detect_inputs,
        "target_feature_layer_index": p3_index,
        "target_feature_layer_type": type(model.model[p3_index]).__name__,
        "criterion_type": type(criterion).__name__,
        "bbox_loss_type": type(criterion.bbox_loss).__name__,
        "focal_eiou": {
            "alpha": cbam_focal_sdc.FOCAL_EIOU_ALPHA,
            "gamma": cbam_focal_sdc.FOCAL_EIOU_GAMMA,
            "eps": cbam_focal_sdc.FOCAL_EIOU_EPS,
            "dfl_preserved": True,
        },
    }


def build_effective_config() -> tuple[dict[str, Any], dict[str, Any]]:
    if STOP_SENTINEL.exists():
        raise RuntimeError(f"Explicit interaction-pilot stop sentinel exists: {STOP_SENTINEL}")
    if not TEMPLATE_PATH.is_file():
        raise RuntimeError(f"Diagnostic template is missing: {TEMPLATE_PATH}")
    state = load_json(PILOT_STATE_PATH)
    if state.get("status") != "completed":
        raise RuntimeError(f"Interaction pilot is not completed: {state.get('status')!r}")
    if int(state.get("current_epoch", -1)) != EXPECTED_EPOCHS or int(state.get("target_epochs", -1)) != EXPECTED_EPOCHS:
        raise RuntimeError(f"Interaction pilot did not complete exactly {EXPECTED_EPOCHS} epochs")
    if state.get("paper_result_eligible") is not False:
        raise RuntimeError("Interaction pilot lost its non-paper scope marker")
    if state.get("object_aware_consistency_started") is not False or state.get("e3_formal_started") is not False:
        raise RuntimeError("Object-Aware consistency or E3 was unexpectedly started")

    run_dir = Path(str(state["run_dir"]))
    args_path = run_dir / "args.yaml"
    checkpoint = run_dir / "weights" / "best.pt"
    runtime_path = run_dir / "e2_sdc_runtime_evidence.json"
    metadata_path = run_dir / "e2_sdc_metadata.json"
    for path in (args_path, checkpoint, runtime_path, metadata_path, run_dir / "results.csv"):
        if not path.is_file():
            raise RuntimeError(f"Completed pilot evidence is missing: {path}")

    epoch_metrics = state.get("sdc_last_epoch_metrics", {})
    reference_cosine = float(epoch_metrics.get("cosine_mean", float("nan")))
    if not math.isfinite(reference_cosine) or not -1.0 <= reference_cosine <= 1.0:
        raise RuntimeError(f"Pilot epoch-30 global cosine is invalid: {reference_cosine!r}")

    config = load_json(TEMPLATE_PATH)
    config["source_run"] = str(run_dir)
    config["source_args_sha256"] = diagnostic.sha256(args_path)
    config["checkpoint"] = str(checkpoint)
    config["checkpoint_sha256"] = diagnostic.sha256(checkpoint)
    config["reference"] = {
        "global_cosine_mean": reference_cosine,
        "source": "completed CBAM plus Focal-EIoU lambda-zero pilot epoch 30 mean",
        "absolute_tolerance": 0.003,
    }
    config["interaction_pilot"].update(
        {
            "pilot_state_sha256": diagnostic.sha256(PILOT_STATE_PATH),
            "pilot_protocol_sha256": diagnostic.sha256(PILOT_PROTOCOL_PATH),
            "runtime_evidence_sha256": diagnostic.sha256(runtime_path),
            "metadata_sha256": diagnostic.sha256(metadata_path),
        }
    )
    diagnostic.load_diagnostic_config(TEMPLATE_PATH)
    diagnostic.write_json(EFFECTIVE_CONFIG_PATH, config)
    diagnostic.load_diagnostic_config(EFFECTIVE_CONFIG_PATH)
    return config, state


def pilot_preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    global _PREFLIGHT_EVIDENCE
    release = pilot.verify_release_manifest()
    require_hash(SETUP_ROOT / "formal_experiment_lock.json", str(config["formal_lock_sha256"]))
    require_hash(SETUP_ROOT / "formal_runtime_amendment.json", str(config["runtime_amendment_sha256"]))
    require_hash(SETUP_ROOT / "e1_mdg_config.json", str(config["mdg_config_sha256"]))
    pilot.verify_anchor_files()

    lock = base.load_lock()
    base.verify_environment(lock)
    base.validate_training_lock(lock)
    if int(lock["common_train"]["epochs"]) != 150:
        raise RuntimeError("The on-disk formal lock no longer records 150 epochs")
    dataset = lock["dataset_protocols"]["seadronessee"]
    data_yaml = Path(dataset["yaml"])
    manifest_path = Path(dataset["manifest_path"])
    require_hash(data_yaml, str(dataset["yaml_sha256"]))
    require_hash(manifest_path, str(dataset["manifest_sha256"]))

    state = load_json(PILOT_STATE_PATH)
    source_run = Path(str(config["source_run"]))
    if source_run != Path(str(state.get("run_dir"))):
        raise RuntimeError("Diagnostic source run no longer matches the completed pilot state")
    args_path = source_run / "args.yaml"
    require_hash(args_path, str(config["source_args_sha256"]))
    source_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    required_args = {
        "data": str(data_yaml),
        "epochs": EXPECTED_EPOCHS,
        "batch": int(config["batch"]),
        "imgsz": int(config["imgsz"]),
        "seed": int(config["seed"]),
        "pretrained": True,
        "deterministic": True,
        "optimizer": lock["common_train"]["optimizer"],
        "resume": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": source_args.get(key)}
        for key, expected in required_args.items()
        if source_args.get(key) != expected
    }
    if mismatches:
        raise RuntimeError("Pilot source args mismatch: " + json.dumps(mismatches, ensure_ascii=False))

    metadata = load_json(source_run / "e2_sdc_metadata.json")
    if metadata.get("paper_result_eligible") is not False or int(metadata.get("pilot_epochs", -1)) != EXPECTED_EPOCHS:
        raise RuntimeError("Pilot metadata scope or epoch count changed")
    if metadata.get("weight_sha256") != lock["weights"]["yolov8s"]["sha256"]:
        raise RuntimeError("Pilot metadata no longer proves locked pretrained initialization")
    if metadata.get("object_aware_consistency_started") is not False:
        raise RuntimeError("Pilot metadata unexpectedly records Object-Aware consistency")

    runtime = load_json(source_run / "e2_sdc_runtime_evidence.json")
    pilot.validate_runtime_evidence(runtime)
    checkpoint = Path(str(config["checkpoint"]))
    require_hash(checkpoint, str(config["checkpoint_sha256"]))
    final_metrics = diagnostic.validate_source_results(source_run, EXPECTED_EPOCHS)
    architecture = verify_checkpoint_architecture(checkpoint)
    _PREFLIGHT_EVIDENCE = {
        "release_manifest": release,
        "pilot_state_sha256": diagnostic.sha256(PILOT_STATE_PATH),
        "pilot_protocol_sha256": diagnostic.sha256(PILOT_PROTOCOL_PATH),
        "runtime_evidence_sha256": diagnostic.sha256(source_run / "e2_sdc_runtime_evidence.json"),
        "metadata_sha256": diagnostic.sha256(source_run / "e2_sdc_metadata.json"),
        "architecture": architecture,
    }
    return {
        "lock": lock,
        "data_yaml": data_yaml,
        "dataset_manifest": manifest_path,
        "source_run": source_run,
        "source_args": source_args,
        "source_args_path": args_path,
        "checkpoint": checkpoint,
        "mdg_path": SETUP_ROOT / "e1_mdg_config.json",
        "config_sha256": diagnostic.sha256(config_path),
        "source_final_metrics": final_metrics,
    }


def finalize_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "diagnostic_summary.json"
    summary = load_json(summary_path)
    gate = summary["e3_object_aware_gate"]
    gate["decision"] = (
        "foreground_gate_passed_stop_for_user_review"
        if gate.get("passed") is True
        else "foreground_gate_failed_do_not_implement_object_aware_or_start_e3"
    )
    summary.update(
        schema="ring3.cbam_focal_region_cosine_diagnostic_summary.v1",
        paper_result_eligible=False,
        interaction_pilot=_PREFLIGHT_EVIDENCE,
        diagnostic_wrapper_sha256=diagnostic.sha256(Path(__file__).resolve()),
        diagnostic_template_sha256=diagnostic.sha256(TEMPLATE_PATH),
        object_aware_consistency_started=False,
        e3_formal_started=False,
        e4_formal_started=False,
        next_action_requires_user_review=True,
    )
    diagnostic.write_json(summary_path, summary)
    diagnostic.write_json(
        DIAGNOSTIC_STATE_PATH,
        {
            "schema": "ring3.cbam_focal_region_diagnostic_state.v1",
            "status": "completed",
            "completed_utc": diagnostic.utc_now(),
            "run_dir": str(run_dir),
            "summary": str(summary_path),
            "gate": gate,
            "object_aware_consistency_started": False,
            "e3_formal_started": False,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gated CBAM/Focal-EIoU lambda-zero regional P3 diagnostic")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = diagnostic.self_test()
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        config, _state = build_effective_config()
        run_dir = Path(str(config["output_root"])) / RUN_NAME
        if run_dir.exists():
            summary_path = run_dir / "diagnostic_summary.json"
            if summary_path.is_file() and load_json(summary_path).get("status") == "completed":
                print(json.dumps(load_json(summary_path), ensure_ascii=False, indent=2))
                return 0
            raise RuntimeError(f"Incomplete diagnostic directory already exists: {run_dir}")

        original_preflight = diagnostic.preflight
        diagnostic.preflight = pilot_preflight
        try:
            run_dir = diagnostic.run_diagnostic(
                EFFECTIVE_CONFIG_PATH,
                requested_batches=None,
                run_name=RUN_NAME,
            )
        finally:
            diagnostic.preflight = original_preflight
        summary = finalize_summary(run_dir)
        print(json.dumps({"summary": str(run_dir / "diagnostic_summary.json"), "gate": summary["e3_object_aware_gate"]}, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        diagnostic.write_json(
            DIAGNOSTIC_STATE_PATH,
            {
                "schema": "ring3.cbam_focal_region_diagnostic_state.v1",
                "status": "failed",
                "failed_utc": diagnostic.utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "object_aware_consistency_started": False,
                "e3_formal_started": False,
            },
        )
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
