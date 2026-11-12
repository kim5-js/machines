from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

import pretrained_sgd_mainline_entry as mainline_entry
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation as alpha
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
from cbam_focal_formal_models import FactorialDetectionModel


CONFIG_SCHEMA = "ring3.wsodd_pretrained_sgd_mainline_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_pretrained_sgd_mainline_release_manifest.v1"
SUPERVISOR_SCHEMA = "ring3.wsodd_pretrained_sgd_mainline_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.wsodd_pretrained_sgd_completion_evidence.v1"
COMPLETION_EVIDENCE_NAME = "wsodd_pretrained_sgd_completion_evidence.json"

EXPECTED_DATA_YAML_SHA256 = "180965bbca7a5416d467b0d089965db2c8fc22751843795df295d1376c62907e"
EXPECTED_DATA_MANIFEST_SHA256 = "a2edac79e4b7782d6b39d3d91bf22bfa30469811c095310e5f258714fe9869a8"
EXPECTED_ANNOTATION_AUDIT_SHA256 = "e53493e4a791595e7e7f18f53a5614f53106f09e6ad61a60b4dd23fb1b2d0efd"
EXPECTED_ALPHA = 1.2
EXPECTED_SPLIT_COUNTS = {"train": 3977, "val": 1084, "test": 2406}
EXPECTED_CLASSES = 14
EXPECTED_OBJECTS = 21911

BASE_TRAINER = impl.MainlineSGDTrainer
BASE_VALIDATE_CONFIG = impl.validate_config
BASE_VALIDATE_RUNTIME_EVIDENCE = impl.validate_runtime_evidence


class WSODDMainlineTrainer(BASE_TRAINER):
    """Final E0-E3 trainer on WSODD; only E3 changes the released weight to alpha=1.2."""

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        if not self.small_weighting:
            return super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        locked = float(self.variant.get("degraded_small_target_weight", -1.0))
        if not math.isclose(locked, EXPECTED_ALPHA, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"WSODD E3 alpha changed: {locked!r}")
        model = alpha.AlphaAblationDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and impl.RANK == -1,
            degraded_small_weight=EXPECTED_ALPHA,
        )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        if not self.small_weighting:
            return batch
        weights = batch.get("e3_gt_weights")
        if not isinstance(weights, torch.Tensor):
            raise RuntimeError("WSODD E3 did not receive released target weights")
        batch["e3_gt_weights"] = torch.where(
            weights > 1.0,
            torch.full_like(weights, EXPECTED_ALPHA),
            torch.ones_like(weights),
        )
        return batch


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected WSODD pretrained-SGD config schema/status")
    runs = config.get("run_order", [])
    observed = [
        (
            run.get("key"),
            run.get("mdg_mode"),
            run.get("dual_view"),
            run.get("degraded_small_weighting"),
            run.get("degraded_small_target_weight"),
            run.get("cbam"),
            run.get("focal_eiou"),
            run.get("consistency_loss"),
        )
        for run in runs
    ]
    expected = [
        ("e0", "none", False, False, 1.0, False, False, False),
        ("e1", "single_view", False, False, 1.0, False, False, False),
        ("e2", "dual_view", True, False, 1.0, False, False, False),
        ("e3", "dual_view", True, True, EXPECTED_ALPHA, False, False, False),
    ]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 4:
        raise ValueError(f"Unexpected WSODD E0-E3 run matrix: {observed!r}")

    proxy = copy.deepcopy(config)
    proxy["schema"] = "ring3.pretrained_sgd_mainline_config.v1"
    for run in proxy["run_order"]:
        run.pop("degraded_small_target_weight", None)
    BASE_VALIDATE_CONFIG(proxy)

    dataset = config.get("dataset", {})
    expected_dataset = {
        "yaml_sha256": EXPECTED_DATA_YAML_SHA256,
        "manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "annotation_semantics_audit_sha256": EXPECTED_ANNOTATION_AUDIT_SHA256,
        "protocol": "wsodd_sequence_disjoint_v1",
        "train_images": 3977,
        "val_images": 1084,
        "test_images": 2406,
        "classes": EXPECTED_CLASSES,
        "sequence_groups": 147,
    }
    mismatches = {
        key: {"expected": value, "observed": dataset.get(key)}
        for key, value in expected_dataset.items()
        if dataset.get(key) != value
    }
    if dataset.get("group_overlap") != {"train-val": 0, "train-test": 0, "val-test": 0}:
        mismatches["group_overlap"] = dataset.get("group_overlap")
    weighting = config.get("small_object_weighting", {})
    if weighting.get("ordinary_target_weight") != 1.0 or weighting.get("degraded_small_target_weight") != EXPECTED_ALPHA:
        mismatches["small_object_weighting"] = weighting
    validation = config.get("post_training_validation", {})
    if validation.get("fixed_split") != "test" or validation.get("fixed_images") != 2406:
        mismatches["post_training_validation"] = validation
    if mismatches:
        raise ValueError(f"WSODD locked protocol changed: {mismatches}")


def verify_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    yaml_path = Path(dataset["yaml"])
    protocol_path = Path(dataset["manifest"])
    audit_path = Path(dataset["annotation_semantics_audit"])
    expected_assets = {
        yaml_path: EXPECTED_DATA_YAML_SHA256,
        protocol_path: EXPECTED_DATA_MANIFEST_SHA256,
        audit_path: EXPECTED_ANNOTATION_AUDIT_SHA256,
    }
    for path, expected_hash in expected_assets.items():
        if not path.is_file() or impl.sha256(path) != expected_hash:
            raise RuntimeError(f"Locked WSODD asset is missing or changed: {path}")

    process = subprocess.run(
        [sys.executable, str(impl.SETUP_ROOT / "verify_yolo_dataset.py"), str(yaml_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError("WSODD verification failed:\n" + process.stdout + process.stderr)
    report = json.loads(process.stdout)
    if report.get("label_error_count") or any(report.get("missing_label_counts", {}).values()):
        raise RuntimeError(f"WSODD verification reported label errors: {report}")
    if any(report.get("counts", {}).get(split) != count for split, count in EXPECTED_SPLIT_COUNTS.items()):
        raise RuntimeError(f"WSODD split counts changed: {report.get('counts')}")

    protocol = impl.load_json(protocol_path)
    if protocol.get("protocol") != "wsodd_sequence_disjoint_v1":
        raise RuntimeError("WSODD protocol name changed")
    if protocol.get("split_images") != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"WSODD protocol split counts changed: {protocol.get('split_images')}")
    if protocol.get("group_overlap") != {"train-val": 0, "train-test": 0, "val-test": 0}:
        raise RuntimeError(f"WSODD sequence groups overlap: {protocol.get('group_overlap')}")
    manifest_checks: dict[str, Any] = {}
    for split, expected in protocol.get("manifests", {}).items():
        path = protocol_path.parent / str(expected["path"])
        observed_lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        observed_hash = impl.sha256(path)
        if observed_hash != expected.get("sha256") or observed_lines != int(expected.get("lines", -1)):
            raise RuntimeError(f"WSODD {split} manifest changed")
        manifest_checks[split] = {"path": str(path), "sha256": observed_hash, "lines": observed_lines}

    audit = impl.load_json(audit_path)
    conclusion = audit.get("conclusion", {})
    if (
        audit.get("status") != "passed"
        or int(audit.get("objects", -1)) != EXPECTED_OBJECTS
        or audit.get("difficult_values") != {"0": EXPECTED_OBJECTS}
        or audit.get("ignore_like_tags") != {}
        or conclusion.get("dedicated_ignore_or_crowd_semantics_present") is not False
        or conclusion.get("seadronessee_ignore_rule_reused") is not False
    ):
        raise RuntimeError(f"WSODD annotation semantics audit failed: {audit}")
    protocol_instances = sum(
        int(count)
        for split_counts in protocol.get("split_class_instances", {}).values()
        for count in split_counts.values()
    )
    if protocol_instances != EXPECTED_OBJECTS:
        raise RuntimeError(f"WSODD object accounting changed: {protocol_instances}")
    report["sequence_protocol"] = {
        "sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "sequence_groups": protocol.get("sequence_groups"),
        "group_overlap": protocol.get("group_overlap"),
        "manifests": manifest_checks,
    }
    report["annotation_semantics"] = audit
    return report


def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    if variant["degraded_small_weighting"]:
        alpha.validate_runtime_evidence(evidence, variant)
        return
    BASE_VALIDATE_RUNTIME_EVIDENCE(evidence, variant)


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.wsodd_pretrained_sgd_mainline_protocol.v1",
        "status": "active_locked",
        "created_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "dataset_report": context["dataset_report"],
        "environment": context["environment"],
        "run_order": context["config"]["run_order"],
        "training": context["config"]["training"],
        "post_training_validation": context["config"]["post_training_validation"],
        "invariants": context["config"]["invariants"],
    }


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.wsodd_pretrained_sgd_mainline_state.v1",
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "protocol_sha256": impl.sha256(impl.PROTOCOL_PATH),
        "run_order": [run["key"] for run in context["config"]["run_order"]],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": 14.0,
        "eta_utc": (now + timedelta(hours=14)).isoformat(),
        "post_training_validation_status": "pending",
        "seadronessee_runs_reused": False,
        "e4_started": False,
    }


def _synthetic_runtime(variant: dict[str, Any], model: Any) -> dict[str, Any]:
    targets: dict[str, int]
    mdg: dict[str, int]
    if variant["mdg_mode"] == "none":
        mdg = {}
        targets = {"images_total": 2, "clean_targets": 2}
    elif variant["mdg_mode"] == "single_view":
        mdg = {"images_total": 2}
        targets = {"images_total": 2, "degraded_targets": 2}
    else:
        mdg = {"images_total": 2}
        targets = {"images_total": 2, "clean_targets": 2, "degraded_targets": 2}
    return {
        "variant": variant,
        **model.factorial_evidence(),
        "mdg_mode": variant["mdg_mode"],
        "dual_view_enabled": variant["dual_view"],
        "degraded_small_weighting_enabled": variant["degraded_small_weighting"],
        "mdg_counts_current_batch": mdg,
        "target_counts_current_batch": targets,
        "optimizer_update_probe": {"finite": True, "parameter_delta_l2": 1.0},
        "consistency_loss_enabled": False,
        "severity_weighting_enabled": False,
        "curriculum_enabled": False,
    }


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = YOLO(str(context["config"]["weights"]["path"])).model
    source_state = source.state_dict()
    tested: list[dict[str, Any]] = []
    for variant in context["config"]["run_order"]:
        if variant["degraded_small_weighting"]:
            model = alpha.AlphaAblationDetectionModel(
                source.yaml,
                ch=3,
                nc=EXPECTED_CLASSES,
                verbose=False,
                degraded_small_weight=EXPECTED_ALPHA,
            )
        else:
            model = FactorialDetectionModel(
                source.yaml,
                ch=3,
                nc=EXPECTED_CLASSES,
                verbose=False,
                use_cbam=False,
                use_focal_eiou=False,
            )
        model.load(source)
        target_state = model.state_dict()
        compatible = [key for key, tensor in source_state.items() if key in target_state and target_state[key].shape == tensor.shape]
        mismatched = [key for key in compatible if not torch.equal(target_state[key].cpu(), source_state[key].cpu())]
        if mismatched or len(compatible) < 300:
            raise RuntimeError(
                f"Pretrained transfer failed for {variant['key']}: compatible={len(compatible)}, mismatched={mismatched[:5]}"
            )
        model.to(device).train()
        model.args = impl.get_cfg(impl.DEFAULT_CFG_DICT)
        model.criterion = model.init_criterion()
        item: dict[str, Any] = {
            "variant": variant,
            "compatible_pretrained_tensors": len(compatible),
            "mismatched_pretrained_tensors": len(mismatched),
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
                "e3_gt_weights": torch.tensor([1.0, 1.0, EXPECTED_ALPHA, EXPECTED_ALPHA], device=device),
                "e3_pair_batch_size": 2,
            }
            loss, loss_items = model.loss(batch)
            total = loss.sum()
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Static WSODD E3 self-test produced a non-finite loss")
            total.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError("Static WSODD E3 gradient check failed")
            item["e3_loss_items"] = {key: float(value) for key, value in loss_items.items()}
        synthetic = _synthetic_runtime(variant, model)
        validate_runtime_evidence(synthetic, variant)
        item["model_evidence"] = model.factorial_evidence()
        tested.append(item)
        del model
        torch.cuda.empty_cache()

    mdg = impl.MaritimeDegradation(impl.load_mdg_config(impl.MDG_CONFIG_PATH))
    probe = torch.rand((2, 3, 160, 160), device=device)
    degraded, counts = mdg.apply_batch(probe.clone())
    if not bool(torch.isfinite(degraded).all()) or int(counts.get("images_total", -1)) != 2:
        raise RuntimeError("Static WSODD MDG self-test failed")
    payload = {
        "schema": "ring3.wsodd_pretrained_sgd_mainline_self_test.v1",
        "status": "passed",
        "verified_utc": impl.utc_now(),
        "device": str(device),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": impl.EXPECTED_WEIGHT_SHA256,
        "tested_variants": tested,
        "mdg_probe_counts": counts,
        "mdg_probe_finite": True,
        "e3_alpha": EXPECTED_ALPHA,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        details = supervisor.run_completion(
            run_dir,
            epochs=int(context["config"]["training"]["epochs"]),
            patience=int(context["config"]["training"]["patience"]),
        )
        if details.get("complete") is not True:
            metrics[variant["key"]] = {"status": "incomplete", "run_dir": str(run_dir), "completion": details}
            continue
        supervisor.validate_completed_variant(run_dir, variant, context)
        metrics[variant["key"]] = {
            "status": "completed",
            "run_dir": str(run_dir),
            "completion_kind": details["completion_kind"],
            "final_epoch": details["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": details["final_row"],
        }
    payload: dict[str, Any] = {
        "schema": "ring3.wsodd_pretrained_sgd_mainline_summary.v1",
        "created_utc": impl.utc_now(),
        "supervisor_schema": SUPERVISOR_SCHEMA,
        "single_seed_mechanism_evidence_only": True,
        "metrics": metrics,
        "post_training_validation_status": "pending",
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
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def configure_runtime() -> None:
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT
    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "wsodd_pretrained_sgd_mainline_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_pretrained_sgd_mainline_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_pretrained_sgd_mainline.py"
    impl.OUTPUT_ROOT = data_root / "formal_wsodd_pretrained_sgd_mainline"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_pretrained_sgd_mainline"
    impl.STATE_PATH = state_root / "wsodd_pretrained_sgd_mainline_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_pretrained_sgd_mainline_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_pretrained_sgd_mainline_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_pretrained_sgd_mainline_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_pretrained_sgd_mainline_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_PRETRAINED_SGD_MAINLINE"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_pretrained_sgd_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_pretrained_sgd_run_metadata.json"
    impl.EXPECTED_DATA_YAML_SHA256 = EXPECTED_DATA_YAML_SHA256
    impl.EXPECTED_DATA_MANIFEST_SHA256 = EXPECTED_DATA_MANIFEST_SHA256
    impl.validate_config = validate_config
    impl.verify_dataset = verify_dataset
    impl.validate_runtime_evidence = validate_runtime_evidence
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = static_self_test
    impl.MainlineSGDTrainer = WSODDMainlineTrainer
    impl.verify_args = mainline_entry.verify_args_with_serialized_device

    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_pretrained_sgd_mainline_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = completion_v2.validate_completed_variant_v2
    supervisor.summary = summary
    supervisor.install()
    supervisor.summary = summary
    impl.summary = summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the locked WSODD pretrained-SGD E0-E3 mainline")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = impl.preflight()
        print("WSODD_PRETRAINED_SGD_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("WSODD_PRETRAINED_SGD_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("WSODD_PRETRAINED_SGD_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.wsodd_pretrained_sgd_mainline_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, failure)
        if impl.STATE_PATH.is_file():
            state = impl.load_json(impl.STATE_PATH)
            state.update(status="supervisor_failed", last_progress_utc=impl.utc_now(), error_type=type(exc).__name__, error=str(exc))
            impl.write_json_atomic(impl.STATE_PATH, state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
