#!/usr/bin/env python3
"""Locked WSODD E1-Mid with image-ID deterministic MDG for seeds 42/123/3407.

Only MDG assignment sampling changes: branch, severity, and numeric parameters
are derived from image_id and stay fixed across epochs, batch order, and seeds.
Every run starts fresh from the same locked pretrained YOLOv8s checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO

import pretrained_sgd_mainline_entry as mainline_entry
import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_alpha_ablation_v2 as completion_v2
import train_pretrained_sgd_mainline as impl
import train_wsodd_pretrained_sgd_mainline as wsodd
import train_wsodd_sequence_e1_mdg_strength_ablation as dose_runner
from cbam_focal_formal_models import FactorialDetectionModel
from image_id_deterministic_mdg import (
    ASSIGNMENT_SCHEMA,
    ImageIdDeterministicMaritimeDegradation,
    canonical_image_id,
    summarize_assignments,
)


CONFIG_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_config.v1"
MANIFEST_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_release.v1"
SUPERVISOR_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_supervisor.v1"
NORMAL_COMPLETION_SCHEMA = "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_completion.v1"
COMPLETION_EVIDENCE_NAME = "wsodd_sequence_e1_deterministic_seeds42_123_3407_completion_evidence.json"

EXPECTED_WEIGHT_SHA256 = "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"
EXPECTED_LOCK_SHA256 = "9ef3f68ef0791f198d1ed894477234a76e5e5a98113aa4a44d10b4c03c4ed3e1"
EXPECTED_AMENDMENT_SHA256 = "c4c6c7f6da5c27b382eed1c65d167106a8681d8d06e627738598ea0944b55d0e"
EXPECTED_DATA_YAML_SHA256 = "180965bbca7a5416d467b0d089965db2c8fc22751843795df295d1376c62907e"
EXPECTED_DATA_MANIFEST_SHA256 = "a2edac79e4b7782d6b39d3d91bf22bfa30469811c095310e5f258714fe9869a8"
EXPECTED_ANNOTATION_AUDIT_SHA256 = "e53493e4a791595e7e7f18f53a5614f53106f09e6ad61a60b4dd23fb1b2d0efd"
EXPECTED_MDG_SHA256 = "41e66e44295fe7dce43fa9f76c5deecae0f75c521abcf60f3d96331bc152b40a"
EXPECTED_TRAIN_MANIFEST_SHA256 = "00c1052014c5bbb96488d419db4da08caa676bbfbb4f54ddfcf602305aeebae1"
EXPECTED_SPLIT_COUNTS = {"train": 3977, "val": 1084, "test": 2406}
EXPECTED_CLASSES = 14
EXPECTED_RUNS = [
    ("e1_deterministic_seed42", 42),
    ("e1_deterministic_seed123", 123),
    ("e1_deterministic_seed3407", 3407),
]
BASE_TRAINER = dose_runner.BASE_TRAINER


class DeterministicWSODDTrainer(BASE_TRAINER):
    """Base E1 trainer with only the MDG assignment mechanism replaced."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.mdg_mode != "single_view":
            raise RuntimeError(f"Deterministic E1 requires single_view MDG: {self.variant}")
        mdg_path = Path(os.environ["RING3_SGD_MAINLINE_MDG_CONFIG"])
        namespace = str(self.variant["mdg_assignment_namespace"])
        self.mdg = ImageIdDeterministicMaritimeDegradation(
            impl.load_mdg_config(mdg_path),
            namespace=namespace,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        raw_ids = batch.get("im_file")
        if not isinstance(raw_ids, (list, tuple)):
            raise RuntimeError(
                f"Deterministic MDG requires batch im_file values, got {type(raw_ids).__name__}"
            )
        image_count = int(batch["img"].shape[0])
        if len(raw_ids) != image_count:
            raise RuntimeError(
                f"Batch image_id count mismatch before MDG: ids={len(raw_ids)}, images={image_count}"
            )
        self.mdg.bind_batch_image_ids(raw_ids)
        return super().preprocess_batch(batch)

    def runtime_evidence(self) -> dict[str, Any]:
        evidence = super().runtime_evidence()
        evidence.update(self.mdg.runtime_evidence())
        evidence["deterministic_assignment_map_sha256"] = os.environ.get(
            "RING3_DETERMINISTIC_ASSIGNMENT_MAP_SHA256"
        )
        return evidence


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected WSODD deterministic E1 config schema/status")
    runs = config.get("run_order", [])
    observed = [(run.get("key"), run.get("seed")) for run in runs]
    if observed != EXPECTED_RUNS or len({str(run.get("name")) for run in runs}) != 3:
        raise ValueError(f"Unexpected deterministic E1 seed matrix: {observed!r}")
    for run in runs:
        expected = {
            "mdg_scale": 1.0,
            "mdg_config": "/root/autodl-tmp/ring3_setup/autodl/e1_mdg_config.json",
            "mdg_config_sha256": EXPECTED_MDG_SHA256,
            "mdg_mode": "single_view",
            "mdg_assignment_mode": "deterministic_by_image_id",
            "mdg_assignment_schema": ASSIGNMENT_SCHEMA,
            "mdg_assignment_namespace": config["deterministic_assignment"]["namespace"],
            "mdg_image_id_source": "filename_stem",
            "dual_view": False,
            "degraded_small_weighting": False,
            "degraded_small_target_weight": 1.0,
            "cbam": False,
            "focal_eiou": False,
            "consistency_loss": False,
        }
        mismatches = {key: {"expected": value, "observed": run.get(key)} for key, value in expected.items() if run.get(key) != value}
        if mismatches:
            raise ValueError(f"Deterministic E1 definition changed for {run.get('key')}: {mismatches}")
    policy = config.get("seed_policy", {})
    if (
        policy.get("aggregate_seeds") != [42, 123, 3407]
        or policy.get("train_new_seeds") != [42, 123, 3407]
        or policy.get("reuse_completed_seeds") != []
    ):
        raise ValueError("Unexpected three-seed policy")
    if int(config.get("training", {}).get("seed", -1)) != 0:
        raise ValueError("The locked training recipe must retain seed 0 as its anchor")
    assignment = config.get("deterministic_assignment", {})
    required_assignment = {
        "schema": ASSIGNMENT_SCHEMA,
        "namespace": config["run_order"][0]["mdg_assignment_namespace"],
        "image_id_source": "filename_stem",
        "locked_image_shape": [960, 960],
        "expected_assignment_map_sha256": "bacbc2b44273e4a8c093b1de395be15a706caf035baa5495361d297e9c998b21",
        "independent_of_epoch": True,
        "independent_of_batch_order": True,
        "independent_of_training_seed": True,
    }
    mismatches = {
        key: {"expected": value, "observed": assignment.get(key)}
        for key, value in required_assignment.items()
        if assignment.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Deterministic assignment contract changed: {mismatches}")
    if any(value is not True for value in config.get("invariants", {}).values()):
        raise ValueError("Deterministic E1 invariants are incomplete")


def _verified_file(path: Path, expected: str, label: str) -> dict[str, str]:
    observed = impl.sha256(path) if path.is_file() else "missing"
    if observed != expected:
        raise RuntimeError(f"Locked anchor changed for {label}: expected={expected}, observed={observed}")
    return {"path": str(path), "sha256": observed}


def verify_reused_anchors(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["reused_anchors"]
    checks = {
        "source_mainline_config": (Path(anchors["source_mainline_config"]), anchors["source_mainline_config_sha256"]),
        "source_mainline_release_manifest": (
            Path(anchors["source_mainline_release_manifest"]),
            anchors["source_mainline_release_manifest_sha256"],
        ),
        "baseline_mdg_config": (Path(anchors["baseline_mdg_config"]), anchors["baseline_mdg_config_sha256"]),
        "fixed_test_summary": (Path(anchors["fixed_test_summary"]), anchors["fixed_test_summary_sha256"]),
        "E0_multiseed_paired_summary": (
            Path(anchors["E0-multiseed-paired-summary"]["path"]),
            anchors["E0-multiseed-paired-summary"]["sha256"],
        ),
        "random_E1_Mid_multiseed_summary": (
            Path(anchors["random-E1-Mid-multiseed-summary"]["path"]),
            anchors["random-E1-Mid-multiseed-summary"]["sha256"],
        ),
    }
    verified = {name: _verified_file(path, expected, name) for name, (path, expected) in checks.items()}
    source = impl.load_json(Path(anchors["source_mainline_config"]))
    if config["training"] != source["training"]:
        raise RuntimeError("The non-seed training recipe differs from the completed fixed-split mainline")
    if config["dataset"]["yaml_sha256"] != source["dataset"]["yaml_sha256"]:
        raise RuntimeError("Dataset YAML differs from the completed fixed-split mainline")
    return verified


def seed_set_sha256(runs: list[dict[str, Any]]) -> str:
    payload = {
        str(run["key"]): {
            "seed": int(run["seed"]),
            "mdg": str(run["mdg_config_sha256"]),
            "assignment_schema": str(run["mdg_assignment_schema"]),
            "assignment_namespace": str(run["mdg_assignment_namespace"]),
        }
        for run in runs
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def deterministic_assignment_report(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["dataset"]["train_manifest"])
    if not manifest_path.is_file() or impl.sha256(manifest_path) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("Locked WSODD train manifest is missing or changed")
    image_sources = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_ids = [canonical_image_id(value) for value in image_sources]
    expected = config["dataset"]["image_id_uniqueness"]
    if (
        len(image_ids) != int(expected["total"])
        or len(set(image_ids)) != int(expected["unique"])
        or len(image_ids) - len(set(image_ids)) != int(expected["duplicates"])
    ):
        raise RuntimeError("WSODD filename stems do not satisfy the locked image_id uniqueness audit")
    transform = ImageIdDeterministicMaritimeDegradation(
        impl.load_mdg_config(impl.MDG_CONFIG_PATH),
        namespace=config["deterministic_assignment"]["namespace"],
    )
    shape = config["deterministic_assignment"]["locked_image_shape"]
    first = summarize_assignments(
        transform,
        image_ids,
        height=int(shape[0]),
        width=int(shape[1]),
    )
    second = summarize_assignments(
        transform,
        reversed(image_ids),
        height=int(shape[0]),
        width=int(shape[1]),
    )
    if first != second:
        raise RuntimeError("Deterministic assignment map changed with manifest ordering")
    expected_assignment = config["deterministic_assignment"]
    if first["assignment_map_sha256"] != expected_assignment["expected_assignment_map_sha256"]:
        raise RuntimeError(
            "Deterministic assignment map differs from the locked release: "
            f"{first['assignment_map_sha256']}"
        )
    if first["counts"] != expected_assignment["expected_counts"]:
        raise RuntimeError(
            f"Deterministic assignment counts differ from the locked release: {first['counts']}"
        )
    branch_total = sum(
        int(first["counts"].get(f"branch_{branch}", 0))
        for branch in ("clean", "low_light", "glare", "mixed", "standard_aug_only")
    )
    active_total = sum(
        int(first["counts"].get(f"branch_{branch}", 0))
        for branch in ("low_light", "glare", "mixed")
    )
    if branch_total != len(image_ids) or active_total <= 0:
        raise RuntimeError(f"Deterministic assignment accounting failed: {first}")
    first["train_manifest_path"] = str(manifest_path)
    first["train_manifest_sha256"] = EXPECTED_TRAIN_MANIFEST_SHA256
    first["repeatability_check"] = "passed"
    first["batch_order_independence_check"] = "passed"
    first["same_map_for_seeds"] = [42, 123, 3407]
    return first


def preflight() -> dict[str, Any]:
    if impl.STOP_SENTINEL.exists():
        raise RuntimeError(f"Stop sentinel exists: {impl.STOP_SENTINEL}")
    config = impl.load_json(impl.CONFIG_PATH)
    validate_config(config)
    release = impl.verify_release_manifest()
    if impl.sha256(impl.HISTORICAL_LOCK_PATH) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Historical formal lock changed")
    if impl.sha256(impl.AMENDMENT_PATH) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("Runtime amendment changed")
    if ultralytics.__version__ != "8.4.124" or torch.__version__ != "2.5.1+cu124":
        raise RuntimeError(f"Locked runtime changed: ultralytics={ultralytics.__version__}, torch={torch.__version__}")
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Locked RTX 4090 runtime is unavailable")
    weight_path = Path(config["weights"]["path"])
    if not weight_path.is_file() or impl.sha256(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Locked yolov8s.pt is missing or changed")
    if impl.sha256(impl.MDG_CONFIG_PATH) != EXPECTED_MDG_SHA256:
        raise RuntimeError("Selected E1-Mid MDG config changed")
    impl.load_mdg_config(impl.MDG_CONFIG_PATH)
    dataset_report = wsodd.verify_dataset(config)
    anchor_report = verify_reused_anchors(config)
    seed_hash = seed_set_sha256(config["run_order"])
    assignment_report = deterministic_assignment_report(config)
    return {
        "config": config,
        "config_sha256": impl.sha256(impl.CONFIG_PATH),
        "release": release,
        "historical_lock_sha256": EXPECTED_LOCK_SHA256,
        "runtime_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "mdg_config_sha256": EXPECTED_MDG_SHA256,
        "seed_set_sha256": seed_hash,
        "deterministic_assignment_report": assignment_report,
        "dataset_report": dataset_report,
        "reused_anchors": anchor_report,
        "environment": {
            "python": impl.sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


def context_for_variant(context: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(context)
    output["config"]["training"]["seed"] = int(variant["seed"])
    output["active_seed"] = int(variant["seed"])
    output["mdg_config_sha256"] = variant["mdg_config_sha256"]
    return output


def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    dose_runner.validate_runtime_evidence(evidence, variant)
    if evidence.get("variant") != variant:
        raise RuntimeError("Runtime evidence does not contain the exact locked seed variant")
    required = {
        "deterministic_degradation_assignment": True,
        "deterministic_assignment_schema": ASSIGNMENT_SCHEMA,
        "deterministic_assignment_namespace": variant["mdg_assignment_namespace"],
        "deterministic_assignment_key": "WSODD filename stem (image_id)",
        "deterministic_assignment_seed_independent": True,
        "deterministic_assignment_epoch_independent": True,
        "deterministic_assignment_batch_order_independent": True,
        "deterministic_assignment_conflict_count": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": evidence.get(key)}
        for key, value in required.items()
        if evidence.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Deterministic MDG runtime evidence is incomplete: {mismatches}")
    batch_assignments = evidence.get("deterministic_assignment_current_batch", [])
    if len(batch_assignments) != int(evidence["mdg_counts_current_batch"]["images_total"]):
        raise RuntimeError("Runtime assignment evidence does not cover the full current batch")
    map_hash = evidence.get("deterministic_assignment_map_sha256")
    if not isinstance(map_hash, str) or len(map_hash) != 64:
        raise RuntimeError("Runtime evidence lacks the full deterministic assignment-map hash")


def verify_resume(run_dir: Path, variant: dict[str, Any], context: dict[str, Any]) -> None:
    dose_runner.verify_resume(run_dir, variant, context_for_variant(context, variant))


def run_variant_seeded(variant: dict[str, Any], context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Run one replication with its locked seed made explicit everywhere."""
    assignment_map_sha256 = context["deterministic_assignment_report"]["assignment_map_sha256"]
    os.environ["RING3_DETERMINISTIC_ASSIGNMENT_MAP_SHA256"] = assignment_map_sha256
    state.update(
        current_seed=int(variant["seed"]),
        deterministic_assignment_map_sha256=assignment_map_sha256,
        last_progress_utc=impl.utc_now(),
    )
    impl.write_json_atomic(impl.STATE_PATH, state)
    return dose_runner.run_variant_scaled(variant, context_for_variant(context, variant), state)


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_protocol.v1",
        "status": "active_locked",
        "created_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest": context["release"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "seed_set_sha256": context["seed_set_sha256"],
        "deterministic_assignment_report": context["deterministic_assignment_report"],
        "dataset_report": context["dataset_report"],
        "reused_anchors": context["reused_anchors"],
        "environment": context["environment"],
        "run_order": context["config"]["run_order"],
        "training_anchor": context["config"]["training"],
        "seed_policy": context["config"]["seed_policy"],
        "invariants": context["config"]["invariants"],
    }


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_state.v1",
        "status": "preflight_passed",
        "started_utc": now.isoformat(),
        "last_progress_utc": now.isoformat(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "historical_lock_sha256": context["historical_lock_sha256"],
        "runtime_amendment_sha256": context["runtime_amendment_sha256"],
        "mdg_config_sha256": context["mdg_config_sha256"],
        "seed_set_sha256": context["seed_set_sha256"],
        "deterministic_assignment_map_sha256": context["deterministic_assignment_report"]["assignment_map_sha256"],
        "protocol_sha256": impl.sha256(impl.PROTOCOL_PATH),
        "run_order": [run["key"] for run in context["config"]["run_order"]],
        "completed_runs": [],
        "failed_runs": [],
        "attempts": {},
        "current_run": None,
        "current_run_key": None,
        "current_seed": None,
        "current_epoch": 0,
        "target_epochs": 150,
        "expected_total_hours": 6.0,
        "eta_utc": (now + timedelta(hours=6)).isoformat(),
        "post_training_validation_status": "pending",
        "reused_seed0": False,
        "all_three_seeds_fresh": True,
        "e2_e3_e4_started": False,
        "e4_started": False,
    }


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    source = YOLO(context["config"]["weights"]["path"]).model
    source_state = source.state_dict()
    tested: list[dict[str, Any]] = []
    assignment_signatures: list[list[str]] = []
    for variant in context["config"]["run_order"]:
        model = FactorialDetectionModel(source.yaml, ch=3, nc=EXPECTED_CLASSES, verbose=False, use_cbam=False, use_focal_eiou=False)
        model.load(source)
        target_state = model.state_dict()
        compatible = [key for key, tensor in source_state.items() if key in target_state and target_state[key].shape == tensor.shape]
        mismatched = [key for key in compatible if not torch.equal(target_state[key].cpu(), source_state[key].cpu())]
        if mismatched or len(compatible) < 300:
            raise RuntimeError(f"Pretrained transfer failed for {variant['key']}: compatible={len(compatible)}")
        torch.manual_seed(0)
        transform = ImageIdDeterministicMaritimeDegradation(
            impl.load_mdg_config(Path(variant["mdg_config"])),
            namespace=variant["mdg_assignment_namespace"],
        )
        image_ids = [f"self_test_wsodd_{index:04d}" for index in range(128)]
        images = torch.rand((128, 3, 96, 96), device=device)
        original = images.clone()
        degraded, counts = transform.apply_batch(images, image_ids)
        repeated, repeated_counts = transform.apply_batch(original.clone(), image_ids)
        if not torch.equal(degraded, repeated) or counts != repeated_counts:
            raise RuntimeError(f"Deterministic MDG repeatability failed for {variant['key']}")
        signatures = [
            item["assignment_sha256"]
            for item in transform._last_assignments
        ]
        assignment_signatures.append(signatures)
        active = sum(int(counts.get(f"branch_{name}", 0)) for name in ("low_light", "glare", "mixed"))
        if not bool(torch.isfinite(degraded).all()) or active <= 0:
            raise RuntimeError(f"MDG self-test failed for {variant['key']}: {counts}")
        tested.append({
            "variant": variant,
            "seed": int(variant["seed"]),
            "compatible_pretrained_tensors": len(compatible),
            "mismatched_pretrained_tensors": len(mismatched),
            "mdg_counts": counts,
            "active_images": active,
            "mean_absolute_pixel_delta": float((degraded - original).abs().mean()),
            "finite": True,
            "same_image_same_assignment_and_tensor": True,
            "assignment_signatures_sha256": hashlib.sha256(
                json.dumps(signatures, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "deterministic_runtime_evidence": transform.runtime_evidence(),
            "model_evidence": model.factorial_evidence(),
        })
        del model, degraded, repeated, images, original
        torch.cuda.empty_cache()
    if not all(signatures == assignment_signatures[0] for signatures in assignment_signatures[1:]):
        raise RuntimeError("Deterministic assignment map changed across training seeds")
    payload = {
        "schema": "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_self_test.v1",
        "status": "passed",
        "verified_utc": impl.utc_now(),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": EXPECTED_WEIGHT_SHA256,
        "tested_variants": tested,
        "same_assignment_map_across_seeds": True,
        "train_assignment_map": context["deterministic_assignment_report"],
        "e2_e3_e4_started": False,
    }
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def validate_completed_variant(run_dir: Path, variant: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return completion_v2.validate_completed_variant_v2(run_dir, variant, context_for_variant(context, variant))


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
            metrics[variant["key"]] = {"status": "incomplete", "seed": variant["seed"], "run_dir": str(run_dir), "completion": details}
            continue
        validate_completed_variant(run_dir, variant, context)
        metrics[variant["key"]] = {
            "status": "completed",
            "label": variant["label"],
            "seed": variant["seed"],
            "mdg_scale": variant["mdg_scale"],
            "run_dir": str(run_dir),
            "completion_kind": details["completion_kind"],
            "final_epoch": details["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": details["final_row"],
        }
    payload = {
        "schema": "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_training_summary.v1",
        "created_utc": impl.utc_now(),
        "supervisor_schema": SUPERVISOR_SCHEMA,
        "metrics": metrics,
        "reused_seed0": False,
        "all_three_seeds_fresh": True,
        "deterministic_assignment_report": context["deterministic_assignment_report"],
        "post_training_validation_status": "pending",
        "three_seed_aggregation_status": "pending",
        "e2_e3_e4_started": False,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def configure_runtime() -> None:
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT

    dose_runner.CONFIG_SCHEMA = CONFIG_SCHEMA
    dose_runner.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    dose_runner.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    dose_runner.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    dose_runner.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    dose_runner._context_for_variant = context_for_variant

    impl.CONFIG_SCHEMA = CONFIG_SCHEMA
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.CONFIG_PATH = setup_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_config.json"
    impl.MANIFEST_PATH = setup_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_release_manifest.json"
    impl.MODEL_MODULE_PATH = setup_root / "train_wsodd_sequence_e1_deterministic_seeds42_123_3407.py"
    impl.MDG_CONFIG_PATH = setup_root / "e1_mdg_config.json"
    impl.OUTPUT_ROOT = data_root / "formal_wsodd_sequence_e1_deterministic_seeds42_123_3407"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "wsodd_sequence_e1_deterministic_seeds42_123_3407"
    impl.STATE_PATH = state_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_state.json"
    impl.EVENTS_PATH = state_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_protocol.json"
    impl.SUMMARY_PATH = state_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_training_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "wsodd_sequence_e1_deterministic_seeds42_123_3407_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_WSODD_SEQUENCE_E1_DETERMINISTIC_SEEDS42_123_3407"
    impl.RUNTIME_EVIDENCE_NAME = "wsodd_sequence_e1_deterministic_seeds42_123_3407_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "wsodd_sequence_e1_deterministic_seeds42_123_3407_run_metadata.json"
    impl.EXPECTED_HISTORICAL_LOCK_SHA256 = EXPECTED_LOCK_SHA256
    impl.EXPECTED_AMENDMENT_SHA256 = EXPECTED_AMENDMENT_SHA256
    impl.EXPECTED_WEIGHT_SHA256 = EXPECTED_WEIGHT_SHA256
    impl.EXPECTED_DATA_YAML_SHA256 = EXPECTED_DATA_YAML_SHA256
    impl.EXPECTED_DATA_MANIFEST_SHA256 = EXPECTED_DATA_MANIFEST_SHA256
    impl.EXPECTED_MDG_SHA256 = EXPECTED_MDG_SHA256
    impl.validate_config = validate_config
    impl.preflight = preflight
    impl.verify_dataset = wsodd.verify_dataset
    impl.validate_runtime_evidence = validate_runtime_evidence
    impl.verify_resume = verify_resume
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = static_self_test
    impl.summary = summary
    impl.MainlineSGDTrainer = DeterministicWSODDTrainer
    impl.verify_args = mainline_entry.verify_args_with_serialized_device

    completion_v2.SUPERVISOR_SCHEMA = SUPERVISOR_SCHEMA
    completion_v2.NORMAL_COMPLETION_SCHEMA = NORMAL_COMPLETION_SCHEMA
    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "wsodd_sequence_e1_deterministic_seeds42_123_3407_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = completion_v2.guarded_original_add_callbacks
    supervisor.add_callbacks = completion_v2.add_callbacks_v2
    supervisor.run_completion = completion_v2.run_completion_v2
    supervisor.completed_run = completion_v2.completed_run_v2
    supervisor.validate_completed_variant = validate_completed_variant
    supervisor.summary = summary
    supervisor.install()
    supervisor.run_variant = run_variant_seeded
    supervisor.validate_completed_variant = validate_completed_variant
    supervisor.summary = summary
    impl.run_variant = run_variant_seeded
    impl.summary = summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked WSODD deterministic-image-id E1-Mid seeds 42/123/3407")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = preflight()
        print("WSODD_E1_DETERMINISTIC_SEEDS42_123_3407_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        guard = completion_v2.guard_self_test()
        print("WSODD_E1_DETERMINISTIC_SEEDS42_123_3407_EARLY_STOP_GUARD " + json.dumps(guard, sort_keys=True), flush=True)
        result = static_self_test(context)
        print("WSODD_E1_DETERMINISTIC_SEEDS42_123_3407_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        failure = {
            "schema": "ring3.wsodd_sequence_e1_deterministic_seeds42_123_3407_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "e2_e3_e4_started": False,
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


