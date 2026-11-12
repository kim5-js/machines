from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

import pretrained_sgd_mainline_supervisor_patch as supervisor
import train_pretrained_sgd_mainline as impl
from pretrained_sgd_mainline_models import MainlineSmallWeightedDetectionModel


CONFIG_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_config.v1"
MANIFEST_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_release_manifest.v1"
SUPERVISOR_SCHEMA = "ring3.pretrained_sgd_alpha_ablation_supervisor.v1"
COMPLETION_EVIDENCE_NAME = "pretrained_sgd_alpha_ablation_completion_evidence.json"
ALLOWED_ALPHAS = {
    "e3_alpha12": 1.2,
    "e3_alpha135": 1.35,
}

BASE_VALIDATE_CONFIG = impl.validate_config
BASE_ADD_CALLBACKS = impl.add_callbacks


# Keeping the released loss base explicit makes it auditable that only the
# constructor guard changes; all loss computation methods are inherited
# unchanged from the released E3 loss.
from e3_small_object_weighting import SmallObjectWeightedDetectionLoss as _ReleasedSmallObjectWeightedDetectionLoss


class AlphaConfigurableSmallObjectWeightedDetectionLoss(_ReleasedSmallObjectWeightedDetectionLoss):
    """Released E3 loss with alpha restricted to the two requested ablation values."""

    def __init__(self, model: torch.nn.Module, *, degraded_small_weight: float) -> None:
        v8DetectionLoss.__init__(self, model)
        self.degraded_small_weight = float(degraded_small_weight)
        if not any(math.isclose(self.degraded_small_weight, value, rel_tol=0.0, abs_tol=1e-12) for value in ALLOWED_ALPHAS.values()):
            raise ValueError(f"Unapproved alpha-ablation weight: {self.degraded_small_weight!r}")
        self._last_evidence: dict[str, Any] = {}


class AlphaAblationDetectionModel(MainlineSmallWeightedDetectionModel):
    """The released E3 model with only its scalar degraded-small weight varied."""

    def init_criterion(self) -> AlphaConfigurableSmallObjectWeightedDetectionLoss:
        return AlphaConfigurableSmallObjectWeightedDetectionLoss(
            self,
            degraded_small_weight=self.degraded_small_weight,
        )


class AlphaAblationTrainer(impl.MainlineSGDTrainer):
    """Existing dual-view E3 trainer parameterized by the locked alpha value."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._last_consumed_epoch_counts: tuple[int, dict[str, int], dict[str, int]] | None = None
        super().__init__(*args, **kwargs)

    @property
    def alpha(self) -> float:
        value = float(self.variant["degraded_small_target_weight"])
        expected = ALLOWED_ALPHAS.get(str(self.variant["key"]))
        if expected is None or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Variant alpha is not locked: {self.variant!r}")
        return value

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        model = AlphaAblationDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and impl.RANK == -1,
            degraded_small_weight=self.alpha,
        )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        weights = batch.get("e3_gt_weights")
        if not isinstance(weights, torch.Tensor):
            raise RuntimeError("Alpha ablation did not receive released E3 target weights")
        batch["e3_gt_weights"] = torch.where(
            weights > 1.0,
            torch.full_like(weights, self.alpha),
            torch.ones_like(weights),
        )
        return batch

    def consume_epoch_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        mdg, targets = super().consume_epoch_counts()
        epoch = int(getattr(self, "epoch", -1))
        if mdg or targets:
            self._last_consumed_epoch_counts = (epoch, dict(mdg), dict(targets))
            return mdg, targets
        # Ultralytics performs a final validation pass after a normal early stop
        # and invokes on_fit_epoch_end again without another training epoch.
        # Reuse the already-audited counts only for that same epoch. This avoids
        # the historical empty-dict bookkeeping failure without hiding a truly
        # empty or skipped training epoch.
        if self._last_consumed_epoch_counts is not None and self._last_consumed_epoch_counts[0] == epoch:
            return dict(self._last_consumed_epoch_counts[1]), dict(self._last_consumed_epoch_counts[2])
        return mdg, targets


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected pretrained-SGD alpha-ablation config schema/status")
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
        ("e3_alpha12", "dual_view", True, True, 1.2, False, False, False),
        ("e3_alpha135", "dual_view", True, True, 1.35, False, False, False),
    ]
    if observed != expected or len({str(run.get("name")) for run in runs}) != 2:
        raise ValueError(f"Unexpected alpha-ablation run matrix: {observed!r}")

    # Reuse the released mainline validator for every training hyperparameter.
    proxy = copy.deepcopy(config)
    proxy["schema"] = "ring3.pretrained_sgd_mainline_config.v1"
    proxy["run_order"] = [
        {"key": "e0", "name": "e0", "mdg_mode": "none", "dual_view": False, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e1", "name": "e1", "mdg_mode": "single_view", "dual_view": False, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e2", "name": "e2", "mdg_mode": "dual_view", "dual_view": True, "degraded_small_weighting": False, "cbam": False, "focal_eiou": False, "consistency_loss": False},
        {"key": "e3", "name": "e3", "mdg_mode": "dual_view", "dual_view": True, "degraded_small_weighting": True, "cbam": False, "focal_eiou": False, "consistency_loss": False},
    ]
    BASE_VALIDATE_CONFIG(proxy)
    if config.get("small_object_weighting", {}).get("new_queue_weights") != [1.2, 1.35]:
        raise ValueError("The requested alpha values changed")


def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    if evidence.get("variant", {}).get("key") != variant["key"]:
        raise RuntimeError("Runtime evidence belongs to a different alpha run")
    if evidence.get("cbam_enabled") is not False or evidence.get("focal_eiou_enabled") is not False:
        raise RuntimeError("CBAM or Focal-EIoU leaked into the alpha ablation")
    if evidence.get("cbam_layer_types") != ["C2f", "C2f"]:
        raise RuntimeError("Unexpected attention topology evidence")
    if int(evidence.get("cbam_attention_parameter_count", -1)) != 0:
        raise RuntimeError("Attention parameters unexpectedly exist")
    if evidence.get("criterion_type") != "AlphaConfigurableSmallObjectWeightedDetectionLoss":
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
    if int(mdg.get("images_total", -1)) != int(targets.get("images_total", -2)):
        raise RuntimeError("MDG image accounting does not close")
    if int(targets.get("clean_targets", -1)) != int(targets.get("degraded_targets", -2)):
        raise RuntimeError("Clean/degraded target counts differ")

    alpha = float(variant["degraded_small_target_weight"])
    required = {
        "ordinary_target_weight": 1.0,
        "normalization_denominator_unchanged": True,
        "background_anchor_weight_unchanged": True,
        "clean_weighted_target_violations": 0,
        "negative_anchor_weight_violations": 0,
    }
    for key, expected in required.items():
        if evidence.get(key) != expected:
            raise RuntimeError(f"Alpha weighting evidence not ready for {key}: {evidence.get(key)!r}")
    if not math.isclose(float(evidence.get("degraded_small_target_weight", -1.0)), alpha, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Runtime degraded-small alpha differs from the locked variant")
    if int(evidence.get("weighted_positive_anchor_count", 0)) <= 0:
        raise RuntimeError("No degraded-small positive anchor has been weighted yet")
    if float(evidence.get("weighting_delta_scaled", 0.0)) <= 0.0:
        raise RuntimeError("Alpha weighting did not increase the detection objective")


def protocol_payload(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ring3.pretrained_sgd_alpha_ablation_protocol.v1",
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
        "existing_references_not_rerun": context["config"]["existing_references_not_rerun"],
        "invariants": context["config"]["invariants"],
    }


def state_template(context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema": "ring3.pretrained_sgd_alpha_ablation_state.v1",
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
        "expected_total_hours": 12.0,
        "eta_utc": (now + timedelta(hours=12)).isoformat(),
        "existing_alpha1_and_alpha15_reused": True,
        "e4_started": False,
    }


def static_self_test(context: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = YOLO(str(context["config"]["weights"]["path"])).model
    source_state = source.state_dict()
    tested: list[dict[str, Any]] = []
    for variant in context["config"]["run_order"]:
        alpha = float(variant["degraded_small_target_weight"])
        model = AlphaAblationDetectionModel(
            source.yaml,
            ch=3,
            nc=5,
            verbose=False,
            degraded_small_weight=alpha,
        )
        model.load(source)
        target_state = model.state_dict()
        compatible = [key for key, tensor in source_state.items() if key in target_state and target_state[key].shape == tensor.shape]
        mismatched = [key for key in compatible if not torch.equal(target_state[key].cpu(), source_state[key].cpu())]
        if mismatched or len(compatible) < 300:
            raise RuntimeError(f"Pretrained transfer failed for {variant['key']}: compatible={len(compatible)}, mismatched={mismatched[:5]}")
        model.to(device).train()
        model.args = impl.get_cfg(impl.DEFAULT_CFG_DICT)
        model.criterion = model.init_criterion()
        torch.manual_seed(0)
        clean = torch.rand((2, 3, 160, 160), device=device)
        degraded = clean.pow(1.15).mul(0.88)
        batch = {
            "img": torch.cat((clean, degraded), dim=0),
            "batch_idx": torch.tensor([0, 1, 2, 3], device=device, dtype=torch.float32),
            "cls": torch.tensor([[0], [1], [0], [1]], device=device, dtype=torch.float32),
            "bboxes": torch.tensor([[0.50, 0.50, 0.10, 0.10], [0.40, 0.40, 0.10, 0.10]] * 2, device=device, dtype=torch.float32),
            "e3_gt_weights": torch.tensor([1.0, 1.0, alpha, alpha], device=device),
            "e3_pair_batch_size": 2,
        }
        loss, loss_items = model.loss(batch)
        total = loss.sum()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Static {variant['key']} self-test produced a non-finite loss")
        total.backward()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        evidence = model.factorial_evidence()
        synthetic_runtime = {
            "variant": variant,
            **evidence,
            "mdg_mode": "dual_view",
            "dual_view_enabled": True,
            "degraded_small_weighting_enabled": True,
            "mdg_counts_current_batch": {"images_total": 2},
            "target_counts_current_batch": {"images_total": 2, "clean_targets": 2, "degraded_targets": 2},
            "optimizer_update_probe": {"finite": True, "parameter_delta_l2": 1.0},
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
        }
        validate_runtime_evidence(synthetic_runtime, variant)
        if not finite_gradients:
            raise RuntimeError(f"Static {variant['key']} gradient check failed")
        tested.append(
            {
                "variant": variant,
                "compatible_pretrained_tensors": len(compatible),
                "mismatched_pretrained_tensors": len(mismatched),
                "loss_items": {key: float(value) for key, value in loss_items.items()},
                "weighting_evidence": evidence,
            }
        )
        del model
        torch.cuda.empty_cache()

    payload = {
        "schema": "ring3.pretrained_sgd_alpha_ablation_self_test.v1",
        "status": "passed",
        "verified_utc": impl.utc_now(),
        "device": str(device),
        "config_sha256": context["config_sha256"],
        "release_manifest_sha256": context["release"]["sha256"],
        "pretrained_source_sha256": impl.EXPECTED_WEIGHT_SHA256,
        "tested_variants": tested,
        "alpha_only_variable": True,
        "e4_started": False,
    }
    impl.write_json_atomic(impl.SELF_TEST_PATH, payload)
    return payload


def alpha_summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for variant in context["config"]["run_order"]:
        run_dir = impl.OUTPUT_ROOT / variant["name"]
        details = supervisor.run_completion(run_dir)
        if details.get("complete") is not True:
            metrics[variant["key"]] = {"status": "incomplete", "alpha": variant["degraded_small_target_weight"], "run_dir": str(run_dir), "completion": details}
            continue
        supervisor.validate_completed_variant(run_dir, variant, context)
        metrics[variant["key"]] = {
            "status": "completed",
            "alpha": variant["degraded_small_target_weight"],
            "run_dir": str(run_dir),
            "completion_kind": details["completion_kind"],
            "final_epoch": details["final_epoch"],
            "best_row": impl.best_row(run_dir / "results.csv"),
            "final_row": details["final_row"],
        }

    payload: dict[str, Any] = {
        "schema": "ring3.pretrained_sgd_alpha_ablation_summary.v1",
        "created_utc": impl.utc_now(),
        "single_seed_mechanism_evidence_only": True,
        "existing_references_not_rerun": context["config"]["existing_references_not_rerun"],
        "new_metrics": metrics,
        "e4_started": False,
    }
    if all(item.get("status") == "completed" for item in metrics.values()):
        fields = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "map50": "metrics/mAP50(B)",
            "map50_95": "metrics/mAP50-95(B)",
        }
        sweep: dict[str, dict[str, float | int]] = copy.deepcopy(context["config"]["existing_references_not_rerun"])
        for variant in context["config"]["run_order"]:
            row = metrics[variant["key"]]["best_row"]
            sweep[f"alpha{variant['degraded_small_target_weight']}"] = {
                "best_epoch": int(float(row["epoch"])),
                **{name: float(row[column]) for name, column in fields.items()},
            }
        baseline = sweep["alpha1.0"]
        payload["full_sweep"] = sweep
        payload["deltas_vs_alpha1.0"] = {
            key: {name: float(item[name]) - float(baseline[name]) for name in fields}
            for key, item in sweep.items()
            if key != "alpha1.0"
        }
    impl.write_json_atomic(impl.SUMMARY_PATH, payload)
    return payload


def configure_runtime() -> None:
    data_root = impl.DATA_ROOT
    setup_root = impl.SETUP_ROOT
    state_root = impl.STATE_ROOT
    impl.CONFIG_PATH = setup_root / "pretrained_sgd_alpha_ablation_config.json"
    impl.MANIFEST_PATH = setup_root / "pretrained_sgd_alpha_ablation_release_manifest.json"
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.MODEL_MODULE_PATH = setup_root / "train_pretrained_sgd_alpha_ablation.py"
    impl.OUTPUT_ROOT = data_root / "formal_pretrained_sgd_alpha_ablation"
    impl.FAILED_ROOT = data_root / "formal_failed_runs" / "pretrained_sgd_alpha_ablation"
    impl.STATE_PATH = state_root / "pretrained_sgd_alpha_ablation_state.json"
    impl.EVENTS_PATH = state_root / "pretrained_sgd_alpha_ablation_events.jsonl"
    impl.PROTOCOL_PATH = state_root / "pretrained_sgd_alpha_ablation_protocol.json"
    impl.SUMMARY_PATH = state_root / "pretrained_sgd_alpha_ablation_summary.json"
    impl.SELF_TEST_PATH = data_root / "diagnostics" / "pretrained_sgd_alpha_ablation_self_test.json"
    impl.STOP_SENTINEL = state_root / "STOP_PRETRAINED_SGD_ALPHA_ABLATION"
    impl.RUNTIME_EVIDENCE_NAME = "pretrained_sgd_alpha_ablation_runtime_evidence.json"
    impl.RUN_METADATA_NAME = "pretrained_sgd_alpha_ablation_run_metadata.json"
    impl.validate_config = validate_config
    impl.validate_runtime_evidence = validate_runtime_evidence
    impl.protocol_payload = protocol_payload
    impl.state_template = state_template
    impl.static_self_test = static_self_test
    impl.MainlineSGDTrainer = AlphaAblationTrainer

    supervisor.PATCH_SCHEMA = SUPERVISOR_SCHEMA
    supervisor.COMPLETION_EVIDENCE_NAME = COMPLETION_EVIDENCE_NAME
    supervisor.RECOVERY_FAILURE_PATH = state_root / "pretrained_sgd_alpha_ablation_failure.json"
    supervisor.ORIGINAL_ADD_CALLBACKS = BASE_ADD_CALLBACKS
    supervisor.install()
    supervisor.summary = alpha_summary
    impl.summary = alpha_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    try:
        context = impl.preflight()
        print("PRETRAINED_SGD_ALPHA_PREFLIGHT " + json.dumps(context, ensure_ascii=False, sort_keys=True), flush=True)
        if args.preflight_only:
            return 0
        result = static_self_test(context)
        print("PRETRAINED_SGD_ALPHA_SELF_TEST " + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if args.self_test_only:
            return 0
        return supervisor.execute_queue(context)
    except BaseException as exc:
        payload = {
            "schema": "ring3.pretrained_sgd_alpha_ablation_failure.v1",
            "status": "supervisor_failed",
            "failed_utc": impl.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "supervisor_schema": SUPERVISOR_SCHEMA,
            "e4_started": False,
        }
        impl.write_json_atomic(supervisor.RECOVERY_FAILURE_PATH, payload)
        impl.append_event("supervisor_failed", error_type=type(exc).__name__, error=str(exc))
        print("PRETRAINED_SGD_ALPHA_FAILURE " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
