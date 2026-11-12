from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from sdc_consistency import P3ConsistencyDetectionModel


CONFIG_SCHEMA = "ring3.e2_p3_consistency_lambda_ablation.v1"
ALLOWED_LAMBDAS = {"lambda0": 0.0, "lambda003": 0.03}


def load_sdc_ablation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_sdc_ablation_config(config)
    return config


def validate_sdc_ablation_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Unexpected lambda-ablation config schema: {config.get('schema')!r}")
    if config.get("stage") != "E2-lambda-ablation":
        raise ValueError(f"Unexpected lambda-ablation stage: {config.get('stage')!r}")
    if config.get("method") != "online_mdg_plus_shared_weight_p3_consistency_lambda_ablation":
        raise ValueError(f"Unexpected lambda-ablation method: {config.get('method')!r}")

    ablation_id = config.get("ablation_id")
    if ablation_id not in ALLOWED_LAMBDAS:
        raise ValueError(f"Unknown lambda-ablation id: {ablation_id!r}")

    dual_view = config.get("dual_view", {})
    required_dual_view_true = (
        "shared_detector_weights",
        "detection_loss_on_clean",
        "detection_loss_on_degraded",
        "validation_uses_locked_original_distribution_only",
        "inference_uses_single_detector_path_only",
    )
    if any(dual_view.get(key) is not True for key in required_dual_view_true):
        raise ValueError(f"Dual-view invariants are incomplete: {dual_view!r}")

    consistency = config.get("consistency", {})
    if consistency.get("feature_scale") != "P3":
        raise ValueError("Lambda ablation must use P3 only")
    if consistency.get("feature_locator") != "first_input_to_detect_head":
        raise ValueError("P3 must be resolved from the first Detect input")
    if consistency.get("loss") != "one_minus_cosine_similarity":
        raise ValueError("Lambda ablation must retain one-minus-cosine consistency")
    weight = float(consistency.get("lambda", float("nan")))
    expected_weight = ALLOWED_LAMBDAS[str(ablation_id)]
    if not math.isfinite(weight) or not math.isclose(weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{ablation_id} must use lambda={expected_weight}, got {weight!r}")
    expected_role = "zero_weight_control" if weight == 0.0 else "lower_weight_consistency"
    if consistency.get("objective_role") != expected_role:
        raise ValueError(f"Unexpected objective role for {ablation_id}: {consistency.get('objective_role')!r}")

    invariants = config.get("invariants", {})
    required_true = (
        "formal_lock_unchanged",
        "runtime_amendment_unchanged",
        "training_hyperparameters_unchanged",
        "dataset_yaml_and_manifest_unchanged",
        "pretrained_weight_unchanged",
        "independent_run_from_same_pretrained_initialization",
        "no_resume_from_lambda_0_1",
        "no_cbam",
        "no_focal_eiou",
        "no_curriculum",
        "p4_p5_consistency_disabled",
    )
    if any(invariants.get(key) is not True for key in required_true):
        raise ValueError(f"Lambda-ablation invariants are incomplete: {invariants!r}")


class LambdaAblationDetectionModel(P3ConsistencyDetectionModel):
    """E2 P3 model that supports both a strict zero control and a lower nonzero lambda."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        consistency_weight: float,
    ) -> None:
        weight = float(consistency_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Invalid lambda-ablation consistency weight: {consistency_weight!r}")
        if not any(math.isclose(weight, allowed, rel_tol=0.0, abs_tol=1e-12) for allowed in ALLOWED_LAMBDAS.values()):
            raise ValueError(f"Unapproved lambda-ablation consistency weight: {consistency_weight!r}")

        DetectionModel.__init__(self, cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.sdc_consistency_weight = weight
        detect = self.model[-1]
        detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
        if detect_inputs != [15, 18, 21]:
            raise RuntimeError(f"Expected locked Detect inputs [15, 18, 21], got {detect_inputs!r}")
        self.sdc_detect_input_indices = detect_inputs
        self.sdc_feature_layer_index = int(detect_inputs[0])
        self.sdc_feature_layer_type = type(self.model[self.sdc_feature_layer_index]).__name__
        if self.sdc_feature_layer_type != "C2f":
            raise RuntimeError(f"Expected P3 target layer 15 to be C2f, got {self.sdc_feature_layer_type!r}")
        self._sdc_capture_enabled = False
        self._sdc_captured_feature: torch.Tensor | None = None
        self._sdc_last_metrics: dict[str, Any] = {}
        self._sdc_gradient_probe: dict[str, Any] = {}
        self.model[self.sdc_feature_layer_index].register_forward_hook(self._capture_p3_feature)

    def _probe_consistency_gradient(self, scaled_consistency: torch.Tensor) -> None:
        if self._sdc_gradient_probe or not self.training:
            return
        parameters = self.sdc_target_parameters()
        gradients = torch.autograd.grad(
            scaled_consistency,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        finite = True
        nonzero_tensors = 0
        squared_norm = torch.zeros((), device=scaled_consistency.device, dtype=torch.float32)
        max_abs = torch.zeros((), device=scaled_consistency.device, dtype=torch.float32)
        for gradient in gradients:
            if gradient is None:
                continue
            detached = gradient.detach().float()
            finite = finite and bool(torch.isfinite(detached).all())
            if bool((detached != 0).any()):
                nonzero_tensors += 1
            squared_norm = squared_norm + detached.square().sum()
            max_abs = torch.maximum(max_abs, detached.abs().max())
        grad_norm = float(squared_norm.sqrt())
        zero_control = math.isclose(self.sdc_consistency_weight, 0.0, rel_tol=0.0, abs_tol=1e-12)
        self._sdc_gradient_probe = {
            "parameter_tensors": len(parameters),
            "gradient_tensors": sum(gradient is not None for gradient in gradients),
            "nonzero_gradient_tensors": nonzero_tensors,
            "finite": finite,
            "l2_norm": grad_norm,
            "max_abs": float(max_abs),
            "expected_behavior": "zero_by_lambda_control" if zero_control else "finite_nonzero_for_enabled_consistency",
        }
        if not finite or not math.isfinite(grad_norm):
            raise FloatingPointError(f"P3 consistency-only gradient probe is non-finite: {self._sdc_gradient_probe}")
        if zero_control:
            if nonzero_tensors != 0 or grad_norm > 1e-12 or float(max_abs) > 1e-12:
                raise FloatingPointError(f"Lambda-zero control leaked a consistency gradient: {self._sdc_gradient_probe}")
        elif len(parameters) != 12 or len(gradients) != 12 or nonzero_tensors != 12 or grad_norm <= 0.0:
            raise FloatingPointError(f"Enabled consistency did not reach all 12 P3 tensors: {self._sdc_gradient_probe}")

    def loss(self, batch: dict[str, Any], preds: Any = None):
        pair_batch_size = batch.get("sdc_pair_batch_size")
        if preds is not None or pair_batch_size is None or not self.training:
            return super(P3ConsistencyDetectionModel, self).loss(batch, preds)
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        pair_batch_size = int(pair_batch_size)
        self._sdc_captured_feature = None
        self._sdc_capture_enabled = True
        try:
            predictions = self.forward(batch["img"])
        finally:
            self._sdc_capture_enabled = False
        feature = self._sdc_captured_feature
        if feature is None:
            raise RuntimeError("P3 forward hook did not fire during lambda-ablation training")

        detection_loss, loss_items = self.criterion(predictions, batch)
        if not bool(torch.isfinite(detection_loss).all()):
            raise FloatingPointError("Dual-view detection loss became non-finite")
        batch_indices = batch["batch_idx"].reshape(-1)
        clean_targets = int((batch_indices < pair_batch_size).sum())
        degraded_targets = int(
            ((batch_indices >= pair_batch_size) & (batch_indices < pair_batch_size * 2)).sum()
        )
        if clean_targets != degraded_targets:
            raise RuntimeError(
                f"Clean/degraded detection target mismatch: clean={clean_targets}, degraded={degraded_targets}"
            )

        raw_consistency, metrics = self._pair_consistency(feature, pair_batch_size)
        weighted_per_pair = raw_consistency * self.sdc_consistency_weight
        scaled_consistency = weighted_per_pair * pair_batch_size
        if not bool(torch.isfinite(scaled_consistency)):
            raise FloatingPointError("P3 lambda-ablation contribution became non-finite")
        self._probe_consistency_gradient(scaled_consistency)
        self._sdc_last_metrics = {
            **metrics,
            "consistency_lambda": self.sdc_consistency_weight,
            "consistency_loss_weighted_per_pair": float(weighted_per_pair.detach()),
            "consistency_loss_scaled_contribution": float(scaled_consistency.detach()),
            "dual_view_detection_loss_scaled": float(detection_loss.detach().sum()),
            "detection_batch_size": int(batch["img"].shape[0]),
            "clean_detection_targets": clean_targets,
            "degraded_detection_targets": degraded_targets,
            "detect_input_indices": list(self.sdc_detect_input_indices),
            "target_feature_layer_index": self.sdc_feature_layer_index,
            "target_feature_layer_type": self.sdc_feature_layer_type,
        }
        self._sdc_captured_feature = None
        return detection_loss + scaled_consistency, loss_items


def self_test(config_path: Path, weights_path: Path, device: str) -> dict[str, Any]:
    config = load_sdc_ablation_config(config_path)
    source = YOLO(str(weights_path)).model
    weight = float(config["consistency"]["lambda"])
    model = LambdaAblationDetectionModel(
        source.yaml,
        ch=3,
        nc=int(source.nc),
        verbose=False,
        consistency_weight=weight,
    )
    model.load(source)
    model.to(device).train()

    torch.manual_seed(0)
    clean = torch.rand((2, 3, 160, 160), device=device)
    degraded = clean.clamp(0.0, 1.0).pow(1.18).mul(0.86)
    paired = torch.cat((clean, degraded), dim=0)
    model._sdc_captured_feature = None
    model._sdc_capture_enabled = True
    try:
        model.forward(paired)
    finally:
        model._sdc_capture_enabled = False
    feature = model._sdc_captured_feature
    if feature is None:
        raise RuntimeError("P3 hook did not fire in the lambda-ablation self-test")
    raw_loss, metrics = model._pair_consistency(feature, 2)
    weighted = raw_loss * weight
    scaled = weighted * 2
    model._probe_consistency_gradient(scaled)
    probe = model.sdc_gradient_probe()
    raw_value = float(raw_loss.detach())
    if not math.isfinite(raw_value) or raw_value <= 0.0:
        raise RuntimeError("Lambda-ablation self-test did not produce a finite nonzero raw P3 discrepancy")
    if weight == 0.0:
        if float(weighted.detach()) != 0.0 or float(scaled.detach()) != 0.0:
            raise RuntimeError("Lambda-zero self-test produced a nonzero weighted contribution")
    elif float(weighted.detach()) <= 0.0 or int(probe.get("nonzero_gradient_tensors", 0)) != 12:
        raise RuntimeError("Enabled lambda-ablation self-test did not produce the required loss and gradients")
    return {
        "schema": "ring3.e2_lambda_ablation_self_test.v1",
        "status": "passed",
        "ablation_id": config["ablation_id"],
        "device": device,
        "detect_input_indices": list(model.sdc_detect_input_indices),
        "target_feature_layer_index": model.sdc_feature_layer_index,
        "target_feature_layer_type": model.sdc_feature_layer_type,
        **metrics,
        "consistency_lambda": weight,
        "consistency_loss_weighted_per_pair": float(weighted.detach()),
        "consistency_loss_scaled_contribution": float(scaled.detach()),
        "consistency_only_gradient_probe": probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate E2 P3 lambda-ablation wiring")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(self_test(args.config, args.weights, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
