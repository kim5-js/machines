from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


CONFIG_SCHEMA = "ring3.e2_p3_consistency_config.v1"


def load_sdc_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_sdc_config(config)
    return config


def validate_sdc_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Unexpected SDC config schema: {config.get('schema')!r}")
    if config.get("stage") != "E2":
        raise ValueError(f"SDC config must target E2, got {config.get('stage')!r}")
    if config.get("method") != "online_mdg_plus_shared_weight_p3_consistency":
        raise ValueError(f"Unexpected E2 method: {config.get('method')!r}")

    dual_view = config.get("dual_view", {})
    required_dual_view_true = (
        "shared_detector_weights",
        "detection_loss_on_clean",
        "detection_loss_on_degraded",
        "validation_uses_locked_original_distribution_only",
        "inference_uses_single_detector_path_only",
    )
    if any(dual_view.get(key) is not True for key in required_dual_view_true):
        raise ValueError(f"E2 dual-view invariants are incomplete: {dual_view!r}")

    consistency = config.get("consistency", {})
    if consistency.get("feature_scale") != "P3":
        raise ValueError("E2 must use P3 consistency only")
    if consistency.get("feature_locator") != "first_input_to_detect_head":
        raise ValueError("E2 P3 must be resolved from the first Detect input")
    if consistency.get("loss") != "one_minus_cosine_similarity":
        raise ValueError("E2 must use one-minus-cosine consistency")
    weight = float(consistency.get("lambda", float("nan")))
    if not math.isfinite(weight) or weight != 0.1:
        raise ValueError(f"E2 consistency lambda must be locked to 0.1, got {weight!r}")

    invariants = config.get("invariants", {})
    required_true = (
        "formal_lock_unchanged",
        "runtime_amendment_unchanged",
        "training_hyperparameters_unchanged",
        "dataset_yaml_and_manifest_unchanged",
        "pretrained_weight_unchanged",
        "independent_run_from_same_pretrained_initialization",
        "no_cbam",
        "no_focal_eiou",
        "no_curriculum",
        "p4_p5_consistency_disabled",
    )
    if any(invariants.get(key) is not True for key in required_true):
        raise ValueError(f"E2 invariants are incomplete: {invariants!r}")


class P3ConsistencyDetectionModel(DetectionModel):
    """YOLO detection model with a training-only shared-weight P3 consistency objective."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        consistency_weight: float = 0.1,
    ) -> None:
        if not math.isfinite(float(consistency_weight)) or float(consistency_weight) <= 0.0:
            raise ValueError(f"Invalid consistency weight: {consistency_weight!r}")
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.sdc_consistency_weight = float(consistency_weight)
        detect = self.model[-1]
        detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
        if len(detect_inputs) < 3 or any(not isinstance(index, int) for index in detect_inputs[:3]):
            raise RuntimeError(f"Expected three integer Detect inputs for P3/P4/P5, got {detect_inputs!r}")
        self.sdc_detect_input_indices = detect_inputs
        self.sdc_feature_layer_index = int(detect_inputs[0])
        self.sdc_feature_layer_type = type(self.model[self.sdc_feature_layer_index]).__name__
        self._sdc_capture_enabled = False
        self._sdc_captured_feature: torch.Tensor | None = None
        self._sdc_last_metrics: dict[str, Any] = {}
        self._sdc_gradient_probe: dict[str, Any] = {}
        self.model[self.sdc_feature_layer_index].register_forward_hook(self._capture_p3_feature)

    def _capture_p3_feature(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not self._sdc_capture_enabled:
            return
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"P3 hook expected a tensor, got {type(output).__name__}")
        self._sdc_captured_feature = output

    def sdc_target_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model[self.sdc_feature_layer_index].parameters() if parameter.requires_grad]

    def sdc_last_metrics(self) -> dict[str, Any]:
        return dict(self._sdc_last_metrics)

    def sdc_gradient_probe(self) -> dict[str, Any]:
        return dict(self._sdc_gradient_probe)

    @staticmethod
    def _pair_consistency(feature: torch.Tensor, pair_batch_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if feature.ndim != 4:
            raise ValueError(f"P3 consistency expects BCHW features, got shape={tuple(feature.shape)}")
        if pair_batch_size <= 0 or int(feature.shape[0]) != pair_batch_size * 2:
            raise ValueError(
                f"P3 paired feature batch mismatch: feature_batch={int(feature.shape[0])}, pairs={pair_batch_size}"
            )
        clean = feature[:pair_batch_size]
        degraded = feature[pair_batch_size:]
        clean_flat = clean.float().flatten(1)
        degraded_flat = degraded.float().flatten(1)
        cosine = F.cosine_similarity(clean_flat, degraded_flat, dim=1, eps=1e-8)
        raw_loss = (1.0 - cosine).mean()
        feature_delta = (clean_flat - degraded_flat).abs().mean()
        return raw_loss, {
            "pair_batch_size": pair_batch_size,
            "concatenated_batch_size": int(feature.shape[0]),
            "feature_shape": [int(value) for value in feature.shape],
            "cosine_mean": float(cosine.detach().mean()),
            "feature_mean_abs_delta": float(feature_delta.detach()),
            "consistency_loss_raw": float(raw_loss.detach()),
        }

    def _probe_consistency_gradient(self, scaled_consistency: torch.Tensor) -> None:
        if self._sdc_gradient_probe or not self.training:
            return
        if float(scaled_consistency.detach()) <= 1e-10:
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
        self._sdc_gradient_probe = {
            "parameter_tensors": len(parameters),
            "gradient_tensors": sum(gradient is not None for gradient in gradients),
            "nonzero_gradient_tensors": nonzero_tensors,
            "finite": finite,
            "l2_norm": grad_norm,
            "max_abs": float(max_abs),
        }
        if not finite or nonzero_tensors <= 0 or not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise FloatingPointError(f"P3 consistency-only gradient probe failed: {self._sdc_gradient_probe}")

    def loss(self, batch: dict[str, Any], preds: Any = None):
        pair_batch_size = batch.get("sdc_pair_batch_size")
        if preds is not None or pair_batch_size is None or not self.training:
            return super().loss(batch, preds)
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        pair_batch_size = int(pair_batch_size)
        self._sdc_captured_feature = None
        self._sdc_capture_enabled = True
        try:
            preds = self.forward(batch["img"])
        finally:
            self._sdc_capture_enabled = False
        feature = self._sdc_captured_feature
        if feature is None:
            raise RuntimeError("P3 forward hook did not fire during E2 training")

        detection_loss, loss_items = self.criterion(preds, batch)
        raw_consistency, metrics = self._pair_consistency(feature, pair_batch_size)
        weighted_per_pair = raw_consistency * self.sdc_consistency_weight
        scaled_consistency = weighted_per_pair * pair_batch_size
        if not bool(torch.isfinite(scaled_consistency)):
            raise FloatingPointError("E2 P3 consistency loss became non-finite")
        self._probe_consistency_gradient(scaled_consistency)
        self._sdc_last_metrics = {
            **metrics,
            "consistency_lambda": self.sdc_consistency_weight,
            "consistency_loss_weighted_per_pair": float(weighted_per_pair.detach()),
            "consistency_loss_scaled_contribution": float(scaled_consistency.detach()),
            "detect_input_indices": list(self.sdc_detect_input_indices),
            "target_feature_layer_index": self.sdc_feature_layer_index,
            "target_feature_layer_type": self.sdc_feature_layer_type,
        }
        self._sdc_captured_feature = None
        return detection_loss + scaled_consistency, loss_items


def self_test(config_path: Path, weights_path: Path, device: str) -> dict[str, Any]:
    config = load_sdc_config(config_path)
    source = YOLO(str(weights_path)).model
    model = P3ConsistencyDetectionModel(
        source.yaml,
        ch=3,
        nc=int(source.nc),
        verbose=False,
        consistency_weight=float(config["consistency"]["lambda"]),
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
        raise RuntimeError("P3 hook did not fire in the SDC self-test")
    raw_loss, metrics = model._pair_consistency(feature, 2)
    scaled = raw_loss * float(config["consistency"]["lambda"]) * 2
    model._probe_consistency_gradient(scaled)
    probe = model.sdc_gradient_probe()
    if float(raw_loss.detach()) <= 0.0 or probe.get("nonzero_gradient_tensors", 0) <= 0:
        raise RuntimeError("SDC self-test did not produce a nonzero loss and gradient")
    return {
        "status": "passed",
        "device": device,
        "detect_input_indices": list(model.sdc_detect_input_indices),
        "target_feature_layer_index": model.sdc_feature_layer_index,
        "target_feature_layer_type": model.sdc_feature_layer_type,
        **metrics,
        "consistency_lambda": float(config["consistency"]["lambda"]),
        "consistency_only_gradient_probe": probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate E2 P3 shared-weight consistency wiring")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(self_test(args.config, args.weights, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
