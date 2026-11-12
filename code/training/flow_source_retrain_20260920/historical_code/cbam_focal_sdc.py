from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.nn.modules import C2f
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist

from sdc_consistency import P3ConsistencyDetectionModel


CONFIG_SCHEMA = "ring3.cbam_focal_interaction_pilot_config.v1"
EXPECTED_CBAM_LAYERS = [4, 6]
EXPECTED_DETECT_INPUTS = [15, 18, 21]
EXPECTED_P3_LAYER = 15
FOCAL_EIOU_ALPHA = 0.25
FOCAL_EIOU_GAMMA = 2.0
FOCAL_EIOU_EPS = 1e-7


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pilot_config(config)
    return config


def validate_pilot_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Unexpected pilot config schema: {config.get('schema')!r}")
    if config.get("stage") != "interaction-pilot" or config.get("paper_result_eligible") is not False:
        raise ValueError("The CBAM/Focal-EIoU run must remain a non-paper interaction pilot")
    if int(config.get("pilot_epochs", 0)) != 30 or int(config.get("seed", -1)) != 0:
        raise ValueError("The interaction pilot is locked to 30 epochs and seed 0")

    architecture = config.get("architecture", {})
    if architecture.get("cbam_layer_indices") != EXPECTED_CBAM_LAYERS:
        raise ValueError("CBAM must be limited to the YOLOv8s backbone P3/P4 C2f layers [4, 6]")
    if architecture.get("detect_input_indices") != EXPECTED_DETECT_INPUTS:
        raise ValueError("Detect inputs must remain [15, 18, 21]")
    if int(architecture.get("consistency_feature_layer_index", -1)) != EXPECTED_P3_LAYER:
        raise ValueError("P3 consistency must remain on layer 15")
    if int(architecture.get("channel_reduction_ratio", 0)) != 16:
        raise ValueError("CBAM channel reduction must remain 16")
    if int(architecture.get("spatial_kernel_size", 0)) != 7:
        raise ValueError("CBAM spatial kernel must remain 7")

    regression = config.get("bbox_regression", {})
    if regression.get("name") != "Focal-EIoU" or regression.get("dfl_preserved") is not True:
        raise ValueError("Focal-EIoU must replace only the bbox regression term while preserving DFL")
    expected_focal = {
        "alpha": FOCAL_EIOU_ALPHA,
        "gamma": FOCAL_EIOU_GAMMA,
        "eps": FOCAL_EIOU_EPS,
    }
    for key, expected in expected_focal.items():
        observed = float(regression.get(key, float("nan")))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unexpected Focal-EIoU {key}: {observed!r}")

    dual_view = config.get("dual_view", {})
    required_dual_view = (
        "shared_detector_weights",
        "detection_loss_on_clean",
        "detection_loss_on_degraded",
        "validation_uses_locked_original_distribution_only",
        "inference_uses_single_detector_path_only",
    )
    if any(dual_view.get(key) is not True for key in required_dual_view):
        raise ValueError(f"Dual-view invariants are incomplete: {dual_view!r}")

    consistency = config.get("consistency", {})
    if consistency.get("feature_scale") != "P3":
        raise ValueError("The interaction pilot must observe P3 only")
    if consistency.get("feature_locator") != "first_input_to_detect_head":
        raise ValueError("P3 must be resolved from the first Detect input")
    if consistency.get("loss") != "one_minus_cosine_similarity":
        raise ValueError("The discrepancy probe must remain one-minus-cosine")
    if not math.isclose(float(consistency.get("lambda", float("nan"))), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The CBAM/Focal-EIoU base pilot must use lambda=0")
    if consistency.get("weighted_consistency_contribution_must_be_zero") is not True:
        raise ValueError("The pilot must explicitly require a zero weighted consistency contribution")

    invariants = config.get("invariants", {})
    required_invariants = (
        "formal_lock_file_unchanged",
        "runtime_amendment_file_unchanged",
        "dataset_yaml_and_manifest_unchanged",
        "locked_training_hyperparameters_unchanged_except_pilot_epochs",
        "independent_initialization_from_locked_yolov8s_pretrained_weight",
        "no_resume_from_E0_E1_E2_or_scratch_checkpoints",
        "p4_p5_consistency_disabled",
        "object_aware_consistency_not_implemented_or_started_by_this_pilot",
        "E3_E4_formal_results_not_started",
    )
    if any(invariants.get(key) is not True for key in required_invariants):
        raise ValueError(f"Pilot invariants are incomplete: {invariants!r}")


class ChannelAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, int(channels) // int(reduction_ratio))
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"Unsupported CBAM spatial kernel: {kernel_size}")
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.max(x, dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class CBAMAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class C2f_CBAM(C2f):
    """C2f followed by channel and spatial attention, preserving the original layer shape."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention = CBAMAttention(c2, reduction_ratio=16, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(super().forward(x))


def _split_xyxy(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return box[..., 0], box[..., 1], box[..., 2], box[..., 3]


def iou_xyxy(pred: torch.Tensor, target: torch.Tensor, eps: float = FOCAL_EIOU_EPS) -> torch.Tensor:
    px1, py1, px2, py2 = _split_xyxy(pred)
    tx1, ty1, tx2, ty2 = _split_xyxy(target)
    inter_w = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0.0)
    inter_h = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(min=0.0)
    inter = inter_w * inter_h
    pred_area = (px2 - px1).clamp(min=eps) * (py2 - py1).clamp(min=eps)
    target_area = (tx2 - tx1).clamp(min=eps) * (ty2 - ty1).clamp(min=eps)
    return inter / (pred_area + target_area - inter + eps)


def efficient_iou_xyxy(
    pred: torch.Tensor, target: torch.Tensor, eps: float = FOCAL_EIOU_EPS
) -> torch.Tensor:
    px1, py1, px2, py2 = _split_xyxy(pred)
    tx1, ty1, tx2, ty2 = _split_xyxy(target)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    iou = iou_xyxy(pred, target, eps)

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5
    enclosing_w = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=eps)
    enclosing_h = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=eps)
    center_penalty = ((pcx - tcx).square() + (pcy - tcy).square()) / (
        enclosing_w.square() + enclosing_h.square() + eps
    )
    width_penalty = (tw - pw).square() / (enclosing_w.square() + eps)
    height_penalty = (th - ph).square() / (enclosing_h.square() + eps)
    return iou - center_penalty - width_penalty - height_penalty


class FocalEIoULoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = FOCAL_EIOU_ALPHA,
        gamma: float = FOCAL_EIOU_GAMMA,
        eps: float = FOCAL_EIOU_EPS,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-1] != 4 or target.shape[-1] != 4:
            raise ValueError("Focal-EIoU expects xyxy tensors with a final dimension of four")
        iou = iou_xyxy(pred, target, self.eps)
        eiou = efficient_iou_xyxy(pred, target, self.eps).clamp(min=-1.0, max=1.0)
        focal_weight = self.alpha + (1.0 - self.alpha) * (1.0 - iou).clamp(min=0.0).pow(self.gamma)
        return (1.0 - eiou).clamp(min=0.0) * focal_weight


class FocalEIoUBboxLoss(BboxLoss):
    """Replace the CIoU term while retaining the installed Ultralytics DFL path."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__(reg_max)
        self.focal_eiou = FocalEIoULoss()

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(fg_mask.any()):
            zero = pred_bboxes.sum() * 0.0
            return zero, zero

        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        focal_eiou = self.focal_eiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).unsqueeze(-1)
        loss_iou = (focal_eiou * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            normalized_pred = pred_dist * stride
            normalized_pred[..., 0::2] /= imgsz[1]
            normalized_pred[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(normalized_pred[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True)
                .mul(weight)
                .sum()
                / target_scores_sum
            )
        return loss_iou, loss_dfl


class v8DetectionLossWithFocalEIoU(v8DetectionLoss):
    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = FocalEIoUBboxLoss(self.reg_max).to(self.device)


def replace_backbone_c2f_with_cbam(model: DetectionModel) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index in EXPECTED_CBAM_LAYERS:
        original = model.model[index]
        if not isinstance(original, C2f) or isinstance(original, C2f_CBAM):
            raise RuntimeError(f"Expected a plain C2f at backbone layer {index}, got {type(original).__name__}")
        c1 = int(original.cv1.conv.in_channels)
        c2 = int(original.cv2.conv.out_channels)
        repeats = len(original.m)
        shortcut = bool(getattr(original.m[0], "add", False)) if repeats else False
        replacement = C2f_CBAM(c1, c2, n=repeats, shortcut=shortcut, g=1, e=0.5)
        reference = next(original.parameters())
        replacement.to(device=reference.device, dtype=reference.dtype)
        incompatible = replacement.load_state_dict(original.state_dict(), strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected or not missing or any(not key.startswith("attention.") for key in missing):
            raise RuntimeError(
                f"Unsafe CBAM replacement at layer {index}: missing={missing}, unexpected={unexpected}"
            )
        for attribute in ("i", "f"):
            if hasattr(original, attribute):
                setattr(replacement, attribute, getattr(original, attribute))
        replacement.type = type(replacement).__name__
        replacement.np = sum(parameter.numel() for parameter in replacement.parameters())
        model.model[index] = replacement
        attention_parameters = sum(parameter.numel() for parameter in replacement.attention.parameters())
        evidence.append(
            {
                "index": index,
                "original_type": type(original).__name__,
                "replacement_type": type(replacement).__name__,
                "c1": c1,
                "c2": c2,
                "repeats": repeats,
                "shortcut": shortcut,
                "attention_parameter_count": attention_parameters,
                "missing_attention_keys_during_structural_copy": missing,
            }
        )
    return evidence


class CBAMFocalP3DetectionModel(P3ConsistencyDetectionModel):
    """Dual-view lambda-zero model with backbone CBAM and Focal-EIoU detection loss."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        consistency_weight: float = 0.0,
    ) -> None:
        weight = float(consistency_weight)
        if not math.isfinite(weight) or not math.isclose(weight, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"The base interaction pilot is locked to lambda=0, got {weight!r}")
        DetectionModel.__init__(self, cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.cbam_replacement_evidence = replace_backbone_c2f_with_cbam(self)
        self.sdc_consistency_weight = weight

        detect = self.model[-1]
        detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
        if detect_inputs != EXPECTED_DETECT_INPUTS:
            raise RuntimeError(f"Detect inputs changed: expected={EXPECTED_DETECT_INPUTS}, observed={detect_inputs}")
        self.sdc_detect_input_indices = detect_inputs
        self.sdc_feature_layer_index = int(detect_inputs[0])
        self.sdc_feature_layer_type = type(self.model[self.sdc_feature_layer_index]).__name__
        if self.sdc_feature_layer_index != EXPECTED_P3_LAYER or self.sdc_feature_layer_type != "C2f":
            raise RuntimeError(
                f"Expected unchanged head P3 layer 15 C2f, got index={self.sdc_feature_layer_index}, "
                f"type={self.sdc_feature_layer_type}"
            )
        self._sdc_capture_enabled = False
        self._sdc_captured_feature: torch.Tensor | None = None
        self._sdc_last_metrics: dict[str, Any] = {}
        self._sdc_gradient_probe: dict[str, Any] = {}
        self.model[self.sdc_feature_layer_index].register_forward_hook(self._capture_p3_feature)

    def init_criterion(self):
        return v8DetectionLossWithFocalEIoU(self)

    def cbam_focal_evidence(self) -> dict[str, Any]:
        return {
            "cbam_layer_indices": [item["index"] for item in self.cbam_replacement_evidence],
            "cbam_layer_types": [type(self.model[index]).__name__ for index in EXPECTED_CBAM_LAYERS],
            "cbam_attention_parameter_count": sum(
                parameter.numel()
                for index in EXPECTED_CBAM_LAYERS
                for parameter in self.model[index].attention.parameters()
            ),
            "cbam_replacement_evidence": self.cbam_replacement_evidence,
            "criterion_type": type(self.criterion).__name__ if getattr(self, "criterion", None) is not None else None,
            "bbox_loss_type": (
                type(self.criterion.bbox_loss).__name__
                if getattr(self, "criterion", None) is not None and hasattr(self.criterion, "bbox_loss")
                else None
            ),
            "focal_eiou": {
                "alpha": FOCAL_EIOU_ALPHA,
                "gamma": FOCAL_EIOU_GAMMA,
                "eps": FOCAL_EIOU_EPS,
                "dfl_preserved": True,
            },
        }

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
        norm = float(squared_norm.sqrt())
        self._sdc_gradient_probe = {
            "parameter_tensors": len(parameters),
            "gradient_tensors": sum(gradient is not None for gradient in gradients),
            "nonzero_gradient_tensors": nonzero_tensors,
            "finite": finite,
            "l2_norm": norm,
            "max_abs": float(max_abs),
            "expected_behavior": "zero_by_lambda_control",
        }
        if not finite or not math.isfinite(norm):
            raise FloatingPointError(f"Lambda-zero consistency gradient is non-finite: {self._sdc_gradient_probe}")
        if len(parameters) != 12 or len(gradients) != 12:
            raise RuntimeError(f"Expected exactly 12 head-P3 parameter tensors: {self._sdc_gradient_probe}")
        if nonzero_tensors != 0 or norm > 1e-12 or float(max_abs) > 1e-12:
            raise FloatingPointError(f"Lambda-zero consistency leaked a gradient: {self._sdc_gradient_probe}")

    def loss(self, batch: dict[str, Any], preds: Any = None):
        pair_batch_size = batch.get("sdc_pair_batch_size")
        if preds is not None or pair_batch_size is None or not self.training:
            return DetectionModel.loss(self, batch, preds)
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
            raise RuntimeError("P3 forward hook did not fire in the CBAM/Focal-EIoU pilot")

        detection_loss, loss_items = self.criterion(predictions, batch)
        if not bool(torch.isfinite(detection_loss).all()):
            raise FloatingPointError("Focal-EIoU dual-view detection loss became non-finite")
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
            raise FloatingPointError("Lambda-zero P3 contribution became non-finite")
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
            **self.cbam_focal_evidence(),
        }
        self._sdc_captured_feature = None
        return detection_loss + scaled_consistency, loss_items


__all__ = [
    "CBAMFocalP3DetectionModel",
    "C2f_CBAM",
    "FocalEIoULoss",
    "FocalEIoUBboxLoss",
    "v8DetectionLossWithFocalEIoU",
    "load_pilot_config",
    "validate_pilot_config",
]
