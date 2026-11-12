from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import bbox2dist, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from cbam_focal_sdc import (
    EXPECTED_CBAM_LAYERS,
    EXPECTED_DETECT_INPUTS,
    EXPECTED_P3_LAYER,
    FOCAL_EIOU_ALPHA,
    FOCAL_EIOU_EPS,
    FOCAL_EIOU_GAMMA,
    FocalEIoULoss,
    replace_backbone_c2f_with_cbam,
    v8DetectionLossWithFocalEIoU,
)


class FactorialDetectionModel(DetectionModel):
    """YOLOv8s with independently switchable CBAM and Focal-EIoU factors."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        use_cbam: bool,
        use_focal_eiou: bool,
    ) -> None:
        self.factorial_use_cbam = bool(use_cbam)
        self.factorial_use_focal_eiou = bool(use_focal_eiou)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.factorial_cbam_replacement_evidence = (
            replace_backbone_c2f_with_cbam(self) if self.factorial_use_cbam else []
        )
        detect = self.model[-1]
        detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
        if detect_inputs != EXPECTED_DETECT_INPUTS:
            raise RuntimeError(
                f"Detect inputs changed: expected={EXPECTED_DETECT_INPUTS}, observed={detect_inputs}"
            )
        self.factorial_detect_input_indices = detect_inputs
        self.factorial_p3_layer_index = int(detect_inputs[0])
        self.factorial_p3_layer_type = type(self.model[self.factorial_p3_layer_index]).__name__
        if self.factorial_p3_layer_index != EXPECTED_P3_LAYER or self.factorial_p3_layer_type != "C2f":
            raise RuntimeError(
                "The head P3 path changed: "
                f"index={self.factorial_p3_layer_index}, type={self.factorial_p3_layer_type}"
            )

    def init_criterion(self):
        if self.factorial_use_focal_eiou:
            return v8DetectionLossWithFocalEIoU(self)
        return v8DetectionLoss(self)

    def p3_target_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model[self.factorial_p3_layer_index].parameters() if parameter.requires_grad]

    def factorial_evidence(self) -> dict[str, Any]:
        criterion = getattr(self, "criterion", None)
        cbam_types = [type(self.model[index]).__name__ for index in EXPECTED_CBAM_LAYERS]
        attention_parameters = 0
        if self.factorial_use_cbam:
            attention_parameters = sum(
                parameter.numel()
                for index in EXPECTED_CBAM_LAYERS
                for parameter in self.model[index].attention.parameters()
            )
        bbox_loss = getattr(criterion, "bbox_loss", None)
        return {
            "cbam_enabled": self.factorial_use_cbam,
            "focal_eiou_enabled": self.factorial_use_focal_eiou,
            "cbam_layer_indices": EXPECTED_CBAM_LAYERS if self.factorial_use_cbam else [],
            "cbam_layer_types": cbam_types,
            "cbam_attention_parameter_count": attention_parameters,
            "cbam_replacement_evidence": self.factorial_cbam_replacement_evidence,
            "criterion_type": type(criterion).__name__ if criterion is not None else None,
            "bbox_loss_type": type(bbox_loss).__name__ if bbox_loss is not None else None,
            "focal_eiou": {
                "alpha": FOCAL_EIOU_ALPHA,
                "gamma": FOCAL_EIOU_GAMMA,
                "eps": FOCAL_EIOU_EPS,
                "dfl_preserved": True,
            }
            if self.factorial_use_focal_eiou
            else None,
            "detect_input_indices": list(self.factorial_detect_input_indices),
            "target_feature_layer_index": self.factorial_p3_layer_index,
            "target_feature_layer_type": self.factorial_p3_layer_type,
            "no_consistency_loss": True,
        }


class SmallObjectWeightedFocalEIoULoss(v8DetectionLoss):
    """Focal-EIoU + DFL with a 1.5 multiplier only on degraded-small positives."""

    def __init__(self, model: torch.nn.Module, *, degraded_small_weight: float = 1.5) -> None:
        super().__init__(model)
        self.degraded_small_weight = float(degraded_small_weight)
        if not math.isclose(self.degraded_small_weight, 1.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unapproved degraded-small weight: {self.degraded_small_weight!r}")
        self.focal_eiou = FocalEIoULoss()
        self._last_evidence: dict[str, Any] = {}

    def last_evidence(self) -> dict[str, Any]:
        return dict(self._last_evidence)

    def _preprocess_gt_weights(
        self,
        raw_weights: torch.Tensor,
        raw_batch_idx: torch.Tensor,
        batch_size: int,
        max_targets: int,
    ) -> torch.Tensor:
        output = torch.ones((batch_size, max_targets), device=self.device, dtype=raw_weights.dtype)
        if raw_weights.numel() == 0:
            return output
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, raw_batch_idx + 1, torch.ones_like(raw_batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(raw_weights.numel(), device=self.device) - offsets[raw_batch_idx]
        output[raw_batch_idx, within_idx] = raw_weights
        return output

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        if "factorial_pair_batch_size" not in batch:
            return super().get_assigned_targets_and_loss(preds, batch)

        loss = torch.zeros(3, device=self.device)
        unit_loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = int(pred_scores.shape[0])
        pair_batch_size = int(batch.get("factorial_pair_batch_size", 0))
        if pair_batch_size <= 0 or batch_size != pair_batch_size * 2:
            raise RuntimeError(
                f"Paired batch mismatch: pair={pair_batch_size}, detection={batch_size}"
            )
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        raw_batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
        raw_cls = batch["cls"].view(-1, 1)
        raw_bboxes = batch["bboxes"]
        raw_weights = batch.get("factorial_gt_weights")
        if raw_weights is None:
            raise RuntimeError("The weighted E3 batch is missing factorial_gt_weights")
        raw_weights = raw_weights.view(-1).to(device=self.device, dtype=dtype)
        if not (raw_weights.numel() == raw_batch_idx.numel() == raw_cls.shape[0] == raw_bboxes.shape[0]):
            raise RuntimeError("Targets and target weights are not aligned")

        targets = torch.cat((raw_batch_idx.view(-1, 1), raw_cls, raw_bboxes), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        gt_weights = self._preprocess_gt_weights(raw_weights, raw_batch_idx, batch_size, int(targets.shape[1]))

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        anchor_weights = torch.ones(fg_mask.shape, device=self.device, dtype=dtype)
        if bool(fg_mask.any()):
            assigned_weights = gt_weights.gather(1, target_gt_idx.long())
            anchor_weights[fg_mask] = assigned_weights[fg_mask]
        if not bool(torch.isfinite(anchor_weights).all()):
            raise FloatingPointError("Assigned-anchor weights became non-finite")
        allowed = torch.isclose(anchor_weights, torch.ones_like(anchor_weights)) | torch.isclose(
            anchor_weights, torch.full_like(anchor_weights, self.degraded_small_weight)
        )
        if not bool(allowed.all()):
            raise RuntimeError("Assigned-anchor weights contain a value outside {1.0, 1.5}")

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        unit_loss[1] = bce_loss.sum() / target_scores_sum
        loss[1] = (bce_loss * anchor_weights.unsqueeze(-1)).sum() / target_scores_sum

        if bool(fg_mask.any()):
            base_box_weight = target_scores[fg_mask].sum(-1, keepdim=True)
            object_weight = anchor_weights[fg_mask].unsqueeze(-1)
            target_bboxes_grid = target_bboxes / stride_tensor
            focal_terms = self.focal_eiou(
                pred_bboxes[fg_mask], target_bboxes_grid[fg_mask]
            ).unsqueeze(-1) * base_box_weight
            unit_loss[0] = focal_terms.sum() / target_scores_sum
            loss[0] = (focal_terms * object_weight).sum() / target_scores_sum

            if self.bbox_loss.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes_grid, self.bbox_loss.dfl_loss.reg_max - 1)
                dfl_terms = self.bbox_loss.dfl_loss(
                    pred_distri[fg_mask].view(-1, self.bbox_loss.dfl_loss.reg_max), target_ltrb[fg_mask]
                ) * base_box_weight
            else:
                target_ltrb = bbox2dist(anchor_points, target_bboxes_grid)
                target_ltrb = target_ltrb * stride_tensor
                target_ltrb[..., 0::2] /= imgsz[1]
                target_ltrb[..., 1::2] /= imgsz[0]
                normalized_pred = pred_distri * stride_tensor
                normalized_pred[..., 0::2] /= imgsz[1]
                normalized_pred[..., 1::2] /= imgsz[0]
                dfl_terms = (
                    F.l1_loss(normalized_pred[fg_mask], target_ltrb[fg_mask], reduction="none")
                    .mean(-1, keepdim=True)
                    .mul(base_box_weight)
                )
            unit_loss[2] = dfl_terms.sum() / target_scores_sum
            loss[2] = (dfl_terms * object_weight).sum() / target_scores_sum

        gains = torch.tensor(
            (float(self.hyp.box), float(self.hyp.cls), float(self.hyp.dfl)),
            device=self.device,
            dtype=loss.dtype,
        )
        loss *= gains
        unit_loss *= gains
        delta = loss - unit_loss
        if not bool(torch.isfinite(loss).all()) or not bool(torch.isfinite(unit_loss).all()):
            raise FloatingPointError("Weighted or unit-control detection loss became non-finite")
        if bool((delta < -1e-7).any()):
            raise RuntimeError(f"Positive weighting reduced a loss component: {delta.detach().tolist()}")

        clean_mask = raw_batch_idx < pair_batch_size
        degraded_mask = (raw_batch_idx >= pair_batch_size) & (raw_batch_idx < pair_batch_size * 2)
        weighted_gt_mask = torch.isclose(
            raw_weights, torch.full_like(raw_weights, self.degraded_small_weight), rtol=0.0, atol=1e-6
        )
        negative_anchor_weight_violations = int(((~fg_mask) & (anchor_weights != 1.0)).sum())
        self._last_evidence = {
            "ordinary_target_weight": 1.0,
            "degraded_small_target_weight": self.degraded_small_weight,
            "weighting_scope": "degraded_small_assigned_positive_anchors_only",
            "normalization_denominator_unchanged": True,
            "background_anchor_weight_unchanged": negative_anchor_weight_violations == 0,
            "clean_detection_targets": int(clean_mask.sum()),
            "degraded_detection_targets": int(degraded_mask.sum()),
            "degraded_small_targets": int((degraded_mask & weighted_gt_mask).sum()),
            "clean_weighted_target_violations": int((clean_mask & weighted_gt_mask).sum()),
            "positive_anchor_count": int(fg_mask.sum()),
            "weighted_positive_anchor_count": int((fg_mask & (anchor_weights > 1.0 + 1e-6)).sum()),
            "negative_anchor_weight_violations": negative_anchor_weight_violations,
            "target_scores_normalizer": float(target_scores_sum.detach()),
            "weighted_loss_per_component": {
                name: float(value) for name, value in zip(self.loss_names, loss.detach(), strict=True)
            },
            "unit_weight_control_loss_per_component": {
                name: float(value) for name, value in zip(self.loss_names, unit_loss.detach(), strict=True)
            },
            "weighting_delta_per_component": {
                name: float(value) for name, value in zip(self.loss_names, delta.detach(), strict=True)
            },
            "weighting_delta_scaled": float((delta * batch_size).detach().sum()),
        }
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach(), strict=True)),
        )


class CBAMFocalSmallWeightedDetectionModel(FactorialDetectionModel):
    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        degraded_small_weight: float = 1.5,
    ) -> None:
        self.degraded_small_weight = float(degraded_small_weight)
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            use_cbam=True,
            use_focal_eiou=True,
        )

    def init_criterion(self):
        return SmallObjectWeightedFocalEIoULoss(
            self, degraded_small_weight=self.degraded_small_weight
        )

    def factorial_evidence(self) -> dict[str, Any]:
        evidence = super().factorial_evidence()
        criterion = getattr(self, "criterion", None)
        evidence["criterion_type"] = type(criterion).__name__ if criterion is not None else None
        evidence["bbox_loss_type"] = type(getattr(criterion, "bbox_loss", None)).__name__ if criterion is not None else None
        evidence["degraded_small_weighting_enabled"] = True
        if isinstance(criterion, SmallObjectWeightedFocalEIoULoss):
            evidence.update(criterion.last_evidence())
        return evidence


__all__ = [
    "CBAMFocalSmallWeightedDetectionModel",
    "FactorialDetectionModel",
    "SmallObjectWeightedFocalEIoULoss",
]
