from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import bbox2dist, bbox_iou, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


CONFIG_SCHEMA = "ring3.e3_small_object_weighting_config.v1"


def load_e3_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_e3_config(config)
    return config


def validate_e3_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"Unexpected E3 config schema: {config.get('schema')!r}")
    if config.get("stage") != "E3":
        raise ValueError(f"E3 config has wrong stage: {config.get('stage')!r}")
    if config.get("method") != "dual_view_degraded_small_object_weighting":
        raise ValueError(f"Unexpected E3 method: {config.get('method')!r}")

    objective = config.get("objective", {})
    expected_true = (
        "shared_detector_weights",
        "detection_loss_on_clean",
        "detection_loss_on_degraded",
        "small_object_weighting_on_degraded_only",
        "background_anchor_weight_unchanged",
        "normalization_denominator_unchanged",
        "consistency_loss_disabled",
        "severity_weighting_disabled",
        "curriculum_disabled",
    )
    if any(objective.get(key) is not True for key in expected_true):
        raise ValueError(f"E3 objective invariants are incomplete: {objective!r}")

    weights = config.get("target_weights", {})
    if not math.isclose(float(weights.get("ordinary", float("nan"))), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("E3 ordinary target weight must be 1.0")
    if not math.isclose(
        float(weights.get("degraded_small", float("nan"))), 1.5, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("E3 degraded-small target weight must be 1.5")

    small = config.get("small_object_definition", {})
    if small.get("coordinate_space") != "post_augmentation_training_pixels":
        raise ValueError("E3 small-object definition must use post-augmentation training pixels")
    if int(small.get("area_lt_px2", -1)) != 32 * 32:
        raise ValueError("E3 first version must use the COCO 32^2 small-object boundary")

    invariants = config.get("invariants", {})
    required_invariants = (
        "pretrained_true",
        "same_yolov8s_pt",
        "same_seadronessee_split",
        "same_training_hyperparameters_as_formal_e2",
        "independent_initialization_from_locked_pretrained",
        "no_cbam",
        "no_focal_eiou",
        "no_severity_weighting",
        "no_curriculum",
        "no_consistency_loss",
    )
    if any(invariants.get(key) is not True for key in required_invariants):
        raise ValueError(f"E3 experiment invariants are incomplete: {invariants!r}")


class SmallObjectWeightedDetectionLoss(v8DetectionLoss):
    """YOLOv8 detection loss with exact per-assigned-GT weights.

    Task-aligned assignment is unchanged. Only positive anchors assigned to a
    degraded-view small GT receive the configured multiplier. Background
    anchors, clean-view targets, and degraded non-small targets remain 1.0.
    The original target-score normalization denominator is intentionally kept.
    """

    def __init__(self, model: torch.nn.Module, *, degraded_small_weight: float) -> None:
        super().__init__(model)
        self.degraded_small_weight = float(degraded_small_weight)
        if not math.isclose(self.degraded_small_weight, 1.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Unapproved degraded-small weight: {self.degraded_small_weight!r}")
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
        # Validation remains the same locked clean-view YOLO validation used by
        # formal E2. Only trainer-preprocessed dual-view batches carry the E3
        # marker and receive degraded-small weighting.
        if "e3_pair_batch_size" not in batch:
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
        pair_batch_size = int(batch.get("e3_pair_batch_size", 0))
        if pair_batch_size <= 0 or batch_size != pair_batch_size * 2:
            raise RuntimeError(
                f"E3 paired batch mismatch: pair_batch_size={pair_batch_size}, detection_batch_size={batch_size}"
            )
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        raw_batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
        raw_cls = batch["cls"].view(-1, 1)
        raw_bboxes = batch["bboxes"]
        raw_weights = batch.get("e3_gt_weights")
        if raw_weights is None:
            raise RuntimeError("E3 batch is missing e3_gt_weights")
        raw_weights = raw_weights.view(-1).to(device=self.device, dtype=dtype)
        if not (raw_weights.numel() == raw_batch_idx.numel() == raw_cls.shape[0] == raw_bboxes.shape[0]):
            raise RuntimeError("E3 target tensors and target-weight tensor are not aligned")

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
            raise FloatingPointError("E3 assigned-anchor weights became non-finite")
        allowed = torch.isclose(anchor_weights, torch.ones_like(anchor_weights)) | torch.isclose(
            anchor_weights, torch.full_like(anchor_weights, self.degraded_small_weight)
        )
        if not bool(allowed.all()):
            raise RuntimeError("E3 assigned-anchor weights contain a value outside {1.0, 1.5}")

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        unit_loss[1] = bce_loss.sum() / target_scores_sum
        loss[1] = (bce_loss * anchor_weights.unsqueeze(-1)).sum() / target_scores_sum

        if bool(fg_mask.any()):
            base_box_weight = target_scores[fg_mask].sum(-1, keepdim=True)
            object_weight = anchor_weights[fg_mask].unsqueeze(-1)
            target_bboxes_grid = target_bboxes / stride_tensor
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes_grid[fg_mask], xywh=False, CIoU=True)
            iou_terms = (1.0 - iou) * base_box_weight
            unit_loss[0] = iou_terms.sum() / target_scores_sum
            loss[0] = (iou_terms * object_weight).sum() / target_scores_sum

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
                    F.l1_loss(normalized_pred[fg_mask], target_ltrb[fg_mask], reduction="none").mean(
                        -1, keepdim=True
                    )
                    * base_box_weight
                )
            unit_loss[2] = dfl_terms.sum() / target_scores_sum
            loss[2] = (dfl_terms * object_weight).sum() / target_scores_sum

        gains = torch.tensor(
            (float(self.hyp.box), float(self.hyp.cls), float(self.hyp.dfl)), device=self.device, dtype=loss.dtype
        )
        loss *= gains
        unit_loss *= gains
        delta = loss - unit_loss
        if not bool(torch.isfinite(loss).all()) or not bool(torch.isfinite(unit_loss).all()):
            raise FloatingPointError("E3 weighted or unit-control detection loss became non-finite")
        if bool((delta < -1e-7).any()):
            raise RuntimeError(f"E3 weighting unexpectedly reduced a loss component: {delta.detach().tolist()}")

        clean_mask = raw_batch_idx < pair_batch_size
        degraded_mask = (raw_batch_idx >= pair_batch_size) & (raw_batch_idx < pair_batch_size * 2)
        weighted_gt_mask = torch.isclose(
            raw_weights, torch.full_like(raw_weights, self.degraded_small_weight), rtol=0.0, atol=1e-6
        )
        clean_targets = int(clean_mask.sum())
        degraded_targets = int(degraded_mask.sum())
        degraded_small_targets = int((degraded_mask & weighted_gt_mask).sum())
        clean_weighted_targets = int((clean_mask & weighted_gt_mask).sum())
        positive_anchor_count = int(fg_mask.sum())
        weighted_positive_anchor_count = int((fg_mask & (anchor_weights > 1.0 + 1e-6)).sum())
        negative_anchor_weight_violations = int(((~fg_mask) & (anchor_weights != 1.0)).sum())

        self._last_evidence = {
            "ordinary_target_weight": 1.0,
            "degraded_small_target_weight": self.degraded_small_weight,
            "weighting_scope": "degraded_small_assigned_positive_anchors_only",
            "normalization_denominator_unchanged": True,
            "background_anchor_weight_unchanged": negative_anchor_weight_violations == 0,
            "no_consistency_loss": True,
            "detection_batch_size": batch_size,
            "pair_batch_size": pair_batch_size,
            "clean_detection_targets": clean_targets,
            "degraded_detection_targets": degraded_targets,
            "degraded_small_targets": degraded_small_targets,
            "degraded_regular_targets": degraded_targets - degraded_small_targets,
            "clean_weighted_target_violations": clean_weighted_targets,
            "positive_anchor_count": positive_anchor_count,
            "weighted_positive_anchor_count": weighted_positive_anchor_count,
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
            "weighted_detection_loss_scaled": float((loss * batch_size).detach().sum()),
            "unit_weight_control_loss_scaled": float((unit_loss * batch_size).detach().sum()),
            "weighting_delta_scaled": float((delta * batch_size).detach().sum()),
        }
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach(), strict=True)),
        )


class SmallObjectWeightedDualViewDetectionModel(DetectionModel):
    """Shared YOLO detector used by clean/degraded views with no consistency term."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        degraded_small_weight: float = 1.5,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.e3_degraded_small_weight = float(degraded_small_weight)
        detect = self.model[-1]
        detect_inputs = list(detect.f) if isinstance(detect.f, (list, tuple)) else [detect.f]
        if detect_inputs != [15, 18, 21]:
            raise RuntimeError(f"Expected locked Detect inputs [15, 18, 21], got {detect_inputs!r}")
        self.e3_detect_input_indices = detect_inputs
        self.e3_p3_layer_index = int(detect_inputs[0])
        self.e3_p3_layer_type = type(self.model[self.e3_p3_layer_index]).__name__
        if self.e3_p3_layer_type != "C2f":
            raise RuntimeError(f"Expected locked layer 15 C2f, got {self.e3_p3_layer_type!r}")

    def init_criterion(self) -> SmallObjectWeightedDetectionLoss:
        return SmallObjectWeightedDetectionLoss(self, degraded_small_weight=self.e3_degraded_small_weight)

    def e3_target_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model[self.e3_p3_layer_index].parameters() if parameter.requires_grad]

    def e3_loss_evidence(self) -> dict[str, Any]:
        criterion = getattr(self, "criterion", None)
        if not isinstance(criterion, SmallObjectWeightedDetectionLoss):
            return {}
        return {
            **criterion.last_evidence(),
            "criterion_type": type(criterion).__name__,
            "detect_input_indices": list(self.e3_detect_input_indices),
            "p3_layer_index": self.e3_p3_layer_index,
            "p3_layer_type": self.e3_p3_layer_type,
        }


def self_test(config_path: Path, weights_path: Path, device: str) -> dict[str, Any]:
    config = load_e3_config(config_path)
    source = YOLO(str(weights_path)).model
    model = SmallObjectWeightedDualViewDetectionModel(
        source.yaml,
        ch=3,
        nc=int(source.nc),
        verbose=False,
        degraded_small_weight=float(config["target_weights"]["degraded_small"]),
    )
    model.load(source)
    # A standalone model loaded from a checkpoint may expose ``args`` as a
    # plain dict, while v8DetectionLoss expects Ultralytics' cfg namespace.
    # DetectionTrainer supplies that namespace during real training; mirror
    # the same type here so the self-test exercises the actual criterion.
    model.args = get_cfg(DEFAULT_CFG_DICT)
    model.to(device).train()

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
        "e3_gt_weights": torch.tensor([1.0, 1.0, 1.5, 1.5], device=device),
        "e3_pair_batch_size": 2,
    }
    loss, loss_items = model.loss(batch)
    total = loss.sum()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("E3 self-test produced a non-finite loss")
    total.backward()
    gradient_tensors = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradient_tensors)
    evidence = model.e3_loss_evidence()
    if not finite_gradients:
        raise FloatingPointError("E3 self-test produced a non-finite gradient")
    if int(evidence.get("clean_weighted_target_violations", -1)) != 0:
        raise RuntimeError(f"E3 self-test leaked weighting into the clean branch: {evidence}")
    if int(evidence.get("weighted_positive_anchor_count", 0)) <= 0:
        raise RuntimeError(f"E3 self-test did not assign a degraded-small positive anchor: {evidence}")
    if float(evidence.get("weighting_delta_scaled", 0.0)) <= 0.0:
        raise RuntimeError(f"E3 self-test did not increase the detection objective: {evidence}")
    return {
        "schema": "ring3.e3_small_object_weighting_self_test.v1",
        "status": "passed",
        "device": device,
        "loss_items": {key: float(value) for key, value in loss_items.items()},
        "total_loss_scaled": float(total.detach()),
        "finite_gradients": finite_gradients,
        **evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact E3 degraded-small target weighting")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(self_test(args.config, args.weights, args.device), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
