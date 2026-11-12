"""YOLO-specific adapters for using Focal-EIoU in Ultralytics training."""

from __future__ import annotations

from typing import Any, Tuple

import torch

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist

from .focal_eiou import FocalEIoULoss


class FocalEIoUBboxLoss(BboxLoss):
    """Replace CIoU with Focal-EIoU while retaining DFL for YOLOv8."""

    def __init__(self, reg_max: int = 16, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7) -> None:
        super().__init__(reg_max)
        self.focal_eiou = FocalEIoULoss(alpha=alpha, gamma=gamma, eps=eps, reduction="none")
        self.eps = eps

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fg_mask.sum() == 0:
            zero = torch.zeros(1, device=pred_bboxes.device, dtype=pred_bboxes.dtype)
            return zero, zero

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_sel = pred_bboxes[fg_mask]
        target_sel = target_bboxes[fg_mask]

        eiou_loss = self.focal_eiou(pred_sel, target_sel).unsqueeze(-1)
        loss_iou = (eiou_loss * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            ).sum() / target_scores_sum
        else:
            loss_dfl = torch.zeros(1, device=pred_bboxes.device, dtype=pred_bboxes.dtype)

        return loss_iou, loss_dfl


class v8DetectionLossWithFocalEIoU(v8DetectionLoss):
    """Detection loss wrapper that swaps the regression term for Focal-EIoU."""

    def __init__(self, model, tal_topk: int = 10, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7) -> None:
        super().__init__(model, tal_topk=tal_topk)
        self.bbox_loss = FocalEIoUBboxLoss(self.reg_max, alpha=alpha, gamma=gamma, eps=eps).to(self.device)

    def __call__(self, preds: Any, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(preds, batch)
