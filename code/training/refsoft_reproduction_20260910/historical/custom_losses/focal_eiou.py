"""Focal-EIoU bounding box regression loss.

This module provides an implementation of the Focal-EIoU loss that combines
Efficient IoU with a focal modulation term to emphasize high-overlap samples.
The implementation follows the optimisation notes provided in 优化改进方案.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn


def _split_xyxy(box: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split an ``(N, 4)`` tensor into ``(x1, y1, x2, y2)`` components."""
    return box[..., 0], box[..., 1], box[..., 2], box[..., 3]


def iou_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Standard IoU for boxes given in ``xyxy`` format."""
    px1, py1, px2, py2 = _split_xyxy(pred)
    tx1, ty1, tx2, ty2 = _split_xyxy(target)

    inter_w = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0.0)
    inter_h = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(min=0.0)
    inter = inter_w * inter_h

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    union = pw * ph + tw * th - inter + eps
    return inter / union


def efficient_iou_xyxy(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Efficient IoU (EIoU) for boxes given in ``xyxy`` format."""
    px1, py1, px2, py2 = _split_xyxy(pred)
    tx1, ty1, tx2, ty2 = _split_xyxy(target)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    # Base IoU
    inter_w = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0.0)
    inter_h = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(min=0.0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    # Center distance term
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5

    cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
    ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    # Aspect ratio alignment terms
    cw_term = (tw - pw).pow(2) / (cw.pow(2) + eps)
    ch_term = (th - ph).pow(2) / (ch.pow(2) + eps)

    return iou - (rho2 / c2 + cw_term + ch_term)


@dataclass
class FocalEIoUConfig:
    """Configuration for :class:`FocalEIoULoss`."""

    alpha: float = 0.25
    gamma: float = 2.0
    eps: float = 1e-7
    reduction: Literal["mean", "sum", "none"] = "mean"


class FocalEIoULoss(nn.Module):
    """Focal-EIoU regression loss for bounding boxes in ``xyxy`` space."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.cfg = FocalEIoUConfig(alpha=alpha, gamma=gamma, eps=eps, reduction=reduction)  # freeze config

    @property
    def alpha(self) -> float:
        return self.cfg.alpha

    @property
    def gamma(self) -> float:
        return self.cfg.gamma

    @property
    def eps(self) -> float:
        return self.cfg.eps

    @property
    def reduction(self) -> str:
        return self.cfg.reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape[-1] != 4 or target.shape[-1] != 4:
            raise ValueError("pred and target must have shape (*, 4) in xyxy format")

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        iou = iou_xyxy(pred, target, self.eps)
        eiou = efficient_iou_xyxy(pred, target, self.eps)
        eiou = torch.clamp(eiou, min=-1.0, max=1.0)

        focal_term = torch.clamp(1.0 - iou, min=0.0) ** self.gamma
        weight = self.alpha + (1.0 - self.alpha) * focal_term
        base_loss = torch.clamp(1.0 - eiou, min=0.0)
        loss = base_loss * weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


__all__ = [
    "FocalEIoULoss",
    "FocalEIoUConfig",
    "efficient_iou_xyxy",
    "iou_xyxy",
]
