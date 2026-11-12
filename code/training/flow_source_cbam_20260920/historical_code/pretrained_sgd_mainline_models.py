from __future__ import annotations

from typing import Any

from cbam_focal_formal_models import FactorialDetectionModel
from e3_small_object_weighting import SmallObjectWeightedDetectionLoss


class MainlineSmallWeightedDetectionModel(FactorialDetectionModel):
    """Plain YOLOv8s plus degraded-small weighting, without CBAM or Focal-EIoU."""

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
            use_cbam=False,
            use_focal_eiou=False,
        )

    def init_criterion(self) -> SmallObjectWeightedDetectionLoss:
        return SmallObjectWeightedDetectionLoss(
            self,
            degraded_small_weight=self.degraded_small_weight,
        )

    def factorial_evidence(self) -> dict[str, Any]:
        evidence = super().factorial_evidence()
        criterion = getattr(self, "criterion", None)
        evidence["criterion_type"] = type(criterion).__name__ if criterion is not None else None
        evidence["bbox_loss_type"] = (
            type(getattr(criterion, "bbox_loss", None)).__name__ if criterion is not None else None
        )
        evidence["degraded_small_weighting_enabled"] = True
        if isinstance(criterion, SmallObjectWeightedDetectionLoss):
            evidence.update(criterion.last_evidence())
        return evidence
