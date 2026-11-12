from __future__ import annotations
import hashlib,json,math,os
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model
from cbam_focal_formal_models import CBAMFocalSmallWeightedDetectionModel,FactorialDetectionModel,SmallObjectWeightedFocalEIoULoss
from maritime_degradation import MaritimeDegradation,load_config as load_mdg_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

class FactorialTrainer(DetectionTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.variant = json.loads(os.environ["RING3_FACTORIAL_VARIANT_JSON"])
        self.dual_view = bool(self.variant["dual_view"])
        self.small_weighting = bool(self.variant["degraded_small_weighting"])
        self.mdg = None
        if self.dual_view:
            mdg_path = Path(os.environ["RING3_FACTORIAL_MDG_CONFIG"])
            if sha256(mdg_path) != os.environ["RING3_FACTORIAL_MDG_CONFIG_SHA256"]:
                raise RuntimeError("MDG config changed before trainer construction")
            self.mdg = MaritimeDegradation(load_mdg_config(mdg_path))
        self._last_mdg_counts: dict[str, int] = {}
        self._last_target_counts: dict[str, int] = {}
        self._mdg_epoch_counts: Counter[str] = Counter()
        self._target_epoch_counts: Counter[str] = Counter()
        self._mdg_total_counts: Counter[str] = Counter()
        self._target_total_counts: Counter[str] = Counter()
        self._optimizer_update_probe: dict[str, Any] = {}
        self._optimizer_step_calls = 0
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        if self.small_weighting:
            model = CBAMFocalSmallWeightedDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                degraded_small_weight=1.5,
            )
        else:
            model = FactorialDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                use_cbam=bool(self.variant["cbam"]),
                use_focal_eiou=bool(self.variant["focal_eiou"]),
            )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        clean = batch["img"]
        pair_batch_size = int(clean.shape[0])
        source_batch_idx = batch["batch_idx"]
        source_cls = batch["cls"]
        source_bboxes = batch["bboxes"]
        if not self.dual_view:
            self._last_target_counts = {
                "images_total": pair_batch_size,
                "clean_targets": int(source_bboxes.shape[0]),
            }
            self._target_epoch_counts.update(self._last_target_counts)
            self._target_total_counts.update(self._last_target_counts)
            return batch

        if self.mdg is None:
            raise RuntimeError("Dual-view run has no MDG instance")
        degraded, mdg_counts = self.mdg.apply_batch(clean.clone())
        batch["img"] = torch.cat((clean, degraded), dim=0)
        batch["batch_idx"] = torch.cat((source_batch_idx, source_batch_idx + pair_batch_size), dim=0)
        batch["cls"] = torch.cat((source_cls, source_cls.clone()), dim=0)
        batch["bboxes"] = torch.cat((source_bboxes, source_bboxes.clone()), dim=0)
        batch["factorial_pair_batch_size"] = pair_batch_size
        if self.small_weighting:
            area_px2 = (
                source_bboxes[:, 2] * float(clean.shape[-1])
                * source_bboxes[:, 3] * float(clean.shape[-2])
            )
            small_mask = area_px2 < 1024.0
            clean_weights = torch.ones(
                source_bboxes.shape[0], device=source_bboxes.device, dtype=source_bboxes.dtype
            )
            degraded_weights = torch.where(
                small_mask, torch.full_like(clean_weights, 1.5), clean_weights
            )
            batch["factorial_gt_weights"] = torch.cat((clean_weights, degraded_weights), dim=0)
        else:
            small_mask = torch.zeros(source_bboxes.shape[0], device=source_bboxes.device, dtype=torch.bool)
        for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad"):
            value = batch.get(key)
            if isinstance(value, list):
                batch[key] = value + list(value)
            elif isinstance(value, tuple) and len(value) == pair_batch_size:
                batch[key] = value + value
        target_counts = {
            "images_total": pair_batch_size,
            "clean_targets": int(source_bboxes.shape[0]),
            "degraded_targets": int(source_bboxes.shape[0]),
            "degraded_small_targets": int(small_mask.sum()),
            "degraded_regular_targets": int((~small_mask).sum()),
        }
        self._last_mdg_counts = dict(mdg_counts)
        self._last_target_counts = target_counts
        self._mdg_epoch_counts.update(mdg_counts)
        self._target_epoch_counts.update(target_counts)
        self._mdg_total_counts.update(mdg_counts)
        self._target_total_counts.update(target_counts)
        return batch

    def optimizer_step(self) -> None:
        self._optimizer_step_calls += 1
        model = unwrap_model(self.model)
        parameter_before = None
        target_parameter = None
        pre_step_gradient: dict[str, Any] = {}
        if not self._optimizer_update_probe and isinstance(model, FactorialDetectionModel):
            parameters = model.p3_target_parameters()
            if parameters:
                target_parameter = parameters[0]
                parameter_before = target_parameter.detach().float().clone()
                if target_parameter.grad is not None:
                    gradient = target_parameter.grad.detach().float()
                    pre_step_gradient = {
                        "finite": bool(torch.isfinite(gradient).all()),
                        "nonzero": bool((gradient != 0).any()),
                        "l2_norm_scaled": float(gradient.square().sum().sqrt()),
                    }
        super().optimizer_step()
        if parameter_before is not None and target_parameter is not None:
            delta = target_parameter.detach().float() - parameter_before
            candidate = {
                "optimizer_step_index": self._optimizer_step_calls,
                "target_feature_layer_index": model.factorial_p3_layer_index,
                "target_parameter_shape": [int(value) for value in target_parameter.shape],
                "pre_step_gradient": pre_step_gradient,
                "finite": bool(torch.isfinite(delta).all()),
                "parameter_delta_l2": float(delta.square().sum().sqrt()),
                "parameter_delta_max_abs": float(delta.abs().max()),
            }
            if candidate["finite"] is not True:
                raise FloatingPointError(f"P3 optimizer update became non-finite: {candidate}")
            if math.isfinite(candidate["parameter_delta_l2"]) and candidate["parameter_delta_l2"] > 0.0:
                self._optimizer_update_probe = candidate

    def runtime_evidence(self) -> dict[str, Any]:
        model = unwrap_model(self.model)
        if not isinstance(model, FactorialDetectionModel):
            raise TypeError(f"Unexpected factorial model type: {type(model).__name__}")
        return {
            "variant": self.variant,
            **model.factorial_evidence(),
            "dual_view_enabled": self.dual_view,
            "degraded_small_weighting_enabled": self.small_weighting,
            "mdg_counts_current_batch": dict(self._last_mdg_counts),
            "target_counts_current_batch": dict(self._last_target_counts),
            "optimizer_update_probe": dict(self._optimizer_update_probe),
            "consistency_loss_enabled": False,
            "severity_weighting_enabled": False,
            "curriculum_enabled": False,
        }

    def consume_epoch_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        mdg, targets = dict(self._mdg_epoch_counts), dict(self._target_epoch_counts)
        self._mdg_epoch_counts.clear()
        self._target_epoch_counts.clear()
        return mdg, targets

    def total_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        return dict(self._mdg_total_counts), dict(self._target_total_counts)

def validate_runtime_evidence(evidence: dict[str, Any], variant: dict[str, Any]) -> None:
    if evidence.get("variant", {}).get("key") != variant["key"]:
        raise RuntimeError("Runtime evidence belongs to a different run")
    if evidence.get("cbam_enabled") is not bool(variant["cbam"]):
        raise RuntimeError("CBAM enablement evidence is wrong")
    if evidence.get("focal_eiou_enabled") is not bool(variant["focal_eiou"]):
        raise RuntimeError("Focal-EIoU enablement evidence is wrong")
    expected_cbam_types = ["C2f_CBAM", "C2f_CBAM"] if variant["cbam"] else ["C2f", "C2f"]
    if evidence.get("cbam_layer_types") != expected_cbam_types:
        raise RuntimeError(f"CBAM topology evidence failed: {evidence.get('cbam_layer_types')}")
    if variant["cbam"] and int(evidence.get("cbam_attention_parameter_count", 0)) <= 0:
        raise RuntimeError("CBAM attention parameters are absent")
    if not variant["cbam"] and int(evidence.get("cbam_attention_parameter_count", -1)) != 0:
        raise RuntimeError("CBAM attention unexpectedly exists in a control")
    criterion_type = str(evidence.get("criterion_type"))
    if variant["degraded_small_weighting"]:
        if criterion_type != "SmallObjectWeightedFocalEIoULoss":
            raise RuntimeError(f"Combined E3 criterion is wrong: {criterion_type}")
    elif variant["focal_eiou"]:
        if criterion_type != "v8DetectionLossWithFocalEIoU":
            raise RuntimeError(f"Focal-EIoU criterion is wrong: {criterion_type}")
    elif criterion_type != "v8DetectionLoss":
        raise RuntimeError(f"Control criterion is wrong: {criterion_type}")
    if variant["focal_eiou"]:
        focal = evidence.get("focal_eiou") or {}
        if focal.get("dfl_preserved") is not True:
            raise RuntimeError("DFL preservation evidence is missing")
        for key, expected in (("alpha", 0.25), ("gamma", 2.0), ("eps", 1e-7)):
            if not math.isclose(float(focal.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Focal-EIoU {key} changed")
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
    if variant["dual_view"]:
        if int(targets.get("clean_targets", -1)) != int(targets.get("degraded_targets", -2)):
            raise RuntimeError("Clean/degraded target counts differ")
        mdg = evidence.get("mdg_counts_current_batch", {})
        if int(mdg.get("images_total", -1)) != int(targets.get("images_total", -2)):
            raise RuntimeError("MDG/pair image counts differ")
    if variant["degraded_small_weighting"]:
        required = {
            "ordinary_target_weight": 1.0,
            "degraded_small_target_weight": 1.5,
            "normalization_denominator_unchanged": True,
            "background_anchor_weight_unchanged": True,
            "clean_weighted_target_violations": 0,
            "negative_anchor_weight_violations": 0,
        }
        for key, expected in required.items():
            if evidence.get(key) != expected:
                raise RuntimeError(f"Combined E3 weighting evidence not ready for {key}: {evidence.get(key)!r}")
        if int(evidence.get("weighted_positive_anchor_count", 0)) <= 0:
            raise RuntimeError("No degraded-small positive anchor has been weighted yet")
        if float(evidence.get("weighting_delta_scaled", 0.0)) <= 0.0:
            raise RuntimeError("Degraded-small weighting did not increase the detection objective")
