#!/usr/bin/env python3
"""Image-ID deterministic variant of the locked E1 maritime degradation.

The original transform samples from Python's process-global RNG each time an
image is seen. This module preserves the exact branch probabilities, severity
probabilities, parameter ranges, and tensor operations, but derives an
independent local RNG from a stable image ID. Therefore an image receives the
same branch, severity, and numeric parameters in every epoch, in every batch
order, and for every training seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

import maritime_degradation as base


ASSIGNMENT_SCHEMA = "ring3.wsodd_image_id_deterministic_mdg.v1"
DEFAULT_NAMESPACE = "ring3/wsodd/sequence-disjoint-v1/e1-mid/image-id-mdg/v1"
DEFAULT_IMAGE_HEIGHT = 960
DEFAULT_IMAGE_WIDTH = 960


def canonical_image_id(value: Any) -> str:
    """Return the filename stem used as the stable WSODD image ID."""
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise ValueError("MDG image_id source is empty")
    filename = text.rsplit("/", 1)[-1]
    if not filename or filename in {".", ".."}:
        raise ValueError(f"Invalid MDG image_id source: {value!r}")
    image_id = filename
    lowered = filename.casefold()
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        if lowered.endswith(suffix):
            image_id = filename[: -len(suffix)]
            break
    if not image_id:
        raise ValueError(f"Image filename has no stable stem: {value!r}")
    return image_id


def _sample_categorical(
    rng: random.Random,
    probabilities: dict[str, Any],
    ordered_keys: tuple[str, ...],
) -> str:
    draw = rng.random()
    cumulative = 0.0
    for key in ordered_keys:
        cumulative += float(probabilities[key])
        if draw < cumulative:
            return key
    return ordered_keys[-1]


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ImageIdDeterministicMaritimeDegradation(base.MaritimeDegradation):
    """Locked MDG with a seed-independent assignment for each image ID."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        super().__init__(config)
        namespace = str(namespace).strip()
        if not namespace:
            raise ValueError("Deterministic MDG namespace must not be empty")
        self.namespace = namespace
        self._pending_image_ids: list[str] | None = None
        self._last_assignments: list[dict[str, Any]] = []
        self._observed_assignment_sha256: dict[str, str] = {}
        self._assignment_recheck_count = 0
        self._assignment_conflict_count = 0

    def _rng(self, image_id: str) -> random.Random:
        material = f"{self.namespace}\0{image_id}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        return random.Random(int.from_bytes(digest, byteorder="big", signed=False))

    def assignment_for_image_id(
        self,
        image_id: Any,
        *,
        height: int = DEFAULT_IMAGE_HEIGHT,
        width: int = DEFAULT_IMAGE_WIDTH,
    ) -> dict[str, Any]:
        image_id = canonical_image_id(image_id)
        if height < 4 or width < 6:
            raise ValueError(f"Image shape is too small for MDG: {(height, width)}")
        rng = self._rng(image_id)
        branch = _sample_categorical(rng, self.branch_probabilities, base.BRANCHES)
        assignment: dict[str, Any] = {
            "schema": ASSIGNMENT_SCHEMA,
            "namespace": self.namespace,
            "image_id": image_id,
            "branch": branch,
            "severity": None,
            "parameters": {},
            "height": int(height),
            "width": int(width),
        }
        if branch in {"clean", "standard_aug_only"}:
            assignment["assignment_sha256"] = _json_sha256(assignment)
            return assignment

        severity = _sample_categorical(rng, self.severity_probabilities, base.SEVERITIES)
        parameters: dict[str, Any] = {}
        if branch in {"low_light", "mixed"}:
            low_light = self.config["low_light"][severity]
            parameters["brightness"] = rng.uniform(
                float(low_light["brightness"][0]),
                float(low_light["brightness"][1]),
            )
            parameters["gamma"] = rng.uniform(
                float(low_light["gamma"][0]),
                float(low_light["gamma"][1]),
            )
        if branch in {"glare", "mixed"}:
            glare = self.config["glare"][severity]
            target_area_ratio = rng.uniform(
                float(glare["target_area_ratio"][0]),
                float(glare["target_area_ratio"][1]),
            )
            aspect = rng.uniform(
                float(self.config["glare"]["aspect_ratio"][0]),
                float(self.config["glare"]["aspect_ratio"][1]),
            )
            target_area = target_area_ratio * float(height * width)
            radius_y = max(2, int(math.sqrt(target_area / (math.pi * aspect))))
            radius_x = max(3, int(radius_y * aspect))
            radius_y = min(radius_y, max(2, height // 2 - 1))
            radius_x = min(radius_x, max(3, width // 2 - 1))
            x_low, x_high = radius_x, width - radius_x - 1
            y_low, y_high = radius_y, height - radius_y - 1
            center_x = rng.randint(x_low, x_high) if x_high >= x_low else width // 2
            center_y = rng.randint(y_low, y_high) if y_high >= y_low else height // 2
            parameters.update(
                {
                    "target_area_ratio": target_area_ratio,
                    "aspect_ratio": aspect,
                    "radius_x": radius_x,
                    "radius_y": radius_y,
                    "center_x": center_x,
                    "center_y": center_y,
                    "softness": rng.uniform(
                        float(self.config["glare"]["softness"][0]),
                        float(self.config["glare"]["softness"][1]),
                    ),
                    "intensity_255": rng.uniform(
                        float(glare["intensity_255"][0]),
                        float(glare["intensity_255"][1]),
                    ),
                }
            )
        assignment["severity"] = severity
        assignment["parameters"] = parameters
        assignment["assignment_sha256"] = _json_sha256(assignment)
        return assignment

    def bind_batch_image_ids(self, image_ids: Sequence[Any]) -> list[str]:
        canonical = [canonical_image_id(value) for value in image_ids]
        if not canonical:
            raise ValueError("Deterministic MDG received an empty image_id batch")
        self._pending_image_ids = canonical
        return canonical

    def _apply_assignment(
        self,
        image: torch.Tensor,
        assignment: dict[str, Any],
    ) -> torch.Tensor:
        branch = str(assignment["branch"])
        parameters = assignment["parameters"]
        if branch in {"low_light", "mixed"}:
            image = image.clamp(0.0, 1.0).pow(float(parameters["gamma"])).mul(
                float(parameters["brightness"])
            )
        if branch in {"glare", "mixed"}:
            yy, xx = self._grid(image)
            radius_x = int(parameters["radius_x"])
            radius_y = int(parameters["radius_y"])
            dx = (xx - float(parameters["center_x"])) / float(max(radius_x, 1))
            dy = (yy - float(parameters["center_y"])) / float(max(radius_y, 1))
            distance = dx.square() + dy.square()
            soft_mask = torch.exp(-distance * float(parameters["softness"])) * (
                distance <= 1.0
            )
            intensity = float(parameters["intensity_255"]) / 255.0
            image = image.add(soft_mask.unsqueeze(0) * intensity).clamp(0.0, 1.0)
        return image

    def apply_batch(
        self,
        images: torch.Tensor,
        image_ids: Sequence[Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"MDG expects BCHW RGB tensors, got shape={tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError(f"MDG expects floating-point normalized tensors, got {images.dtype}")
        if image_ids is None:
            image_ids = self._pending_image_ids
        self._pending_image_ids = None
        if image_ids is None:
            raise RuntimeError("Deterministic MDG image IDs were not bound before apply_batch")
        canonical = [canonical_image_id(value) for value in image_ids]
        if len(canonical) != int(images.shape[0]):
            raise RuntimeError(
                "Deterministic MDG image count mismatch: "
                f"ids={len(canonical)}, images={int(images.shape[0])}"
            )

        height, width = (int(value) for value in images.shape[-2:])
        assignments: list[dict[str, Any]] = []
        counts: Counter[str] = Counter(images_total=int(images.shape[0]))
        for index, image_id in enumerate(canonical):
            assignment = self.assignment_for_image_id(
                image_id,
                height=height,
                width=width,
            )
            assignments.append(assignment)
            branch = str(assignment["branch"])
            counts[f"branch_{branch}"] += 1
            severity = assignment.get("severity")
            if severity is not None:
                counts[f"severity_{severity}"] += 1
            signature = str(assignment["assignment_sha256"])
            previous = self._observed_assignment_sha256.get(image_id)
            if previous is None:
                self._observed_assignment_sha256[image_id] = signature
            else:
                self._assignment_recheck_count += 1
                if previous != signature:
                    self._assignment_conflict_count += 1
                    raise RuntimeError(f"Deterministic assignment drifted for image_id={image_id}")
            if branch not in {"clean", "standard_aug_only"}:
                images[index] = self._apply_assignment(images[index], assignment)
        self._last_assignments = assignments

        if not bool(torch.isfinite(images).all()):
            raise FloatingPointError("Deterministic MDG produced a non-finite image tensor")
        if bool((images < 0.0).any()) or bool((images > 1.0).any()):
            raise FloatingPointError("Deterministic MDG produced values outside [0, 1]")
        return images, dict(counts)

    def runtime_evidence(self) -> dict[str, Any]:
        compact = [
            {
                "image_id": item["image_id"],
                "branch": item["branch"],
                "severity": item["severity"],
                "assignment_sha256": item["assignment_sha256"],
            }
            for item in self._last_assignments
        ]
        return {
            "deterministic_degradation_assignment": True,
            "deterministic_assignment_schema": ASSIGNMENT_SCHEMA,
            "deterministic_assignment_namespace": self.namespace,
            "deterministic_assignment_key": "WSODD filename stem (image_id)",
            "deterministic_assignment_seed_independent": True,
            "deterministic_assignment_epoch_independent": True,
            "deterministic_assignment_batch_order_independent": True,
            "deterministic_assignment_current_batch": compact,
            "deterministic_assignment_current_batch_sha256": _json_sha256(compact),
            "deterministic_assignment_unique_ids_observed": len(
                self._observed_assignment_sha256
            ),
            "deterministic_assignment_recheck_count": self._assignment_recheck_count,
            "deterministic_assignment_conflict_count": self._assignment_conflict_count,
        }


def summarize_assignments(
    transform: ImageIdDeterministicMaritimeDegradation,
    image_ids: Iterable[Any],
    *,
    height: int = DEFAULT_IMAGE_HEIGHT,
    width: int = DEFAULT_IMAGE_WIDTH,
) -> dict[str, Any]:
    canonical = [canonical_image_id(value) for value in image_ids]
    if len(set(canonical)) != len(canonical):
        duplicates = [
            key
            for key, count in Counter(canonical).items()
            if count > 1
        ]
        raise RuntimeError(f"image_id values are not unique: {duplicates[:10]}")
    assignments = [
        transform.assignment_for_image_id(image_id, height=height, width=width)
        for image_id in sorted(canonical)
    ]
    counts: Counter[str] = Counter()
    for item in assignments:
        counts[f"branch_{item['branch']}"] += 1
        if item.get("severity") is not None:
            counts[f"severity_{item['severity']}"] += 1
    compact = [
        {
            "image_id": item["image_id"],
            "branch": item["branch"],
            "severity": item["severity"],
            "parameters": item["parameters"],
            "assignment_sha256": item["assignment_sha256"],
        }
        for item in assignments
    ]
    return {
        "schema": "ring3.wsodd_image_id_deterministic_mdg_map_summary.v1",
        "assignment_schema": ASSIGNMENT_SCHEMA,
        "namespace": transform.namespace,
        "image_id_source": "filename_stem",
        "image_count": len(canonical),
        "unique_image_id_count": len(set(canonical)),
        "height": int(height),
        "width": int(width),
        "counts": dict(sorted(counts.items())),
        "assignment_map_sha256": _json_sha256(compact),
    }


def self_test(config_path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    transform = ImageIdDeterministicMaritimeDegradation(base.load_config(config_path))
    image_ids = ["wsodd_demo_0001", "wsodd_demo_0002", "wsodd_demo_0003"]
    torch.manual_seed(0)
    images = torch.rand((len(image_ids), 3, 96, 96), dtype=torch.float32)
    first, first_counts = transform.apply_batch(images.clone(), image_ids)
    second, second_counts = transform.apply_batch(images.clone(), image_ids)
    reverse, reverse_counts = transform.apply_batch(
        images.flip(0).clone(),
        list(reversed(image_ids)),
    )
    if not torch.equal(first, second) or not torch.equal(first, reverse.flip(0)):
        raise RuntimeError("Deterministic MDG repeatability or batch-order test failed")
    if first_counts != second_counts or first_counts != reverse_counts:
        raise RuntimeError("Deterministic MDG accounting changed across equivalent batches")
    payload: dict[str, Any] = {
        "schema": "ring3.wsodd_image_id_deterministic_mdg_self_test.v1",
        "status": "passed",
        "same_image_same_tensor": True,
        "batch_order_independent": True,
        "seed_independent_by_construction": True,
        "finite": bool(torch.isfinite(first).all()),
        "counts": first_counts,
        "runtime_evidence": transform.runtime_evidence(),
    }
    if manifest_path is not None:
        entries = [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        first_summary = summarize_assignments(transform, entries)
        second_summary = summarize_assignments(transform, reversed(entries))
        if first_summary != second_summary:
            raise RuntimeError("Full assignment map is not order-independent")
        payload["train_assignment_map"] = first_summary
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate image-ID deterministic maritime degradation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            self_test(args.config, args.manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
