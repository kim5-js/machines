from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch


BRANCHES = ("clean", "low_light", "glare", "mixed", "standard_aug_only")
SEVERITIES = ("mild", "medium", "strong")


def _ordered_range(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element JSON list")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{label} must be finite and ordered, got {value!r}")
    return low, high


def _validate_probabilities(payload: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    if set(payload) != set(keys):
        raise ValueError(f"{label} must contain exactly {keys}, got {tuple(payload)}")
    values = [float(payload[key]) for key in keys]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{label} contains an invalid probability: {payload!r}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} must sum to 1.0, got {sum(values):.12f}")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "ring3.e1_mdg_config.v1":
        raise ValueError(f"Unexpected MDG config schema: {config.get('schema')!r}")
    if config.get("stage") != "E1":
        raise ValueError(f"MDG config must target E1, got {config.get('stage')!r}")

    _validate_probabilities(config["branch_probabilities"], BRANCHES, "branch_probabilities")
    _validate_probabilities(config["severity_probabilities"], SEVERITIES, "severity_probabilities")

    for severity in SEVERITIES:
        brightness = _ordered_range(config["low_light"][severity]["brightness"], f"low_light.{severity}.brightness")
        gamma = _ordered_range(config["low_light"][severity]["gamma"], f"low_light.{severity}.gamma")
        if not (0.0 < brightness[0] <= brightness[1] <= 1.0):
            raise ValueError(f"Invalid brightness range for {severity}: {brightness}")
        if gamma[0] < 1.0:
            raise ValueError(f"Gamma must be at least 1.0 for {severity}: {gamma}")

        area = _ordered_range(config["glare"][severity]["target_area_ratio"], f"glare.{severity}.target_area_ratio")
        intensity = _ordered_range(config["glare"][severity]["intensity_255"], f"glare.{severity}.intensity_255")
        if not (0.0 < area[0] <= area[1] <= 0.25):
            raise ValueError(f"Invalid glare area range for {severity}: {area}")
        if not (0.0 <= intensity[0] <= intensity[1] <= 255.0):
            raise ValueError(f"Invalid glare intensity range for {severity}: {intensity}")

    aspect = _ordered_range(config["glare"]["aspect_ratio"], "glare.aspect_ratio")
    softness = _ordered_range(config["glare"]["softness"], "glare.softness")
    if aspect[0] <= 0.0 or softness[0] <= 0.0:
        raise ValueError("Glare aspect ratio and softness must be positive")

    invariants = config.get("invariants", {})
    required_true = (
        "train_only",
        "validation_unchanged",
        "labels_unchanged",
        "dataset_manifests_unchanged",
        "applied_after_standard_yolo_augmentation",
    )
    if any(invariants.get(key) is not True for key in required_true):
        raise ValueError(f"E1 invariants are incomplete: {invariants!r}")
    if invariants.get("consistency_loss") is not False or invariants.get("curriculum") is not False:
        raise ValueError("E1 must not enable consistency loss or curriculum")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _sample_categorical(probabilities: dict[str, Any], ordered_keys: tuple[str, ...]) -> str:
    draw = random.random()
    cumulative = 0.0
    for key in ordered_keys:
        cumulative += float(probabilities[key])
        if draw < cumulative:
            return key
    return ordered_keys[-1]


class MaritimeDegradation:
    """Apply train-only online maritime photometric degradation to normalized BCHW tensors."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_config(config)
        self.config = config
        self.branch_probabilities = config["branch_probabilities"]
        self.severity_probabilities = config["severity_probabilities"]
        self._grid_cache: dict[tuple[str, int | None, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _uniform(bounds: list[float]) -> float:
        return random.uniform(float(bounds[0]), float(bounds[1]))

    def _grid(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        device = image.device
        key = (device.type, device.index, image.dtype, height, width)
        cached = self._grid_cache.get(key)
        if cached is None:
            yy = torch.arange(height, device=device, dtype=image.dtype).view(height, 1)
            xx = torch.arange(width, device=device, dtype=image.dtype).view(1, width)
            cached = (yy, xx)
            self._grid_cache[key] = cached
        return cached

    def _apply_low_light(self, image: torch.Tensor, severity: str) -> torch.Tensor:
        spec = self.config["low_light"][severity]
        brightness = self._uniform(spec["brightness"])
        gamma = self._uniform(spec["gamma"])
        return image.clamp(0.0, 1.0).pow(gamma).mul(brightness)

    def _apply_glare(self, image: torch.Tensor, severity: str) -> torch.Tensor:
        height, width = image.shape[-2:]
        spec = self.config["glare"][severity]
        target_area = self._uniform(spec["target_area_ratio"]) * float(height * width)
        aspect = self._uniform(self.config["glare"]["aspect_ratio"])
        radius_y = max(2, int(math.sqrt(target_area / (math.pi * aspect))))
        radius_x = max(3, int(radius_y * aspect))
        radius_y = min(radius_y, max(2, height // 2 - 1))
        radius_x = min(radius_x, max(3, width // 2 - 1))

        x_low, x_high = radius_x, width - radius_x - 1
        y_low, y_high = radius_y, height - radius_y - 1
        center_x = random.randint(x_low, x_high) if x_high >= x_low else width // 2
        center_y = random.randint(y_low, y_high) if y_high >= y_low else height // 2

        yy, xx = self._grid(image)
        dx = (xx - float(center_x)) / float(max(radius_x, 1))
        dy = (yy - float(center_y)) / float(max(radius_y, 1))
        distance = dx.square() + dy.square()
        softness = self._uniform(self.config["glare"]["softness"])
        soft_mask = torch.exp(-distance * softness) * (distance <= 1.0)
        intensity = self._uniform(spec["intensity_255"]) / 255.0
        return image.add(soft_mask.unsqueeze(0) * intensity).clamp(0.0, 1.0)

    def apply_batch(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"MDG expects BCHW RGB tensors, got shape={tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError(f"MDG expects floating-point normalized tensors, got {images.dtype}")

        counts: Counter[str] = Counter(images_total=int(images.shape[0]))
        for index in range(int(images.shape[0])):
            branch = _sample_categorical(self.branch_probabilities, BRANCHES)
            counts[f"branch_{branch}"] += 1
            if branch in {"clean", "standard_aug_only"}:
                continue

            severity = _sample_categorical(self.severity_probabilities, SEVERITIES)
            counts[f"severity_{severity}"] += 1
            image = images[index]
            if branch in {"low_light", "mixed"}:
                image = self._apply_low_light(image, severity)
            if branch in {"glare", "mixed"}:
                image = self._apply_glare(image, severity)
            images[index] = image

        if not bool(torch.isfinite(images).all()):
            raise FloatingPointError("MDG produced a non-finite image tensor")
        if bool((images < 0.0).any()) or bool((images > 1.0).any()):
            raise FloatingPointError("MDG produced values outside [0, 1]")
        return images, dict(counts)


def self_test(config_path: Path) -> dict[str, Any]:
    random.seed(0)
    torch.manual_seed(0)
    config = load_config(config_path)
    transform = MaritimeDegradation(config)
    images = torch.rand((128, 3, 96, 96), dtype=torch.float32)
    original = images.clone()
    output, counts = transform.apply_batch(images)
    branch_total = sum(counts.get(f"branch_{name}", 0) for name in BRANCHES)
    active = sum(counts.get(f"branch_{name}", 0) for name in ("low_light", "glare", "mixed"))
    if branch_total != images.shape[0] or active <= 0:
        raise RuntimeError(f"MDG branch accounting failed: {counts}")
    if torch.equal(original, output):
        raise RuntimeError("MDG self-test did not modify any image")
    return {
        "status": "passed",
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "min": float(output.min()),
        "max": float(output.max()),
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the E1 online maritime degradation implementation")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(self_test(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
