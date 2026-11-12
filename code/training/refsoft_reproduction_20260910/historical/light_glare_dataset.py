from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_entries(dataset_yaml: Path, split_key: str) -> tuple[dict[str, Any], list[Path]]:
    config = read_yaml(dataset_yaml)
    raw_base = Path(config.get("path", dataset_yaml.parent))
    base = raw_base if raw_base.is_absolute() else (dataset_yaml.parent / raw_base).resolve()
    raw_entry = config.get(split_key)
    if raw_entry is None:
        raise RuntimeError(f"Dataset yaml has no '{split_key}' entry: {dataset_yaml}")

    raw_entries = raw_entry if isinstance(raw_entry, list) else [raw_entry]
    resolved_entries = [(base / Path(item)).resolve() for item in raw_entries]
    return config, resolved_entries


def iter_images_from_entry(entry: Path) -> Iterable[Path]:
    if entry.is_dir():
        for path in sorted(entry.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path.resolve()
        return

    if entry.is_file() and entry.suffix.lower() == ".txt":
        for line in entry.read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            yield Path(candidate).expanduser().resolve()
        return

    raise RuntimeError(f"Unsupported dataset entry: {entry}")


def resolve_image_paths(dataset_yaml: Path, split_key: str) -> list[Path]:
    _, entries = resolve_entries(dataset_yaml, split_key)
    images: list[Path] = []
    for entry in entries:
        images.extend(iter_images_from_entry(entry))
    if not images:
        raise RuntimeError(f"No images found for split '{split_key}' in {dataset_yaml}")
    return images


def resolve_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def apply_low_light(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    brightness = float(rng.uniform(0.82, 0.95))
    gamma = float(rng.uniform(1.05, 1.30))
    adjusted = np.power(np.clip(normalized, 0.0, 1.0), gamma) * brightness
    return np.clip(adjusted * 255.0, 0.0, 255.0).astype(np.uint8)


def apply_reflection_glare(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    max_area_ratio: float = 0.05,
    min_area_factor: float = 0.35,
    aspect_min: float = 1.4,
    aspect_max: float = 3.2,
    intensity_min: float = 22.0,
    intensity_max: float = 48.0,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    height, width = output.shape[:2]
    total_area = float(height * width)

    max_area = max(total_area * max_area_ratio, 64.0)
    target_area = float(rng.uniform(max_area * min_area_factor, max_area))
    aspect = float(rng.uniform(aspect_min, aspect_max))
    radius_y = max(10, int(np.sqrt(target_area / (np.pi * aspect))))
    radius_x = max(14, int(radius_y * aspect))

    center_x = int(rng.integers(radius_x, max(radius_x + 1, width - radius_x)))
    center_y = int(rng.integers(radius_y, max(radius_y + 1, height - radius_y)))

    yy, xx = np.ogrid[:height, :width]
    dx = (xx - center_x) / max(radius_x, 1)
    dy = (yy - center_y) / max(radius_y, 1)
    distance = dx * dx + dy * dy

    hard_mask = distance <= 1.0
    soft_mask = np.exp(-distance * float(rng.uniform(1.6, 2.4))) * hard_mask

    intensity = float(rng.uniform(intensity_min, intensity_max))
    glare = soft_mask[..., None] * intensity
    output = output + glare

    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def choose_transform(
    rng: np.random.Generator,
    *,
    low_light_prob: float,
    reflection_prob: float,
    allow_combined: bool,
) -> tuple[bool, bool]:
    use_low_light = bool(rng.random() < low_light_prob)
    use_reflection = bool(rng.random() < reflection_prob)

    if not use_low_light and not use_reflection:
        if low_light_prob >= reflection_prob:
            use_low_light = True
        else:
            use_reflection = True

    if not allow_combined and use_low_light and use_reflection:
        if rng.random() < 0.5:
            use_low_light = False
        else:
            use_reflection = False

    return use_low_light, use_reflection


def build_augmented_dataset(
    *,
    data_yaml: Path,
    output_root: Path,
    seed: int = 42,
    sample_ratio: float = 0.30,
    low_light_prob: float = 0.50,
    reflection_prob: float = 0.50,
    allow_combined: bool = False,
    reflection_max_area_ratio: float = 0.05,
    reflection_min_area_factor: float = 0.35,
    reflection_aspect_min: float = 1.4,
    reflection_aspect_max: float = 3.2,
    reflection_intensity_min: float = 22.0,
    reflection_intensity_max: float = 48.0,
    refresh: bool = False,
) -> Path:
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if not (0.0 < reflection_max_area_ratio <= 1.0):
        raise ValueError(f"reflection_max_area_ratio must be in (0, 1], got {reflection_max_area_ratio}")
    if not (0.0 < reflection_min_area_factor <= 1.0):
        raise ValueError(f"reflection_min_area_factor must be in (0, 1], got {reflection_min_area_factor}")
    if not (reflection_aspect_min > 0 and reflection_aspect_max >= reflection_aspect_min):
        raise ValueError(
            f"reflection aspect range must be positive and ordered, got {reflection_aspect_min}..{reflection_aspect_max}"
        )
    if not (reflection_intensity_min >= 0 and reflection_intensity_max >= reflection_intensity_min):
        raise ValueError(
            f"reflection intensity range must be non-negative and ordered, got {reflection_intensity_min}..{reflection_intensity_max}"
        )

    config = read_yaml(data_yaml)
    train_images = resolve_image_paths(data_yaml, "train")
    val_images = resolve_image_paths(data_yaml, "val")

    output_root = output_root.expanduser().resolve()
    aug_images_dir = output_root / "aug_train" / "images"
    aug_labels_dir = output_root / "aug_train" / "labels"
    train_manifest = output_root / "train_images.txt"
    val_manifest = output_root / "val_images.txt"
    output_yaml = output_root / "dataset.yaml"
    metadata_path = output_root / "metadata.json"

    desired_config = {
        "source_data_yaml": str(data_yaml.resolve()),
        "seed": seed,
        "sample_ratio": sample_ratio,
        "low_light_prob": low_light_prob,
        "reflection_prob": reflection_prob,
        "allow_combined": allow_combined,
        "reflection_max_area_ratio": reflection_max_area_ratio,
        "reflection_min_area_factor": reflection_min_area_factor,
        "reflection_aspect_min": reflection_aspect_min,
        "reflection_aspect_max": reflection_aspect_max,
        "reflection_intensity_min": reflection_intensity_min,
        "reflection_intensity_max": reflection_intensity_max,
    }

    if refresh and output_root.exists():
        shutil.rmtree(output_root)

    if output_yaml.exists() and metadata_path.exists() and not refresh:
        try:
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            current = None
        if current == desired_config:
            return output_yaml

    ensure_dir(output_root)
    ensure_dir(aug_images_dir)
    ensure_dir(aug_labels_dir)

    rng = np.random.default_rng(seed)
    sampled_indices = sorted(rng.choice(len(train_images), size=max(1, int(len(train_images) * sample_ratio)), replace=False))

    augmented_image_paths: list[Path] = []
    for index in sampled_indices:
        image_path = train_images[index]
        label_path = resolve_label_path(image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found for {image_path}: {label_path}")

        local_rng = np.random.default_rng(seed + index)
        use_low_light, use_reflection = choose_transform(
            local_rng,
            low_light_prob=low_light_prob,
            reflection_prob=reflection_prob,
            allow_combined=allow_combined,
        )

        image = np.asarray(Image.open(image_path).convert("RGB"))
        if use_low_light:
            image = apply_low_light(image, local_rng)
        if use_reflection:
            image = apply_reflection_glare(
                image,
                local_rng,
                max_area_ratio=reflection_max_area_ratio,
                min_area_factor=reflection_min_area_factor,
                aspect_min=reflection_aspect_min,
                aspect_max=reflection_aspect_max,
                intensity_min=reflection_intensity_min,
                intensity_max=reflection_intensity_max,
            )

        suffix_tokens = []
        if use_low_light:
            suffix_tokens.append("ll")
        if use_reflection:
            suffix_tokens.append("rf")
        variant_name = f"{image_path.stem}_{'_'.join(suffix_tokens)}{image_path.suffix.lower()}"

        out_image = aug_images_dir / variant_name
        out_label = aug_labels_dir / f"{Path(variant_name).stem}.txt"

        if out_image.exists():
            out_image.unlink()
        Image.fromarray(image).save(out_image, quality=95)
        if out_label.exists():
            out_label.unlink()
        shutil.copy2(label_path, out_label)
        augmented_image_paths.append(out_image.resolve())

    all_train_paths = [str(path) for path in train_images] + [str(path) for path in augmented_image_paths]
    train_manifest.write_text("\n".join(all_train_paths) + "\n", encoding="utf-8")
    val_manifest.write_text("\n".join(str(path) for path in val_images) + "\n", encoding="utf-8")

    output_config = {
        key: value
        for key, value in config.items()
        if key not in {"path", "train", "val"}
    }
    output_config["path"] = str(output_root)
    output_config["train"] = train_manifest.name
    output_config["val"] = val_manifest.name
    write_yaml(output_yaml, output_config)

    metadata_path.write_text(json.dumps(desired_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only low-light/reflection augmented YOLO dataset overlay")
    parser.add_argument("--data", required=True, help="Path to source dataset.yaml")
    parser.add_argument("--output-root", required=True, help="Directory that will hold the generated dataset overlay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-ratio", type=float, default=0.30)
    parser.add_argument("--low-light-prob", type=float, default=0.50)
    parser.add_argument("--reflection-prob", type=float, default=0.50)
    parser.add_argument("--allow-combined", action="store_true")
    parser.add_argument("--reflection-max-area-ratio", type=float, default=0.05)
    parser.add_argument("--reflection-min-area-factor", type=float, default=0.35)
    parser.add_argument("--reflection-aspect-min", type=float, default=1.4)
    parser.add_argument("--reflection-aspect-max", type=float, default=3.2)
    parser.add_argument("--reflection-intensity-min", type=float, default=22.0)
    parser.add_argument("--reflection-intensity-max", type=float, default=48.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    dataset_yaml = build_augmented_dataset(
        data_yaml=Path(args.data).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        seed=args.seed,
        sample_ratio=args.sample_ratio,
        low_light_prob=args.low_light_prob,
        reflection_prob=args.reflection_prob,
        allow_combined=args.allow_combined,
        reflection_max_area_ratio=args.reflection_max_area_ratio,
        reflection_min_area_factor=args.reflection_min_area_factor,
        reflection_aspect_min=args.reflection_aspect_min,
        reflection_aspect_max=args.reflection_aspect_max,
        reflection_intensity_min=args.reflection_intensity_min,
        reflection_intensity_max=args.reflection_intensity_max,
        refresh=args.refresh,
    )
    print(dataset_yaml)


if __name__ == "__main__":
    main()
