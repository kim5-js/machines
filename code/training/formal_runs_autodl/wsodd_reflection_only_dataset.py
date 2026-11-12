from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def resolve_entries(dataset_yaml: Path, split: str) -> tuple[dict[str, Any], list[Path]]:
    config = read_yaml(dataset_yaml)
    raw_root = Path(config.get("path", dataset_yaml.parent))
    root = raw_root if raw_root.is_absolute() else (dataset_yaml.parent / raw_root).resolve()
    raw = config.get(split)
    if raw is None:
        raise RuntimeError(f"Dataset has no {split!r} split: {dataset_yaml}")
    entries = raw if isinstance(raw, list) else [raw]
    return config, [(root / Path(value)).resolve() if not Path(value).is_absolute() else Path(value).resolve() for value in entries]


def iter_images(entry: Path) -> Iterable[Path]:
    if entry.is_dir():
        yield from sorted(
            (path.resolve() for path in entry.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=str,
        )
        return
    if entry.is_file() and entry.suffix.lower() == ".txt":
        for line in entry.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield Path(line.strip()).expanduser().resolve()
        return
    raise RuntimeError(f"Unsupported dataset entry: {entry}")


def resolve_images(dataset_yaml: Path, split: str) -> list[Path]:
    _, entries = resolve_entries(dataset_yaml, split)
    images = [path for entry in entries for path in iter_images(entry)]
    if not images:
        raise RuntimeError(f"No images found for {split!r}")
    if len(images) != len(set(images)):
        raise RuntimeError(f"Duplicate image paths found in {split!r}")
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        raise RuntimeError(f"Cannot map image to label: {image_path}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def apply_reflection_glare(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    max_area_ratio: float,
    min_area_factor: float,
    aspect_min: float,
    aspect_max: float,
    intensity_min: float,
    intensity_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = image.astype(np.float32).copy()
    height, width = output.shape[:2]
    max_area = max(float(height * width) * max_area_ratio, 64.0)
    target_area = float(rng.uniform(max_area * min_area_factor, max_area))
    aspect = float(rng.uniform(aspect_min, aspect_max))
    radius_y = max(10, int(np.sqrt(target_area / (np.pi * aspect))))
    radius_x = max(14, int(radius_y * aspect))
    center_x = int(rng.integers(radius_x, max(radius_x + 1, width - radius_x)))
    center_y = int(rng.integers(radius_y, max(radius_y + 1, height - radius_y)))
    yy, xx = np.ogrid[:height, :width]
    distance = ((xx - center_x) / max(radius_x, 1)) ** 2 + ((yy - center_y) / max(radius_y, 1)) ** 2
    hard_mask = distance <= 1.0
    falloff = float(rng.uniform(1.6, 2.4))
    soft_mask = np.exp(-distance * falloff) * hard_mask
    intensity = float(rng.uniform(intensity_min, intensity_max))
    output += soft_mask[..., None] * intensity
    transformed = np.clip(output, 0.0, 255.0).astype(np.uint8)
    absolute_delta = np.abs(transformed.astype(np.int16) - image.astype(np.int16))
    changed = np.any(absolute_delta > 0, axis=2)
    evidence = {
        "height": height,
        "width": width,
        "target_area": target_area,
        "target_area_ratio": target_area / float(height * width),
        "aspect": aspect,
        "radius_x": radius_x,
        "radius_y": radius_y,
        "center_x": center_x,
        "center_y": center_y,
        "falloff": falloff,
        "intensity": intensity,
        "changed_pixel_count": int(changed.sum()),
        "changed_pixel_ratio": float(changed.mean()),
        "mean_absolute_delta": float(absolute_delta.mean()),
        "max_absolute_delta": int(absolute_delta.max()),
    }
    if evidence["changed_pixel_count"] <= 0 or evidence["mean_absolute_delta"] <= 0.0:
        raise RuntimeError("Reflection transform produced an unchanged image")
    return transformed, evidence


def expected_spec(config: dict[str, Any]) -> dict[str, Any]:
    reflection = config["reflection_only"]
    return {
        "schema": "ring3.wsodd_reflection_only_overlay.v1",
        "source_data_yaml": str(Path(config["dataset"]["yaml"]).resolve()),
        "source_data_yaml_sha256": config["dataset"]["yaml_sha256"],
        "source_protocol": config["dataset"]["protocol"],
        "sample_ratio": reflection["sample_ratio"],
        "sample_seed": reflection["sample_seed"],
        "low_light_probability": reflection["low_light_probability"],
        "reflection_probability": reflection["reflection_probability"],
        "allow_combined": reflection["allow_combined"],
        "mask": reflection["mask"],
        "area_preset": reflection["area_preset"],
        "reflection_max_area_ratio": reflection["reflection_max_area_ratio"],
        "reflection_min_area_factor": reflection["reflection_min_area_factor"],
        "reflection_aspect_min": reflection["reflection_aspect_min"],
        "reflection_aspect_max": reflection["reflection_aspect_max"],
        "reflection_intensity_min": reflection["reflection_intensity_min"],
        "reflection_intensity_max": reflection["reflection_intensity_max"],
        "jpeg_quality": reflection["jpeg_quality"],
    }


def validate_overlay(output_root: Path, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = output_root / "metadata.json"
    dataset_yaml = output_root / "dataset.yaml"
    if not metadata_path.is_file() or not dataset_yaml.is_file():
        raise RuntimeError(f"Incomplete reflection overlay: {output_root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("spec") != spec:
        raise RuntimeError("Reflection overlay metadata does not match the locked specification")
    counts = metadata.get("counts", {})
    expected_counts = {
        "source_train": int(config["dataset"]["train_images"]),
        "augmented_train": int(config["reflection_only"]["augmented_train_copies"]),
        "effective_train": int(config["reflection_only"]["effective_train_images_per_epoch"]),
        "val": int(config["dataset"]["val_images"]),
        "test": int(config["dataset"]["test_images"]),
    }
    if counts != expected_counts:
        raise RuntimeError(f"Reflection overlay counts changed: {counts} != {expected_counts}")
    for name, expected_hash in metadata.get("artifact_sha256", {}).items():
        artifact = output_root / name
        if not artifact.is_file() or sha256(artifact) != expected_hash:
            raise RuntimeError(f"Reflection overlay artifact changed: {artifact}")
    records = metadata.get("augmented_records", [])
    if len(records) != expected_counts["augmented_train"]:
        raise RuntimeError("Reflection overlay record count changed")
    for record in records:
        image = output_root / record["output_image"]
        label = output_root / record["output_label"]
        if not image.is_file() or sha256(image) != record["output_image_sha256"]:
            raise RuntimeError(f"Augmented image changed: {image}")
        if not label.is_file() or sha256(label) != record["output_label_sha256"]:
            raise RuntimeError(f"Augmented label changed: {label}")
        source_label = Path(record["source_label"])
        if not source_label.is_file() or sha256(source_label) != record["output_label_sha256"]:
            raise RuntimeError(f"Augmented label is not identical to source: {label}")
    return metadata


def build_or_validate_overlay(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    dataset_yaml = Path(config["dataset"]["yaml"])
    output_root = Path(config["execution"]["overlay_root"])
    spec = expected_spec(config)
    if output_root.exists():
        return output_root / "dataset.yaml", validate_overlay(output_root, spec, config)

    train_images = resolve_images(dataset_yaml, "train")
    val_images = resolve_images(dataset_yaml, "val")
    test_images = resolve_images(dataset_yaml, "test")
    if len(train_images) != int(config["dataset"]["train_images"]):
        raise RuntimeError("Source train image count changed")
    if len(val_images) != int(config["dataset"]["val_images"]):
        raise RuntimeError("Source validation image count changed")
    if len(test_images) != int(config["dataset"]["test_images"]):
        raise RuntimeError("Source test image count changed")
    train_set, val_set, test_set = set(train_images), set(val_images), set(test_images)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Source image paths overlap across train, validation, and test")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.building.{os.getpid()}.{secrets.token_hex(4)}"
    temporary.mkdir(parents=False, exist_ok=False)
    images_dir = temporary / "aug_train" / "images"
    labels_dir = temporary / "aug_train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    reflection = config["reflection_only"]
    sample_count = int(reflection["augmented_train_copies"])
    expected_sample_count = max(1, int(len(train_images) * float(reflection["sample_ratio"])))
    if sample_count != expected_sample_count:
        raise RuntimeError(f"Locked reflection sample count is inconsistent: {sample_count} != {expected_sample_count}")
    rng = np.random.default_rng(int(reflection["sample_seed"]))
    selected_indices = sorted(int(value) for value in rng.choice(len(train_images), size=sample_count, replace=False))
    records: list[dict[str, Any]] = []
    final_augmented_paths: list[Path] = []
    for ordinal, index in enumerate(selected_indices):
        source_image = train_images[index]
        source_label = label_path_for_image(source_image)
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing source label: {source_label}")
        source_array = np.asarray(Image.open(source_image).convert("RGB"))
        local_rng = np.random.default_rng(int(reflection["sample_seed"]) + index)
        transformed, glare = apply_reflection_glare(
            source_array,
            local_rng,
            max_area_ratio=float(reflection["reflection_max_area_ratio"]),
            min_area_factor=float(reflection["reflection_min_area_factor"]),
            aspect_min=float(reflection["reflection_aspect_min"]),
            aspect_max=float(reflection["reflection_aspect_max"]),
            intensity_min=float(reflection["reflection_intensity_min"]),
            intensity_max=float(reflection["reflection_intensity_max"]),
        )
        source_token = hashlib.sha256(str(source_image).encode("utf-8")).hexdigest()[:12]
        suffix = source_image.suffix.lower() if source_image.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        filename = f"{ordinal:04d}_{source_token}_{source_image.stem}_rf{suffix}"
        output_image = images_dir / filename
        output_label = labels_dir / f"{Path(filename).stem}.txt"
        save_kwargs = {"quality": int(reflection["jpeg_quality"])} if suffix in {".jpg", ".jpeg", ".webp"} else {}
        Image.fromarray(transformed).save(output_image, **save_kwargs)
        shutil.copy2(source_label, output_label)
        final_image = output_root / output_image.relative_to(temporary)
        final_augmented_paths.append(final_image)
        records.append(
            {
                "source_index": index,
                "source_image": str(source_image),
                "source_image_sha256": sha256(source_image),
                "source_label": str(source_label),
                "output_image": str(output_image.relative_to(temporary)),
                "output_image_sha256": sha256(output_image),
                "output_label": str(output_label.relative_to(temporary)),
                "output_label_sha256": sha256(output_label),
                "reflection": glare,
            }
        )

    manifests = {
        "train_images.txt": [*map(str, train_images), *map(str, final_augmented_paths)],
        "val_images.txt": list(map(str, val_images)),
        "test_images.txt": list(map(str, test_images)),
    }
    for name, values in manifests.items():
        (temporary / name).write_text("\n".join(values) + "\n", encoding="utf-8")
    source_config = read_yaml(dataset_yaml)
    overlay_config = {
        "path": str(output_root),
        "train": "train_images.txt",
        "val": "val_images.txt",
        "test": "test_images.txt",
        "nc": source_config["nc"],
        "names": source_config["names"],
    }
    (temporary / "dataset.yaml").write_text(
        yaml.safe_dump(overlay_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    artifact_names = ["dataset.yaml", *manifests]
    metadata = {
        "schema": "ring3.wsodd_reflection_only_overlay_evidence.v1",
        "status": "completed",
        "spec": spec,
        "counts": {
            "source_train": len(train_images),
            "augmented_train": len(records),
            "effective_train": len(manifests["train_images.txt"]),
            "val": len(val_images),
            "test": len(test_images),
        },
        "split_path_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "selected_sources_subset_of_train": all(Path(record["source_image"]) in train_set for record in records),
        "selected_source_list_sha256": sha256_lines(record["source_image"] for record in records),
        "artifact_sha256": {name: sha256(temporary / name) for name in artifact_names},
        "augmented_records": records,
    }
    write_json_atomic(temporary / "metadata.json", metadata)
    temporary.rename(output_root)
    return output_root / "dataset.yaml", validate_overlay(output_root, spec, config)


__all__ = ["build_or_validate_overlay", "resolve_images", "sha256", "validate_overlay"]
