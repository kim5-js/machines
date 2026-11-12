from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def _resolve_base(yaml_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", yaml_path.parent))).expanduser()
    return raw.resolve() if raw.is_absolute() else (yaml_path.parent / raw).resolve()


def _iter_images(entry: Path) -> Iterable[Path]:
    if entry.is_dir():
        yield from (
            path.resolve()
            for path in sorted(entry.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        return
    if entry.is_file() and entry.suffix.lower() == ".txt":
        for raw_line in entry.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if line:
                candidate = Path(line).expanduser()
                yield candidate.resolve() if candidate.is_absolute() else (entry.parent / candidate).resolve()
        return
    raise FileNotFoundError(f"Unsupported or missing split entry: {entry}")


def _split_images(yaml_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    raw_entry = config.get(split)
    if raw_entry in (None, "", []):
        return []
    base = _resolve_base(yaml_path, config)
    entries = raw_entry if isinstance(raw_entry, list) else [raw_entry]
    result: list[Path] = []
    for item in entries:
        path = Path(str(item)).expanduser()
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        result.extend(_iter_images(path))
    return result


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    image_positions = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_positions:
        parts[image_positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _validate_label(path: Path, nc: int) -> tuple[list[str], Counter[int]]:
    errors: list[str] = []
    class_counts: Counter[int] = Counter()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"cannot read {path}: {exc}"], class_counts
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            errors.append(f"{path}:{line_number}: expected at least 5 fields, got {len(fields)}")
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric field")
            continue
        class_id = values[0]
        if not class_id.is_integer() or not 0 <= int(class_id) < nc:
            errors.append(f"{path}:{line_number}: class {class_id:g} outside [0, {nc - 1}]")
        else:
            class_counts[int(class_id)] += 1
        if any(not math.isfinite(value) for value in values[1:]):
            errors.append(f"{path}:{line_number}: non-finite coordinate")
        elif any(value < 0.0 or value > 1.0 for value in values[1:]):
            errors.append(f"{path}:{line_number}: coordinate outside [0, 1]")
    return errors, class_counts


def verify(yaml_path: Path, require_test: bool) -> tuple[dict[str, Any], bool]:
    yaml_path = yaml_path.expanduser().resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")
    names = config.get("names")
    nc = int(config.get("nc", len(names) if names is not None else 0))
    if nc <= 0:
        raise ValueError("Dataset YAML must define a positive nc or non-empty names")

    images_by_split = {split: _split_images(yaml_path, config, split) for split in SPLITS}
    missing_labels: dict[str, list[str]] = defaultdict(list)
    label_errors: list[str] = []
    class_counts: Counter[int] = Counter()
    inode_splits: dict[tuple[int, int], set[str]] = defaultdict(set)
    names_by_split: dict[str, set[str]] = {}

    for split, images in images_by_split.items():
        names_by_split[split] = {path.name.casefold() for path in images}
        for image in images:
            try:
                stat = image.stat()
                inode_splits[(stat.st_dev, stat.st_ino)].add(split)
            except OSError:
                pass
            label = _label_path(image)
            if not label.is_file():
                missing_labels[split].append(str(label))
                continue
            current_errors, current_counts = _validate_label(label, nc)
            label_errors.extend(current_errors)
            class_counts.update(current_counts)

    exact_path_overlap: dict[str, int] = {}
    filename_overlap: dict[str, int] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}-{right}"
            exact_path_overlap[key] = len(set(images_by_split[left]) & set(images_by_split[right]))
            filename_overlap[key] = len(names_by_split[left] & names_by_split[right])
    hardlink_overlap = sum(1 for splits in inode_splits.values() if len(splits) > 1)

    warnings: list[str] = []
    if not images_by_split["test"]:
        warnings.append("independent test split is missing")
    if filename_overlap and any(filename_overlap.values()):
        warnings.append("same image filename appears in more than one split")
    unused_class_ids = sorted(set(range(nc)) - set(class_counts))
    if unused_class_ids:
        warnings.append(f"configured class IDs absent from all labels: {unused_class_ids}")
    if names is not None and len(names) != nc:
        warnings.append(f"names has {len(names)} entries but nc is {nc}")

    report: dict[str, Any] = {
        "yaml": str(yaml_path),
        "base": str(_resolve_base(yaml_path, config)),
        "nc": nc,
        "counts": {split: len(images) for split, images in images_by_split.items()},
        "missing_label_counts": {split: len(missing_labels[split]) for split in SPLITS},
        "missing_label_examples": {
            split: paths[:10] for split, paths in missing_labels.items() if paths
        },
        "label_error_count": len(label_errors),
        "label_error_examples": label_errors[:20],
        "class_instance_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "unused_class_ids": unused_class_ids,
        "exact_path_overlap": exact_path_overlap,
        "filename_overlap": filename_overlap,
        "hardlink_overlap": hardlink_overlap,
        "warnings": warnings,
    }
    split_leak = any(exact_path_overlap.values()) or hardlink_overlap > 0
    missing_required = not images_by_split["train"] or not images_by_split["val"]
    failed = (
        missing_required
        or any(len(paths) for paths in missing_labels.values())
        or bool(label_errors)
        or split_leak
        or (require_test and not images_by_split["test"])
    )
    return report, not failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a YOLO dataset before scientific runs")
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--require-test", action="store_true")
    args = parser.parse_args()
    report, ok = verify(args.yaml, args.require_test)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
