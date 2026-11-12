#!/usr/bin/env python3
"""Materialize the strict FloW split beside the original YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_lookup(root: Path) -> dict[str, tuple[Path, Path, str]]:
    found: dict[str, tuple[Path, Path, str]] = {}
    for old_split in ("train", "val"):
        image_dir = root / old_split / "images"
        label_dir = root / old_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source directories under {root}: {old_split}")
        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = image.name.casefold()
            if key in found:
                raise ValueError(f"Duplicate source image filename: {image.name}")
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing source label: {label}")
            found[key] = (image, label, old_split)
    return found


def place(source: Path, target: Path, mode: str) -> str:
    if target.exists():
        if not target.is_file() or sha256(source) != sha256(target):
            raise RuntimeError(f"Existing target does not match its source: {target}")
        return "existing"
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)
    return mode


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"image_name", "split", "source_group", "current_split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid source manifest schema: {path}")
    return rows


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()

    rows = read_manifest(args.source_manifest)
    if len(rows) != 1200:
        raise ValueError(f"Expected 1200 manifest records, found {len(rows)}")
    names = [row["image_name"].casefold() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image_name values in source manifest")
    if set(row["split"] for row in rows) != set(SPLITS):
        raise ValueError("Manifest must contain train, val, and test")

    sources = source_lookup(args.source_root)
    if len(sources) != 1200:
        raise ValueError(f"Expected 1200 source images, found {len(sources)}")
    expected_by_split = {
        split: {row["image_name"].casefold() for row in rows if row["split"] == split}
        for split in SPLITS
    }
    if any(expected_by_split[left] & expected_by_split[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Manifest image names overlap across output splits")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        image_dir = args.output_root / split / "images"
        label_dir = args.output_root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        existing_images = {
            path.name.casefold()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        existing_labels = {path.stem.casefold() for path in label_dir.glob("*.txt") if path.is_file()}
        expected_images = expected_by_split[split]
        expected_stems = {Path(name).stem.casefold() for name in expected_images}
        if existing_images - expected_images or existing_labels - expected_stems:
            raise RuntimeError(f"Output split contains unexpected pre-existing files: {split}")

    placed = {split: {"hardlink": 0, "symlink": 0, "copy": 0, "existing": 0} for split in SPLITS}
    source_origin_counts = {split: {"train": 0, "val": 0} for split in SPLITS}
    for row in rows:
        split = row["split"]
        source = sources.get(row["image_name"].casefold())
        if source is None:
            raise FileNotFoundError(f"Source image missing: {row['image_name']}")
        image, label, old_split = source
        if old_split != row["current_split"]:
            raise ValueError(f"Source location drifted for {row['image_name']}: {old_split}")
        image_out = args.output_root / split / "images" / image.name
        label_out = args.output_root / split / "labels" / label.name
        placed[split][place(image, image_out, args.mode)] += 1
        placed[split][place(label, label_out, args.mode)] += 1
        source_origin_counts[split][old_split] += 1

    payload = {
        "schema": "ring3.flow_source_temporal_materialization.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "output_root": str(args.output_root.resolve()),
        "mode": args.mode,
        "split_images": {split: len(expected_by_split[split]) for split in SPLITS},
        "placed_files": placed,
        "source_origin_images": source_origin_counts,
        "original_dataset_modified": False,
    }
    atomic_json(args.output_root / "materialization_report.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
