#!/usr/bin/env python3
"""Audit the materialized FloW source/temporal split and its fixed test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 653, "val": 203, "test": 344}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def list_images(root: Path, split: str) -> list[Path]:
    directory = root / split / "images"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )


def read_source_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "image_name",
        "split",
        "source_group",
        "raw_folder",
        "original_filename",
        "original_path",
        "capture_timestamp",
        "width",
        "height",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid source manifest: {path}")
    if len(rows) != 1200:
        raise ValueError(f"Expected 1200 source records, found {len(rows)}")
    return rows


def pairwise_overlap(values: dict[str, set[str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(values[left] & values[right])
        output[f"{left}_{right}"] = {"count": len(overlap), "examples": overlap[:25]}
    return output


def audit_labels(root: Path, split: str, images: list[Path], metadata: dict[str, dict[str, str]]) -> dict[str, Any]:
    label_dir = root / split / "labels"
    missing: list[str] = []
    malformed: list[dict[str, Any]] = []
    bins: Counter[str] = Counter()
    instances = 0
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            missing.append(image.name)
            continue
        row = metadata[image.name.casefold()]
        width = int(row["width"])
        height = int(row["height"])
        for line_number, raw in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                values = [float(value) for value in line.split()]
            except ValueError:
                malformed.append({"file": label.name, "line": line_number, "reason": "non_numeric"})
                continue
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                malformed.append({"file": label.name, "line": line_number, "reason": "schema_or_nonfinite"})
                continue
            class_id, center_x, center_y, box_w, box_h = values
            if class_id != 0 or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
                malformed.append({"file": label.name, "line": line_number, "reason": "class_or_bbox_range"})
                continue
            area = box_w * width * box_h * height
            bins["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
            instances += 1
    return {
        "missing_label_count": len(missing),
        "missing_label_examples": missing[:25],
        "malformed_label_count": len(malformed),
        "malformed_label_examples": malformed[:25],
        "valid_instances": instances,
        "area_bins_original_pixels": {
            "small": bins["small"],
            "medium": bins["medium"],
            "large": bins["large"],
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    args = parser.parse_args()

    rows = read_source_manifest(args.source_manifest)
    metadata = {row["image_name"].casefold(): row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError("Duplicate image names in source manifest")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    if protocol.get("protocol") != "flow_source_temporal_disjoint_v1":
        raise ValueError("Unexpected protocol identity")
    if protocol.get("manifests", {}).get("source_manifest", {}).get("sha256") != sha256(args.source_manifest):
        raise ValueError("Source manifest hash does not match protocol")

    images = {split: list_images(args.dataset_root, split) for split in SPLITS}
    actual_names = {split: {path.name.casefold() for path in paths} for split, paths in images.items()}
    expected_names = {
        split: {row["image_name"].casefold() for row in rows if row["split"] == split}
        for split in SPLITS
    }
    name_match = {
        split: {
            "matches": actual_names[split] == expected_names[split],
            "missing_count": len(expected_names[split] - actual_names[split]),
            "unexpected_count": len(actual_names[split] - expected_names[split]),
        }
        for split in SPLITS
    }
    labels = {split: audit_labels(args.dataset_root, split, images[split], metadata) for split in SPLITS}

    hashes: dict[str, set[str]] = {}
    inodes: dict[str, set[str]] = {}
    perceptual: dict[str, list[tuple[str, int]]] = {}
    for split, paths in images.items():
        hashes[split] = {sha256(path) for path in paths}
        inodes[split] = {f"{path.stat().st_dev}:{path.stat().st_ino}" for path in paths}
        perceptual[split] = [(path.name, dhash(path)) for path in paths]

    near_duplicates: dict[str, dict[str, Any]] = {}
    threshold = int(args.near_duplicate_hamming)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        candidates: list[dict[str, Any]] = []
        for left_name, left_hash in perceptual[left]:
            for right_name, right_hash in perceptual[right]:
                distance = hamming(left_hash, right_hash)
                if distance <= threshold:
                    candidates.append({left: left_name, right: right_name, "dhash_hamming": distance})
        candidates.sort(key=lambda row: (row["dhash_hamming"], row[left], row[right]))
        near_duplicates[f"{left}_{right}"] = {"count": len(candidates), "examples": candidates[:50]}

    source_values = {
        split: {row["source_group"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    original_filenames = {
        split: {
            row["original_filename"].strip().casefold()
            for row in rows
            if row["split"] == split and row["original_filename"].strip()
        }
        for split in SPLITS
    }
    original_paths = {
        split: {
            row["original_path"].strip().replace("/", "\\").casefold()
            for row in rows
            if row["split"] == split and row["original_path"].strip()
        }
        for split in SPLITS
    }
    dated_ranges: dict[str, dict[str, float | int]] = {}
    for split in SPLITS:
        timestamps = [float(row["capture_timestamp"]) for row in rows if row["split"] == split and row["capture_timestamp"]]
        dated_ranges[split] = {"min": min(timestamps), "max": max(timestamps), "dated_images": len(timestamps)}
    chronology_passed = dated_ranges["train"]["max"] < dated_ranges["val"]["min"] and dated_ranges["val"]["max"] < dated_ranges["test"]["min"]

    count_match = {split: len(images[split]) == EXPECTED_COUNTS[split] for split in SPLITS}
    filename_overlap = pairwise_overlap(actual_names)
    content_overlap = pairwise_overlap(hashes)
    inode_overlap = pairwise_overlap(inodes)
    source_overlap = pairwise_overlap(source_values)
    original_filename_overlap = pairwise_overlap(original_filenames)
    original_path_overlap = pairwise_overlap(original_paths)
    all_zero = lambda report: all(value["count"] == 0 for value in report.values())
    passed = (
        all(count_match.values())
        and all(item["matches"] for item in name_match.values())
        and all(report["missing_label_count"] == 0 and report["malformed_label_count"] == 0 for report in labels.values())
        and all_zero(filename_overlap)
        and all_zero(content_overlap)
        and all_zero(inode_overlap)
        and all_zero(source_overlap)
        and all_zero(original_filename_overlap)
        and all_zero(original_path_overlap)
        and all(item["count"] == 0 for item in near_duplicates.values())
        and chronology_passed
    )
    payload = {
        "schema": "ring3.flow_source_temporal_split_audit.v1",
        "status": "passed" if passed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "split_counts": {split: len(images[split]) for split in SPLITS},
        "expected_counts": EXPECTED_COUNTS,
        "count_match": count_match,
        "manifest_name_match": name_match,
        "labels": labels,
        "cross_split_overlap": {
            "renamed_filename": filename_overlap,
            "content_sha256": content_overlap,
            "hardlink_inode": inode_overlap,
            "source_group": source_overlap,
            "original_filename": original_filename_overlap,
            "original_path": original_path_overlap,
        },
        "temporal_order": {
            "passed": chronology_passed,
            "dated_ranges": dated_ranges,
            "undated_source_policy": "myPerson kept whole in train",
        },
        "perceptual_near_duplicate_screen": {
            "method": "64-bit dHash",
            "hamming_threshold": threshold,
            "pairwise": near_duplicates,
        },
        "claim_boundary": protocol.get("claim_boundary"),
    }
    atomic_json(args.output, payload)
    summary = {
        "schema": payload["schema"],
        "status": payload["status"],
        "split_counts": payload["split_counts"],
        "labels": labels,
        "cross_split_overlap": payload["cross_split_overlap"],
        "temporal_order": payload["temporal_order"],
        "perceptual_near_duplicate_screen": payload["perceptual_near_duplicate_screen"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
