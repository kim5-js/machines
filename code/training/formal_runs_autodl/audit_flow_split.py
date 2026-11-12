#!/usr/bin/env python3
"""Audit the fixed FloW YOLO split without changing dataset contents.

The audit distinguishes exact file leakage from source/sequence leakage.  It
can prove the former from the materialized split, but it deliberately leaves
the latter unverified when no source-video manifest is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


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


def numeric_stems(paths: list[Path]) -> list[int] | None:
    values: list[int] = []
    for path in paths:
        match = re.fullmatch(r"0*(\d+)", path.stem)
        if match is None:
            return None
        values.append(int(match.group(1)))
    return sorted(values)


def audit_labels(root: Path, split: str, paths: list[Path]) -> dict[str, Any]:
    label_dir = root / split / "labels"
    missing: list[str] = []
    malformed: list[dict[str, Any]] = []
    empty = 0
    instances = 0
    for image in paths:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            missing.append(image.name)
            continue
        lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            empty += 1
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            try:
                values = [float(value) for value in fields]
            except ValueError:
                malformed.append({"file": str(label), "line": line_number, "reason": "non_numeric"})
                continue
            if len(values) != 5 or int(values[0]) != values[0] or int(values[0]) != 0:
                malformed.append({"file": str(label), "line": line_number, "reason": "schema_or_class"})
                continue
            if not all(0.0 <= value <= 1.0 for value in values[1:]):
                malformed.append({"file": str(label), "line": line_number, "reason": "bbox_range"})
                continue
            instances += 1
    return {
        "missing_label_count": len(missing),
        "missing_label_examples": missing[:20],
        "malformed_label_count": len(malformed),
        "malformed_label_examples": malformed[:20],
        "empty_label_files": empty,
        "valid_instances": instances,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only FloW split audit")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    splits = {split: list_images(root, split) for split in ("train", "val")}
    labels = {split: audit_labels(root, split, paths) for split, paths in splits.items()}

    names = {
        split: {path.name.casefold(): str(path) for path in paths}
        for split, paths in splits.items()
    }
    filename_overlap = sorted(set(names["train"]) & set(names["val"]))

    hashes: dict[str, dict[str, list[str]]] = {}
    inodes: dict[str, dict[str, list[str]]] = {}
    perceptual: dict[str, list[tuple[str, int]]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split, paths in splits.items():
        hash_map: dict[str, list[str]] = defaultdict(list)
        inode_map: dict[str, list[str]] = defaultdict(list)
        perceptual_rows: list[tuple[str, int]] = []
        manifest_rows: list[dict[str, Any]] = []
        for path in paths:
            digest = sha256(path)
            stat = path.stat()
            inode = f"{stat.st_dev}:{stat.st_ino}"
            phash = dhash(path)
            hash_map[digest].append(path.name)
            inode_map[inode].append(path.name)
            perceptual_rows.append((path.name, phash))
            manifest_rows.append(
                {
                    "file": path.name,
                    "sha256": digest,
                    "bytes": stat.st_size,
                    "device_inode": inode,
                    "dhash64": f"{phash:016x}",
                }
            )
        hashes[split] = dict(hash_map)
        inodes[split] = dict(inode_map)
        perceptual[split] = perceptual_rows
        manifests[split] = manifest_rows

    content_overlap = sorted(set(hashes["train"]) & set(hashes["val"]))
    hardlink_overlap = sorted(set(inodes["train"]) & set(inodes["val"]))
    near_duplicates: list[dict[str, Any]] = []
    threshold = int(args.near_duplicate_hamming)
    for train_name, train_hash in perceptual["train"]:
        for val_name, val_hash in perceptual["val"]:
            distance = hamming(train_hash, val_hash)
            if distance <= threshold:
                near_duplicates.append(
                    {"train": train_name, "val": val_name, "dhash_hamming": distance}
                )
    near_duplicates.sort(key=lambda row: (row["dhash_hamming"], row["train"], row["val"]))

    numeric = {split: numeric_stems(paths) for split, paths in splits.items()}
    numeric_summary: dict[str, Any] = {}
    if all(values for values in numeric.values()):
        train_values = numeric["train"] or []
        val_values = numeric["val"] or []
        numeric_summary = {
            "all_stems_numeric": True,
            "train_min": min(train_values),
            "train_max": max(train_values),
            "val_min": min(val_values),
            "val_max": max(val_values),
            "boundary_gap": min(val_values) - max(train_values),
            "contiguous_boundary": min(val_values) - max(train_values) == 1,
        }
    else:
        numeric_summary = {"all_stems_numeric": False}

    exact_passed = (
        not filename_overlap
        and not content_overlap
        and not hardlink_overlap
        and all(report["missing_label_count"] == 0 and report["malformed_label_count"] == 0 for report in labels.values())
    )
    source_manifest_candidates = [
        root / name
        for name in ("protocol.json", "source_manifest.json", "sequence_manifest.json", "metadata.json")
    ]
    source_manifests = [str(path) for path in source_manifest_candidates if path.is_file()]
    source_sequence_status = "verified" if source_manifests else "unverified_no_source_or_sequence_manifest"
    status = "passed" if exact_passed and source_sequence_status == "verified" else (
        "passed_exact_overlap_only" if exact_passed else "failed"
    )
    payload = {
        "schema": "ring3.flow_split_audit.v1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "read_only": True,
        "split_counts": {split: len(paths) for split, paths in splits.items()},
        "labels": labels,
        "exact_overlap": {
            "passed": exact_passed,
            "filename_overlap_count": len(filename_overlap),
            "filename_overlap_examples": filename_overlap[:20],
            "content_sha256_overlap_count": len(content_overlap),
            "content_sha256_overlap_examples": content_overlap[:20],
            "hardlink_inode_overlap_count": len(hardlink_overlap),
            "hardlink_inode_overlap_examples": hardlink_overlap[:20],
        },
        "perceptual_screen": {
            "method": "64-bit dHash; screening evidence only, not source-disjoint proof",
            "hamming_threshold": threshold,
            "cross_split_candidate_count": len(near_duplicates),
            "candidate_examples": near_duplicates[:100],
        },
        "numeric_filename_structure": numeric_summary,
        "source_sequence_audit": {
            "status": source_sequence_status,
            "manifest_candidates_found": source_manifests,
            "claim_boundary": "Do not call this source/sequence-disjoint without an authoritative source grouping manifest.",
        },
        "manifests": manifests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: payload[key] for key in ("schema", "status", "split_counts", "exact_overlap", "perceptual_screen", "numeric_filename_structure", "source_sequence_audit")}, indent=2))
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
