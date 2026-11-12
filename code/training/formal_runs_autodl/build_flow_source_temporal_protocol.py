#!/usr/bin/env python3
"""Build the strict source/temporal split for the available 1,200 FloW images.

The XML annotations retain original acquisition folders and, except for the
``myPerson`` source, Unix timestamps.  Renamed YOLO IDs such as 000001.jpg are
therefore identifiers only and are never interpreted as capture order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_FOLDER_COUNTS = {
    "Saved_Pic": 103,
    "Saved_Pic_2": 134,
    "Saved_Pic_3": 107,
    "images": 2,
    "myPerson": 391,
    "ningbo1_select_select_new": 117,
    "ningbo4_select_select_new": 110,
    "ningbo_2_select_select_new": 33,
    "ningbo_3_select_select_new": 14,
    "ningbo_5_select_select_new": 189,
}

SOURCE_GROUP = {
    "Saved_Pic": "saved_pic_continuous",
    "Saved_Pic_2": "saved_pic_continuous",
    "Saved_Pic_3": "saved_pic_continuous",
    "images": "ningbo1_session",
    "ningbo1_select_select_new": "ningbo1_session",
    "myPerson": "myPerson",
    "ningbo4_select_select_new": "ningbo4_select_select_new",
    "ningbo_2_select_select_new": "ningbo_2_select_select_new",
    "ningbo_3_select_select_new": "ningbo_3_select_select_new",
    "ningbo_5_select_select_new": "ningbo_5_select_select_new",
}

# ``myPerson`` has no reliable capture time, so its complete acquisition is
# conservatively train-only.  Every dated source/session is strictly ordered.
GROUP_ASSIGNMENT = {
    "myPerson": "train",
    "ningbo1_session": "train",
    "ningbo4_select_select_new": "train",
    "ningbo_2_select_select_new": "train",
    "ningbo_3_select_select_new": "val",
    "ningbo_5_select_select_new": "val",
    "saved_pic_continuous": "test",
}

DATED_GROUP_ORDER = (
    "ningbo1_session",
    "ningbo4_select_select_new",
    "ningbo_2_select_select_new",
    "ningbo_3_select_select_new",
    "ningbo_5_select_select_new",
    "saved_pic_continuous",
)

EXPECTED_SPLIT_COUNTS = {"train": 653, "val": 203, "test": 344}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def timestamp_from_filename(value: str) -> float | None:
    stem = Path(value).stem
    if re.fullmatch(r"1\d{9}(?:\.\d+)?", stem) is None:
        return None
    timestamp = float(stem)
    return timestamp if 946684800 <= timestamp <= 1893456000 else None


def area_bin(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def locate_current_yolo(root: Path) -> dict[str, tuple[str, Path, Path]]:
    found: dict[str, tuple[str, Path, Path]] = {}
    for split in ("train", "val"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing current YOLO directories for {split}: {root}")
        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = image.name.casefold()
            if key in found:
                raise ValueError(f"Duplicate image filename in current YOLO tree: {image.name}")
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image}")
            found[key] = (split, image.resolve(), label.resolve())
    return found


def parse_annotation(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    raw_folder = (root.findtext("folder") or "").strip()
    if raw_folder not in SOURCE_GROUP:
        raise ValueError(f"Unexpected source folder in {path}: {raw_folder!r}")
    original_filename = (root.findtext("filename") or "").strip()
    original_path = (root.findtext("path") or "").strip()
    width = int(root.findtext("size/width") or 0)
    height = int(root.findtext("size/height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {path}: {width}x{height}")

    bins: Counter[str] = Counter()
    object_count = 0
    for object_node in root.findall("object"):
        name = (object_node.findtext("name") or "").strip().casefold()
        if name != "bottle":
            raise ValueError(f"Unexpected class in {path}: {name!r}")
        box = object_node.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {path}")
        xmin = float(box.findtext("xmin") or "nan")
        ymin = float(box.findtext("ymin") or "nan")
        xmax = float(box.findtext("xmax") or "nan")
        ymax = float(box.findtext("ymax") or "nan")
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(f"Invalid bndbox in {path}: {(xmin, ymin, xmax, ymax)}")
        bins[area_bin((xmax - xmin) * (ymax - ymin))] += 1
        object_count += 1
    if object_count == 0:
        raise ValueError(f"Annotation has no objects: {path}")

    timestamp = None if raw_folder == "myPerson" else timestamp_from_filename(original_filename)
    if raw_folder != "myPerson" and timestamp is None:
        raise ValueError(f"Expected a Unix timestamp filename in {path}: {original_filename!r}")
    return {
        "image_id": path.stem,
        "image_name": f"{path.stem}.jpg",
        "raw_folder": raw_folder,
        "source_group": SOURCE_GROUP[raw_folder],
        "split": GROUP_ASSIGNMENT[SOURCE_GROUP[raw_folder]],
        "original_filename": original_filename,
        "original_path": original_path,
        "capture_timestamp": timestamp,
        "width": width,
        "height": height,
        "object_count": object_count,
        "small_objects": bins["small"],
        "medium_objects": bins["medium"],
        "large_objects": bins["large"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--current-yolo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotation_paths = sorted(args.annotations.glob("*.xml"), key=lambda path: path.name)
    if len(annotation_paths) != 1200:
        raise ValueError(f"Expected exactly 1200 XML files, found {len(annotation_paths)}")
    records = [parse_annotation(path) for path in annotation_paths]
    if len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Duplicate renamed image IDs in XML annotations")

    raw_counts = Counter(row["raw_folder"] for row in records)
    if dict(raw_counts) != RAW_FOLDER_COUNTS:
        raise ValueError(f"Raw folder counts changed: {dict(raw_counts)}")
    split_counts = Counter(row["split"] for row in records)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")

    current = locate_current_yolo(args.current_yolo_root)
    if len(current) != 1200:
        raise ValueError(f"Expected 1200 current YOLO images, found {len(current)}")
    for row in records:
        item = current.get(row["image_name"].casefold())
        if item is None:
            raise FileNotFoundError(f"Current YOLO image not found: {row['image_name']}")
        current_split, image, label = item
        label_lines = [line for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(label_lines) != row["object_count"]:
            raise ValueError(
                f"XML/YOLO object count mismatch for {row['image_name']}: "
                f"{row['object_count']} != {len(label_lines)}"
            )
        row["current_split"] = current_split
        row["current_image"] = str(image)
        row["current_label"] = str(label)

    group_times: dict[str, list[float]] = defaultdict(list)
    raw_times: dict[str, list[float]] = defaultdict(list)
    for row in records:
        timestamp = row["capture_timestamp"]
        if timestamp is not None:
            group_times[row["source_group"]].append(timestamp)
            raw_times[row["raw_folder"]].append(timestamp)
    time_ranges = {
        group: {"min": min(values), "max": max(values), "images": len(values)}
        for group, values in group_times.items()
    }
    for left, right in zip(DATED_GROUP_ORDER, DATED_GROUP_ORDER[1:]):
        if time_ranges[left]["max"] >= time_ranges[right]["min"]:
            raise ValueError(f"Dated sessions overlap or are out of order: {left}, {right}")
    saved_gaps = [
        min(raw_times["Saved_Pic_2"]) - max(raw_times["Saved_Pic"]),
        min(raw_times["Saved_Pic_3"]) - max(raw_times["Saved_Pic_2"]),
    ]
    if not all(0 <= gap <= 60 for gap in saved_gaps):
        raise ValueError(f"Saved_Pic sessions are not demonstrably continuous: {saved_gaps}")
    ningbo1_min = min(raw_times["ningbo1_select_select_new"])
    ningbo1_max = max(raw_times["ningbo1_select_select_new"])
    if not all(ningbo1_min <= value <= ningbo1_max for value in raw_times["images"]):
        raise ValueError("The two generic 'images' records do not fall inside the ningbo1 session")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        selected = sorted(
            (row for row in records if row["split"] == split),
            key=lambda row: (
                row["capture_timestamp"] is None,
                row["capture_timestamp"] or 0,
                row["image_name"],
            ),
        )
        path = args.output / f"{split}.txt"
        atomic_text(path, "".join(f"{row['image_name']}\n" for row in selected))
        manifest_paths[split] = path
    trainval = args.output / "trainval.txt"
    atomic_text(
        trainval,
        "".join(
            f"{row['image_name']}\n"
            for row in sorted(
                (row for row in records if row["split"] in {"train", "val"}),
                key=lambda row: row["image_name"],
            )
        ),
    )
    manifest_paths["trainval"] = trainval

    source_manifest = args.output / "source_manifest.tsv"
    temporary = source_manifest.with_suffix(".tsv.tmp")
    fields = [
        "image_id",
        "image_name",
        "split",
        "source_group",
        "raw_folder",
        "original_filename",
        "original_path",
        "capture_timestamp",
        "width",
        "height",
        "object_count",
        "small_objects",
        "medium_objects",
        "large_objects",
        "current_split",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in sorted(records, key=lambda item: item["image_name"]):
            writer.writerow(row)
    os.replace(temporary, source_manifest)
    manifest_paths["source_manifest"] = source_manifest

    group_counts = Counter(row["source_group"] for row in records)
    split_groups = {
        split: sorted(group for group, assigned in GROUP_ASSIGNMENT.items() if assigned == split)
        for split in ("train", "val", "test")
    }
    object_bins = {
        split: {
            "total": sum(row["object_count"] for row in records if row["split"] == split),
            "small": sum(row["small_objects"] for row in records if row["split"] == split),
            "medium": sum(row["medium_objects"] for row in records if row["split"] == split),
            "large": sum(row["large_objects"] for row in records if row["split"] == split),
        }
        for split in ("train", "val", "test")
    }
    payload = {
        "schema": "ring3.flow_source_temporal_disjoint.v1",
        "protocol": "flow_source_temporal_disjoint_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "available_images": len(records),
        "source_of_truth": "official FloW XML acquisition metadata supplied with the available 1,200-image subset",
        "renamed_numeric_id_policy": "identifier_only_not_capture_time",
        "claim_boundary": (
            "Source/session-disjoint for all 1,200 available images and chronological for all dated sessions. "
            "The undated myPerson acquisition is kept whole in train; therefore this is not proof of a "
            "global chronology relative to that one source."
        ),
        "grouping": {
            "Saved_Pic+Saved_Pic_2+Saved_Pic_3": "one continuous session; adjacent gaps <=60 seconds",
            "ningbo1_select_select_new+images": "one session; generic images timestamps lie inside ningbo1 range",
            "other_folders": "one source/session per XML folder",
        },
        "assignment_rule": (
            "keep source sessions intact; put earliest dated sessions in train, next dated sessions in val, "
            "and latest continuous Saved_Pic session in fixed test; keep undated myPerson wholly in train"
        ),
        "group_assignment": GROUP_ASSIGNMENT,
        "split_images": dict(split_counts),
        "split_ratios": {split: split_counts[split] / len(records) for split in ("train", "val", "test")},
        "split_source_groups": split_groups,
        "source_group_counts": dict(group_counts),
        "raw_folder_counts": dict(raw_counts),
        "dated_group_order": list(DATED_GROUP_ORDER),
        "dated_group_ranges": time_ranges,
        "saved_pic_interfolder_gaps_seconds": saved_gaps,
        "object_area_bins_original_pixels": object_bins,
    }
    # Build overlap and manifest sections without self-referential protocol hashing.
    payload["source_group_overlap"] = {
        "train_val": len(set(split_groups["train"]) & set(split_groups["val"])),
        "train_test": len(set(split_groups["train"]) & set(split_groups["test"])),
        "val_test": len(set(split_groups["val"]) & set(split_groups["test"])),
    }
    payload["manifests"] = {
        name: {
            "path": path.name,
            "sha256": sha256(path),
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
        }
        for name, path in manifest_paths.items()
    }
    protocol = args.output / "protocol.json"
    atomic_text(protocol, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
