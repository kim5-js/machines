#!/usr/bin/env python3
"""Independently re-read the release and official XML; do not import the builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath

from PIL import Image


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_one(row, source, output):
    official_dir = "training" if row["official_split"] == "train" else "test"
    stem = row["official_image_id"]
    original_image = source / official_dir / "images" / (stem + ".jpg")
    original_xml = source / official_dir / "annotations" / (stem + ".xml")
    copied_image, label = output / row["image"], output / row["label"]
    check(digest(original_image) == digest(copied_image) == row["image_sha256"], f"Image hash mismatch: {stem}")
    check(digest(original_xml) == digest(output / row["archived_xml"]) == row["xml_sha256"], f"XML hash mismatch: {stem}")
    check(digest(label) == row["label_sha256"], f"Label hash mismatch: {label}")
    xml = ET.parse(original_xml).getroot()
    w, h = (int(xml.findtext(f"size/{axis}")) for axis in ["width", "height"])
    with Image.open(copied_image) as im:
        rgb = im.convert("RGB")
        check(rgb.size == (w, h), f"Decoded image dimensions disagree with XML: {copied_image}")
        pixel_digest = hashlib.sha256(f"RGB:{w}x{h}:".encode("ascii"))
        pixel_digest.update(rgb.tobytes())
        check(pixel_digest.hexdigest() == row["pixel_sha256"], f"Pixel digest mismatch: {copied_image}")
    reference_boxes = []
    for obj in xml.findall("object"):
        check((obj.findtext("name") or "").strip().casefold() == "bottle", f"Unexpected official class: {original_xml}")
        reference_boxes.append(tuple(float(obj.findtext(f"bndbox/{axis}")) for axis in ["xmin", "ymin", "xmax", "ymax"]))
    recovered_boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        check(len(parts) == 5 and parts[0] == "0", f"Invalid YOLO class/fields: {label}")
        x, y, bw, bh = map(float, parts[1:])
        check(all(math.isfinite(v) and 0 <= v <= 1 for v in [x, y, bw, bh]) and bw > 0 and bh > 0, f"Invalid normalized box: {label}")
        recovered_boxes.append(((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    canonical = lambda bb: Counter(tuple(round(v, 6) for v in box) for box in bb)
    check(canonical(reference_boxes) == canonical(recovered_boxes), f"YOLO boxes differ from official XML: {label}")
    check(len(reference_boxes) == len(recovered_boxes) == int(row["box_count"]) > 0, f"Object count mismatch: {label}")
    folder = (xml.findtext("folder") or "").strip()
    filename = (xml.findtext("filename") or "").strip()
    path = (xml.findtext("path") or "").strip()
    parent = str(PureWindowsPath(path).parent) if path else ""
    key = re.sub(r"\.(?:jpg|jpeg|png|bmp)$", "", PureWindowsPath(filename).name, flags=re.IGNORECASE)
    check(folder == row["raw_folder"] and filename == row["original_filename"] and path == row["original_path"] and parent == row["original_parent"] and key == row["original_filename_key"], f"Source metadata mismatch: {original_xml}")
    supported_time = float(key) if folder != "myPerson" and re.fullmatch(r"1\d{9}\.\d+", key) else None
    if supported_time is None:
        check(row["split"] == "train", "Uncertain acquisition time assigned outside train")
    else:
        check(supported_time == float(row["timestamp_seconds"]), "Timestamp mismatch")
    return {"split": row["split"], "source_group": row["source_group"], "folder": folder, "parent": parent, "key": key, "time": supported_time, "boxes": len(reference_boxes), "pixel_hash": pixel_digest.hexdigest(), "image": row["image"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("D:/download/FloW_IMG/FloW_IMG"))
    parser.add_argument("--output", type=Path, default=Path("D:/ring3/02_当前实验/数据集/FloW_1124_335_541"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    with (output / "audit/manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == 2000, "Expected 2000 manifest rows")
    all_originals = {(split, p.stem) for split, folder in [("train", "training"), ("test", "test")] for p in (source / folder / "images").glob("*.jpg")}
    manifest_originals = {(r["official_split"], r["official_image_id"]) for r in rows}
    check(len(manifest_originals) == 2000 and manifest_originals == all_originals, "Some official images omitted or repeated")
    for kind, field in [("images", "image"), ("labels", "label")]:
        actual = {p.relative_to(output).as_posix() for p in (output / kind).rglob("*") if p.is_file()}
        expected = {r[field] for r in rows}
        check(len(expected) == 2000 and actual == expected, f"Missing or extra output {kind}")
    observations = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(lambda row: verify_one(row, source, output), rows), 1):
            observations.append(result)
            if index % 500 == 0:
                print(f"Independently verified {index}/2000 image/XML/YOLO pairs.", flush=True)
    overlap_counts = {}
    for field in ["source_group", "folder", "parent", "key", "pixel_hash"]:
        assignments = defaultdict(set)
        for observation in observations:
            if observation[field]:
                assignments[observation[field].casefold()].add(observation["split"])
        overlaps = {key: sorted(splits) for key, splits in assignments.items() if len(splits) > 1}
        check(not overlaps, f"Cross-split overlap: {field}: {overlaps}")
        overlap_counts[field] = len(overlaps)
    counts = {}
    times = {}
    for split in ["train", "val", "test"]:
        selected = [r for r in observations if r["split"] == split]
        counts[split] = {"images": len(selected), "boxes": sum(r["boxes"] for r in selected)}
        values = [r["time"] for r in selected if r["time"] is not None]
        times[split] = {"earliest": min(values), "latest": max(values), "supported_time_images": len(values)}
    check(counts == {"train": {"images": 1124, "boxes": 2665}, "val": {"images": 335, "boxes": 1517}, "test": {"images": 541, "boxes": 1090}}, "Unexpected final split counts")
    check(times["train"]["latest"] < times["val"]["earliest"] and times["val"]["latest"] < times["test"]["earliest"], "Chronological order failed for supported capture times")
    saved = [r for r in observations if r["folder"].startswith("Saved_Pic")]
    check(len(saved) == 541 and {r["split"] for r in saved} == {"test"}, "Saved_Pic continuous acquisition was split")
    duplicate_frame = [r for r in observations if r["key"] == "1601097462.9573567"]
    check(len(duplicate_frame) == 2 and {r["split"] for r in duplicate_frame} == {"train"}, "Known reencoded duplicate frame was split")
    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    for key, target in [("train", "images/train"), ("val", "images/val"), ("test", "images/test")]:
        found = re.search(rf"^{key}:\s*(\S+)\s*$", yaml_text, re.MULTILINE)
        check(found and found.group(1) == target and (output / target).is_dir(), f"Wrong YAML path: {key}")
    visual = json.loads((output / "audit/visual_duplicate_audit.json").read_text(encoding="utf-8"))
    check(visual["cross_split_candidates"] == [], "Unreviewed visual duplicate candidates")
    result = {
        "status": "passed", "official_images_used_exactly_once": 2000,
        "official_annotation_boxes_preserved": 5272, "split_counts": counts,
        "image_and_xml_byte_hashes_match_official_inputs": True,
        "images_fully_decoded": 2000, "label_files_checked_against_official_xml": 2000,
        "bbox_comparison": "Order-independent multiset of inverse-converted YOLO edges, rounded to 6 decimal pixels; no missing or extra boxes.",
        "missing_labels": 0, "empty_labels": 0, "cross_split_overlap_counts": overlap_counts,
        "supported_time_ranges": times,
        "supported_capture_times_strictly_ordered_train_val_test": True,
        "uncertain_capture_time_images_train_only": sum(r["time"] is None for r in observations),
        "saved_pic_entire_541_image_continuous_acquisition_in_test": True,
        "known_near_identical_original_frame_both_copies_in_train": True,
        "validation_and_test_are_different_directories_and_images": True,
        "global_chronology_for_all_images_claimed": False,
        "unknown_video_provenance_independence_guaranteed": False,
        "original_inputs_modified": False,
        "manifest_sha256": digest(output / "audit/manifest.csv"),
    }
    (output / "audit/independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(Path(__file__), output / "audit" / Path(__file__).name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
