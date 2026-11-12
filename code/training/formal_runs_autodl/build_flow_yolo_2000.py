#!/usr/bin/env python3
"""Build repaired FloW YOLO labels using official train and shared val/test subsets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from PIL import Image

from build_flow_source_temporal_protocol import SOURCE_GROUP

COORDS = ("xmin", "ymin", "xmax", "ymax")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def annotation(path: Path):
    data = path.read_bytes()
    root = ET.fromstring(data)
    size = (int(root.findtext("size/width")), int(root.findtext("size/height")))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        x1, y1, x2, y2 = (float(obj.findtext("bndbox/" + key)) for key in COORDS)
        if name != "bottle" or not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError(f"Invalid annotation: {path}: {name}, {(x1, y1, x2, y2)}")
        boxes.append((name, x1, y1, x2, y2))
    if len(boxes) != len(set(boxes)):
        raise ValueError(f"Duplicate boxes: {path}")
    return root, size, boxes, sha(data)


def acquisition(root):
    folder = root.findtext("folder") or ""
    name = root.findtext("filename") or ""
    if folder not in SOURCE_GROUP:
        raise ValueError(f"Unexpected original acquisition directory: {folder!r}")
    # Numeric myPerson names are identifiers, not Unix capture timestamps.
    timestamp = None
    if folder != "myPerson":
        value = name[:-4] if name.lower().endswith(".jpg") else name
        timestamp = float(value)
        if not 1577836800 <= timestamp < 1609459200:
            raise ValueError(f"Unexpected dated acquisition: {folder}, {name}")
    return {
        "raw_folder": folder, "source_group": SOURCE_GROUP[folder],
        "original_filename": name, "original_path": root.findtext("path") or "",
        "capture_timestamp": timestamp,
        "capture_datetime_utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else "",
    }


def audit_coco(root: Path):
    findings = {}
    for split in ("train", "test"):
        path = root / split / "_annotations.coco.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        images = data.get("images", [])
        date_created = data.get("info", {}).get("date_created")
        findings[split] = {
            "path": str(path), "sha256": sha(path.read_bytes()),
            "images": len(images), "annotations": len(data.get("annotations", [])),
            "date_created": date_created,
            "date_captured_counts": dict(Counter(row.get("date_captured", "") for row in images)),
            "all_capture_dates_equal_export_creation_date": all(row.get("date_captured") == date_created for row in images),
            "images_with_source_fields": {
                key: sum(bool(row.get(key)) for row in images)
                for key in ("folder", "path", "video_id", "sequence_id", "original_filename", "extra")
            },
        }
    return {
        "findings": findings,
        "conclusion": "This COCO export contains one shared export timestamp, not usable per-image acquisition dates. It does not recover the missing 800-image provenance.",
    }


def main():
    project = Path('D:/ring3')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, default=project / "03_历史归档/数据集/早期FloW导出/FloW_IMG_VOC.v1i.voc")
    parser.add_argument("--reference-root", type=Path, default=project / "03_历史归档/数据集/早期FloW导出/原始训练标注")
    parser.add_argument("--coco-root", type=Path, default=project / "03_历史归档/数据集/早期FloW导出/FloW_IMG_VOC.v1i.coco")
    parser.add_argument("--output-root", type=Path, default=project / "03_历史归档/数据集/旧FloW_1200_800")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    records, repairs = [], []
    for split, count in (("train", 1200), ("test", 800)):
        folder = args.voc_root.resolve() / split
        images = sorted(folder.glob("*.jpg"))
        if len(images) != count or {p.stem for p in images} != {p.stem for p in folder.glob("*.xml")}:
            raise ValueError(f"Unexpected image/XML inventory: {folder}")
        for image in images:
            xml = image.with_suffix(".xml")
            _, size, exported, exported_hash = annotation(xml)
            image_id = image.name.split("_jpg.rf.")[0]
            label_source, label_hash, boxes = xml, exported_hash, exported
            provenance = {k: "" for k in ("raw_folder", "source_group", "original_filename", "original_path", "capture_timestamp", "capture_datetime_utc")}
            missing = []
            if split == "train":
                label_source = args.reference_root.resolve() / f"{image_id}.xml"
                root, ref_size, boxes, label_hash = annotation(label_source)
                if size != ref_size or Counter(exported) - Counter(boxes):
                    raise ValueError(f"Unexpected reference differences: {image_id}")
                provenance = acquisition(root)
                missing = list((Counter(boxes) - Counter(exported)).elements())
                if missing:
                    repairs.append({
                        "image": image.name, "image_id": image_id, "reference_xml": str(label_source),
                        "boxes_before": len(exported), "boxes_after": len(boxes),
                        "restored_boxes": [dict(zip(("class_name", *COORDS), box)) for box in missing],
                    })
            records.append({
                "source_image": image, "source_split": split, "source_xml": xml, "source_xml_sha256": exported_hash,
                "label_source": label_source, "label_source_sha256": label_hash, "image_id": image_id,
                "size": size, "boxes": boxes, "boxes_before": len(exported), "restored_boxes": len(missing),
                "provenance": provenance,
            })
    if len({r["source_image"].name for r in records}) != 2000:
        raise ValueError("Image filenames are not globally unique")
    if len(repairs) != 49 or sum(r["restored_boxes"] for r in records) != 52:
        raise ValueError("The verified 49-image / 52-box repair inventory changed")
    if sum(len(r["boxes"]) for r in records) != 5243:
        raise ValueError("The verified final 5243-box inventory changed")
    print("PREFLIGHT_OK: 2000 images; 49 affected images; 52 boxes restored; official 1200 train / 800 shared val-test", flush=True)

    for split in ("train", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    audit = out / "audit"
    audit.mkdir()
    (out / "BUILD_IN_PROGRESS.txt").write_text("Dataset build is in progress; use only after validation_report.json reports passed.\n", encoding="utf-8")

    def convert(record):
        source = record["source_image"]
        image = out / "images" / record["source_split"] / source.name
        label = out / "labels" / record["source_split"] / f"{source.stem}.txt"
        image_bytes = source.read_bytes()
        image_hash = sha(image_bytes)
        image.write_bytes(image_bytes)
        if sha(image.read_bytes()) != image_hash:
            raise ValueError(f"Image copy mismatch: {image}")
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            if decoded.size != record["size"] or decoded.mode != "RGB":
                raise ValueError(f"Image size/mode disagrees with XML: {source}")
            pixel_hash = sha(str(decoded.size).encode() + decoded.tobytes())
        w, h = record["size"]
        lines = []
        for _, x1, y1, x2, y2 in record["boxes"]:
            values = ((x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h)
            lines.append("0 " + " ".join(f"{v:.10f}" for v in values))
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written = label.read_text(encoding="utf-8").splitlines()
        if len(written) != len(record["boxes"]):
            raise ValueError(f"Label count mismatch: {label}")
        error = 0.0
        for line, (_, x1, y1, x2, y2) in zip(written, record["boxes"]):
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Invalid YOLO row: {label}")
            xc, yc, bw, bh = map(float, fields[1:])
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f"Invalid normalized box: {label}")
            back = ((xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h)
            error = max(error, *(abs(a-b) for a, b in zip(back, (x1, y1, x2, y2))))
        if error > 0.000001:
            raise ValueError(f"Coordinate round-trip error: {label}: {error}")
        return {
            "image": image.relative_to(out).as_posix(), "label": label.relative_to(out).as_posix(),
            "source_split": record["source_split"], "source_image": str(source),
            "source_xml": str(record["source_xml"]), "source_xml_sha256": record["source_xml_sha256"],
            "label_source": str(record["label_source"]), "label_source_sha256": record["label_source_sha256"],
            "image_sha256": image_hash, "pixel_sha256": pixel_hash, "label_sha256": sha(label.read_bytes()),
            "width": w, "height": h, "boxes_before_repair": record["boxes_before"], "boxes": len(written),
            "restored_boxes": record["restored_boxes"], "coordinate_roundtrip_error_pixels": error,
        }

    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(convert, records), 1):
            manifest.append(row)
            if index % 250 == 0:
                print(f"COPIED_AND_VERIFIED {index}/2000", flush=True)
    for split, count in (("train", 1200), ("test", 800)):
        actual_images = {p.stem for p in (out/"images"/split).glob("*.jpg")}
        actual_labels = {p.stem for p in (out/"labels"/split).glob("*.txt")}
        if len(actual_images) != count or actual_images != actual_labels:
            raise ValueError(f"Final image/label pairing mismatch: {split}")
    source_rows = [
        {"image_name": r["source_image"].name, "source_split": r["source_split"], "assigned_split": ("train" if r["source_split"] == "train" else "val/test_shared"),
         **r["provenance"], "provenance_status": "verified_original_xml" if r["provenance"]["source_group"] else "missing_original_source_metadata",
         "provenance_xml": str(r["label_source"]) if r["provenance"]["source_group"] else ""}
        for r in records
    ]
    for name, rows in (("manifest.csv", manifest), ("source_groups.csv", source_rows)):
        with (audit / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    hashes = defaultdict(list)
    for row in manifest:
        hashes[row["pixel_sha256"]].append(row["image"])
    stats = {
        split: {"images": sum(r["source_split"] == split for r in manifest),
                "boxes": sum(r["boxes"] for r in manifest if r["source_split"] == split),
                "empty_labels": sum(r["boxes"] == 0 for r in manifest if r["source_split"] == split)}
        for split in ("train", "test")
    }
    cross_subset_duplicates = [paths for paths in hashes.values() if len({Path(p).parts[1] for p in paths}) > 1]
    (out / "data.yaml").write_text(
        "# Official 1200/800 membership. Validation and testing use the SAME 800 images.\n"
        "# Paths are relative to this YAML file; pass its absolute path to Ultralytics.\n"
        "train: images/train\nval: images/test\ntest: images/test\nnc: 1\nnames:\n  0: bottle\n", encoding="utf-8"
    )
    report = {
        "status": "passed" if not cross_subset_duplicates else "passed_with_exact_duplicate_warning",
        "created_utc": datetime.now(timezone.utc).isoformat(), "total_images": 2000, "total_boxes": 5243,
        "repaired_images": 49, "restored_boxes": 52, "output_root": str(out),
        "original_sources_modified": False, "source_split_statistics": stats,
        "encoded_image_bytes_preserved": True, "all_images_decoded": True,
        "all_image_label_pairs_verified": True, "all_coordinate_roundtrips_verified": True,
        "max_coordinate_roundtrip_error_pixels": max(r["coordinate_roundtrip_error_pixels"] for r in manifest),
        "coordinate_convention": "xc=(xmin+xmax)/(2W), yc=(ymin+ymax)/(2H), w=(xmax-xmin)/W, h=(ymax-ymin)/H; no 1-pixel offset",
        "exact_pixel_duplicate_groups": [paths for paths in hashes.values() if len(paths) > 1],
        "source_metadata_known_images": sum(bool(r["source_group"]) for r in source_rows),
        "source_metadata_missing_images": sum(not r["source_group"] for r in source_rows),
        "known_source_group_counts": dict(Counter(r["source_group"] for r in source_rows if r["source_group"])),
        "dated_images": sum(bool(r["capture_datetime_utc"]) for r in source_rows),
        "source_group_disjoint_split_verified": False, "training_yaml_created": True,
        "split_protocol": "official_train_1200_shared_validation_test_800",
        "val_and_test_share_same_800_images": True, "independent_test_set": False,
        "official_membership_preserved": True,
        "train_evaluation_exact_pixel_duplicates": cross_subset_duplicates,
        "data_yaml_sha256": sha((out/"data.yaml").read_bytes()),
        "manifest_sha256": sha((audit/"manifest.csv").read_bytes()),
        "source_groups_sha256": sha((audit/"source_groups.csv").read_bytes()),
    }
    save_json(audit / "annotation_repairs.json", repairs)
    save_json(audit / "validation_report.json", report)
    save_json(audit / "coco_metadata_audit.json", audit_coco(args.coco_root.resolve()))
    (out / "classes.txt").write_text("bottle\n", encoding="utf-8")
    (audit / "source_readmes").mkdir()
    for p in args.voc_root.glob("README*.txt"):
        shutil.copyfile(p, audit / "source_readmes" / p.name)
    (out / "README.md").write_text(
        "# FloW IMG YOLO 2000\n\n"
        "已补齐标注，按原数据包 1200 张 train / 800 张 test 成员生成标准 YOLO detection 格式。\n\n"
        "| 用途 | 图片 | 目标框 | 空标签 |\n|---|---:|---:|---:|\n"
        "| train | 1200 | 3248 | 0 |\n| val / test 共用 | 800 | 1995 | 4 |\n"
        "| 合计（800 张只计一次） | 2000 | 5243 | 4 |\n\n"
        "## 使用\n\n"
        "配置文件：data.yaml。train 指向 images/train；val 和 test 都指向 images/test，共用同一组 800 张图片。\n"
        "标签位于 labels/train、labels/test，与图片同名一一对应。类别为 0=bottle。每行为 class_id x_center y_center width height，坐标归一化到 [0,1]。\n"
        "请向 Ultralytics 传入 data.yaml 的绝对路径；配置里的图片目录相对于该 YAML，整个目录可一起迁移。\n"
        "示例：yolo detect train model=yolov8s.pt data=\"D:/ring3/03_历史归档/数据集/旧FloW_1200_800/data.yaml\"\n\n"
        "## 标注和图像来源\n\n"
        "图片字节直接复制自 FloW_IMG_VOC.v1i.voc，没有重新压缩。1200 张 train 使用 FloW_IMG/annotations 的已核实原始标注，补回 49 张图片中缺失的 52 个框，从 3196 个恢复至 3248 个。\n"
        "另外 800 张使用下载包的 1995 个框，保留其中 4 个空标签。VOC 坐标转换不额外减 1 像素，与现有本地 FloW YOLO 标签约定一致。原始数据未修改。\n\n"
        "## 评估范围\n\n"
        "按用户确认沿用官方成员划分，不另行随机拆分这 800 张。val 与 test 是同一个评估集，因此不是独立测试集。\n"
        "此版本没有验证采集来源或时间隔离。1200 张可恢复原始目录信息，其中 809 张有可靠时间戳；另外 800 张缺少原始采集来源，不能声称无时间泄露。\n"
        "COCO 版本的 2000 个 date_captured 均为 2024-03-26T07:36:48+00:00，与导出时间一致；这些值未被当作真实采集日期。\n\n"
        "## 核查文件\n\n"
        "- audit/manifest.csv：2000 张图片和标签的来源、SHA-256、尺寸、框数及转换精度。\n"
        "- audit/annotation_repairs.json：补回的 52 个框及其原始 XML 来源。\n"
        "- audit/source_groups.csv：现有来源元数据、缺失状态及最终用途。\n"
        "- audit/coco_metadata_audit.json：COCO 日期和来源字段的核查证据。\n"
        "- audit/validation_report.json：图片解码、拷贝、配对、坐标还原和完全重复检查。\n\n"
        "YOLO 格式依据：https://docs.ultralytics.com/datasets/detect/\n", encoding="utf-8"
    )
    (out / "BUILD_IN_PROGRESS.txt").unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
