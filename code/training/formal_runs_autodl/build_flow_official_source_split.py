#!/usr/bin/env python3
"""Build an auditable, source-grouped YOLO release from the official 2,000 images.

Requires Pillow and NumPy. Refuses to overwrite an existing output directory.
Original acquisition metadata, rather than renamed numeric image IDs, defines
the split. Only fractional Unix-second filenames support chronological checks;
undated identifiers and unverified millisecond names remain train-only.
"""

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
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image


SOURCE_GROUPS = {
    "undated_static_collection": {
        "split": "train",
        "folders": ["myPerson"],
        "reason": "All 753 images share the original collection name; no reliable capture time or original path is available. Keep the complete collection in train.",
    },
    "ningbo_20200925": {
        "split": "train",
        "folders": ["images"] + [f"ningbo{i}_select_select_new" for i in range(1, 6)],
        "reason": "Keep the full September 25 acquisition together. The two images-folder timestamps lie inside the ningbo1 session.",
    },
    "ningbo_20200926_early_plus_added": {
        "split": "train",
        "folders": ["ningbo_1_select_select_new", "ningbo_2_select_select_new", "新添加图片"],
        "reason": "Keep the two early acquisition directories together. The entire mixed added-images directory joins them because official train/001163 and test/000736 share original name 1601097462.9573567 and are near-identical encodings. Its uncertain-time images remain train-only.",
    },
    "ningbo_20200926_later": {
        "split": "val",
        "folders": [f"ningbo_{i}_select_select_new" for i in range(3, 6)],
        "reason": "Keep all three later September 26 directories together, after the early train acquisition. The closest train/val supported timestamps are separated by 34 minutes 49.9904 seconds.",
    },
    "saved_pic_20200928_continuous": {
        "split": "test",
        "folders": ["Saved_Pic"] + [f"Saved_Pic_{i}" for i in range(2, 6)],
        "reason": "Merge all five directories and their drive/path aliases: consecutive directory boundaries have only 2.96 to 7.37 second gaps and belong to a continuous acquisition. Keep the entire session as test.",
    },
}
FOLDER_TO_GROUP = {folder: group for group, rule in SOURCE_GROUPS.items() for folder in rule["folders"]}
EXPECTED_FOLDERS = {
    "myPerson": 753, "images": 2, "新添加图片": 33,
    "ningbo1_select_select_new": 120, "ningbo2_select_select_new": 4,
    "ningbo3_select_select_new": 5, "ningbo4_select_select_new": 112,
    "ningbo5_select_select_new": 50, "ningbo_1_select_select_new": 12,
    "ningbo_2_select_select_new": 33, "ningbo_3_select_select_new": 14,
    "ningbo_4_select_select_new": 122, "ningbo_5_select_select_new": 199,
    "Saved_Pic": 104, "Saved_Pic_2": 136, "Saved_Pic_3": 112,
    "Saved_Pic_4": 96, "Saved_Pic_5": 93,
}
EXPECTED_IMAGES = {"train": 1124, "val": 335, "test": 541}
EXPECTED_BOXES = {"train": 2665, "val": 1517, "test": 1090}
TIME_CONFIRMED = "fractional_unix_seconds_in_original_filename"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def original_key(filename):
    # Path.stem would incorrectly discard fractional seconds without a .jpg suffix.
    return re.sub(r"\.(?:jpe?g|png|bmp)$", "", PureWindowsPath(filename).name, flags=re.I)


def timestamp_evidence(filename, folder):
    key = original_key(filename)
    if folder != "myPerson" and re.fullmatch(r"1\d{9}\.\d+", key):
        return float(key), TIME_CONFIRMED
    if folder != "myPerson" and re.fullmatch(r"1\d{12}", key):
        return float(key) / 1000, "unverified_millisecond_timestamp_candidate"
    return None, "undated_identifier"


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp is not None else ""


def parse_records(source):
    records = []
    for official_split, subdir, expected in [("train", "training", 1200), ("test", "test", 800)]:
        images = {p.stem: p for p in (source / subdir / "images").glob("*.jpg")}
        annotations = {p.stem: p for p in (source / subdir / "annotations").glob("*.xml")}
        require(len(images) == expected and images.keys() == annotations.keys(), f"Incomplete official {official_split} image/XML pairs")
        for image_id in sorted(images):
            annotation = annotations[image_id]
            node = ET.parse(annotation).getroot()
            folder = (node.findtext("folder") or "").strip()
            require(folder in FOLDER_TO_GROUP, f"Unknown acquisition folder: {folder!r}")
            group = FOLDER_TO_GROUP[folder]
            split = SOURCE_GROUPS[group]["split"]
            filename = (node.findtext("filename") or "").strip()
            original_path = (node.findtext("path") or "").strip()
            parent = str(PureWindowsPath(original_path).parent) if original_path else ""
            width, height = (int(node.findtext(f"size/{axis}") or 0) for axis in ("width", "height"))
            require(width > 0 and height > 0, f"Invalid dimensions: {annotation}")
            boxes = []
            for obj in node.findall("object"):
                require((obj.findtext("name") or "").strip().lower() == "bottle", f"Unexpected class: {annotation}")
                box = tuple(float(obj.findtext(f"bndbox/{key}") or "nan") for key in ("xmin", "ymin", "xmax", "ymax"))
                x1, y1, x2, y2 = box
                require(all(math.isfinite(v) for v in box) and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"Invalid box: {annotation}: {box}")
                boxes.append(box)
            require(boxes, f"Unexpected empty official annotation: {annotation}")
            timestamp, evidence = timestamp_evidence(filename, folder)
            if evidence != TIME_CONFIRMED:
                require(split == "train", f"Uncertain capture time outside train: {annotation}")
            stem = f"official_{official_split}_{image_id}"
            records.append({
                "official_split": official_split, "official_image_id": image_id,
                "split": split, "source_group": group, "raw_folder": folder,
                "original_filename": filename, "original_filename_key": original_key(filename),
                "original_path": original_path, "original_parent": parent,
                "timestamp_seconds": timestamp if timestamp is not None else "",
                "datetime_utc": iso(timestamp), "time_evidence": evidence,
                "width": width, "height": height, "box_count": len(boxes),
                "source_image": images[image_id].relative_to(source).as_posix(),
                "source_xml": annotation.relative_to(source).as_posix(),
                "image": f"images/{split}/{stem}.jpg", "label": f"labels/{split}/{stem}.txt",
                "archived_xml": f"audit/official_annotations/{stem}.xml", "_boxes": boxes,
            })
    require(Counter(r["raw_folder"] for r in records) == EXPECTED_FOLDERS, "Official acquisition folder counts changed")
    require(Counter(r["split"] for r in records) == EXPECTED_IMAGES, "Unexpected split image counts")
    return records


def grouping_checks(records):
    for field in ["source_group", "raw_folder", "original_parent", "original_filename_key"]:
        assignments = defaultdict(set)
        for r in records:
            if r[field]:
                assignments[r[field].casefold()].add(r["split"])
        require(all(len(splits) == 1 for splits in assignments.values()), f"Cross-split overlap in {field}")
    times = {}
    for split in EXPECTED_IMAGES:
        values = [float(r["timestamp_seconds"]) for r in records if r["split"] == split and r["time_evidence"] == TIME_CONFIRMED]
        require(values, f"No supported capture times for {split}")
        times[split] = {"images": len(values), "min": min(values), "max": max(values), "first_utc": iso(min(values)), "last_utc": iso(max(values))}
    require(times["train"]["max"] < times["val"]["min"] < times["val"]["max"] < times["test"]["min"], "Supported capture times are not chronologically separated")
    return times


def materialize(record, source, output):
    row = dict(record)
    image = source / row["source_image"]
    xml = source / row["source_xml"]
    image_hash, xml_hash = sha256(image), sha256(xml)
    with Image.open(image) as opened:
        rgb = opened.convert("RGB")
        require(rgb.size == (row["width"], row["height"]), f"Image/XML size mismatch: {image}")
        digest = hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        row["pixel_sha256"] = digest.hexdigest()
        feature = np.asarray(rgb.resize((32, 18), Image.Resampling.LANCZOS), dtype=np.float32).reshape(-1)
    shutil.copyfile(image, output / row["image"])
    shutil.copyfile(xml, output / row["archived_xml"])
    require(sha256(output / row["image"]) == image_hash, f"Image copy mismatch: {image}")
    require(sha256(output / row["archived_xml"]) == xml_hash, f"XML copy mismatch: {xml}")
    width, height = row["width"], row["height"]
    lines = []
    for x1, y1, x2, y2 in row.pop("_boxes"):
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
        lines.append("0 " + " ".join(f"{v:.10f}" for v in values))
    (output / row["label"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    row.update(image_sha256=image_hash, xml_sha256=xml_hash, label_sha256=sha256(output / row["label"]))
    return row, feature


def visual_checks(rows, features):
    repeated = defaultdict(list)
    for row in rows:
        repeated[row["pixel_sha256"]].append(row)
    exact = [[r["image"] for r in rr] for rr in repeated.values() if len(rr) > 1]
    for rr in repeated.values():
        require(len({r["split"] for r in rr}) == 1, "Exact pixel duplicate across subsets")
    a = np.stack(features)
    squared = np.sum(a * a, axis=1)
    distances = np.sqrt(np.maximum(0, (squared[:, None] + squared[None, :] - 2 * a @ a.T) / a.shape[1]))
    np.fill_diagonal(distances, np.inf)
    cross_candidates, cross_folder_candidates, nearest = [], [], {}
    for i, row in enumerate(rows):
        for j in np.where(distances[i, i + 1:] <= 3)[0] + i + 1:
            candidate = {"image_a": row["image"], "image_b": rows[j]["image"], "thumbnail_rgb_rmse_255": float(distances[i, j])}
            if row["split"] != rows[j]["split"]:
                cross_candidates.append(candidate)
            if row["raw_folder"] != rows[j]["raw_folder"]:
                cross_folder_candidates.append(candidate)
    require(not cross_candidates, "Near-duplicate candidates across subsets require review")
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        ii = [i for i, r in enumerate(rows) if r["split"] == left]
        jj = [i for i, r in enumerate(rows) if r["split"] == right]
        sub = distances[np.ix_(ii, jj)]
        x, y = np.unravel_index(np.argmin(sub), sub.shape)
        nearest[f"{left}/{right}"] = {"image_a": rows[ii[x]]["image"], "image_b": rows[jj[y]]["image"], "thumbnail_rgb_rmse_255": float(sub[x, y])}
    return {
        "exact_pixel_duplicate_groups": exact,
        "near_duplicate_screening": "32x18 RGB Lanczos thumbnails, RMSE <= 3 in 0..255 units. A screening check, not proof of video provenance or absence of all visual similarity.",
        "cross_split_candidates": cross_candidates, "cross_folder_candidates": cross_folder_candidates,
        "nearest_cross_split_pairs": nearest,
    }


def write_readme(output, counts, times):
    text = """# FloW IMG: 官方 2000 张，来源分组 YOLO 版本

本版本使用官方原始 JPG 和完整 XML 标注，单类别 `0: bottle`。图片只复制，不缩放、不重新压缩。原来的官方划分和旧 YOLO 版本均不修改。

| 子集 | 图片 | 标注框 | 分组依据 |
|---|---:|---:|---|
| train | 1124 | 2665 | myPerson 全部；9 月 25 日全部；9 月 26 日较早两个目录；新添加图片整个目录 |
| val | 335 | 1517 | 9 月 26 日较晚三个采集目录整体保留 |
| test | 541 | 1090 | Saved_Pic 至 Saved_Pic_5 整段连续采集全部保留 |
| 合计 | 2000 | 5272 | 每个官方文件恰好使用一次 |

最初期望 1200 张训练，其余分成 val/test；保留完整来源组并保持可验证时间先后时，实际采用 1124/335/541，不随机抽帧凑数。val 和 test 是不同图片、不同来源组。训练中选模型和调参使用 val，最终评估使用 test。

## 来源与时间依据

- 同一 XML `folder`、同一非空原始采集目录、同一原始文件名均不跨子集；不同磁盘上的来源别名按采集记录合并。
- `Saved_Pic` 到 `Saved_Pic_5` 的相邻目录时间间隔仅约 2.96–7.37 秒，因此合并整段放在 test。
- `images` 中两张图片的时间落在 `ningbo1` 采集内，与 9 月 25 日来源一起保留。
- “新添加图片”中 `official_test_000736` 与 `official_train_001163` 同名、画面近乎相同；整个新增目录和宁波早段均放在 train。保留官方 2000 个文件，因此这两个不同压缩版本也保留，但都在 train。
- 仅小数秒 Unix 文件名用于时间顺序检查。`myPerson` 的数字编号不是采集时间；25 个毫秒样式文件名缺少原始采集语义，不据此认定日期。这些时间未确认的 784 张全部只进入 train。

已支持采集时间的图片为 train 340 张、val 335 张、test 541 张。UTC 时间段如下：

| 子集 | 最早 | 最晚 |
|---|---|---|
"""
    for split in EXPECTED_IMAGES:
        text += f"| {split} | {times[split]['first_utc']} | {times[split]['last_utc']} |\n"
    text += """
有时间依据的 train 最后一张与 val 第一张相隔约 34 分 50 秒；val 最后一张与 test 第一张相隔约 39 小时 53 分。时间字段统一保存为 UTC；Saved_Pic 所在目录标注的 09_28 对应中国当地日期。

该划分解决现有元数据能够识别的同来源、连续帧跨集合问题。官方包没有完整的原始视频 ID 映射，部分采集时间仍缺失，因此不能证明所有未知来源都绝对独立，也不能宣称全部 2000 张都满足严格全局时间截止。未凭空补写时间或视频 ID。图像相似性筛查只是补充检查。test 集主要来自一个连续采集会话，其指标反映该会话上的表现。

## 使用

`data.yaml` 中 train、val、test 分别指向独立的 images/train、images/val、images/test，匹配标签位于 labels 下。类别编号从 0 开始，YOLO 标签每行为 `class x_center y_center width height`，坐标按图片宽高归一化。VOC 坐标不做额外 -1 或裁剪，保持既有项目采用的边界坐标约定；转换后可逆核对至原 XML 框。

例如，在当前项目根目录运行：

```text
yolo detect train data=FloW_IMG_YOLO_2000_source_disjoint/data.yaml model=yolov8n.pt
```

本目录没有训练权重，也未启动模型训练。调整划分后，旧划分上的实验指标不能当作本划分的结果，需用此 data.yaml 重新训练和评估。旧训练图片中有一部分已进入新的 val/test，因此应从未使用 FloW 的通用预训练权重或随机初始化开始，不能直接继续旧 FloW 权重，否则权重已经见过新评估集，会产生训练泄露。

## 可追溯记录

- `audit/manifest.csv`：逐图记录旧编号、新子集、原始目录/文件名/时间证据、文件 SHA-256。
- `audit/source_groups.csv` 与 `audit/protocol.json`：固定分组、分配原因、时间区间、标注框数量。
- `audit/official_annotations/`：每张图对应的官方原始 XML 字节副本。
- `audit/visual_duplicate_audit.json`：全图像素哈希和缩略图相似性筛查结果。
- `audit/build_report.json`：构建统计；完成后另有独立复核报告。
- `audit/build_flow_official_source_split.py`：本次构建脚本的副本。

使用官方 XML 保留全部 5272 个框，包括先前 Roboflow 导出遗漏的左边界目标。不会以空标签代替已有官方标注。所有输入保留不变。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("D:/download/FloW_IMG/FloW_IMG"))
    parser.add_argument("--output", type=Path, default=Path("D:/ring3/02_当前实验/数据集/FloW_1124_335_541"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(source.is_dir(), f"Missing official dataset: {source}")
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    require(not output.is_relative_to(source), "Output must not be inside the official input")
    records = parse_records(source)
    times = grouping_checks(records)
    for split in EXPECTED_IMAGES:
        require(sum(r["box_count"] for r in records if r["split"] == split) == EXPECTED_BOXES[split], f"Unexpected box count: {split}")
        for kind in ["images", "labels"]:
            (output / kind / split).mkdir(parents=True, exist_ok=True)
    audit = output / "audit"
    (audit / "official_annotations").mkdir(parents=True)
    print("Source metadata and chronological group checks passed; copying official data.", flush=True)
    rows, features = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (row, feature) in enumerate(pool.map(lambda r: materialize(r, source, output), records), 1):
            rows.append(row)
            features.append(feature)
            if i % 400 == 0:
                print(f"Copied and checked {i}/2000 images and annotations.", flush=True)
    visual = visual_checks(rows, features)
    write_json(audit / "visual_duplicate_audit.json", visual)
    with (audit / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for group, rule in SOURCE_GROUPS.items():
        rr = [r for r in rows if r["source_group"] == group]
        tt = [float(r["timestamp_seconds"]) for r in rr if r["time_evidence"] == TIME_CONFIRMED]
        groups.append({"source_group": group, "split": rule["split"], "images": len(rr), "boxes": sum(r["box_count"] for r in rr), "folders": " | ".join(rule["folders"]), "supported_time_images": len(tt), "first_utc": iso(min(tt)) if tt else "", "last_utc": iso(max(tt)) if tt else "", "reason": rule["reason"]})
    with (audit / "source_groups.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    counts = {split: {"images": EXPECTED_IMAGES[split], "boxes": EXPECTED_BOXES[split]} for split in EXPECTED_IMAGES}
    protocol = {
        "protocol": "flow_official2000_source_temporal_grouped_v1",
        "source_root": str(source), "output_root": str(output), "groups": SOURCE_GROUPS,
        "counts": counts, "supported_capture_times": times,
        "train_to_val_gap_seconds": times["val"]["min"] - times["train"]["max"],
        "val_to_test_gap_seconds": times["test"]["min"] - times["val"]["max"],
        "uncertain_time_images": sum(r["time_evidence"] != TIME_CONFIRMED for r in rows),
        "all_uncertain_time_images_in_train": True,
        "chronology_claim": "Strict ordering only for supported fractional Unix-second filenames; no global cutoff claim for undated or unverified millisecond identifiers.",
        "source_claim": "Disjoint observed source folders, original parent paths, normalized original filenames, merged known continuous acquisitions. Missing original video IDs prevent an absolute proof of all unknown source independence.",
        "coordinate_convention": "Preserve XML edge coordinates without +/-1 adjustment; normalize center and xmax-xmin/ymax-ymin extents; 10 decimal places.",
        "selection": "Fixed source assignments, no random frame split and no model-performance-based assignment.",
    }
    write_json(audit / "protocol.json", protocol)
    (output / "data.yaml").write_text("# Official FloW images; source-grouped, separate validation and test.\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: bottle\n", encoding="utf-8")
    write_readme(output, counts, times)
    shutil.copyfile(Path(__file__), audit / Path(__file__).name)
    report = {
        "status": "built; independent annotation round-trip verification is separate",
        "images": len(rows), "boxes": sum(r["box_count"] for r in rows), "split_counts": counts,
        "source_groups": len(groups), "raw_source_folders": len(EXPECTED_FOLDERS),
        "all_copied_image_and_xml_hashes_match_official": True,
        "cross_split_source_group_overlap": 0, "cross_split_raw_folder_overlap": 0,
        "cross_split_original_parent_overlap": 0, "cross_split_original_filename_overlap": 0,
        "cross_split_exact_pixel_duplicates": 0, "cross_split_thumbnail_candidates_rmse_at_most_3": 0,
        "supported_capture_time_ordered": True, "original_inputs_modified": False,
        "manifest_sha256": sha256(audit / "manifest.csv"), "builder_sha256": sha256(Path(__file__)),
    }
    write_json(audit / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
