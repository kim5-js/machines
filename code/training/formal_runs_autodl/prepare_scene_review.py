from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENE_LABELS = ("Normal", "Glare", "Low-light", "Mixed")
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(image: dict[str, Any], origin: str) -> str:
    source = image.get("source")
    if not isinstance(source, dict):
        return f"singleton:{origin}:{image['file_name']}"
    drone = str(source.get("drone") or "unknown_drone")
    folder = str(source.get("folder_name") or "unknown_folder")
    video = str(source.get("video") or "")
    if video:
        return f"video:{drone}|{folder}|{video}"
    if source.get("image_name"):
        return f"still_acquisition:{drone}|{folder}"
    return f"singleton:{origin}:{image['file_name']}"


def load_source_map(dataset_root: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for origin in ("train", "val"):
        annotation_path = dataset_root / "annotations" / f"instances_{origin}.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        for image in payload["images"]:
            name = Path(str(image["file_name"])).name
            if name in mapping:
                raise RuntimeError(f"Duplicate image filename in source metadata: {name}")
            mapping[name] = {
                "origin": origin,
                "source_group": source_group(image, origin),
                "source": image.get("source"),
            }
    return mapping


def load_val_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        pieces = raw.split("\t")
        if len(pieces) != 2:
            raise RuntimeError(f"Invalid val manifest line {line_number}: {raw!r}")
        origin, name = pieces
        records[name] = origin
    return records


def image_statistics(path: Path) -> dict[str, float | int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, 640.0 / float(max(height, width)))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    neutral_highlight = ((gray >= 220.0) & (saturation <= 0.35)).astype(np.uint8)
    component_fraction = 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(neutral_highlight, connectivity=8)
    if count > 1:
        component_fraction = float(stats[1:, cv2.CC_STAT_AREA].max()) / float(gray.size)
    values = np.percentile(gray, [5, 10, 50, 90, 95])
    return {
        "width": int(width),
        "height": int(height),
        "mean_luma": float(gray.mean()),
        "p05_luma": float(values[0]),
        "p10_luma": float(values[1]),
        "p50_luma": float(values[2]),
        "p90_luma": float(values[3]),
        "p95_luma": float(values[4]),
        "dark_frac_32": float((gray < 32.0).mean()),
        "dark_frac_64": float((gray < 64.0).mean()),
        "bright_frac_220": float((gray >= 220.0).mean()),
        "bright_frac_245": float((gray >= 245.0).mean()),
        "neutral_highlight_frac": float(neutral_highlight.mean()),
        "largest_neutral_highlight_frac": component_fraction,
        "mean_saturation": float(saturation.mean()),
    }


def heuristic_label(stats: dict[str, float | int]) -> str:
    low_light = (
        float(stats["mean_luma"]) < 92.0
        or float(stats["p50_luma"]) < 78.0
        or float(stats["dark_frac_64"]) > 0.52
    )
    glare = (
        float(stats["largest_neutral_highlight_frac"]) > 0.008
        and float(stats["neutral_highlight_frac"]) > 0.012
        and float(stats["p95_luma"]) >= 220.0
    )
    if low_light and glare:
        return "Mixed"
    if low_light:
        return "Low-light"
    if glare:
        return "Glare"
    return "Normal"


def slug_for_group(group: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", group).strip("_")[:80]
    return f"{readable}_{hashlib.sha256(group.encode('utf-8')).hexdigest()[:10]}"


def font() -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, 15)
    return ImageFont.load_default()


def make_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    columns: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    label_height = 54
    cell_width = thumb_width
    cell_height = thumb_height + label_height
    sheet_rows = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * cell_width, sheet_rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    text_font = font()
    for index, row in enumerate(rows):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        with Image.open(row["image_path"]) as source:
            source = source.convert("RGB")
            source.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_width, thumb_height), (24, 24, 24))
            tile.paste(source, ((thumb_width - source.width) // 2, (thumb_height - source.height) // 2))
        canvas.paste(tile, (x, y))
        decision = row.get("manual_label") or "UNLABELED"
        line1 = f"{row['image_name']}  [{decision}]"
        line2 = (
            f"mean={row['mean_luma']:.0f} p50={row['p50_luma']:.0f} "
            f"hi={100.0 * row['neutral_highlight_frac']:.1f}% auto={row['heuristic_label']}"
        )
        draw.rectangle((x, y + thumb_height, x + cell_width, y + cell_height), fill="white")
        draw.text((x + 4, y + thumb_height + 3), line1, fill="black", font=text_font)
        draw.text((x + 4, y + thumb_height + 27), line2, fill=(55, 55, 55), font=text_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)


def load_decisions(path: Path | None) -> tuple[dict[str, str], dict[str, str], dict[str, Any] | None]:
    if path is None:
        return {}, {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ring3.robustness_scene_label_decisions.v1":
        raise RuntimeError(f"Unexpected decision schema: {payload.get('schema')!r}")
    groups = {str(key): str(value) for key, value in payload.get("group_labels", {}).items()}
    overrides = {str(key): str(value) for key, value in payload.get("image_overrides", {}).items()}
    bad = {value for value in [*groups.values(), *overrides.values()] if value not in SCENE_LABELS}
    if bad:
        raise RuntimeError(f"Invalid manual labels: {sorted(bad)}")
    return groups, overrides, payload


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare auditable manual scene review sheets for locked SeaDronesSee val")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=180)
    args = parser.parse_args()
    if args.columns <= 0 or args.page_size <= 0:
        raise SystemExit("--columns and --page-size must be positive")

    source_map = load_source_map(args.dataset_root)
    val_manifest = load_val_manifest(args.val_manifest)
    image_paths = sorted(
        (path for path in args.val_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if len(image_paths) != 1555 or len(val_manifest) != 1555:
        raise RuntimeError(f"Expected 1555 locked val images, found images={len(image_paths)} manifest={len(val_manifest)}")
    if {path.name for path in image_paths} != set(val_manifest):
        raise RuntimeError("Locked val image directory and val.tsv disagree")

    group_labels, overrides, decision_payload = load_decisions(args.decisions)
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        metadata = source_map.get(image_path.name)
        if metadata is None:
            raise RuntimeError(f"Source metadata missing for {image_path.name}")
        if metadata["origin"] != val_manifest[image_path.name]:
            raise RuntimeError(f"Origin mismatch for {image_path.name}: {metadata['origin']} != {val_manifest[image_path.name]}")
        stats = image_statistics(image_path)
        group = str(metadata["source_group"])
        manual = overrides.get(image_path.name, group_labels.get(group, ""))
        rows.append(
            {
                "review_index": index,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "origin": metadata["origin"],
                "source_group": group,
                **stats,
                "heuristic_label": heuristic_label(stats),
                "manual_label": manual,
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["source_group"]].append(row)
    sheets_root = args.output / "contact_sheets"
    group_summaries: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_group.items(), key=lambda item: (-len(item[1]), item[0])):
        slug = slug_for_group(group)
        summary_rows = [group_rows[round(value)] for value in np.linspace(0, len(group_rows) - 1, min(25, len(group_rows)))]
        make_sheet(
            summary_rows,
            sheets_root / f"SUMMARY_{slug}.jpg",
            columns=args.columns,
            thumb_width=args.thumb_width,
            thumb_height=args.thumb_height,
        )
        for start in range(0, len(group_rows), args.page_size):
            make_sheet(
                group_rows[start : start + args.page_size],
                sheets_root / slug / f"page_{start // args.page_size + 1:03d}.jpg",
                columns=args.columns,
                thumb_width=args.thumb_width,
                thumb_height=args.thumb_height,
            )
        values = {key: [float(row[key]) for row in group_rows] for key in ("mean_luma", "p50_luma", "neutral_highlight_frac")}
        group_summaries.append(
            {
                "source_group": group,
                "slug": slug,
                "images": len(group_rows),
                "first_image": group_rows[0]["image_name"],
                "last_image": group_rows[-1]["image_name"],
                "heuristic_counts": dict(Counter(row["heuristic_label"] for row in group_rows)),
                "manual_counts": dict(Counter(row["manual_label"] for row in group_rows if row["manual_label"])),
                "mean_luma_quantiles": [float(value) for value in np.quantile(values["mean_luma"], [0, 0.1, 0.5, 0.9, 1])],
                "p50_luma_quantiles": [float(value) for value in np.quantile(values["p50_luma"], [0, 0.1, 0.5, 0.9, 1])],
                "neutral_highlight_fraction_quantiles": [
                    float(value) for value in np.quantile(values["neutral_highlight_frac"], [0, 0.1, 0.5, 0.9, 1])
                ],
            }
        )

    fields = [
        "review_index",
        "image_name",
        "image_path",
        "origin",
        "source_group",
        "width",
        "height",
        "mean_luma",
        "p05_luma",
        "p10_luma",
        "p50_luma",
        "p90_luma",
        "p95_luma",
        "dark_frac_32",
        "dark_frac_64",
        "bright_frac_220",
        "bright_frac_245",
        "neutral_highlight_frac",
        "largest_neutral_highlight_frac",
        "mean_saturation",
        "heuristic_label",
        "manual_label",
    ]
    write_csv(args.output / "scene_review_manifest.csv", rows, fields)
    labeled = [row for row in rows if row["manual_label"]]
    if decision_payload is not None:
        if len(labeled) != len(rows):
            missing = [row["image_name"] for row in rows if not row["manual_label"]]
            raise RuntimeError(f"Manual decisions leave {len(missing)} images unlabeled; first={missing[:10]}")
        write_csv(
            args.output / "scene_labels.csv",
            rows,
            ["image_name", "manual_label", "source_group", "origin", "review_index"],
        )
    report = {
        "schema": "ring3.robustness_scene_review_manifest.v1",
        "status": "completed" if len(labeled) == len(rows) else "awaiting_manual_labels",
        "dataset_root": str(args.dataset_root),
        "val_images": str(args.val_images),
        "val_manifest": str(args.val_manifest),
        "val_manifest_sha256": sha256(args.val_manifest),
        "validation_image_count": len(rows),
        "source_group_count": len(by_group),
        "allowed_labels": list(SCENE_LABELS),
        "manual_label_count": len(labeled),
        "manual_label_counts": dict(Counter(row["manual_label"] for row in labeled)),
        "heuristic_labels_are_review_order_only": True,
        "decision_file": str(args.decisions) if args.decisions else None,
        "decision_file_sha256": sha256(args.decisions) if args.decisions else None,
        "groups": group_summaries,
    }
    report_path = args.output / "scene_review_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
