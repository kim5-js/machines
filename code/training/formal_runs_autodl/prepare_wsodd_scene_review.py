from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_wsodd_split import IMAGE_SUFFIXES, build_groups
from prepare_scene_review import (
    SCENE_LABELS,
    heuristic_label,
    image_statistics,
    load_decisions,
    make_sheet,
    sha256,
    slug_for_group,
    write_csv,
)


EXPECTED_TEST_IMAGES = 2406
EXPECTED_GROUPS = 32
EXPECTED_TEST_MANIFEST_SHA256 = "786bd086a5924fc8cda5ea339eaee8d30928ddd1fe7c90312fb19c1186a1c136"


def read_names(path: Path) -> list[str]:
    return [
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare auditable WSODD sequence-level scene review sheets")
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--gap-seconds", type=float, default=60.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=180)
    args = parser.parse_args()

    if sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Locked WSODD test manifest hash changed")
    all_images = sorted(
        (path for path in args.source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    by_name = {path.name.casefold(): path for path in all_images}
    group_by_name = build_groups(all_images, args.gap_seconds)
    names = read_names(args.test_manifest)
    if len(names) != EXPECTED_TEST_IMAGES or len({name.casefold() for name in names}) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Unexpected WSODD test manifest size: {len(names)}")
    missing = [name for name in names if name.casefold() not in by_name]
    if missing:
        raise RuntimeError(f"Missing WSODD source images: {missing[:10]}")

    group_labels, overrides, decision_payload = load_decisions(args.decisions)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        image_path = by_name[name.casefold()]
        group = group_by_name[name.casefold()]
        stats = image_statistics(image_path)
        manual = overrides.get(name, group_labels.get(group, ""))
        rows.append(
            {
                "review_index": index,
                "image_name": name,
                "image_path": str(image_path),
                "source_group": group,
                **stats,
                "heuristic_label": heuristic_label(stats),
                "manual_label": manual,
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["source_group"]].append(row)
    if len(by_group) != EXPECTED_GROUPS:
        raise RuntimeError(f"Unexpected WSODD test sequence-group count: {len(by_group)}")

    sheets_root = args.output / "contact_sheets"
    group_summaries: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_group.items(), key=lambda item: (-len(item[1]), item[0])):
        slug = slug_for_group(group)
        sample_indices = np.linspace(0, len(group_rows) - 1, min(25, len(group_rows))).round().astype(int)
        summary_rows = [group_rows[int(index)] for index in sample_indices]
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
        group_summaries.append(
            {
                "source_group": group,
                "slug": slug,
                "images": len(group_rows),
                "first_image": group_rows[0]["image_name"],
                "last_image": group_rows[-1]["image_name"],
                "heuristic_counts": dict(Counter(row["heuristic_label"] for row in group_rows)),
                "manual_counts": dict(Counter(row["manual_label"] for row in group_rows if row["manual_label"])),
                "summary_sheet": str(sheets_root / f"SUMMARY_{slug}.jpg"),
            }
        )

    fields = [
        "review_index",
        "image_name",
        "image_path",
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
            missing_labels = [row["image_name"] for row in rows if not row["manual_label"]]
            raise RuntimeError(f"Manual decisions leave {len(missing_labels)} images unlabeled; first={missing_labels[:10]}")
        write_csv(
            args.output / "scene_labels.csv",
            rows,
            ["image_name", "manual_label", "source_group", "review_index"],
        )

    report = {
        "schema": "ring3.wsodd_scene_review_manifest.v1",
        "status": "completed" if len(labeled) == len(rows) else "awaiting_manual_labels",
        "source_images": str(args.source_images),
        "test_manifest": str(args.test_manifest),
        "test_manifest_sha256": sha256(args.test_manifest),
        "test_image_count": len(rows),
        "sequence_group_count": len(by_group),
        "gap_seconds": args.gap_seconds,
        "allowed_labels": list(SCENE_LABELS),
        "manual_label_count": len(labeled),
        "manual_label_counts": dict(Counter(row["manual_label"] for row in labeled)),
        "heuristic_labels_are_review_order_only": True,
        "decision_file": str(args.decisions) if args.decisions else None,
        "decision_file_sha256": sha256(args.decisions) if args.decisions else None,
        "groups": group_summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scene_review_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
