#!/usr/bin/env python3
"""Materialize the completed WSODD test-scene review decisions.

The image-statistics heuristic is review metadata only.  All labels below are
group decisions or explicit per-image overrides fixed by visual review of the
32 sequence summaries and all 115 full 5x5 contact-sheet pages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_IMAGES = 2406
EXPECTED_GROUPS = 32
SEQUENCE_0048 = "ip:192.168.0.64_01_:sequence:0048"
SEQUENCE_0051 = "ip:192.168.0.64_01_:sequence:0051"
PLAIN_0002 = "plain_timestamp:sequence:0002"
PLAIN_0003 = "plain_timestamp:sequence:0003"
WIN_0004 = "win:sequence:0004"

MIXED_IMAGES = {
    "192.168.0.64_01_20200729193250471.jpg",
    "192.168.0.64_01_20200729193254703.jpg",
}

GLARE_IMAGES_0051 = {
    "192.168.0.64_01_20200807073759505.jpg",
    "192.168.0.64_01_202008070738087.jpg",
    "192.168.0.64_01_20200807073819426.jpg",
    "192.168.0.64_01_20200807073826321.jpg",
    "192.168.0.64_01_20200807073839581.jpg",
    "192.168.0.64_01_20200807073855243.jpg",
    "192.168.0.64_01_20200807073914198.jpg",
    "192.168.0.64_01_20200807073925102.jpg",
    "192.168.0.64_01_20200807073930656.jpg",
    "192.168.0.64_01_20200807073947940.jpg",
    "192.168.0.64_01_20200807073951497.jpg",
    "192.168.0.64_01_20200807073955506.jpg",
    "192.168.0.64_01_20200807074003743.jpg",
    "192.168.0.64_01_20200807074010607.jpg",
    "192.168.0.64_01_20200807074025614.jpg",
}

GLARE_IMAGES_PLAIN_0002 = {
    "20201103110118.jpg",
    "20201103110126.jpg",
    "20201103110140.jpg",
    "20201103110147.jpg",
    "20201103110153.jpg",
    "20201103110159.jpg",
    "20201103110204.jpg",
    "20201103110210.jpg",
    "20201103110216.jpg",
}

GLARE_IMAGES_WIN_0004 = {
    "WIN_20200724_18_01_30_Pro.jpg",
    "WIN_20200724_18_02_00_Pro.jpg",
    "WIN_20200724_18_02_02_Pro.jpg",
    "WIN_20200724_18_02_21_Pro.jpg",
    "WIN_20200724_18_03_18_Pro.jpg",
    "WIN_20200724_18_05_28_Pro.jpg",
    "WIN_20200724_18_05_39_Pro.jpg",
    "WIN_20200724_18_05_56_Pro.jpg",
    "WIN_20200724_18_05_58_Pro.jpg",
    "WIN_20200724_18_06_11_Pro.jpg",
    "WIN_20200724_18_06_22_Pro.jpg",
    "WIN_20200724_18_08_12_Pro.jpg",
    "WIN_20200724_18_09_06_Pro.jpg",
    "WIN_20200724_18_09_07_Pro.jpg",
    "WIN_20200724_18_09_14_Pro.jpg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited manual WSODD scene decisions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} locked test rows, found {len(rows)}")
    image_names = {row["image_name"] for row in rows}
    if len(image_names) != EXPECTED_IMAGES:
        raise RuntimeError("Duplicate image names in WSODD review manifest")
    groups = sorted({row["source_group"] for row in rows})
    if len(groups) != EXPECTED_GROUPS:
        raise RuntimeError(f"Expected {EXPECTED_GROUPS} sequence groups, found {len(groups)}")
    required_groups = {SEQUENCE_0048, SEQUENCE_0051, PLAIN_0002, PLAIN_0003, WIN_0004}
    if not required_groups.issubset(groups):
        raise RuntimeError(f"Required reviewed groups missing: {sorted(required_groups - set(groups))}")

    group_labels = {group: "Normal" for group in groups}
    group_labels[SEQUENCE_0048] = "Low-light"
    group_labels[PLAIN_0003] = "Low-light"

    overrides: dict[str, str] = {}
    for name in MIXED_IMAGES:
        overrides[name] = "Mixed"
    for name in GLARE_IMAGES_0051 | GLARE_IMAGES_PLAIN_0002 | GLARE_IMAGES_WIN_0004:
        overrides[name] = "Glare"
    unknown = sorted(set(overrides) - image_names)
    if unknown:
        raise RuntimeError(f"Reviewed overrides are absent from the locked manifest: {unknown}")

    final_labels = {
        row["image_name"]: overrides.get(row["image_name"], group_labels[row["source_group"]])
        for row in rows
    }
    counts = Counter(final_labels.values())
    expected = {"Normal": 2344, "Glare": 39, "Low-light": 21, "Mixed": 2}
    if dict(counts) != expected:
        raise RuntimeError(f"Manual label count drift: {dict(counts)} != {expected}")

    payload = {
        "schema": "ring3.robustness_scene_label_decisions.v1",
        "status": "completed_manual_review",
        "review_protocol": "ROBUSTNESS_SCENE_LABELING_PROTOCOL.md",
        "review_scope": {
            "locked_test_images": EXPECTED_IMAGES,
            "source_summaries_reviewed": EXPECTED_GROUPS,
            "full_contact_sheet_pages_reviewed": 115,
            "contact_sheet_layout": "5x5",
            "labels": ["Normal", "Glare", "Low-light", "Mixed"],
        },
        "review_basis": [
            "Normal: usable exposure without a dominant water-surface or lens glare patch.",
            "Glare: usable exposure with a pronounced specular patch, reflection band, or direct sun glare.",
            "Low-light: broadly depressed twilight or night exposure without dominant glare.",
            "Mixed: broadly depressed exposure together with pronounced reflected artificial light.",
            "Heuristic candidates were ordering aids only; final labels are manual visual decisions.",
        ],
        "manual_boundaries": {
            "sequence_0048": "Low-light group default; two night waterfront frames with dominant light reflections are Mixed.",
            "sequence_0051": "First fifteen backlit river frames contain a broad silver specular field and are Glare; later frames are Normal.",
            "plain_sequence_0002": "Nine frames with direct sun and a vertical water reflection band are Glare; first and final frames are Normal.",
            "win_sequence_0004": "Fifteen explicitly listed panoramas with direct sun, lens flare, or saturated water reflection are Glare.",
            "plain_sequence_0003": "Single globally dark overhead frame is Low-light.",
        },
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "label_counts": expected,
        "group_labels": group_labels,
        "image_overrides": dict(sorted(overrides.items(), key=lambda item: item[0].casefold())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output), "counts": expected}))


if __name__ == "__main__":
    main()
