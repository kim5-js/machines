#!/usr/bin/env python3
"""Materialize the completed manual scene review as an auditable decision file.

The exposure and reflection boundaries below were fixed only after reviewing all
10 source summaries and all 66 full 5x5 contact-sheet pages.  The precomputed
heuristic label is used solely to reproduce the manually approved exposure
split inside the heterogeneous Trinity source; it is not used for the other
sources or as an unattended scene classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


TRINITY = "still_acquisition:trinity|multispectral"
DJI_0062 = "video:mavic|DJI_0062|DJI_0062.MP4"
DJI_0113 = "video:mavic|NorthSea_Fri_DJI_0113|NorthSea_Fri_DJI_0113.MP4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_stem(name: str) -> int:
    return int(Path(name).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited manual Normal/Glare/Low-light/Mixed labels")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1555:
        raise RuntimeError(f"Expected 1555 locked validation rows, found {len(rows)}")
    if len({row["image_name"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate image names in review manifest")

    groups = sorted({row["source_group"] for row in rows})
    if len(groups) != 10 or TRINITY not in groups or DJI_0062 not in groups or DJI_0113 not in groups:
        raise RuntimeError(f"Unexpected source groups: {groups}")

    # Stable video acquisitions are Normal by default.  Trinity is the only
    # heterogeneous still-acquisition group and is Low-light by default.
    group_labels = {group: "Normal" for group in groups}
    group_labels[TRINITY] = "Low-light"
    overrides: dict[str, str] = {}

    for row in rows:
        name = row["image_name"]
        group = row["source_group"]
        number = numeric_stem(name)
        if group == TRINITY:
            if row["heuristic_label"] == "Normal":
                # Manually approved after full-page visual review.
                overrides[name] = "Normal"
            elif 434 <= number <= 617 or 744 <= number <= 820:
                # Coherent dark sequences with pronounced water-surface glint.
                overrides[name] = "Mixed"
        elif group == DJI_0113 and 5682 <= number <= 6010:
            # Broad specular band is visible through 6010; 6011 onward is Normal.
            overrides[name] = "Glare"
        elif group == DJI_0062 and 5236 <= number <= 5239:
            # Four terminal frames contain a clear sun-reflection patch.
            overrides[name] = "Glare"

    final_labels = {
        row["image_name"]: overrides.get(row["image_name"], group_labels[row["source_group"]]) for row in rows
    }
    counts = Counter(final_labels.values())
    expected = {"Normal": 841, "Glare": 333, "Low-light": 224, "Mixed": 157}
    if dict(counts) != expected:
        raise RuntimeError(f"Manual label count drift: {dict(counts)} != {expected}")

    payload = {
        "schema": "ring3.robustness_scene_label_decisions.v1",
        "status": "completed_manual_review",
        "review_protocol": "ROBUSTNESS_SCENE_LABELING_PROTOCOL.md",
        "review_scope": {
            "locked_validation_images": 1555,
            "source_summaries_reviewed": 10,
            "full_contact_sheet_pages_reviewed": 66,
            "contact_sheet_layout": "5x5",
            "labels": ["Normal", "Glare", "Low-light", "Mixed"],
        },
        "review_basis": [
            "Normal: usable exposure without a dominant water-surface specular patch.",
            "Glare: usable exposure with a pronounced water-surface reflection patch or band.",
            "Low-light: broadly depressed exposure without a pronounced specular patch.",
            "Mixed: broadly depressed exposure together with pronounced water-surface glint.",
            "Heuristic candidates were review aids only; every full contact-sheet page was visually reviewed.",
        ],
        "manual_boundaries": {
            "trinity_exposure": "Full-sheet review approved the manifest exposure candidates; Normal candidates override the Low-light group default.",
            "trinity_mixed_sequences": ["434-617", "744-820"],
            "northsea_dji_0113_glare_sequence": "5682-6010 inclusive; 6011 onward Normal",
            "dji_0062_glare_frames": ["5236.jpg", "5237.jpg", "5238.jpg", "5239.jpg"],
        },
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "label_counts": expected,
        "group_labels": group_labels,
        "image_overrides": dict(sorted(overrides.items(), key=lambda item: item[0].casefold())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output), "counts": expected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
