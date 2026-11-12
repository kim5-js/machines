#!/usr/bin/env python3
"""Build compact all-page atlases for manual WSODD scene review.

This utility never assigns scene labels.  It only tiles the already generated
5x5 review pages so a reviewer can scan every test frame in each sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile WSODD review pages by sequence")
    parser.add_argument("--contact-sheets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--page-width", type=int, default=800)
    args = parser.parse_args()

    if args.columns <= 0 or args.page_width <= 0:
        raise ValueError("columns and page-width must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for sequence_dir in sorted(path for path in args.contact_sheets.iterdir() if path.is_dir()):
        pages = sorted(sequence_dir.glob("page_*.jpg"))
        if not pages:
            continue
        rendered: list[Image.Image] = []
        for path in pages:
            with Image.open(path) as source:
                source = source.convert("RGB")
                height = max(1, round(source.height * args.page_width / source.width))
                page = ImageOps.contain(source, (args.page_width, height), Image.Resampling.LANCZOS)
                rendered.append(page.copy())
        cell_height = max(page.height for page in rendered)
        rows = (len(rendered) + args.columns - 1) // args.columns
        atlas = Image.new("RGB", (args.columns * args.page_width, rows * cell_height), "white")
        for index, page in enumerate(rendered):
            x = (index % args.columns) * args.page_width
            y = (index // args.columns) * cell_height
            atlas.paste(page, (x, y))
        output = args.output / f"ATLAS_{sequence_dir.name}.jpg"
        atlas.save(output, quality=90, optimize=True)
        records.append(
            {
                "sequence": sequence_dir.name,
                "pages": len(pages),
                "atlas": str(output),
                "width": atlas.width,
                "height": atlas.height,
            }
        )

    report = {
        "schema": "ring3.wsodd_scene_review_atlases.v1",
        "status": "passed",
        "sequences": len(records),
        "source_pages": sum(int(record["pages"]) for record in records),
        "records": records,
    }
    (args.output / "atlas_index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "sequences", "source_pages")}))


if __name__ == "__main__":
    main()
