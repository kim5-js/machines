from __future__ import annotations

import argparse
import json
from pathlib import Path

from materialize_yolo_split import place_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a SeaDronesSee origin/filename TSV using links or copies")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    args = parser.parse_args()

    records: list[tuple[str, str]] = []
    for line_number, raw in enumerate(args.manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) != 2 or fields[0] not in {"train", "val"}:
            raise ValueError(f"Invalid manifest record at {args.manifest}:{line_number}: {raw!r}")
        records.append((fields[0], fields[1]))
    names = [filename.casefold() for _, filename in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate destination filenames in manifest")

    image_output = args.output / "images"
    label_output = args.output / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    counts = {"hardlink": 0, "symlink": 0, "copy": 0, "existing": 0}
    missing: list[str] = []
    for origin, filename in records:
        image = args.dataset_root / "images" / origin / filename
        label = args.dataset_root / "labels" / origin / f"{Path(filename).stem}.txt"
        if not image.is_file() or not label.is_file():
            missing.append(f"{origin}\t{filename}")
            continue
        counts[place_file(image, image_output / image.name, args.mode)] += 1
        counts[place_file(label, label_output / label.name, args.mode)] += 1
    report = {
        "manifest": str(args.manifest),
        "requested_images": len(records),
        "missing_pairs": missing,
        "placed_files": counts,
        "output": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(f"Missing {len(missing)} pairs")


if __name__ == "__main__":
    main()
