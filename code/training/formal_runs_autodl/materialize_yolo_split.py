from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def place_file(source: Path, target: Path, mode: str) -> str:
    if target.exists():
        if source.stat().st_size != target.stat().st_size:
            raise RuntimeError(f"Existing target differs in size: {target}")
        return "existing"
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)
    return mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("hardlink", "symlink", "copy"), default="hardlink"
    )
    args = parser.parse_args()

    names = [
        line.strip().lstrip("\ufeff")
        for line in args.manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise SystemExit(f"Duplicate filenames in manifest: {args.manifest}")

    image_output = args.output / "images"
    label_output = args.output / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    counts = {"hardlink": 0, "symlink": 0, "copy": 0, "existing": 0}
    missing: list[str] = []
    for name in names:
        image = args.images / name
        label = args.labels / f"{Path(name).stem}.txt"
        if not image.is_file() or not label.is_file():
            missing.append(name)
            continue
        counts[place_file(image, image_output / image.name, args.mode)] += 1
        counts[place_file(label, label_output / label.name, args.mode)] += 1

    report = {
        "manifest": str(args.manifest),
        "requested_images": len(names),
        "missing_pairs": missing,
        "placed_files": counts,
        "output": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(f"Missing {len(missing)} image/label pairs")


if __name__ == "__main__":
    main()

