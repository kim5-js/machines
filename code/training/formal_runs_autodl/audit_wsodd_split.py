from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


IP_TIMESTAMP = re.compile(r"^(?P<source>.+_)(?P<stamp>\d{14})(?P<fraction>\d{1,3})$")
WIN_TIMESTAMP = re.compile(r"^(?P<source>WIN_)(?P<date>\d{8})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_Pro$")
PLAIN_TIMESTAMP = re.compile(r"^(?P<stamp>\d{14})$")
PLAIN_NUMBER = re.compile(r"^\d+$")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_timestamp(path: Path) -> tuple[str, datetime] | None:
    stem = path.stem
    match = IP_TIMESTAMP.match(stem)
    if match:
        stamp = datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")
        fraction = match.group("fraction").ljust(3, "0")[:3]
        return f"ip:{match.group('source')}", stamp.replace(microsecond=int(fraction) * 1000)
    match = WIN_TIMESTAMP.match(stem)
    if match:
        raw = (
            match.group("date")
            + match.group("hour")
            + match.group("minute")
            + match.group("second")
        )
        return "win", datetime.strptime(raw, "%Y%m%d%H%M%S")
    match = PLAIN_TIMESTAMP.match(stem)
    if match:
        return "plain_timestamp", datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")
    return None


def build_groups(images: list[Path], gap_seconds: float) -> dict[str, str]:
    timelines: dict[str, list[tuple[datetime, Path]]] = defaultdict(list)
    group_by_name: dict[str, str] = {}
    numeric_misc: list[Path] = []
    for image in images:
        parsed = parse_timestamp(image)
        if parsed is not None:
            source, timestamp = parsed
            timelines[source].append((timestamp, image))
        elif PLAIN_NUMBER.match(image.stem):
            numeric_misc.append(image)
        else:
            group_by_name[image.name.casefold()] = f"singleton:{image.name.casefold()}"

    for source, entries in timelines.items():
        entries.sort(key=lambda item: (item[0], item[1].name.casefold()))
        group_index = 0
        previous: datetime | None = None
        for timestamp, image in entries:
            if previous is not None and (timestamp - previous).total_seconds() > gap_seconds:
                group_index += 1
            group_by_name[image.name.casefold()] = f"{source}:sequence:{group_index:04d}"
            previous = timestamp

    # The small integer filenames form one short burst in this copy. Keeping
    # them together is safer than randomly assigning near-adjacent frames.
    for image in numeric_misc:
        group_by_name[image.name.casefold()] = "plain_numeric:sequence:0000"
    return group_by_name


def read_manifest(path: Path) -> list[str]:
    return [
        Path(line.strip()).name.casefold()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect temporal sequence leakage in a WSODD split")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--gap-seconds", type=float, default=60.0)
    args = parser.parse_args()

    images = sorted(
        (path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    group_by_name = build_groups(images, args.gap_seconds)
    manifests = {"train": args.train, "val": args.val}
    if args.test is not None:
        manifests["test"] = args.test
    split_names = {split: read_manifest(path) for split, path in manifests.items()}

    group_splits: dict[str, set[str]] = defaultdict(set)
    group_images: Counter[str] = Counter()
    missing_names: dict[str, list[str]] = defaultdict(list)
    for split, names in split_names.items():
        for name in names:
            group = group_by_name.get(name)
            if group is None:
                missing_names[split].append(name)
                continue
            group_splits[group].add(split)
            group_images[group] += 1

    contaminated = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    group_sizes = Counter(group_by_name.values())
    report = {
        "gap_seconds": args.gap_seconds,
        "image_count": len(images),
        "sequence_group_count": len(group_sizes),
        "largest_groups": [
            {"group": group, "images": count}
            for group, count in group_sizes.most_common(20)
        ],
        "split_counts": {split: len(names) for split, names in split_names.items()},
        "missing_manifest_names": {split: names[:20] for split, names in missing_names.items()},
        "contaminated_group_count": len(contaminated),
        "images_in_contaminated_groups": sum(group_images[group] for group in contaminated),
        "contaminated_examples": [
            {
                "group": group,
                "splits": sorted(splits),
                "group_images": group_sizes[group],
            }
            for group, splits in sorted(contaminated.items())[:30]
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(2 if contaminated or missing_names else 0)


if __name__ == "__main__":
    main()
