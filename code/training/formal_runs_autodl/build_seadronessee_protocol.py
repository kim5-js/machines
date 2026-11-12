from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val")


def label_counts(path: Path, nc: int = 5) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            value = float(line.split()[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid class ID at {path}:{line_number}") from exc
        if not value.is_integer() or not 0 <= int(value) < nc:
            raise ValueError(
                f"Class ID outside 0..{nc - 1} at {path}:{line_number}: {value}"
            )
        counts[int(value)] += 1
    return counts


def group_key(image: dict[str, Any], origin: str) -> str:
    source = image.get("source")
    if not isinstance(source, dict):
        return f"singleton:{origin}:{image['file_name']}"
    drone = str(source.get("drone") or "unknown_drone")
    folder = str(source.get("folder_name") or "unknown_folder")
    video = str(source.get("video") or "")
    if video:
        return f"video:{drone}|{folder}|{video}"
    if source.get("image_name"):
        # Trinity still images expose no flight/session ID beyond the source
        # acquisition folder. Keeping the whole folder together is conservative
        # but prevents adjacent captures and repeated source names from leaking
        # between development train and validation.
        return f"still_acquisition:{drone}|{folder}"
    return f"singleton:{origin}:{image['file_name']}"


def score_assignment(
    assignment: dict[str, str],
    groups: dict[str, list[tuple[str, str]]],
    group_classes: dict[str, Counter[int]],
    ratios: dict[str, float],
    min_val_groups: int,
) -> float:
    total_images = sum(len(records) for records in groups.values())
    total_classes: Counter[int] = Counter()
    for counts in group_classes.values():
        total_classes.update(counts)
    image_counts: Counter[str] = Counter()
    class_counts = {split: Counter() for split in SPLITS}
    source_group_counts: Counter[str] = Counter()
    for group, split in assignment.items():
        source_group_counts[split] += 1
        image_counts[split] += len(groups[group])
        class_counts[split].update(group_classes[group])
    image_error = sum(
        ((image_counts[split] - ratios[split] * total_images) / max(ratios[split] * total_images, 1.0)) ** 2
        for split in SPLITS
    )
    class_error = 0.0
    missing_penalty = 0.0
    for split in SPLITS:
        for class_id, total in total_classes.items():
            target = ratios[split] * total
            actual = class_counts[split][class_id]
            class_error += ((actual - target) / max(target, 1.0)) ** 2
            if target >= 1.0 and actual == 0:
                missing_penalty += 20.0
    group_penalty = 10.0 * max(0, min_val_groups - source_group_counts["val"]) ** 2
    return 8.0 * image_error + class_error + missing_penalty + group_penalty


def assign_groups(
    groups: dict[str, list[tuple[str, str]]],
    group_classes: dict[str, Counter[int]],
    ratios: dict[str, float],
    seed: int,
    attempts: int,
    min_val_groups: int,
) -> tuple[dict[str, str], float]:
    rng = random.Random(seed)
    group_ids = sorted(groups)
    best: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(attempts):
        order = sorted(
            group_ids,
            key=lambda group: -(len(groups[group]) * (0.65 + 0.70 * rng.random())),
        )
        assignment: dict[str, str] = {}
        for group in order:
            candidates: list[tuple[float, float, str]] = []
            for split in SPLITS:
                assignment[group] = split
                candidates.append(
                    (
                        score_assignment(
                            assignment, groups, group_classes, ratios, min_val_groups
                        ),
                        rng.random(),
                        split,
                    )
                )
            assignment[group] = min(candidates)[2]
        current = score_assignment(assignment, groups, group_classes, ratios, min_val_groups)
        if current < best_score:
            best = assignment.copy()
            best_score = current
    if best is None:
        raise RuntimeError("Could not assign video groups")

    for _ in range(max(5000, attempts * 10)):
        candidate = best.copy()
        if rng.random() < 0.45:
            group = rng.choice(group_ids)
            candidate[group] = "val" if candidate[group] == "train" else "train"
        else:
            left, right = rng.sample(group_ids, 2)
            if candidate[left] == candidate[right]:
                continue
            candidate[left], candidate[right] = candidate[right], candidate[left]
        current = score_assignment(candidate, groups, group_classes, ratios, min_val_groups)
        if current < best_score:
            best = candidate
            best_score = current
    return best, best_score


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-video-disjoint SeaDronesSee v2 development splits")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=3000)
    parser.add_argument("--min-val-groups", type=int, default=5)
    args = parser.parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio}
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise SystemExit(f"Ratios must be positive and sum to 1.0: {ratios}")

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    group_classes: dict[str, Counter[int]] = defaultdict(Counter)
    group_origins: dict[str, set[str]] = defaultdict(set)
    filenames: set[str] = set()
    for origin in SPLITS:
        annotation_path = args.dataset_root / "annotations" / f"instances_{origin}.json"
        coco = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        for image in coco["images"]:
            filename = str(image["file_name"])
            if Path(filename).name != filename:
                raise ValueError(f"Expected a flat filename, got {filename}")
            if filename.casefold() in filenames:
                raise ValueError(f"Duplicate filename across labeled folders: {filename}")
            filenames.add(filename.casefold())
            image_path = args.dataset_root / "images" / origin / filename
            label_path = args.dataset_root / "labels" / origin / f"{Path(filename).stem}.txt"
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing image/label pair: {image_path}, {label_path}")
            group = group_key(image, origin)
            groups[group].append((origin, filename))
            group_origins[group].add(origin)
            group_classes[group].update(label_counts(label_path))

    if args.min_val_groups < 1:
        raise SystemExit("--min-val-groups must be at least 1")
    assignment, optimization_score = assign_groups(
        groups, group_classes, ratios, args.seed, args.attempts, args.min_val_groups
    )
    split_records = {
        split: sorted(
            (
                record
                for group, records in groups.items()
                if assignment[group] == split
                for record in records
            ),
            key=lambda record: (record[1].casefold(), record[0]),
        )
        for split in SPLITS
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    for split, records in split_records.items():
        path = args.output / f"{split}.tsv"
        path.write_text("".join(f"{origin}\t{filename}\n" for origin, filename in records), encoding="utf-8")
        manifests[split] = path
    trainval_path = args.output / "trainval.tsv"
    trainval_records = sorted(
        (record for records in split_records.values() for record in records),
        key=lambda record: (record[1].casefold(), record[0]),
    )
    trainval_path.write_text("".join(f"{origin}\t{filename}\n" for origin, filename in trainval_records), encoding="utf-8")
    manifests["trainval"] = trainval_path

    split_groups = {
        split: {group for group, assigned in assignment.items() if assigned == split}
        for split in SPLITS
    }
    split_classes = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        split_classes[split].update(group_classes[group])
    report = {
        "protocol": "seadronessee_source_disjoint_v1",
        "purpose": "development split only; final model retrains on trainval then uses the official unlabeled test server",
        "grouping_rule": {
            "video_frames": "exact source drone|folder_name|video",
            "still_images": "whole source drone|folder_name acquisition because no flight/session ID is available",
            "fallback": "singleton only when source metadata has neither video nor image_name",
        },
        "seed": args.seed,
        "ratios": ratios,
        "minimum_validation_source_groups": args.min_val_groups,
        "optimization_score": optimization_score,
        "total_labeled_images": sum(len(records) for records in groups.values()),
        "source_groups": len(groups),
        "video_groups": sum(group.startswith("video:") for group in groups),
        "still_acquisition_groups": sum(group.startswith("still_acquisition:") for group in groups),
        "singleton_groups": sum(group.startswith("singleton:") for group in groups),
        "official_train_val_mixed_source_groups": sum(1 for origins in group_origins.values() if len(origins) > 1),
        "split_images": {split: len(records) for split, records in split_records.items()},
        "split_source_groups": {split: len(values) for split, values in split_groups.items()},
        "split_class_instances": {
            split: {str(key): counts[key] for key in sorted(counts)}
            for split, counts in split_classes.items()
        },
        "source_group_overlap": len(split_groups["train"] & split_groups["val"]),
        "largest_source_groups": [
            {"group": group, "images": len(records), "assigned": assignment[group]}
            for group, records in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:20]
        ],
        "manifests": {
            split: {"path": path.name, "sha256": sha256(path), "lines": len(path.read_text(encoding="utf-8").splitlines())}
            for split, path in manifests.items()
        },
    }
    protocol_path = args.output / "protocol.json"
    protocol_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
