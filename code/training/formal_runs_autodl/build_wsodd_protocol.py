from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from audit_wsodd_split import IMAGE_SUFFIXES, build_groups


SPLITS = ("train", "val", "test")


def read_class_counts(label: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_value = float(fields[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid class ID at {label}:{line_number}") from exc
        if not class_value.is_integer() or class_value < 0:
            raise ValueError(f"Invalid class ID at {label}:{line_number}: {class_value}")
        counts[int(class_value)] += 1
    return counts


def split_score(
    assignment: dict[str, str],
    groups: dict[str, list[Path]],
    group_classes: dict[str, Counter[int]],
    image_targets: dict[str, float],
    class_targets: dict[str, dict[int, float]],
    class_ids: list[int],
) -> float:
    image_counts: Counter[str] = Counter()
    for group, split in assignment.items():
        image_counts[split] += len(groups[group])
    class_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    for group, split in assignment.items():
        class_counts[split].update(group_classes[group])

    image_error = sum(
        ((image_counts[split] - image_targets[split]) / max(image_targets[split], 1.0)) ** 2
        for split in SPLITS
    )
    class_error = 0.0
    missing_penalty = 0.0
    for split in SPLITS:
        for class_id in class_ids:
            target = class_targets[split][class_id]
            actual = class_counts[split][class_id]
            class_error += ((actual - target) / max(target, 1.0)) ** 2
            if target >= 1.0 and actual == 0:
                missing_penalty += 10.0
    # Treat each class as a first-class constraint. Dividing this term by the
    # class count lets rare categories collapse into one split even when a
    # better whole-sequence allocation exists.
    return 4.0 * image_error + class_error + missing_penalty


def build_assignment(
    groups: dict[str, list[Path]],
    group_classes: dict[str, Counter[int]],
    ratios: dict[str, float],
    seed: int,
    attempts: int,
) -> tuple[dict[str, str], float]:
    total_images = sum(len(images) for images in groups.values())
    total_classes: Counter[int] = Counter()
    for counts in group_classes.values():
        total_classes.update(counts)
    class_ids = sorted(total_classes)
    image_targets = {split: ratios[split] * total_images for split in SPLITS}
    class_targets = {
        split: {class_id: ratios[split] * total_classes[class_id] for class_id in class_ids}
        for split in SPLITS
    }

    rng = random.Random(seed)
    group_ids = sorted(groups)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(attempts):
        # Large temporal sequences are placed first. Random jitter explores
        # alternative deterministic allocations around that constraint.
        order = sorted(
            group_ids,
            key=lambda group: -(len(groups[group]) * (0.70 + 0.60 * rng.random())),
        )
        assignment: dict[str, str] = {}
        for group in order:
            candidates: list[tuple[float, float, str]] = []
            for split in SPLITS:
                assignment[group] = split
                candidates.append(
                    (
                        split_score(
                            assignment,
                            groups,
                            group_classes,
                            image_targets,
                            class_targets,
                            class_ids,
                        ),
                        rng.random(),
                        split,
                    )
                )
            _, _, chosen = min(candidates)
            assignment[group] = chosen
        score = split_score(
            assignment, groups, group_classes, image_targets, class_targets, class_ids
        )
        if score < best_score:
            best_assignment = assignment.copy()
            best_score = score

    if best_assignment is None:
        raise RuntimeError("Failed to construct a split assignment")

    # Deterministic local search: moving or swapping whole sequences can repair
    # rare-class imbalance without ever leaking a sequence across splits.
    for _ in range(max(5000, attempts * 20)):
        candidate = best_assignment.copy()
        left = rng.choice(group_ids)
        if rng.random() < 0.45:
            choices = [split for split in SPLITS if split != candidate[left]]
            candidate[left] = rng.choice(choices)
        else:
            right = rng.choice(group_ids)
            if candidate[left] == candidate[right]:
                continue
            candidate[left], candidate[right] = candidate[right], candidate[left]
        score = split_score(candidate, groups, group_classes, image_targets, class_targets, class_ids)
        if score < best_score:
            best_assignment = candidate
            best_score = score
    return best_assignment, best_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sequence-disjoint WSODD train/val/test protocol")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.56)
    parser.add_argument("--val-ratio", type=float, default=0.14)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--gap-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=600)
    args = parser.parse_args()

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise SystemExit(f"Ratios must be positive and sum to 1.0: {ratios}")

    images = sorted(
        (path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise SystemExit(f"No images found in {args.images}")
    group_by_name = build_groups(images, args.gap_seconds)
    groups: dict[str, list[Path]] = defaultdict(list)
    group_classes: dict[str, Counter[int]] = defaultdict(Counter)
    missing_labels: list[str] = []
    for image in images:
        group = group_by_name[image.name.casefold()]
        label = args.labels / f"{image.stem}.txt"
        if not label.is_file():
            missing_labels.append(image.name)
            continue
        groups[group].append(image)
        group_classes[group].update(read_class_counts(label))
    if missing_labels:
        raise SystemExit(f"Missing labels for {len(missing_labels)} images: {missing_labels[:20]}")

    assignment, score = build_assignment(
        groups, group_classes, ratios, args.seed, args.attempts
    )
    args.output.mkdir(parents=True, exist_ok=True)
    split_images = {
        split: sorted(
            (image for group, images_in_group in groups.items() if assignment[group] == split for image in images_in_group),
            key=lambda path: path.name.casefold(),
        )
        for split in SPLITS
    }
    manifest_paths: dict[str, Path] = {}
    for split, split_paths in split_images.items():
        path = args.output / f"{split}.txt"
        path.write_text("".join(f"{image.name}\n" for image in split_paths), encoding="utf-8")
        manifest_paths[split] = path
    trainval_path = args.output / "trainval.txt"
    trainval_names = sorted(
        [image.name for split in ("train", "val") for image in split_images[split]],
        key=str.casefold,
    )
    trainval_path.write_text("".join(f"{name}\n" for name in trainval_names), encoding="utf-8")
    manifest_paths["trainval"] = trainval_path

    split_class_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    split_groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for group, split in assignment.items():
        split_groups[split].add(group)
        split_class_counts[split].update(group_classes[group])
    total_class_counts: Counter[int] = Counter()
    for counts in group_classes.values():
        total_class_counts.update(counts)
    class_group_support = {
        str(class_id): {
            "instances": total_class_counts[class_id],
            "groups_with_class": sum(
                1 for counts in group_classes.values() if counts[class_id] > 0
            ),
            "largest_groups": [
                {"group": group, "instances": count}
                for group, count in sorted(
                    (
                        (group, counts[class_id])
                        for group, counts in group_classes.items()
                        if counts[class_id] > 0
                    ),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
            ],
        }
        for class_id in sorted(total_class_counts)
    }
    protocol = {
        "protocol": "wsodd_sequence_disjoint_v1",
        "rationale": (
            "56/14/30 development split preserves the source paper's 70/30 "
            "train-test ratio while reserving 20% of the 70% development pool for validation"
        ),
        "seed": args.seed,
        "gap_seconds": args.gap_seconds,
        "ratios": ratios,
        "optimization_score": score,
        "total_images": len(images),
        "sequence_groups": len(groups),
        "split_images": {split: len(paths) for split, paths in split_images.items()},
        "split_groups": {split: len(group_ids) for split, group_ids in split_groups.items()},
        "split_class_instances": {
            split: {str(key): counts[key] for key in sorted(counts)}
            for split, counts in split_class_counts.items()
        },
        "class_group_support": class_group_support,
        "group_overlap": {
            f"{left}-{right}": len(split_groups[left] & split_groups[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        },
        "manifests": {
            split: {
                "path": path.name,
                "sha256": sha256(path),
                "lines": len(path.read_text(encoding="utf-8").splitlines()),
            }
            for split, path in manifest_paths.items()
        },
    }
    protocol_path = args.output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
