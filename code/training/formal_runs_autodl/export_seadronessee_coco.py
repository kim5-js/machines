from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_ROOT / ".config" / "Ultralytics"))

import ultralytics
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SeaDronesSee v2 test predictions in official COCO format")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--images",
        type=Path,
        default=DATA_ROOT / "datasets" / "SeaDronesSee" / "images" / "test",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    test_images = sorted(
        (path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: int(path.stem),
    )
    if len(test_images) != 3750:
        raise SystemExit(f"Expected 3750 official test images, found {len(test_images)} in {args.images}")
    if len({int(path.stem) for path in test_images}) != len(test_images):
        raise SystemExit("Test image IDs are not unique numeric filename stems")

    model = YOLO(str(args.weights.resolve()))
    predictions: list[dict[str, object]] = []
    seen_images: set[int] = set()
    results = model.predict(
        source=[str(path) for path in test_images],
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=0,
        save=False,
        verbose=False,
    )
    for result in results:
        image_id = int(Path(result.path).stem)
        seen_images.add(image_id)
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        for coordinates, class_value, score in zip(xyxy, class_ids, scores, strict=True):
            class_id = int(class_value)
            if class_id < 0 or class_id > 4:
                raise RuntimeError(f"Predicted YOLO class outside 0..4: {class_id}")
            x1, y1, x2, y2 = coordinates
            predictions.append(
                {
                    "image_id": image_id,
                    # Official COCO categories are 1..5. Category 0 is an
                    # ignore region and is never a trainable YOLO class.
                    "category_id": class_id + 1,
                    "score": float(score),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                }
            )
    expected_ids = {int(path.stem) for path in test_images}
    if seen_images != expected_ids:
        missing = sorted(expected_ids - seen_images)
        extra = sorted(seen_images - expected_ids)
        raise RuntimeError(f"Prediction stream image mismatch; missing={missing[:20]}, extra={extra[:20]}")

    predictions.sort(key=lambda item: (int(item["image_id"]), -float(item["score"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")
    metadata = {
        "purpose": "SeaDronesSee Object Detection v2 official test submission",
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "ultralytics": ultralytics.__version__,
        "images": str(args.images.resolve()),
        "image_count": len(test_images),
        "prediction_count": len(predictions),
        "category_mapping": {
            "0": "ignored (not predicted)",
            "1": "swimmer",
            "2": "boat",
            "3": "jetski",
            "4": "life_saving_appliances",
            "5": "buoy",
        },
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
