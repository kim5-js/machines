from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
os.environ.setdefault("PIP_CACHE_DIR", str(DATA_ROOT / ".cache" / "pip"))
os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_ROOT / ".config" / "Ultralytics"))

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Short, non-scientific AutoDL runtime calibration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {args.data}")

    output_root = DATA_ROOT / "calibration"
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Materialize the CUDA context before resetting peak counters. Some AutoDL
    # driver/runtime combinations reject a reset on a merely enumerated device.
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = YOLO(args.model).train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        cache=False,
        project=str(output_root),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        seed=42,
        deterministic=True,
        plots=False,
        save=False,
        verbose=False,
    )
    wall_seconds = time.perf_counter() - started
    summary = {
        "purpose": "runtime calibration only; not a scientific result",
        "model": args.model,
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "gpu": torch.cuda.get_device_name(device),
        "wall_seconds": round(wall_seconds, 3),
        "seconds_per_epoch_including_validation": round(wall_seconds / args.epochs, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "save_dir": str(result.save_dir),
    }
    summary_path = Path(result.save_dir) / "runtime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
