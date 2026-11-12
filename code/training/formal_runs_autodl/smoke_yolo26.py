from __future__ import annotations

import json
import os
from pathlib import Path

# Configure all writable caches before importing Torch or Ultralytics.  AutoDL's
# system disk is small, while /root/autodl-tmp is the persistent data disk.
DATA_ROOT = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
os.environ.setdefault("PIP_CACHE_DIR", str(DATA_ROOT / ".cache" / "pip"))
os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_ROOT / ".config" / "Ultralytics"))

import torch
from ultralytics import YOLO


def main() -> None:
    data_root = DATA_ROOT
    smoke_root = data_root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    os.chdir(smoke_root)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    model = YOLO("yolo26n.pt")
    result = model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=320,
        batch=2,
        device=0,
        workers=0,
        cache=False,
        project=str(smoke_root),
        name="yolo26n_train",
        exist_ok=True,
        verbose=False,
    )
    summary = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "save_dir": str(result.save_dir),
        "results_csv_exists": (Path(result.save_dir) / "results.csv").is_file(),
        "best_weight_exists": (Path(result.save_dir) / "weights" / "best.pt").is_file(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
