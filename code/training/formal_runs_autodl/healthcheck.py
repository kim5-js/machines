from __future__ import annotations

import json
import platform
import sys

import torch
import torchvision
import ultralytics


def main() -> None:
    cuda_available = torch.cuda.is_available()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda_available else None
        ),
        "ultralytics": ultralytics.__version__,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not cuda_available:
        raise SystemExit("CUDA is not available in the MDC environment")


if __name__ == "__main__":
    main()

