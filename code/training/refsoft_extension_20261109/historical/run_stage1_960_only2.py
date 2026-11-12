"""Stage1 (960) runner for cbam_eiou_light_glare_20260506.

This is the recommended training entrypoint for this environment.

Goals:
- Run from any cwd (script will chdir into its folder)
- Route B only (YAML builds model with C2f_CBAM)
- No pretrain (pretrained=False)
- Training args fixed to the train_matrix.py FloW v8s stage1 960 protocol
- Keep Focal-EIoU enabled via bbox_loss=focal_eiou
- Add train-only lightweight low-light/reflection augmentation by default
- Keep tiled inference as an optional post-train validation step
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
from datetime import datetime
from pathlib import Path


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _assert_running_in_env_venv(env_dir: Path) -> None:
    """Fail fast if script is not executed by this environment's venv python."""
    import sys

    expected_candidates = [
        (env_dir / "yolo_cbam_eiou_light_glare_20260506" / "Scripts" / "python.exe"),
        (env_dir / "venv" / "Scripts" / "python.exe"),
    ]
    expected: Path | None = None
    for candidate in expected_candidates:
        if candidate.exists():
            expected = candidate.resolve()
            break

    actual = Path(sys.executable).resolve()
    if expected is None:
        print(
            "[warn] Strict venv check: expected python.exe not found under this env folder; "
            "skip enforcement. (Looked for yolo_cbam_eiou_light_glare_20260506/Scripts/python.exe and venv/Scripts/python.exe)"
        )
        return

    if actual != expected:
        raise RuntimeError(
            "Wrong Python interpreter detected.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n\n"
            "Fix: run with this environment's python. Recommended: activate_venv.ps1,"
            " or execute .\\yolo_cbam_eiou_light_glare_20260506\\Scripts\\python.exe explicitly."
        )


def _load_light_glare_builder(env_dir: Path):
    module_path = env_dir / "light_glare_dataset.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Augmentation builder not found: {module_path}")

    spec = importlib.util.spec_from_file_location("low_light_reflection_dataset", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load augmentation builder from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_key_from_path(data_config: Path) -> str:
    path_text = str(data_config).lower()
    if "flow_img" in path_text:
        return "FloW_IMG"
    if "seadronessee" in path_text:
        return "SeaDronesSee_v2"
    if "wsodd" in path_text:
        return "WSODD_USV"
    fallback = data_config.parent.name or data_config.stem
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in fallback)
    return safe.strip("_") or "dataset"


def main() -> None:
    env_dir = Path(__file__).resolve().parent
    workspace_root = env_dir.parent.parent.resolve()
    os.chdir(env_dir)
    ultralytics_config_dir = env_dir / ".ultralytics"
    ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_config_dir))

    default_data_config = Path(r"D:\ring3\FloW_IMG\training\dataset.yaml")
    model_config = (env_dir / "yolov8s_cbam.yaml").resolve()
    project_root = (env_dir / "cbam_eiou_light_glare_training").resolve()

    parser = argparse.ArgumentParser(description="Stage1 (960) CBAM+Focal-EIoU runner with lightweight train-only low-light/reflection augmentation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--amp", type=str, default=None)
    parser.add_argument("--data", type=str, default=None, help="Path to dataset.yaml")
    parser.add_argument("--project", type=str, default=None, help="Ultralytics project output root")
    parser.add_argument("--name", type=str, default=None, help="Ultralytics run name")
    parser.add_argument("--plots", type=str, default=None)
    parser.add_argument("--verbose", type=str, default=None)
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument(
        "--use-light-glare-aug",
        dest="use_light_glare_aug",
        action="store_true",
        help="Build a train-only low-light/reflection augmented dataset overlay (default)",
    )
    aug_group.add_argument(
        "--no-light-glare-aug",
        dest="use_light_glare_aug",
        action="store_false",
        help="Disable the low-light/reflection overlay for a no-augmentation control run",
    )
    parser.set_defaults(use_light_glare_aug=True)
    parser.add_argument("--aug-dataset-root", type=str, default=None, help="Optional output directory for the generated augmented dataset overlay")
    parser.add_argument("--aug-refresh", action="store_true", help="Regenerate the augmented dataset even if a cached overlay exists")
    parser.add_argument("--aug-sample-ratio", type=float, default=0.10, help="Fraction of training images to duplicate with custom low-light/reflection augmentation")
    parser.add_argument("--low-light-prob", type=float, default=0.50, help="Probability that a sampled duplicate uses low-light augmentation")
    parser.add_argument("--reflection-prob", type=float, default=0.50, help="Probability that a sampled duplicate uses reflection augmentation")
    parser.add_argument("--reflection-max-area-ratio", type=float, default=0.05, help="Maximum reflection glare area as a fraction of image area")
    parser.add_argument("--reflection-min-area-factor", type=float, default=0.35, help="Minimum reflection area as a fraction of the configured max area")
    parser.add_argument("--reflection-aspect-min", type=float, default=1.4, help="Minimum reflection ellipse aspect ratio")
    parser.add_argument("--reflection-aspect-max", type=float, default=3.2, help="Maximum reflection ellipse aspect ratio")
    parser.add_argument("--reflection-intensity-min", type=float, default=22.0, help="Minimum additive reflection intensity")
    parser.add_argument("--reflection-intensity-max", type=float, default=48.0, help="Maximum additive reflection intensity")
    parser.add_argument("--allow-combined-aug", action="store_true", help="Allow low-light and reflection to be stacked on the same duplicate sample")
    parser.add_argument("--dry-run", action="store_true", help="Validate args and print the planned run; do not train")
    parser.add_argument("--skip-check", action="store_true", help="Skip integration check before running")
    parser.add_argument(
        "--run-tiled-val",
        action="store_true",
        help="Run tiled validation on best.pt after training finishes",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size for tiled validation. Default: imgsz",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="Tile stride for tiled validation. Default: tile_size // 2",
    )
    parser.add_argument(
        "--tile-batch",
        type=int,
        default=4,
        help="How many tiles to infer together during tiled validation",
    )
    parser.add_argument(
        "--tile-mode",
        choices=["auto", "force"],
        default="auto",
        help="auto: only tile sufficiently large images; force: tile every image",
    )
    parser.add_argument(
        "--tile-trigger-ratio",
        type=float,
        default=1.5,
        help="In auto mode, tile only when max(image_w, image_h) >= tile_size * this ratio",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.6,
        help="IoU threshold used when merging detections from different tiles",
    )
    parser.add_argument(
        "--tiled-output",
        type=str,
        default=None,
        help="Optional output directory for tiled validation artifacts",
    )
    parser.add_argument(
        "--no-strict-venv",
        action="store_true",
        help="Disable strict venv interpreter check (not recommended)",
    )
    args = parser.parse_args()

    if not args.no_strict_venv:
        _assert_running_in_env_venv(env_dir)

    seed = 42
    data_config = Path(args.data).expanduser().resolve() if args.data else default_data_config
    if not data_config.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_config}")
    if not model_config.exists():
        raise FileNotFoundError(f"Model config not found: {model_config} (cwd={Path.cwd()})")

    if args.use_light_glare_aug:
        dataset_key = _dataset_key_from_path(data_config)
        aug_output_root = (
            Path(args.aug_dataset_root).expanduser().resolve()
            if args.aug_dataset_root
            else env_dir / "generated_datasets" / "light_glare_20260506" / dataset_key
        )
        if args.dry_run:
            print(f"Dry-run: light/glare overlay would be built under: {aug_output_root}")
        else:
            builder = _load_light_glare_builder(env_dir)
            data_config = builder.build_augmented_dataset(
                data_yaml=data_config,
                output_root=aug_output_root,
                seed=seed,
                sample_ratio=float(args.aug_sample_ratio),
                low_light_prob=float(args.low_light_prob),
                reflection_prob=float(args.reflection_prob),
                allow_combined=bool(args.allow_combined_aug),
                reflection_max_area_ratio=float(args.reflection_max_area_ratio),
                reflection_min_area_factor=float(args.reflection_min_area_factor),
                reflection_aspect_min=float(args.reflection_aspect_min),
                reflection_aspect_max=float(args.reflection_aspect_max),
                reflection_intensity_min=float(args.reflection_intensity_min),
                reflection_intensity_max=float(args.reflection_intensity_max),
                refresh=bool(args.aug_refresh),
            )
            print(f"Using train-only low-light/reflection overlay: {data_config}")

    if args.dry_run:
        print("Model line: yolov8s_cbam_eiou_light_glare")
        print(f"Model config: {model_config}")
        print(f"Light/glare augmentation: {'enabled' if args.use_light_glare_aug else 'disabled'}")
        print(f"Data config: {data_config}")
        print("Dry-run requested: dataset, model config, and augmentation args validated. Exiting without model build/training.")
        return

    if not args.skip_check:
        check_script = env_dir / "check_cbam_eiou_integration.py"
        if not check_script.exists():
            raise FileNotFoundError(f"Integration check script not found: {check_script}")
        import subprocess
        import sys

        print("Running integration check before training...")
        subprocess.run([sys.executable, str(check_script)], check=True)

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cuda_ready = torch.cuda.is_available()
    device_arg: str | int = 0 if cuda_ready else "cpu"
    if not cuda_ready:
        print("CUDA device not detected, fallback to CPU training.")
    if args.device is not None:
        device_arg = int(args.device) if args.device.isdigit() else args.device

    from ultralytics import YOLO

    print("Building YOLOv8s CBAM model from config...")
    model = YOLO(str(model_config))

    focal_defaults = {
        "bbox_loss": "focal_eiou",
        "focal_eiou_alpha": 0.25,
        "focal_eiou_gamma": 2.0,
        "focal_eiou_eps": 1e-7,
    }
    if isinstance(model.model.args, dict):
        model.model.args.update(focal_defaults)
    else:
        for key, value in focal_defaults.items():
            setattr(model.model.args, key, value)
    if isinstance(model.overrides, dict):
        model.overrides.update(focal_defaults)
    else:
        for key, value in focal_defaults.items():
            setattr(model.overrides, key, value)

    stage1_args = {
        "data": str(data_config),
        "epochs": 150,
        "batch": 4,
        "imgsz": 960,
        "device": device_arg,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.006,
        "momentum": 0.937,
        "weight_decay": 6.25e-05,
        "box": 4.0,
        "cls": 0.5,
        "dfl": 1.3,
        "nbs": 8,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.2,
        "mosaic": 0.7,
        "mixup": 0.05,
        "close_mosaic": 10,
        "warmup_epochs": 3,
        "seed": 42,
        "translate": 0.1,
        "scale": 0.45,
        "fliplr": 0.5,
        "iou": 0.7,
        "copy_paste": 0.0,
        "erasing": 0.2,
        "project": str(project_root),
        "name": f"stage1_960_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "save_period": -1,
        "val": True,
        "save": True,
        "verbose": True,
        "plots": True,
        "patience": 100,
        "pretrained": False,
        "deterministic": True,
        "amp": True,
        **focal_defaults,
    }

    if args.epochs is not None:
        stage1_args["epochs"] = args.epochs
    if args.batch is not None:
        stage1_args["batch"] = args.batch
    if args.workers is not None:
        stage1_args["workers"] = args.workers
    if args.amp is not None:
        stage1_args["amp"] = _str2bool(args.amp)
    if args.project is not None:
        stage1_args["project"] = str(Path(args.project).expanduser().resolve())
    if args.name is not None:
        stage1_args["name"] = args.name
    if args.plots is not None:
        stage1_args["plots"] = _str2bool(args.plots)
    if args.verbose is not None:
        stage1_args["verbose"] = _str2bool(args.verbose)

    print("=" * 70)
    print("Training parameters")
    for key, value in stage1_args.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    train_results = model.train(**stage1_args)

    if args.run_tiled_val:
        import subprocess
        import sys

        save_dir = Path(train_results.save_dir).resolve()
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"best.pt not found after training: {best_weights}")

        tiled_output_dir = Path(args.tiled_output).expanduser().resolve() if args.tiled_output else save_dir
        tiled_cmd = [
            sys.executable,
            str((env_dir / "validate_tiled_best_model.py").resolve()),
            "--weights",
            str(best_weights),
            "--data",
            str(data_config),
            "--imgsz",
            str(stage1_args["imgsz"]),
            "--tile-batch",
            str(args.tile_batch),
            "--tile-mode",
            str(args.tile_mode),
            "--tile-trigger-ratio",
            str(args.tile_trigger_ratio),
            "--merge-iou",
            str(args.merge_iou),
            "--output-dir",
            str(tiled_output_dir),
        ]
        if args.tile_size is not None:
            tiled_cmd += ["--tile-size", str(args.tile_size)]
        if args.tile_stride is not None:
            tiled_cmd += ["--tile-stride", str(args.tile_stride)]

        print("Run tiled validation on best.pt ...")
        subprocess.run(tiled_cmd, check=True)


if __name__ == "__main__":
    main()
