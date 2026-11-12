from __future__ import annotations

from typing import Any

import train_wsodd_legacy_split_sgd_mainline as legacy


def validate_config_v2(config: dict[str, Any]) -> None:
    if config.get("schema") != legacy.CONFIG_SCHEMA or config.get("status") != "locked_requested":
        raise ValueError("Unexpected legacy WSODD pretrained-SGD config schema/status")
    observed = [
        (
            run.get("key"),
            run.get("mdg_mode"),
            run.get("dual_view"),
            run.get("degraded_small_weighting"),
            run.get("degraded_small_target_weight"),
            run.get("cbam"),
            run.get("focal_eiou"),
            run.get("consistency_loss"),
        )
        for run in config.get("run_order", [])
    ]
    expected_runs = [
        ("e0", "none", False, False, 1.0, False, False, False),
        ("e1", "single_view", False, False, 1.0, False, False, False),
        ("e2", "dual_view", True, False, 1.0, False, False, False),
        ("e3", "dual_view", True, True, legacy.EXPECTED_ALPHA, False, False, False),
    ]
    if observed != expected_runs or len({str(run.get("name")) for run in config.get("run_order", [])}) != 4:
        raise ValueError(f"Unexpected legacy WSODD E0-E3 run matrix: {observed!r}")

    training = config.get("training", {})
    locked_training = {
        "model": "/root/autodl-tmp/weights/yolov8s.pt",
        "pretrained": True,
        "epochs": 150,
        "patience": 100,
        "batch": 8,
        "imgsz": 960,
        "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.006,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0000625,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "nbs": 8,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.2,
        "translate": 0.1,
        "scale": 0.45,
        "fliplr": 0.5,
        "mosaic": 0.7,
        "mixup": 0.05,
        "erasing": 0.2,
        "close_mosaic": 10,
        "save_period": -1,
        "seed": 0,
    }
    mismatches = {
        key: {"expected": expected, "observed": training.get(key)}
        for key, expected in locked_training.items()
        if training.get(key) != expected
    }
    dataset = config.get("dataset", {})
    expected_dataset = {
        "yaml_sha256": legacy.EXPECTED_DATA_YAML_SHA256,
        "annotation_semantics_audit_sha256": legacy.EXPECTED_ANNOTATION_AUDIT_SHA256,
        "protocol": "wsodd_legacy_image_split_v1",
        "train_images": 6222,
        "val_images": 1245,
        "test_images": 0,
        "classes": legacy.EXPECTED_CLASSES,
        "sequence_groups": 147,
        "contaminated_sequence_groups": 98,
        "val_images_in_contaminated_groups": 1243,
        "independent_test_available": False,
    }
    mismatches.update(
        {
            f"dataset.{key}": {"expected": expected, "observed": dataset.get(key)}
            for key, expected in expected_dataset.items()
            if dataset.get(key) != expected
        }
    )
    weighting = config.get("small_object_weighting", {})
    if weighting.get("ordinary_target_weight") != 1.0 or weighting.get("degraded_small_target_weight") != legacy.EXPECTED_ALPHA:
        mismatches["small_object_weighting"] = weighting
    validation = config.get("post_training_validation", {})
    if validation.get("fixed_split") != "val" or validation.get("fixed_images") != 1245:
        mismatches["post_training_validation"] = validation
    if config.get("execution", {}).get("output_root") != "/root/autodl-tmp/validation_wsodd_legacy_split_sgd_e0_e3":
        mismatches["output_root"] = config.get("execution", {}).get("output_root")
    if str(training.get("optimizer", "")).lower() == "auto":
        mismatches["optimizer"] = "optimizer=auto is forbidden"
    if any(value is not True for value in config.get("invariants", {}).values()):
        mismatches["invariants"] = config.get("invariants")
    if mismatches:
        raise ValueError(f"Legacy WSODD reproduction contract changed: {mismatches}")


def main() -> int:
    legacy.validate_config = validate_config_v2
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
