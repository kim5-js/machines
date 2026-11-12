#!/usr/bin/env python3
"""Compatibility entrypoint for the locked WSODD post-training evaluator.

The locked area evaluator intentionally omits ``positive_gt_count`` when an
area range has no positive classes.  The validation summary originally
assumed that field was always present, so a manually reviewed scene with no
small-object ground truth raised ``KeyError``.  This entrypoint leaves every
locked evaluator/config/checkpoint untouched and patches only the summary
adapter so an empty area range reports a ground-truth count of zero.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import evaluate_pretrained_sgd_robustness as robust


SETUP_ROOT = Path(os.environ.get("RING3_SETUP_ROOT", "/root/autodl-tmp/ring3_setup/autodl"))
EXPECTED_BASE_HASHES = {
    "evaluate_pretrained_sgd_robustness.py": "cb42325d69011f24043170bbdee3b12371cd433d4dcf7d3fe611568a76c1794f",
    "evaluate_wsodd_pretrained_sgd_validation.py": "9f4e88e3924ef70eca183bf07d797ad9a438274be5524d627342aacb8df680ad",
    "evaluate_e3_coco_areas.py": "c1eeb2e5bdf72e627a856ac14f4f679f844de2a9ff0a205cfce50323de9b61cc",
    "wsodd_pretrained_sgd_validation_config.json": "844e06e1787638368a8c50c9213c82df6f6c112e62c1e02e57f6c4bf5831174b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_base() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_BASE_HASHES.items():
        path = SETUP_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Locked WSODD validation file is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Locked WSODD validation hash mismatch for {path}: expected {expected}, found {actual}"
            )
        observed[relative] = actual
    return observed


def positive_gt_count(area_metrics: dict[str, Any]) -> int:
    if "positive_gt_count" in area_metrics:
        return int(area_metrics["positive_gt_count"])
    per_class = area_metrics.get("per_class")
    if not isinstance(per_class, list):
        raise RuntimeError("Area metrics without positive_gt_count must contain a per_class list")
    if not per_class:
        return 0
    if any(not isinstance(item, dict) or "positive_gt_count" not in item for item in per_class):
        raise RuntimeError("Non-empty per_class metrics are missing positive_gt_count")
    return sum(int(item["positive_gt_count"]) for item in per_class)


def metric_summary_empty_area_safe(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "mAP50": metrics["all"]["AP50"],
        "mAP50-95": metrics["all"]["AP50_95"],
        "APsmall": metrics["small"]["AP50_95"],
        "APmedium": metrics["medium"]["AP50_95"],
        "APlarge": metrics["large"]["AP50_95"],
        "ground_truth_count": positive_gt_count(metrics["all"]),
        "small_ground_truth_count": positive_gt_count(metrics["small"]),
    }


def regression_self_test() -> dict[str, Any]:
    empty = {"AP50": None, "AP50_95": None, "per_class": []}
    populated = {
        "AP50": 0.5,
        "AP50_95": 0.25,
        "per_class": [{"positive_gt_count": 3}, {"positive_gt_count": 4}],
    }
    metrics = {
        "all": populated,
        "small": empty,
        "medium": empty,
        "large": empty,
    }
    summary = metric_summary_empty_area_safe(metrics)
    if summary["ground_truth_count"] != 7 or summary["small_ground_truth_count"] != 0:
        raise RuntimeError(f"WSODD empty-area metric-summary regression failed: {summary}")
    if summary["APsmall"] is not None:
        raise RuntimeError(f"WSODD empty-area APsmall must remain null: {summary}")
    return {
        "status": "passed",
        "empty_area_positive_gt_count": 0,
        "populated_area_positive_gt_count_from_per_class": 7,
        "empty_area_ap_preserved_as_null": True,
    }


def main() -> int:
    locked_hashes = verify_locked_base()
    self_test = regression_self_test()
    robust.metric_summary = metric_summary_empty_area_safe

    import evaluate_wsodd_pretrained_sgd_validation as wsodd

    evidence = {
        "schema": "ring3.wsodd_validation_empty_area_hotfix.v1",
        "status": "passed",
        "scope": "summary_adapter_only",
        "locked_files_modified": False,
        "training_started": False,
        "checkpoint_updates": False,
        "e4_started": False,
        "locked_hashes": locked_hashes,
        "regression_self_test": self_test,
    }
    print("WSODD_VALIDATION_HOTFIX_V1 " + json.dumps(evidence, sort_keys=True), flush=True)
    return int(wsodd.main())


if __name__ == "__main__":
    raise SystemExit(main())
