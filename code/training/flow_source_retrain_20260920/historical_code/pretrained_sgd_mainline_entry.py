from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

import train_pretrained_sgd_mainline as implementation


def verify_args_with_serialized_device(
    run_dir: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> None:
    """Match locked args while accepting Ultralytics' string serialization of device 0."""

    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        raise RuntimeError(f"Run args.yaml is missing: {run_dir}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected = implementation.training_kwargs(config, variant)
    expected["model"] = config["weights"]["path"]
    ignored = {"project", "name", "exist_ok", "verbose"}
    mismatches = []
    for key, value in expected.items():
        if key in ignored:
            continue
        observed = args.get(key)
        if key == "device":
            equal = str(observed) == str(value)
        elif isinstance(value, float) and isinstance(observed, (float, int)):
            equal = math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == value
        if not equal:
            mismatches.append(f"{key}: expected {value!r}, found {observed!r}")
    if mismatches:
        raise RuntimeError("Run args differ from the locked SGD recipe:\n" + "\n".join(mismatches))


implementation.verify_args = verify_args_with_serialized_device


if __name__ == "__main__":
    raise SystemExit(implementation.main())
