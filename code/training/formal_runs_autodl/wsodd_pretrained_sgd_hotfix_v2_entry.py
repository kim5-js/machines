#!/usr/bin/env python3
"""Narrow entry-point hotfix for WSODD base-schema validation isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import train_wsodd_pretrained_sgd_mainline as wsodd


EXPECTED_BASE_MANIFEST_SHA256 = "fe106f29d4bbd424bb3962d8a459b151e6e5810e50cc8eefebef35e2ba0b117e"
EXPECTED_BASE_RUNNER_SHA256 = "256ac80772bc3c0f23d6ac8e07877c43b333bb681e5b52898e39dbca0d79fd21"
BASE_CONFIG_SCHEMA = "ring3.pretrained_sgd_mainline_config.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_release() -> None:
    setup_root = Path(__file__).resolve().parent
    expected = {
        setup_root / "wsodd_pretrained_sgd_mainline_release_manifest.json": EXPECTED_BASE_MANIFEST_SHA256,
        setup_root / "train_wsodd_pretrained_sgd_mainline.py": EXPECTED_BASE_RUNNER_SHA256,
    }
    for path, locked_hash in expected.items():
        if not path.is_file() or sha256(path) != locked_hash:
            raise RuntimeError(f"WSODD base release changed before v2 entry: {path}")


_released_validate_config = wsodd.validate_config


def validate_config_with_base_schema_isolation(config: dict) -> None:
    """Let the reused base validator see its own schema, then restore WSODD."""
    original_schema = wsodd.impl.CONFIG_SCHEMA
    try:
        wsodd.impl.CONFIG_SCHEMA = BASE_CONFIG_SCHEMA
        _released_validate_config(config)
    finally:
        wsodd.impl.CONFIG_SCHEMA = original_schema


def main() -> int:
    verify_base_release()
    wsodd.validate_config = validate_config_with_base_schema_isolation
    return wsodd.main()


if __name__ == "__main__":
    raise SystemExit(main())
