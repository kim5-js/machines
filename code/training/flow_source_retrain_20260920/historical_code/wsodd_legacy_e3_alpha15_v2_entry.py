from __future__ import annotations

from typing import Any

import train_pretrained_sgd_mainline as impl
import train_wsodd_legacy_e3_alpha15 as release


MANIFEST_SCHEMA = "ring3.wsodd_legacy_e3_alpha15_release_manifest.v2"
_released_configure_runtime = release.configure_runtime
_released_validate_config = release.validate_config


def validate_config_v2(config: dict[str, Any]) -> None:
    # The released SeaDronesSee validator resolves CONFIG_SCHEMA dynamically
    # from its module. Temporarily restore that original schema while it checks
    # the synthetic four-run proxy, then restore this release's schema.
    rebound_schema = impl.CONFIG_SCHEMA
    try:
        impl.CONFIG_SCHEMA = "ring3.pretrained_sgd_mainline_config.v1"
        _released_validate_config(config)
    finally:
        impl.CONFIG_SCHEMA = rebound_schema


def configure_runtime_v2() -> None:
    _released_configure_runtime()
    impl.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    impl.MANIFEST_PATH = impl.SETUP_ROOT / "wsodd_legacy_e3_alpha15_v2_release_manifest.json"
    impl.validate_config = validate_config_v2


def main() -> int:
    release.validate_config = validate_config_v2
    release.configure_runtime = configure_runtime_v2
    return release.main()


if __name__ == "__main__":
    raise SystemExit(main())
