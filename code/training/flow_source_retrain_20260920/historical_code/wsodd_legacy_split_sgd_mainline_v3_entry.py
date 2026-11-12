from __future__ import annotations

import train_pretrained_sgd_mainline as impl
import train_wsodd_legacy_split_sgd_mainline as legacy
from wsodd_legacy_split_sgd_mainline_v2_entry import validate_config_v2


_released_configure_runtime = legacy.configure_runtime


def configure_runtime_v3() -> None:
    _released_configure_runtime()
    impl.MANIFEST_PATH = impl.SETUP_ROOT / "wsodd_legacy_split_sgd_mainline_v3_release_manifest.json"


def main() -> int:
    legacy.validate_config = validate_config_v2
    legacy.configure_runtime = configure_runtime_v3
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
