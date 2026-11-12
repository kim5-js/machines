from __future__ import annotations

# Importing the original entry installs the audited device-serialization args check.
import pretrained_sgd_mainline_entry  # noqa: F401
import pretrained_sgd_mainline_supervisor_patch as supervisor_patch


if __name__ == "__main__":
    raise SystemExit(supervisor_patch.main())
