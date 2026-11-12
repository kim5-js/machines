#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RING3_SETUP_ROOT="${RING3_SETUP_ROOT:-/root/autodl-tmp/ring3_setup/autodl}"
source "${RING3_SETUP_ROOT}/remote_env.sh"
export PYTHONPATH="${RING3_SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -u "${SCRIPT_DIR}/p3_region_cosine_diagnostic.py" \
  --config "${SCRIPT_DIR}/p3_region_cosine_diagnostic_config.json" \
  "$@"
