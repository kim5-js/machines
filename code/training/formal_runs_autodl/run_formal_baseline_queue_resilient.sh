#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="${RING3_SETUP_ROOT:-/root/autodl-tmp/ring3_setup/autodl}"
LOG_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}/logs"

source "${SETUP_ROOT}/remote_env.sh"
mkdir -p "${LOG_ROOT}"

python -u "${SETUP_ROOT}/formal_baseline_queue_resilient.py" \
  2>&1 | tee -a "${LOG_ROOT}/formal_baseline_queue.log"
