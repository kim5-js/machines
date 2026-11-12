#!/usr/bin/env bash
set -Eeuo pipefail

SETUP_ROOT="/root/autodl-tmp/ring3_setup/autodl"
LOG_ROOT="/root/autodl-tmp/logs"
STATE_ROOT="/root/autodl-tmp/formal_state"

mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"
source "${SETUP_ROOT}/remote_env.sh"

python -u "${SETUP_ROOT}/formal_baseline_queue.py" \
  2>&1 | tee -a "${LOG_ROOT}/formal_baseline_queue.log"
exit "${PIPESTATUS[0]}"
