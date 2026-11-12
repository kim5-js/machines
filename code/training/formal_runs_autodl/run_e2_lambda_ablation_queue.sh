#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="${RING3_SETUP_ROOT:-/root/autodl-tmp/ring3_setup/autodl}"
DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
LOG_ROOT="${DATA_ROOT}/logs"
STATE_ROOT="${DATA_ROOT}/formal_state"

source "${SETUP_ROOT}/remote_env.sh"
mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"

exec 9>"${STATE_ROOT}/e2_lambda_ablation_queue_supervisor.lock"
if ! flock -n 9; then
  echo "An E2 lambda-ablation queue supervisor already holds the runtime lock." >&2
  exit 4
fi

python -u "${SETUP_ROOT}/train_e2_lambda_ablation.py" --mode queue \
  2>&1 | tee -a "${LOG_ROOT}/e2_lambda_ablation_queue.log"
exit "${PIPESTATUS[0]}"
