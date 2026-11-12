#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON="${DATA_ROOT}/envs/mdc-yolo26/bin/python"
LOG_ROOT="${DATA_ROOT}/logs"
LOG_PATH="${LOG_ROOT}/cbam_focal_formal_series.log"

mkdir -p "${LOG_ROOT}"
cd "${SETUP_ROOT}"
export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"

"${PYTHON}" train_cbam_focal_formal_series.py --mode queue 2>&1 | tee -a "${LOG_PATH}"

