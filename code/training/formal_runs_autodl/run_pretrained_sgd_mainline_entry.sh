#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
ENV_ROOT="${DATA_ROOT}/envs/mdc-yolo26"
STATE_ROOT="${DATA_ROOT}/formal_state"
LOG_ROOT="${DATA_ROOT}/logs"
STOP_SENTINEL="${STATE_ROOT}/STOP_PRETRAINED_SGD_MAINLINE"
LOG_PATH="${LOG_ROOT}/pretrained_sgd_mainline.log"

mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

if [[ -e "${STOP_SENTINEL}" ]]; then
  echo "Refusing to start because stop sentinel exists: ${STOP_SENTINEL}" | tee -a "${LOG_PATH}"
  exit 2
fi

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

exec "${ENV_ROOT}/bin/python" -u "${SETUP_ROOT}/pretrained_sgd_mainline_entry.py" 2>&1 | tee -a "${LOG_PATH}"
