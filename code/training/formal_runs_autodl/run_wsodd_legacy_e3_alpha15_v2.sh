#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHON_BIN
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"
export RING3_AREA_CHUNK_SIZE=4

LOG_PATH="${DATA_ROOT}/logs/wsodd_legacy_e3_alpha15.log"
mkdir -p "$(dirname "${LOG_PATH}")"
cd "${SETUP_ROOT}"
"${PYTHON_BIN}" -u wsodd_legacy_e3_alpha15_v2_entry.py "$@" 2>&1 | tee -a "${LOG_PATH}"
