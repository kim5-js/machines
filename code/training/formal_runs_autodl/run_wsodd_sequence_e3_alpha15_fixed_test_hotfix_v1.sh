#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"
LOG_PATH="${DATA_ROOT}/logs/wsodd_sequence_e3_alpha15_fixed_test_hotfix_v1.log"

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

mkdir -p "$(dirname "${LOG_PATH}")"
cd "${SETUP_ROOT}"
exec "${PYTHON_BIN}" -u wsodd_sequence_e3_alpha15_fixed_test_hotfix_v1.py \
  --config "${SETUP_ROOT}/wsodd_sequence_e3_alpha15_config.json" >>"${LOG_PATH}" 2>&1
