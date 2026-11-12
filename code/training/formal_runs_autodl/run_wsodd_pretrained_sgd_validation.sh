#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

cd "${SETUP_ROOT}"
"${PYTHON_BIN}" -u evaluate_wsodd_pretrained_sgd_validation.py \
  --config "${SETUP_ROOT}/wsodd_pretrained_sgd_validation_config.json"
