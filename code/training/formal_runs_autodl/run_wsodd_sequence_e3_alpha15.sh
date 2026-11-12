#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/root/autodl-tmp/envs/mdc-yolo26/bin/python"
SETUP_ROOT="/root/autodl-tmp/ring3_setup/autodl"
LOG_PATH="/root/autodl-tmp/logs/wsodd_sequence_e3_alpha15.log"

mkdir -p "$(dirname "${LOG_PATH}")"
exec "${PYTHON_BIN}" -u "${SETUP_ROOT}/train_wsodd_sequence_e3_alpha15.py" >>"${LOG_PATH}" 2>&1
