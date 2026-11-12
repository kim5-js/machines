#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"
STATE_PATH="${DATA_ROOT}/formal_state/wsodd_pretrained_sgd_mainline_state.json"

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHON_BIN
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

cd "${SETUP_ROOT}"
"${PYTHON_BIN}" -u wsodd_pretrained_sgd_hotfix_v3_entry.py

QUEUE_STATUS="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${STATE_PATH}', encoding='utf-8')).get('status', 'missing'))")"
if [[ "${QUEUE_STATUS}" != "completed" ]]; then
  echo "WSODD mainline ended without a successful completed state: ${QUEUE_STATUS}" >&2
  exit 3
fi

"${SETUP_ROOT}/run_wsodd_pretrained_sgd_validation.sh"
