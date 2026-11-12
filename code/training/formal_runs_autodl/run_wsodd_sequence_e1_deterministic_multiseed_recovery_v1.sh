#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"
STATE_ROOT="${DATA_ROOT}/formal_state"
LOG_ROOT="${DATA_ROOT}/logs"
STATE_PATH="${STATE_ROOT}/wsodd_sequence_e1_deterministic_multiseed_state.json"
STOP_SENTINEL="${STATE_ROOT}/STOP_WSODD_SEQUENCE_E1_DETERMINISTIC_MULTISEED"
LOG_PATH="${LOG_ROOT}/wsodd_sequence_e1_deterministic_multiseed.log"

mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

if [[ -e "${STOP_SENTINEL}" ]]; then
  echo "Refusing recovery because stop sentinel exists: ${STOP_SENTINEL}" | tee -a "${LOG_PATH}"
  exit 2
fi

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHON_BIN
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

cd "${SETUP_ROOT}"
echo "Starting audited deterministic E1 multiseed missing-return recovery" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" -u wsodd_sequence_e1_deterministic_multiseed_recovery_v1.py 2>&1 | tee -a "${LOG_PATH}"

QUEUE_STATUS="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${STATE_PATH}', encoding='utf-8')).get('status', 'missing'))")"
if [[ "${QUEUE_STATUS}" != "completed" ]]; then
  echo "Recovered deterministic E1 multiseed queue ended without completed state: ${QUEUE_STATUS}" | tee -a "${LOG_PATH}"
  exit 3
fi

echo "Starting locked 2406-image fixed-test evaluation for deterministic E1 seeds 0/1/2" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" -u evaluate_wsodd_sequence_e1_deterministic_multiseed_fixed_test.py \
  --config "${SETUP_ROOT}/wsodd_sequence_e1_deterministic_multiseed_config.json" 2>&1 | tee -a "${LOG_PATH}"
