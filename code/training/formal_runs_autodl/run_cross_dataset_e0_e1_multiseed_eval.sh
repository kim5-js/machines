#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"
STATE_ROOT="${DATA_ROOT}/formal_state"
LOG_ROOT="${DATA_ROOT}/logs"
STATE_PATH="${STATE_ROOT}/cross_dataset_e0_e1_multiseed_eval_state.json"
STOP_SENTINEL="${STATE_ROOT}/STOP_CROSS_DATASET_E0_E1_EVAL"
LOG_PATH="${LOG_ROOT}/cross_dataset_e0_e1_multiseed_eval.log"

mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

if [[ -e "${STOP_SENTINEL}" ]]; then
  echo "Refusing to start because stop sentinel exists: ${STOP_SENTINEL}" | tee -a "${LOG_PATH}"
  exit 2
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "Refusing to start evaluation while a GPU compute process is active" | tee -a "${LOG_PATH}"
  exit 3
fi

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

cd "${SETUP_ROOT}"
echo "Starting evaluation-only matched E0/E1 seed 0/1/2 cross-dataset fixed-split and strong-robustness evaluation" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" -u evaluate_cross_dataset_e0_e1_multiseed.py \
  --config "${SETUP_ROOT}/cross_dataset_e0_e1_multiseed_eval_config.json" 2>&1 | tee -a "${LOG_PATH}"

STATUS="$(${PYTHON_BIN} -c "import json; print(json.load(open('${STATE_PATH}', encoding='utf-8')).get('status', 'missing'))")"
if [[ "${STATUS}" != "completed" ]]; then
  echo "Cross-dataset evaluation ended without completed status: ${STATUS}" | tee -a "${LOG_PATH}"
  exit 4
fi

echo "Cross-dataset evaluation completed; no training was started" | tee -a "${LOG_PATH}"
