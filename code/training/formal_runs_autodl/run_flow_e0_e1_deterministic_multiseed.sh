#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"
STATE_ROOT="${DATA_ROOT}/formal_state"
LOG_ROOT="${DATA_ROOT}/logs"
SEA_E0_STATE="${STATE_ROOT}/seadronessee_e0_multiseed_state.json"
SEA_E1_STATE="${STATE_ROOT}/seadronessee_e1_deterministic_multiseed_state.json"
E0_STATE="${STATE_ROOT}/flow_e0_multiseed_state.json"
E1_STATE="${STATE_ROOT}/flow_e1_deterministic_multiseed_state.json"
STOP_ALL="${STATE_ROOT}/STOP_FLOW_CROSS_DATASET"
STOP_E0="${STATE_ROOT}/STOP_FLOW_E0_MULTISEED"
STOP_E1="${STATE_ROOT}/STOP_FLOW_E1_DETERMINISTIC_MULTISEED"
LOG_PATH="${LOG_ROOT}/flow_e0_e1_deterministic_multiseed.log"

mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"
for sentinel in "${STOP_ALL}" "${STOP_E0}" "${STOP_E1}"; do
  if [[ -e "${sentinel}" ]]; then
    echo "Refusing to start because stop sentinel exists: ${sentinel}" | tee -a "${LOG_PATH}"
    exit 2
  fi
done

for sea_state in "${SEA_E0_STATE}" "${SEA_E1_STATE}"; do
  if [[ ! -f "${sea_state}" ]]; then
    echo "Refusing to start before SeaDronesSee completes: missing ${sea_state}" | tee -a "${LOG_PATH}"
    exit 3
  fi
  sea_status="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${sea_state}', encoding='utf-8')).get('status', 'missing'))")"
  if [[ "${sea_status}" != "completed" ]]; then
    echo "Refusing to start before SeaDronesSee completes: ${sea_state} status=${sea_status}" | tee -a "${LOG_PATH}"
    exit 4
  fi
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "Refusing to start FloW while a GPU compute process is active" | tee -a "${LOG_PATH}"
  exit 5
fi

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHON_BIN
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

cd "${SETUP_ROOT}"
echo "Starting FloW source-temporal-disjoint E0 seeds 0, 1, and 2" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" -u train_flow_e0_multiseed.py 2>&1 | tee -a "${LOG_PATH}"

E0_STATUS="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${E0_STATE}', encoding='utf-8')).get('status', 'missing'))")"
if [[ "${E0_STATUS}" != "completed" ]]; then
  echo "FloW E0 queue ended without completed status: ${E0_STATUS}" | tee -a "${LOG_PATH}"
  exit 6
fi

echo "Starting FloW deterministic-image-ID E1-Mid seeds 0, 1, and 2" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" -u train_flow_e1_deterministic_multiseed.py 2>&1 | tee -a "${LOG_PATH}"

E1_STATUS="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${E1_STATE}', encoding='utf-8')).get('status', 'missing'))")"
if [[ "${E1_STATUS}" != "completed" ]]; then
  echo "FloW deterministic E1 queue ended without completed status: ${E1_STATUS}" | tee -a "${LOG_PATH}"
  exit 7
fi

echo "FloW E0/E1 training completed; fixed-test evaluation is a separately locked stage" | tee -a "${LOG_PATH}"
