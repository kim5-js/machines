#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="${RING3_SETUP_ROOT:-/root/autodl-tmp/ring3_setup/autodl}"
DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
LOG_ROOT="${DATA_ROOT}/logs"
STATE_ROOT="${DATA_ROOT}/formal_state"
MODE="${1:-formal}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "formal" ]]; then
  echo "usage: $0 {smoke|formal}" >&2
  exit 64
fi

source "${SETUP_ROOT}/remote_env.sh"
mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"

exec 9>"${STATE_ROOT}/e2_sdc_${MODE}_supervisor.lock"
if ! flock -n 9; then
  echo "An E2 ${MODE} supervisor already holds the runtime lock." >&2
  exit 4
fi

if [[ "${MODE}" == "smoke" ]]; then
  python -u "${SETUP_ROOT}/maritime_degradation.py" \
    --config "${SETUP_ROOT}/e1_mdg_config.json" \
    2>&1 | tee -a "${LOG_ROOT}/e2_sdc_smoke.log"
  python -u "${SETUP_ROOT}/sdc_consistency.py" \
    --config "${SETUP_ROOT}/e2_p3_consistency_config.json" \
    --weights "${DATA_ROOT}/weights/yolov8s.pt" \
    --device cuda:0 \
    2>&1 | tee -a "${LOG_ROOT}/e2_sdc_smoke.log"
  python -u "${SETUP_ROOT}/train_e2_sdc.py" --mode smoke \
    2>&1 | tee -a "${LOG_ROOT}/e2_sdc_smoke.log"
  exit "${PIPESTATUS[0]}"
fi

while true; do
  set +e
  python -u "${SETUP_ROOT}/train_e2_sdc.py" --mode formal \
    2>&1 | tee -a "${LOG_ROOT}/e2_sdc.log"
  rc="${PIPESTATUS[0]}"
  set -e
  case "${rc}" in
    0)
      exit 0
      ;;
    3)
      echo "E2 formal attempts are exhausted; supervisor is stopping." | tee -a "${LOG_ROOT}/e2_sdc.log"
      exit 3
      ;;
    *)
      echo "E2 formal attempt exited with rc=${rc}; retrying the locked run in 10 seconds." | tee -a "${LOG_ROOT}/e2_sdc.log"
      sleep 10
      ;;
  esac
done
