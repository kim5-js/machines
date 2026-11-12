#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SETUP_ROOT="${RING3_SETUP_ROOT:-${DATA_ROOT}/ring3_setup/autodl}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/envs/mdc-yolo26/bin/python}"
STOP_SENTINEL="${DATA_ROOT}/formal_state/STOP_WSODD_LEGACY_SPLIT_SGD_MAINLINE"
EXPECTED_ADAPTER_SHA256="2aa4655c94a347b006e849703984fb6dacf3fc7e35f350816291161b4c5bdbf2"

if [[ -e "${STOP_SENTINEL}" ]]; then
  echo "Refusing area recovery because stop sentinel exists: ${STOP_SENTINEL}" >&2
  exit 42
fi

observed_sha256="$(sha256sum "${SETUP_ROOT}/recover_wsodd_legacy_coco_areas.py" | cut -c1-64)"
if [[ "${observed_sha256}" != "${EXPECTED_ADAPTER_SHA256}" ]]; then
  echo "Area recovery adapter hash mismatch: ${observed_sha256}" >&2
  exit 43
fi

export AUTODL_DATA_ROOT="${DATA_ROOT}"
export RING3_SETUP_ROOT="${SETUP_ROOT}"
export PYTHONPATH="${SETUP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RING3_AREA_CHUNK_SIZE="${RING3_AREA_CHUNK_SIZE:-4}"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"

cd "${SETUP_ROOT}"
exec "${PYTHON_BIN}" -u recover_wsodd_legacy_coco_areas.py \
  --config "${SETUP_ROOT}/wsodd_legacy_split_sgd_mainline_config.json" \
  --state "${DATA_ROOT}/formal_state/wsodd_legacy_split_sgd_mainline_state.json" \
  --summary "${DATA_ROOT}/formal_state/wsodd_legacy_split_sgd_mainline_summary.json" \
  --output-root "${DATA_ROOT}/formal_state/wsodd_legacy_area_recovery" \
  --imgsz 960 \
  --batch 8 \
  --device 0
