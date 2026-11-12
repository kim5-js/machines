#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="/root/autodl-tmp/ring3_setup/autodl"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "${LOG_ROOT}"

source "${SETUP_ROOT}/remote_env.sh"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model rtdetr-l.pt \
  --data "${SETUP_ROOT}/datasets/flow_current.yaml" \
  --name rtdetr_l_flow_960_e1 \
  --epochs 1 \
  --imgsz 960 \
  --batch 4 \
  --workers 8 \
  2>&1 | tee "${LOG_ROOT}/rtdetr_flow_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model rtdetr-l.pt \
  --data "${SETUP_ROOT}/datasets/wsodd_sequence_disjoint_v1.yaml" \
  --name rtdetr_l_wsodd_sequence_disjoint_960_e1 \
  --epochs 1 \
  --imgsz 960 \
  --batch 4 \
  --workers 8 \
  2>&1 | tee "${LOG_ROOT}/rtdetr_wsodd_runtime_calibration.log"
