#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="/root/autodl-tmp/ring3_setup/autodl"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "${LOG_ROOT}"

source "${SETUP_ROOT}/remote_env.sh"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model yolo26s.pt \
  --data "${SETUP_ROOT}/datasets/wsodd_sequence_disjoint_v1.yaml" \
  --name yolo26s_wsodd_sequence_disjoint_960_e1 \
  --epochs 1 \
  --imgsz 960 \
  --batch 8 \
  --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolo26s_wsodd_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model yolov8s.pt \
  --data "${SETUP_ROOT}/datasets/seadronessee_source_disjoint_v1.yaml" \
  --name yolov8s_seadronessee_source_disjoint_960_e1 \
  --epochs 1 \
  --imgsz 960 \
  --batch 8 \
  --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolov8s_seadronessee_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/autodl-tmp/weights/yolo11s.pt \
  --data "${SETUP_ROOT}/datasets/seadronessee_source_disjoint_v1.yaml" \
  --name yolo11s_seadronessee_source_disjoint_960_e1 \
  --epochs 1 \
  --imgsz 960 \
  --batch 8 \
  --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolo11s_seadronessee_runtime_calibration.log"
