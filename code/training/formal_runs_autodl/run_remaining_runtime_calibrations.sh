#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="/root/autodl-tmp/ring3_setup/autodl"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "${LOG_ROOT}"

source "${SETUP_ROOT}/remote_env.sh"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/autodl-tmp/weights/yolo11s.pt \
  --data "${SETUP_ROOT}/datasets/seadronessee_source_disjoint_v1.yaml" \
  --name yolo11s_seadronessee_source_disjoint_960_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolo11s_seadronessee_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/yolov8s.pt \
  --data "${SETUP_ROOT}/datasets/flow_current.yaml" \
  --name yolov8s_flow_960_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolov8s_flow_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/yolov8s.pt \
  --data "${SETUP_ROOT}/datasets/wsodd_sequence_disjoint_v1.yaml" \
  --name yolov8s_wsodd_sequence_disjoint_960_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolov8s_wsodd_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/autodl-tmp/weights/yolo11s.pt \
  --data "${SETUP_ROOT}/datasets/flow_current.yaml" \
  --name yolo11s_flow_960_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolo11s_flow_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/autodl-tmp/weights/yolo11s.pt \
  --data "${SETUP_ROOT}/datasets/wsodd_sequence_disjoint_v1.yaml" \
  --name yolo11s_wsodd_sequence_disjoint_960_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/yolo11s_wsodd_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/rtdetr-l.pt \
  --data "${SETUP_ROOT}/datasets/seadronessee_source_disjoint_v1.yaml" \
  --name rtdetr_l_seadronessee_source_disjoint_960_batch8_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/rtdetr_seadronessee_batch8_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/rtdetr-l.pt \
  --data "${SETUP_ROOT}/datasets/flow_current.yaml" \
  --name rtdetr_l_flow_960_batch8_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/rtdetr_flow_batch8_runtime_calibration.log"

python "${SETUP_ROOT}/calibrate_runtime.py" \
  --model /root/rtdetr-l.pt \
  --data "${SETUP_ROOT}/datasets/wsodd_sequence_disjoint_v1.yaml" \
  --name rtdetr_l_wsodd_sequence_disjoint_960_batch8_e1 \
  --epochs 1 --imgsz 960 --batch 8 --workers 8 \
  2>&1 | tee "${LOG_ROOT}/rtdetr_wsodd_batch8_runtime_calibration.log"
