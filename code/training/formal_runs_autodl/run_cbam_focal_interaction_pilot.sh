#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/autodl-tmp/envs/mdc-yolo26/bin/python
SETUP_ROOT=/root/autodl-tmp/ring3_setup/autodl
LOG_ROOT=/root/autodl-tmp/logs
LOG_PATH="${LOG_ROOT}/cbam_focal_interaction_pilot.log"

mkdir -p "${LOG_ROOT}"
cd "${SETUP_ROOT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] interaction pilot launcher started" | tee -a "${LOG_PATH}"
"${PYTHON}" train_cbam_focal_interaction_pilot.py --mode queue 2>&1 | tee -a "${LOG_PATH}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pilot completed; starting fixed 200-batch diagnostic" | tee -a "${LOG_PATH}"
"${PYTHON}" cbam_focal_region_diagnostic.py 2>&1 | tee -a "${LOG_PATH}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] diagnostic completed; stopping before Object-Aware/E3" | tee -a "${LOG_PATH}"
