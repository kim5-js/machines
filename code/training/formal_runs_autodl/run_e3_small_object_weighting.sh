#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/root/autodl-tmp"
SETUP_ROOT="$DATA_ROOT/ring3_setup/autodl"
PYTHON="$DATA_ROOT/envs/mdc-yolo26/bin/python"
LOG_FILE="$DATA_ROOT/logs/e3_small_object_weighting.log"
STATE_ROOT="$DATA_ROOT/formal_state"
STOP_SENTINEL="$STATE_ROOT/STOP_E3_SMALL_OBJECT_WEIGHTING"
RUN_DIR="$DATA_ROOT/formal_e3_small_object_weighting/seed0_yolov8s_seadronessee_e3_dualview_smallobj_alpha15_img960_e150"
OUTPUT="$STATE_ROOT/e3_small_object_weighting_coco_area_metrics.json"

mkdir -p "$DATA_ROOT/logs" "$STATE_ROOT"
exec 9>"$STATE_ROOT/e3_small_object_weighting_launcher.lock"
if ! flock -n 9; then
  echo "Another E3 small-object-weighting launcher already holds the runtime lock." >&2
  exit 23
fi
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] E3 launcher started"
if [[ -e "$STOP_SENTINEL" ]]; then
  echo "E3 stop sentinel exists: $STOP_SENTINEL" >&2
  exit 24
fi

"$PYTHON" "$SETUP_ROOT/evaluate_e3_coco_areas.py" --self-test
"$PYTHON" "$SETUP_ROOT/train_e3_small_object_weighting.py" --mode queue

if [[ -e "$STOP_SENTINEL" ]]; then
  echo "E3 stop sentinel appeared before final evaluation: $STOP_SENTINEL" >&2
  exit 25
fi

"$PYTHON" "$SETUP_ROOT/evaluate_e3_coco_areas.py" \
  --run-dir "$RUN_DIR" \
  --data "$SETUP_ROOT/datasets/seadronessee_source_disjoint_v1.yaml" \
  --output "$OUTPUT" \
  --imgsz 960 \
  --batch 8 \
  --device 0

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] E3 training and area evaluation completed"
