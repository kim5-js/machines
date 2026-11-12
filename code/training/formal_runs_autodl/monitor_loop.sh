#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MONITOR_PYTHON:-/root/miniconda3/bin/python}"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-60}"
STATE_DIR="${DATA_ROOT}/monitor"

mkdir -p "${STATE_DIR}"

while true; do
  tmp_file="${STATE_DIR}/latest.json.tmp"
  "${PYTHON}" "${SCRIPT_DIR}/status_snapshot.py" --root "${DATA_ROOT}" > "${tmp_file}"
  mv -f "${tmp_file}" "${STATE_DIR}/latest.json"
  "${PYTHON}" "${SCRIPT_DIR}/status_snapshot.py" --root "${DATA_ROOT}" --compact \
    >> "${STATE_DIR}/history.jsonl"
  sleep "${INTERVAL}"
done

