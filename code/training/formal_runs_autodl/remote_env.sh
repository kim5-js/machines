#!/usr/bin/env bash

export AUTODL_DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
export RING3_REMOTE_ROOT="${RING3_REMOTE_ROOT:-${AUTODL_DATA_ROOT}/ring3}"
export MDC_ENV_PREFIX="${MDC_ENV_PREFIX:-${AUTODL_DATA_ROOT}/envs/mdc-yolo26}"
export AUTODL_WEIGHTS_ROOT="${AUTODL_WEIGHTS_ROOT:-${AUTODL_DATA_ROOT}/weights}"
export AUTODL_RUNS_ROOT="${AUTODL_RUNS_ROOT:-${AUTODL_DATA_ROOT}/runs}"
export PIP_CACHE_DIR="${AUTODL_DATA_ROOT}/.cache/pip"
export TORCH_HOME="${AUTODL_DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${AUTODL_DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${AUTODL_DATA_ROOT}/.config/Ultralytics"
export PATH="${MDC_ENV_PREFIX}/bin:${PATH}"
