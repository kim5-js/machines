#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
ENV_PREFIX="${LEGACY_ENV_PREFIX:-${DATA_ROOT}/envs/ring3-legacy}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${DATA_ROOT}/envs" "${DATA_ROOT}/.cache/pip"
export PIP_CACHE_DIR="${DATA_ROOT}/.cache/pip"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib-legacy"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics-legacy"
export PYTHONUNBUFFERED=1

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_EXE}" create \
    --override-channels \
    --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    --prefix "${ENV_PREFIX}" \
    python=3.10.8 \
    pip \
    -y
fi

PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
"${PYTHON}" -m pip install \
  --index-url https://pypi.org/simple \
  --requirement "${SCRIPT_DIR}/legacy_requirements.txt"
"${PYTHON}" -m pip check
"${PYTHON}" "${SCRIPT_DIR}/healthcheck.py"

