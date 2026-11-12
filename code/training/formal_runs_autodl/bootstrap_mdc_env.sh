#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
ENV_PREFIX="${MDC_ENV_PREFIX:-${DATA_ROOT}/envs/mdc-yolo26}"

mkdir -p \
  "${DATA_ROOT}/envs" \
  "${DATA_ROOT}/logs" \
  "${DATA_ROOT}/.cache/pip" \
  "${DATA_ROOT}/.cache/torch" \
  "${DATA_ROOT}/.cache/matplotlib" \
  "${DATA_ROOT}/.config/Ultralytics"

export PIP_CACHE_DIR="${DATA_ROOT}/.cache/pip"
export TORCH_HOME="${DATA_ROOT}/.cache/torch"
export MPLCONFIGDIR="${DATA_ROOT}/.cache/matplotlib"
export YOLO_CONFIG_DIR="${DATA_ROOT}/.config/Ultralytics"
export PYTHONUNBUFFERED=1

if [[ -r /etc/network_turbo ]]; then
  # AutoDL's optional network acceleration only affects this setup process.
  set +u
  source /etc/network_turbo
  set -u
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_EXE}" create \
    --override-channels \
    --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    --prefix "${ENV_PREFIX}" \
    python=3.10 \
    pip \
    -y
fi

PYTHON="${ENV_PREFIX}/bin/python"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
"${PYTHON}" -m pip install \
  --index-url https://pypi.org/simple \
  ultralytics==8.4.124 \
  numpy==2.2.6 \
  opencv-python==4.12.0.88 \
  pandas==2.3.2 \
  scipy==1.15.3 \
  scikit-learn==1.7.1 \
  matplotlib==3.10.5 \
  seaborn==0.13.2

"${PYTHON}" -m pip check
"${PYTHON}" "$(dirname "$0")/healthcheck.py"
