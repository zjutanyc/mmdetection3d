#!/usr/bin/env bash
# Evaluate SECOND PTQ (quantized SparseEncoder) on a single node.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DEFAULT_CONFIG="${SCRIPT_DIR}/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py"
DEFAULT_CKPT="work_dirs/SECOND_qat/sparse_encoder_ptq.pth"

# Activate openmmlab env if available
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "openmmlab" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate openmmlab
    else
        echo "[ERROR] Conda env 'openmmlab' not available." >&2
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"5"}

CONFIG=$DEFAULT_CONFIG
CHECKPOINT=$DEFAULT_CKPT
if [[ $# -gt 0 ]]; then
    CONFIG=$1
    shift
fi
if [[ $# -gt 0 ]]; then
    CHECKPOINT=$1
    shift
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

export PYTHONPATH="${REPO_DIR}":${PYTHONPATH:-}
cd "$REPO_DIR"

python "${REPO_DIR}/tanyc/SECOND_qat/CUDA-SECOND/qat/tools/second_ptq_eval.py" \
    "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    "$@"
