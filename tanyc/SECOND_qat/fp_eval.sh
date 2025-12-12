#!/usr/bin/env bash
# Run single-GPU evaluation for SECOND FLOAT experiments.
set -euo pipefail

# Ensure we operate from repository root even if invoked elsewhere.
SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DEFAULT_CONFIG="${SCRIPT_DIR}/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py"
DEFAULT_CKPT="work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth"

# Activate the designated Conda environment if it is available.
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "openmmlab" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate openmmlab
    else
        echo "[ERROR] Conda is not available to activate 'openmmlab'." >&2
        exit 1
    fi
fi

# Allow zero-argument execution by falling back to the known config/ckpt.
CONFIG=${1:-$DEFAULT_CONFIG}
CHECKPOINT=${2:-$DEFAULT_CKPT}
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

VISIBLE_GPUS=${VISIBLE_GPUS:-"5"}

# Embrace randomness: no fixed seeds, enable benchmark autotune, unset deterministic flags.
export PYTHONHASHSEED=${PYTHONHASHSEED:-$RANDOM}
unset CUBLAS_WORKSPACE_CONFIG
export CUDNN_DETERMINISTIC=0
export CUDNN_BENCHMARK=1

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"

export PYTHONPATH="${REPO_DIR}":${PYTHONPATH:-}

cd "$REPO_DIR"

python "${REPO_DIR}/tools/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher none \
    "$@"
