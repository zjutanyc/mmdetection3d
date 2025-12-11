#!/usr/bin/env bash
# Run distributed evaluation for SECOND FLOAT experiments using GPUs 4,5,6,7.
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
if [[ $# -gt 0 ]]; then
    CONFIG=$1
    shift
else
    CONFIG=$DEFAULT_CONFIG
fi

if [[ $# -gt 0 ]]; then
    CHECKPOINT=$1
    shift
else
    CHECKPOINT=$DEFAULT_CKPT
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

VISIBLE_GPUS=${VISIBLE_GPUS:-"5"}
IFS=',' read -ra GPU_IDS <<< "$VISIBLE_GPUS"
GPUS=${GPUS:-${#GPU_IDS[@]}}

export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export PYTHONPATH="${REPO_DIR}":${PYTHONPATH:-}

cd "$REPO_DIR"

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "${REPO_DIR}/tools/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "$@"
