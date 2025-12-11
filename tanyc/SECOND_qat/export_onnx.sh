#!/usr/bin/env bash
# Export SECOND (KITTI Car) to ONNX using mmdeploy torch2onnx.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# Conda env
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "openmmlab" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate openmmlab
    else
        echo "[ERROR] conda not found to activate openmmlab" >&2
        exit 1
    fi
fi

DEPLOY_CFG=${DEPLOY_CFG:-"${SCRIPT_DIR}/deploy_trt_int8.py"}
MODEL_CFG=${MODEL_CFG:-"${SCRIPT_DIR}/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py"}
CKPT=${CKPT:-"${SCRIPT_DIR}/pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth"}
PCD=${PCD:-"${REPO_DIR}/data/kitti/training/velodyne_reduced/000000.bin"}
WORK_DIR=${WORK_DIR:-"${SCRIPT_DIR}"}
SAVE_NAME=${SAVE_NAME:-"end2end.onnx"}
DEVICE=${DEVICE:-"cuda:7"}
export DEPLOY_CFG MODEL_CFG CKPT PCD WORK_DIR SAVE_NAME DEVICE

if [[ ! -f "$DEPLOY_CFG" ]]; then
    echo "[ERROR] deploy cfg not found: $DEPLOY_CFG" >&2
    exit 1
fi
if [[ ! -f "$MODEL_CFG" ]]; then
    echo "[ERROR] model cfg not found: $MODEL_CFG" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "[ERROR] checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$PCD" ]]; then
    echo "[ERROR] sample point cloud not found: $PCD" >&2
    echo "Set PCD env to an existing .bin file" >&2
    exit 1
fi
mkdir -p "$WORK_DIR"

export PYTHONPATH="${REPO_DIR}":${PYTHONPATH:-}

python - <<'PY'
import os
import tempfile
from mmdeploy.apis import torch2onnx
import torch
from mmengine import Config

pcd = os.environ['PCD']
work_dir = os.environ['WORK_DIR']
save_name = os.environ['SAVE_NAME']
deploy_cfg_path = os.environ['DEPLOY_CFG']
model_cfg = os.environ['MODEL_CFG']
ckpt = os.environ['CKPT']
device = os.environ.get('DEVICE', 'cuda:0')

deploy_cfg_to_use = deploy_cfg_path

if device.startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('[ERROR] torch.cuda.is_available() is False. Please run on a machine with CUDA to export ONNX for SECOND/spconv.')

torch2onnx(
    pcd,
    work_dir,
    save_name,
    deploy_cfg_to_use,
    model_cfg,
    model_checkpoint=ckpt,
    device=device)

print(f"[INFO] ONNX exported to {os.path.join(work_dir, save_name)}")
PY
