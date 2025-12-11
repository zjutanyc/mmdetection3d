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
CKPT=${CKPT:-"${REPO_DIR}/work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth"}
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
from pathlib import Path

import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules

pcd = os.environ['PCD']
work_dir = os.environ['WORK_DIR']
save_name = os.environ['SAVE_NAME']
deploy_cfg_path = os.environ['DEPLOY_CFG']
model_cfg = os.environ['MODEL_CFG']
ckpt = os.environ['CKPT']
device = os.environ.get('DEVICE', 'cuda:0')

if device.startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('[ERROR] torch.cuda.is_available() is False. Please run on a machine with CUDA to export ONNX for SECOND/spconv.')

register_all_modules()
deploy_cfg = Config.fromfile(deploy_cfg_path)
ir_cfg = deploy_cfg.get('ir_config', {})
input_names = ir_cfg.get('input_names', ['voxels', 'num_points', 'coors'])
output_names = ir_cfg.get('output_names', [
    'cls_score0', 'cls_score1',
    'bbox_pred0', 'bbox_pred1',
    'dir_cls_pred0', 'dir_cls_pred1'
])
dynamic_axes = ir_cfg.get('dynamic_axes', {
    'voxels': {0: 'voxels', 1: 'points_per_voxel'},
    'num_points': {0: 'voxels'},
    'coors': {0: 'voxels'},
})
opset = ir_cfg.get('opset_version', 11)

# Build one sample through calib dataset pipeline to get voxelized tensors.
calib_cfg = deploy_cfg.get('calib_config', {})
dataset_cfg = calib_cfg.get('dataset')
if dataset_cfg is None:
    raise SystemExit('[ERROR] calib_config.dataset not found in deploy config.')
dataset_cfg = dataset_cfg.copy()
dataset_cfg.setdefault('test_mode', True)
dataset = DATASETS.build(dataset_cfg)
data_item = dataset[0]
collated = pseudo_collate([data_item])

model = init_model(model_cfg, ckpt, device=device)
model.eval()
processed = model.data_preprocessor(collated, False)
vox = processed['inputs']['voxels']
voxels = vox['voxels']
num_points = vox['num_points']
coors = vox['coors']


class SecondWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, voxels, num_points, coors):
        inputs = {
            'voxels': {
                'voxels': voxels,
                'num_points': num_points,
                'coors': coors
            }
        }
        cls_list, bbox_list, dir_list = self.model(inputs, mode='tensor')
        return (*cls_list, *bbox_list, *dir_list)


wrapper = SecondWrapper(model)
output_path = os.path.join(work_dir, save_name)
Path(work_dir).mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    wrapper,
    (voxels, num_points, coors),
    output_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=opset,
    dynamic_axes=dynamic_axes,
    keep_initializers_as_inputs=True,
    verbose=False)

print(f"[INFO] ONNX exported to {output_path}")
PY
