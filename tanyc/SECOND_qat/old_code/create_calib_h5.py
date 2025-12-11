#!/usr/bin/env python
"""Generate TensorRT INT8 calibration HDF5 for SECOND (voxels/num_points/coors).

This script:
1) Reads deploy config (default: deploy_trt_int8.py) to fetch calib_config/dataset.
2) Builds the calibration dataloader.
3) Runs the model data_preprocessor to produce voxelized inputs.
4) Saves HDF5 as expected by mmdeploy TensorRT calibrator:
   calib_data/end2end/{voxels,num_points,coors}/0,1,...

Usage:

python tanyc/SECOND_qat/create_calib_h5.py \
--deploy-cfg tanyc/SECOND_qat/deploy_trt_int8.py \
--model-cfg tanyc/SECOND_qat/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
--checkpoint work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth \
--work-dir work_dirs/SECOND_qat/mmdeploy/second_trt_int8 \
--device cuda:0

"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader

from mmdet3d.apis import init_model
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='Create TRT INT8 calib HDF5')
    parser.add_argument('--deploy-cfg', default='tanyc/SECOND_qat/deploy_trt_int8.py')
    parser.add_argument('--model-cfg', default='configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth')
    parser.add_argument('--work-dir', default='work_dirs/SECOND_qat/mmdeploy/second_trt_int8')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--subset', type=int, default=None, help='Override calib_config.subset')
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules()

    deploy_cfg = Config.fromfile(args.deploy_cfg)
    calib_cfg = deploy_cfg.get('calib_config', {})
    if not calib_cfg.get('create_calib', False):
        raise ValueError('calib_config.create_calib is False/absent in deploy config')

    subset = args.subset if args.subset is not None else calib_cfg.get('subset', 500)
    batch_size = calib_cfg.get('batch_size', 1)
    num_workers = calib_cfg.get('num_workers', 2)
    calib_file = calib_cfg.get('calib_file', 'calib_file.h5')

    dataset_cfg = calib_cfg['dataset']
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg.setdefault('test_mode', True)

    dataset = DATASETS.build(dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pseudo_collate,
        drop_last=False)

    model = init_model(args.model_cfg, args.checkpoint, device=args.device)
    model.eval()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    calib_path = work_dir / calib_file

    with h5py.File(calib_path, 'w') as f:
        calib_group = f.create_group('calib_data')
        end2end_group = calib_group.create_group('end2end')
        vox_group = end2end_group.create_group('voxels')
        np_group = end2end_group.create_group('num_points')
        coors_group = end2end_group.create_group('coors')

        count = 0
        with torch.no_grad():
            for data in dataloader:
                processed = model.data_preprocessor(data, False)
                voxels = processed['inputs']['voxels']
                vox_np = voxels['voxels'].cpu().numpy().astype(np.float32)
                num_points_np = voxels['num_points'].cpu().numpy().astype(np.float32)
                coors_np = voxels['coors'].cpu().numpy().astype(np.float32)

                vox_group.create_dataset(str(count), data=vox_np, compression='gzip', compression_opts=4)
                np_group.create_dataset(str(count), data=num_points_np, compression='gzip', compression_opts=4)
                coors_group.create_dataset(str(count), data=coors_np, compression='gzip', compression_opts=4)

                count += 1
                if subset is not None and count >= subset:
                    break

    print(f'[INFO] Saved calibration data to {calib_path} (num_batches={count}, batch_size={batch_size})')


if __name__ == '__main__':
    main()
