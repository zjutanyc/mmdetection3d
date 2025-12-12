#!/usr/bin/env python
"""PTQ calibration for SECOND SparseEncoder (traveller59/spconv)."""

import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.runner.checkpoint import save_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

from sparseencoder_quantization import (
    calibrate_model,
    initialize,
    print_quantizer_status,
    quantize_sparse_encoder,
    set_quantizer_fast,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize SECOND SparseEncoder (PTQ)")
    parser.add_argument(
        "--config",
        default="tanyc/SECOND_qat/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py",
        help="Path to SECOND config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth",
        help="FP checkpoint under work_dirs",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/SECOND_qat",
        help="Where to dump quantized checkpoint",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=200,
        help="Number of training batches for histogram calibration",
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="Filename for quantized checkpoint (auto if None)",
    )
    return parser.parse_args()


def build_calib_dataloader(cfg: Config):
    dataloader_cfg = deepcopy(cfg.train_dataloader)
    dataset_cfg = dataloader_cfg.pop("dataset")
    dataset = DATASETS.build(dataset_cfg)
    dataloader_cfg["dataset"] = dataset
    dataloader_cfg.setdefault("shuffle", False)
    dataloader_cfg.pop("seed", None)
    return Runner.build_dataloader(dataloader_cfg)


def main():
    args = parse_args()
    initialize()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir or cfg.get("work_dir", args.work_dir)

    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()

    quantize_sparse_encoder(model.middle_encoder)
    set_quantizer_fast(model.middle_encoder)

    calib_loader = build_calib_dataloader(cfg)

    @torch.no_grad()
    def calib_step(data_batch):
        processed = model.data_preprocessor(data_batch, False)
        vox = processed["inputs"]["voxels"]
        feat = model.voxel_encoder(vox["voxels"], vox["num_points"], vox["coors"])
        batch_size = vox["coors"][-1, 0].item() + 1
        model.middle_encoder(feat, vox["coors"], batch_size)

    print(f"🔥 start calibrate SparseEncoder with {args.calib_batches} batches 🔥")
    calibrate_model(model.middle_encoder, calib_loader, calib_step, num_batch=args.calib_batches)
    print_quantizer_status(model.middle_encoder)

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_name = args.save_name or f"{base_name}_calib{args.calib_batches}.pth"
    save_path = os.path.join("work_dirs/SECOND_qat", save_name)
    save_checkpoint(model, save_path)
    print(f"[INFO] Quantized SparseEncoder checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
