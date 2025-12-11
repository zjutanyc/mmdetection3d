#!/usr/bin/env python
"""PTQ/QAT bootstrap for SECOND SparseEncoder (traveller59/spconv only)."""

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
    parser = argparse.ArgumentParser(description="Quantize SECOND SparseEncoder (PTQ/QAT prep)")
    parser.add_argument("--config", required=True, help="Path to SECOND config file")
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth",
        help="FP checkpoint under work_dirs (used as initialization)",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/SECOND_qat",
        help="Where to dump quantized checkpoint (defaults to config work_dir)",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=200,
        help="Number of training batches for histogram calibration",
    )
    parser.add_argument(
        "--save-name",
        default="sparse_encoder_ptq.pth",
        help="Filename for quantized checkpoint inside work-dir",
    )
    return parser.parse_args()


def build_calib_dataloader(cfg: Config):
    """Instantiate the train dataloader for calibration."""
    dataloader_cfg = deepcopy(cfg.train_dataloader)
    dataset_cfg = dataloader_cfg.pop("dataset")
    dataset = DATASETS.build(dataset_cfg)
    dataloader_cfg["dataset"] = dataset
    dataloader_cfg.setdefault("shuffle", False)
    # mmengine Runner.build_dataloader doesn't accept `seed` kwarg
    dataloader_cfg.pop("seed", None)
    return Runner.build_dataloader(dataloader_cfg)


def main():
    args = parse_args()
    initialize()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = os.path.join(
            "./work_dirs", os.path.splitext(os.path.basename(args.config))[0]
        )

    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda().eval()

    # Quantize SparseEncoder only
    quantize_sparse_encoder(model.middle_encoder)
    set_quantizer_fast(model.middle_encoder)
    model.eval()

    calib_loader = build_calib_dataloader(cfg)

    @torch.no_grad()
    def calib_step(data_batch):
        # Use the model's data_preprocessor to voxelize then run voxel_encoder + SparseEncoder only.
        processed = model.data_preprocessor(data_batch, False)
        voxels = processed["inputs"]["voxels"]
        voxel_features = model.voxel_encoder(
            voxels["voxels"], voxels["num_points"], voxels["coors"]
        )
        batch_size = voxels["coors"][-1, 0].item() + 1
        model.middle_encoder(voxel_features, voxels["coors"], batch_size)

    print(f"🔥 start calibrate SparseEncoder with {args.calib_batches} batches 🔥")
    calibrate_model(model.middle_encoder, calib_loader, calib_step, num_batch=args.calib_batches)
    print_quantizer_status(model.middle_encoder)

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(cfg.work_dir, args.save_name)
    save_checkpoint(model, save_path)
    print(f"[INFO] Quantized SparseEncoder checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
