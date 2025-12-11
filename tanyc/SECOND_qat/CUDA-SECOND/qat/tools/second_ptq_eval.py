#!/usr/bin/env python
"""Run PTQ evaluation for SECOND with quantized SparseEncoder."""

import argparse
import os

import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet3d.utils import register_all_modules

from sparseencoder_quantization import quantize_sparse_encoder


def _load_quant_checkpoint(model, ckpt_path: str):
    """Load checkpoint that may or may not wrap weights in state_dict."""
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    elif hasattr(obj, "state_dict"):
        # Handle checkpoints saved via torch.save(model, ...)
        state_dict = obj.state_dict()
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(obj)}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SECOND PTQ (SparseEncoder only)")
    parser.add_argument("config", help="Config file")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Quantized checkpoint produced by second_sparse_encoder_qat.py",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Work dir for eval logs/outputs (defaults to config work_dir)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, key=value",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = os.path.join(
            "./work_dirs", os.path.splitext(os.path.basename(args.config))[0]
        )

    # Avoid Runner auto-loading weights; we load after quantization.
    cfg.load_from = None
    cfg.resume = False
    cfg.launcher = "none"

    init_default_scope(cfg.get("default_scope", "mmdet3d"))

    runner = Runner.from_cfg(cfg)

    # Patch SparseEncoder to quantized modules then load PTQ weights.
    quantize_sparse_encoder(runner.model.middle_encoder)
    _load_quant_checkpoint(runner.model, args.checkpoint)

    runner.model.eval()
    runner.test()


if __name__ == "__main__":
    main()
