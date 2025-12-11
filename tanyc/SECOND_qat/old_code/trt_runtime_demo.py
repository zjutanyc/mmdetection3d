#!/usr/bin/env python
"""Quick runtime check for TensorRT engine via mmdeploy_runtime.VoxelDetection."""
import argparse
import json

from mmdeploy_runtime import VoxelDetection


def parse_args():
    parser = argparse.ArgumentParser(description='Run TensorRT engine on one KITTI sample')
    parser.add_argument(
        '--engine',
        default='work_dirs/SECOND_qat/mmdeploy/second_trt_int8/end2end.engine',
        help='Path to TensorRT engine')
    parser.add_argument(
        '--pcd',
        default='data/kitti/training/velodyne_reduced/000000.bin',
        help='Point cloud (.bin) to run inference on')
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device name for runtime, e.g., cuda or cpu')
    parser.add_argument(
        '--model-id',
        type=int,
        default=0,
        help='Device id when using cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    detector = VoxelDetection(
        model_path=args.engine, model_id=args.model_id, device_name=args.device)
    result = detector(args.pcd)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
