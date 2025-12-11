#!/usr/bin/env python
"""Convert exported ONNX to TensorRT INT8 engine using mmdeploy.apis.tensorrt."""
import argparse

from mmdeploy.apis.tensorrt import onnx2tensorrt


def parse_args():
    parser = argparse.ArgumentParser(description='Build TensorRT INT8 engine for SECOND')
    parser.add_argument(
        '--deploy-cfg',
        default='tanyc/SECOND_qat/deploy_trt_int8.py',
        help='Path to deploy config')
    parser.add_argument(
        '--onnx',
        default='work_dirs/SECOND_qat/mmdeploy/second_trt_int8/end2end.onnx',
        help='Path to ONNX model')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/SECOND_qat/mmdeploy/second_trt_int8',
        help='Working directory (should contain calib file)')
    parser.add_argument(
        '--save-file',
        default='end2end.engine',
        help='Engine filename to save (basename, inside work-dir)')
    parser.add_argument(
        '--model-id', type=int, default=0, help='Model index in deploy config')
    parser.add_argument(
        '--device', default='cuda:0', help='CUDA device, e.g. cuda:0')
    return parser.parse_args()


def main():
    try:
        import pycuda.autoinit  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            '[ERROR] pycuda is required for TensorRT conversion. Install it in your openmmlab env, '
            'e.g. `pip install pycuda` (ensure CUDA toolkit matches).') from e

    args = parse_args()
    onnx2tensorrt(
        work_dir=args.work_dir,
        save_file=args.save_file,
        model_id=args.model_id,
        deploy_cfg=args.deploy_cfg,
        onnx_model=args.onnx,
        device=args.device)
    print(f'[INFO] TensorRT engine saved to {args.work_dir}/{args.save_file}')


if __name__ == '__main__':
    main()
