## SECOND SparseEncoder Quantization

This folder adapts the CenterPoint sparse-convolution QAT workflow to the SECOND
model. Only the `SparseEncoder` (traveller59/spconv) is quantized; everything
else stays in FP.

### Dependencies
- [`pytorch-quantization`](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) (`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`)
- spconv from traveller59 (the same one used by SECOND)

### Workflow (PTQ / QAT bootstrap)
1. Make sure your FP checkpoint is in `work_dirs`, e.g.
   `work_dirs/second_hv_secfpn_8xb6-80e_kitti-3d-car/latest.pth`.
2. Run calibration to insert fake-quant nodes for `SparseEncoder` only:
   ```bash
   PYTHONPATH=. python tanyc/SECOND_qat/CUDA-SECOND/qat/tools/second_sparse_encoder_qat.py \
     --config tanyc/SECOND_qat/configs/second_hv_secfpn_8xb6-80e_kitti-3d-car.py \
     --checkpoint work_dirs/SECOND_qat/pretarined/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth \
     --work-dir work_dirs/second_sparse_encoder_qat \
     --calib-batches 200
   ```
   This:
   - replaces spconv ops in `SparseEncoder` with quantized versions,
   - inserts a quantized residual add in every `SparseBasicBlock`,
   - calibrates histogram amax using the train dataloader (voxelized via the model data preprocessor),
   - saves `sparse_encoder_ptq.pth` under the given work-dir.

3. (Optional) For QAT, resume from the saved checkpoint with your usual
   `tools/train.py` run; only the SparseEncoder modules carry quantizers.

### Notes
- Calibration runs the voxel encoder + SparseEncoder only; heads/backbone stay
  untouched.
- Quantizer defaults follow the CenterPoint script (8-bit histogram for
  activations, per-channel weights).
- The scripts avoid touching anything outside `CUDA-SECOND`, and weights stay
  under `work_dirs` as requested.
