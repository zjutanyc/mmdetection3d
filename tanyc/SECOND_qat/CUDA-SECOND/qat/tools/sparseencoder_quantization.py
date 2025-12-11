# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Adapted for SECOND SparseEncoder (traveller59/spconv) quantization only.
#
# Only SparseEncoder is touched; rest of the model remains FP.

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch_quantization import calib, tensor_quant
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging

from cumm import tensorview as tv
from spconv.core import ConvAlgo
from spconv.pytorch.conv import SparseConvolution, SparseConvTensor
import spconv.pytorch as spconv
from tqdm import tqdm

from mmdet3d.models.layers.sparse_block import SparseBasicBlock, replace_feature


class QuantAdd(nn.Module, _utils.QuantInputMixin):
    """Quantized residual add for SparseBasicBlock."""

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self):
        super().__init__()
        self.init_quantizer(self.default_quant_desc_input)

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.add(self._input_quantizer(lhs), self._input_quantizer(rhs))


class SparseConvolutionQuant(SparseConvolution, _utils.QuantMixin):
    """Quantized spconv module (per-tensor act, per-channel weights)."""

    default_quant_desc_input = QuantDescriptor(num_bits=8, calib_method="histogram")
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: Union[int, List[int], Tuple[int, ...]] = 0,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        subm: bool = False,
        output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
        transposed: bool = False,
        inverse: bool = False,
        indice_key: Optional[str] = None,
        algo: Optional[ConvAlgo] = None,
        fp32_accum: Optional[bool] = None,
        record_voxel_count: bool = False,
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
        act_alpha: float = 0,
        act_beta: float = 0,
        name=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            ndim,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
            subm=subm,
            output_padding=output_padding,
            transposed=transposed,
            inverse=inverse,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            record_voxel_count=record_voxel_count,
            act_type=act_type,
            act_alpha=act_alpha,
            act_beta=act_beta,
        )
        self.init_quantizer(self.default_quant_desc_input, self.default_quant_desc_weight)

    def _quant(self, input: SparseConvTensor, add_input: Optional[SparseConvTensor]):
        def _maybe_replace_features(tensor: SparseConvTensor, feats: torch.Tensor):
            # spconv v2 exposes replace_feature; fallback to direct assignment for safety.
            if hasattr(tensor, "replace_feature"):
                tensor = tensor.replace_feature(feats)
            else:
                tensor._features = feats  # type: ignore[attr-defined]
            return tensor

        if input is not None:
            input = _maybe_replace_features(input, self._input_quantizer(input.features))

        if add_input is not None:
            add_input = _maybe_replace_features(add_input, self._input_quantizer(add_input.features))

        quant_weight = self._weight_quantizer(self.weight) if self.weight is not None else None
        return input, add_input, quant_weight

    def forward(self, input: SparseConvTensor, add_input: Optional[SparseConvTensor] = None):
        input, add_input, quant_weight = self._quant(input, add_input)
        return self._conv_forward(
            self.training,
            input,
            quant_weight,
            self.bias,
            add_input,
            name=self.name,
            sparse_unique_name=self._sparse_unique_name,
            act_type=self.act_type,
            act_alpha=self.act_alpha,
            act_beta=self.act_beta,
        )


def _clone_spconv_to_quant(nninstance: torch.nn.Module) -> SparseConvolutionQuant:
    """Copy a pretrained spconv layer into the quantized wrapper without reinit."""
    quant_instance = SparseConvolutionQuant.__new__(SparseConvolutionQuant)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)
    # Init quantizers on the cloned instance
    SparseConvolutionQuant.__init__(
        quant_instance,
        nninstance.ndim,
        nninstance.in_channels,
        nninstance.out_channels,
        nninstance.kernel_size,
        nninstance.stride,
        nninstance.padding,
        nninstance.dilation,
        nninstance.groups,
        bias=nninstance.bias is not None,
        subm=nninstance.subm,
        output_padding=nninstance.output_padding,
        transposed=nninstance.transposed,
        inverse=nninstance.inverse,
        indice_key=nninstance.indice_key,
        algo=nninstance.algo,
        fp32_accum=getattr(nninstance, "fp32_accum", None),
        record_voxel_count=getattr(nninstance, "record_voxel_count", False),
        act_type=getattr(nninstance, "act_type", tv.gemm.Activation.None_),
        act_alpha=getattr(nninstance, "act_alpha", 0),
        act_beta=getattr(nninstance, "act_beta", 0),
    )
    # Preserve trained weights
    quant_instance.weight = nninstance.weight
    quant_instance.bias = nninstance.bias
    quant_instance.training = nninstance.training
    return quant_instance


def _patch_sparse_basic_block(block: SparseBasicBlock):
    """Insert quantized add and patch the forward to use it."""
    if hasattr(block, "quant_add"):
        return
    block.quant_add = QuantAdd()

    def forward_quant(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x).features

        out = replace_feature(out, self.quant_add(out.features, identity))
        out = replace_feature(out, self.relu(out.features))
        return out

    block.forward = forward_quant.__get__(block, SparseBasicBlock)  # type: ignore[assignment]


def quantize_sparse_encoder(module: torch.nn.Module):
    """Replace spconv layers + residual adds under a SparseEncoder."""

    def replace_module(mod: torch.nn.Module):
        for name, child in list(mod.named_children()):
            replace_module(child)
            if isinstance(child, (spconv.SubMConv3d, spconv.SparseConv3d)):
                mod._modules[name] = _clone_spconv_to_quant(child)
            elif isinstance(child, SparseBasicBlock):
                _patch_sparse_basic_block(child)

    replace_module(module)


def initialize():
    """Set a quiet log level for pytorch-quantization."""
    quant_logging.set_verbosity(quant_logging.ERROR)


def set_quantizer_fast(module: torch.nn.Module):
    """Use torch.histogram calibrator for speed."""
    for _, m in module.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            if isinstance(m._calibrator, calib.HistogramCalibrator):
                m._calibrator._torch_hist = True  # type: ignore[attr-defined]


def _enable_calibration(module: torch.nn.Module, enabled: bool):
    for _, m in module.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            if m._calibrator is not None:
                if enabled:
                    m.disable_quant()
                    m.enable_calib()
                else:
                    m.enable_quant()
                    m.disable_calib()
            else:
                m.disable() if enabled else m.enable()


def calibrate_model(
    module: torch.nn.Module,
    dataloader,
    forward_step: Callable[[object], None],
    num_batch: int = 200,
):
    """Collect stats then load amax for quantizers inside `module`.

    Args:
        module: module containing TensorQuantizers (SparseEncoder for SECOND).
        dataloader: iterable of batches from training data.
        forward_step: callable that runs a forward pass given one batch.
        num_batch: number of batches to use for calibration.
    """

    def compute_amax(target: torch.nn.Module, **kwargs):
        for _, m in target.named_modules():
            if isinstance(m, quant_nn.TensorQuantizer) and m._calibrator is not None:
                if isinstance(m._calibrator, calib.MaxCalibrator):
                    m.load_calib_amax()
                else:
                    m.load_calib_amax(**kwargs)

    _enable_calibration(module, enabled=True)
    module.eval()
    total = num_batch
    try:
        total = min(len(dataloader), num_batch)  # type: ignore[arg-type]
    except Exception:
        pass
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=total, desc="Calibrating")):
            forward_step(data)
            if i + 1 >= num_batch:
                break
    _enable_calibration(module, enabled=False)
    compute_amax(module, method="mse")


class disable_quantization:
    """Context manager to toggle quantizers off (for export/inference prep)."""

    def __init__(self, module: torch.nn.Module):
        self.module = module

    def apply(self, disabled: bool = True):
        for _, m in self.module.named_modules():
            if isinstance(m, quant_nn.TensorQuantizer):
                m._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    """Context manager to ensure quantizers are enabled."""

    def __init__(self, module: torch.nn.Module):
        self.module = module

    def apply(self, enabled: bool = True):
        for _, m in self.module.named_modules():
            if isinstance(m, quant_nn.TensorQuantizer):
                m._disabled = not enabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def print_quantizer_status(module: torch.nn.Module):
    for name, m in module.named_modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            print(f"TensorQuantizer name:{name} disabled:{m._disabled} module:{m}")


def have_quantizer(module: torch.nn.Module) -> bool:
    return any(isinstance(m, quant_nn.TensorQuantizer) for _, m in module.named_modules())
