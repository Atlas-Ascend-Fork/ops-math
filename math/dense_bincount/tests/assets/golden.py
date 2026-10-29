# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import ClassVar

import numpy as np

__spec__ = {
    "dense_bincount": "DenseBincountKernelSpec",
    "DenseBincount": "DenseBincountKernelSpec",
    "tf.raw_ops.DenseBincount": "DenseBincountTfSpec",
}

_TOLERANCE = {
    # Weighted bins may sum hundreds of float32 values in a different legal
    # order on NPU and GPU. Keep L1 relative-error limits and allow the
    # corresponding RMSE ratio for that reduction-only difference.
    "float32": {"standard": "cross_check", "level": "L1", "rmse_ratio": 5.0},
}

_E2E_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1", "rmse_ratio": 5.0},
}

# TTK resolves tolerance from output dtype, not input dtype or attributes. The
# 950 definition has one FP32 output, so weighted cases use cross_check here to
# retain the GPU leg. Run binary_output/empty-weight counting cases in an
# additional CLI --compare binary_equal round for exact two-leg validation;
# TestSpec has no per-attribute tolerance routing.


def _bool_attr(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _to_torch(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "numpy"):
        value = value.numpy()
    return torch.from_numpy(np.ascontiguousarray(value))


def _cpu_compute_tensor(value):
    import torch

    tensor = _to_torch(value)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


def _prepare_indices(input_tensor, size_tensor):
    import torch

    size = int(size_tensor.reshape(-1)[0].item())
    rank = input_tensor.dim()
    if rank not in (1, 2):
        raise ValueError(f"DenseBincount input rank must be 1 or 2, got {rank}")

    rows = 1 if rank == 1 else input_tensor.shape[0]
    cols = input_tensor.numel() if rank == 1 else input_tensor.shape[1]
    bins = input_tensor.reshape(-1).to(torch.int64)
    if size <= 0:
        empty = bins[:0]
        return size, rows, empty, torch.zeros_like(bins, dtype=torch.bool)
    if rank == 1 or cols == 0:
        mapped_rows = torch.zeros_like(bins)
    else:
        mapped_rows = (
            torch.arange(bins.numel(), device=bins.device, dtype=torch.int64) // cols
        )
        negative = bins < 0
        mapped_rows = mapped_rows + torch.where(
            negative, torch.div(bins, size, rounding_mode="floor"), 0
        )
        bins = torch.where(negative, torch.remainder(bins, size), bins)

    valid = (bins >= 0) & (bins < size) & (mapped_rows >= 0) & (mapped_rows < rows)
    linear_indices = mapped_rows[valid] * size + bins[valid]
    return size, rows, linear_indices, valid


def _torch_golden(input_tensor, size_tensor, weights_tensor, binary_output=False):
    import torch

    input_tensor = _to_torch(input_tensor)
    size_tensor = _to_torch(size_tensor)
    weights_tensor = _cpu_compute_tensor(weights_tensor)
    size, rows, linear_indices, valid = _prepare_indices(input_tensor, size_tensor)
    output_shape = (size,) if input_tensor.dim() == 1 else (rows, size)
    if size <= 0:
        return [torch.zeros(output_shape, dtype=weights_tensor.dtype).cpu().numpy()]

    use_binary = _bool_attr(binary_output)
    if use_binary or weights_tensor.numel() == 0:
        values = torch.ones(linear_indices.numel(), dtype=weights_tensor.dtype)
    else:
        values = weights_tensor.reshape(-1)[valid]

    output = (
        torch.bincount(linear_indices, weights=values, minlength=rows * size)
        .to(weights_tensor.dtype)
        .reshape(output_shape)
    )
    if use_binary:
        output = (output != 0).to(output.dtype)
    return [output.detach().cpu().numpy()]


class _DenseBincountCompose:
    def __init__(self, binary_output=False, **kwargs):
        del kwargs
        self.binary_output = _bool_attr(binary_output)

    def __call__(self, input, size, weights, **kwargs):
        import torch

        del kwargs
        bin_count = int(size.reshape(-1)[0].item())
        rows = 1 if input.dim() == 1 else input.shape[0]
        output_shape = (bin_count,) if input.dim() == 1 else (rows, bin_count)
        if bin_count <= 0:
            return [torch.zeros(output_shape, dtype=weights.dtype, device=input.device)]

        bins = input.reshape(-1).to(torch.int64)
        if input.dim() == 1:
            mapped_rows = torch.zeros_like(bins)
        else:
            cols = input.shape[1]
            mapped_rows = (
                torch.arange(bins.numel(), device=bins.device, dtype=torch.int64)
                // cols
            )
            negative = bins < 0
            signed_remainder = torch.fmod(bins, bin_count)
            trunc_quotient = torch.div(bins, bin_count, rounding_mode="trunc")
            row_delta = trunc_quotient - (signed_remainder != 0).to(torch.int64)
            bins = torch.where(
                negative,
                torch.where(
                    signed_remainder == 0,
                    torch.zeros_like(bins),
                    signed_remainder + bin_count,
                ),
                bins,
            )
            mapped_rows = mapped_rows + torch.where(
                negative, row_delta, torch.zeros_like(row_delta)
            )
        valid = (
            (bins >= 0) & (bins < bin_count) & (mapped_rows >= 0) & (mapped_rows < rows)
        )
        linear_indices = mapped_rows[valid] * bin_count + bins[valid]

        if self.binary_output or weights.numel() == 0:
            values = torch.ones(
                linear_indices.numel(), dtype=weights.dtype, device=input.device
            )
        else:
            values = weights.reshape(-1)[valid]
        output = torch.bincount(
            linear_indices, weights=values, minlength=rows * bin_count
        ).to(values.dtype)
        if self.binary_output:
            output = (output != 0).to(output.dtype)
        return [output.reshape(output_shape)]


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(value)


def _tensorflow_golden(input_tensor, size_tensor, weights_tensor, binary_output=False):
    """TensorFlow composition matching the A5 negative-index compatibility rule."""
    import tensorflow as tf

    input_tensor = tf.convert_to_tensor(_to_numpy(input_tensor))
    size_tensor = tf.reshape(tf.convert_to_tensor(_to_numpy(size_tensor)), [])
    weights_tensor = tf.convert_to_tensor(_to_numpy(weights_tensor), dtype=tf.float32)
    size = int(size_tensor.numpy())
    rank = input_tensor.shape.rank
    if rank not in (1, 2):
        raise ValueError(f"DenseBincount input rank must be 1 or 2, got {rank}")
    rows = 1 if rank == 1 else int(input_tensor.shape[0])
    output_shape = (size,) if rank == 1 else (rows, size)
    if size <= 0:
        return [np.zeros(output_shape, dtype=np.float32)]

    bins = tf.cast(tf.reshape(input_tensor, [-1]), tf.int64)
    if rank == 1:
        mapped_rows = tf.zeros_like(bins)
    else:
        cols = int(input_tensor.shape[1])
        mapped_rows = tf.range(tf.size(bins), dtype=tf.int64) // cols
        negative = bins < 0
        mapped_rows += tf.where(negative, tf.math.floordiv(bins, size), 0)
        bins = tf.where(negative, tf.math.floormod(bins, size), bins)

    valid = (bins >= 0) & (bins < size) & (mapped_rows >= 0) & (mapped_rows < rows)
    linear_indices = mapped_rows[valid] * size + bins[valid]
    if _bool_attr(binary_output) or int(tf.size(weights_tensor)) == 0:
        values = tf.ones(tf.size(linear_indices), dtype=tf.float32)
    else:
        values = tf.boolean_mask(tf.reshape(weights_tensor, [-1]), valid)
    output = tf.scatter_nd(tf.expand_dims(linear_indices, 1), values, [rows * size])
    if _bool_attr(binary_output):
        output = tf.cast(output != 0.0, tf.float32)
    return [tf.reshape(output, output_shape).numpy()]


class _DenseBincountTfCompose:
    """TensorFlow competitor used by the TF E2E cross-check provider."""

    def __init__(self, binary_output=False, **kwargs):
        del kwargs
        self.binary_output = _bool_attr(binary_output)

    def __call__(self, input, size, weights, **kwargs):
        del kwargs
        import tensorflow as tf

        return [
            tf.raw_ops.DenseBincount(
                input=tf.convert_to_tensor(_to_numpy(input)),
                size=tf.reshape(tf.convert_to_tensor(_to_numpy(size)), []),
                weights=tf.convert_to_tensor(_to_numpy(weights), dtype=tf.float32),
                binary_output=self.binary_output,
            )
        ]


class DenseBincountKernelSpec:
    def golden(*inputs, **kwargs):
        input_tensor, size_tensor, weights_tensor = inputs
        return _tensorflow_golden(
            input_tensor,
            size_tensor,
            weights_tensor,
            binary_output=kwargs.get("binary_output", False),
        )

    third_party: ClassVar[dict] = {"torch": _DenseBincountCompose}
    tolerance: ClassVar[dict] = _TOLERANCE


class DenseBincountTfSpec:
    """TensorFlow E2E spec for ``tf.raw_ops.DenseBincount``."""

    @staticmethod
    def golden(input, size, weights, binary_output=False, **kwargs):
        del kwargs
        return _DenseBincountTfCompose(binary_output=binary_output)(
            input, size, weights
        )

    third_party: ClassVar[dict] = {"tf": _DenseBincountTfCompose}
    tolerance: ClassVar[dict] = _E2E_TOLERANCE


def dense_bincount_golden(input_tensor, size_tensor, weights_tensor, **kwargs):
    return DenseBincountKernelSpec.golden(
        input_tensor, size_tensor, weights_tensor, **kwargs
    )


# No ACLNN or Torch E2E spec: DenseBincount exposes neither interface. The
# TensorFlow E2E path is registered above from framework/dense_bincount_tf_plugin.cpp.
