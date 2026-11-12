#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Scale TestSpec: kernel/GEIR and ACLNN, using PyTorch mul/add composition.

CPU golden computes in the supplied dtype; TTK owns reference promotion.
The GPU reference widens FP16/BF16 to FP32 and narrows the final output,
matching the kernel's computation dtype. Neither path uses NumPy arithmetic.
Shape/attribute validation belongs to the operator. The reference only places
scale/bias on the requested axes; it does not duplicate device admission checks.
"""

import torch

__spec__ = {"scale": "ScaleKernelSpec", "aclnnScale": "ScaleAclnnSpec"}

_L0 = {
    dtype: {"standard": "cross_check", "level": "L0"}
    for dtype in ("float16", "bfloat16", "float32")
}


def _tensor(value):
    """Convert the kernel's NumPy input without changing values or dtype."""
    if value is None or isinstance(value, torch.Tensor):
        return value
    if value.dtype.name == "bfloat16":
        return torch.from_numpy(value.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(value)


def _broadcast_shape(x, scale, axis, num_axes, scale_from_blob):
    """Place coefficients on x's axes without imposing device input limits."""
    rank = x.ndim
    axis = axis + rank if axis < 0 else axis
    shape = [1] * rank
    if scale.numel() == 1 and (not scale_from_blob or num_axes == 0):
        return shape
    # Full-rank descriptors are already aligned, including descriptors expanded
    # by GE before the runtime Tiling callback.
    if scale.ndim == rank:
        return list(scale.shape)
    shape[axis : axis + scale.ndim] = scale.shape
    return shape


def _scale_compute(x, scale, bias, axis, num_axes, scale_from_blob):
    """Broadcast, then multiply and optionally add, in the supplied dtype."""
    shape = _broadcast_shape(x, scale, axis, num_axes, scale_from_blob)
    result = torch.mul(x, scale.reshape(shape))
    if bias is not None and bias.numel() != 0:
        result = torch.add(result, bias.reshape(shape))
    return result


def _golden(x, scale, bias, axis, num_axes, scale_from_blob):
    x_ref, scale_ref, bias_ref = _tensor(x), _tensor(scale), _tensor(bias)
    result = _scale_compute(
        x_ref,
        scale_ref,
        bias_ref,
        axis,
        num_axes,
        scale_from_blob,
    )
    if isinstance(x, torch.Tensor):
        return [result]
    result = result.detach().cpu()
    if result.dtype == torch.bfloat16:
        return [result.view(torch.int16).numpy().view(x.dtype)]
    return [result.numpy()]


def scale_golden(
    x, scale, bias=None, axis=1, num_axes=1, scale_from_blob=True, **kwargs
):
    return _golden(x, scale, bias, axis, num_axes, scale_from_blob)


def aclnn_scale_golden(
    x, scale, bias=None, axis=1, numAxes=1, scaleFromBlob=True, out=None, **kwargs
):
    # The output placeholder must not narrow the promoted CPU reference.
    return _golden(x, scale, bias, axis, numAxes, scaleFromBlob)


def _third_party(x, scale, bias, axis, num_axes, scale_from_blob):
    output_dtype = x.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        x, scale = x.float(), scale.float()
        bias = bias.float() if bias is not None else None
    result = _scale_compute(x, scale, bias, axis, num_axes, scale_from_blob)
    return [result.to(output_dtype)]


class _TorchScaleCompose:
    def __call__(
        self, x, scale, bias=None, axis=1, num_axes=1, scale_from_blob=True, **kwargs
    ):
        return _third_party(x, scale, bias, axis, num_axes, scale_from_blob)


class _AclnnScaleCompose:
    def __call__(
        self, x, scale, bias=None, axis=1, numAxes=1, scaleFromBlob=True, **kwargs
    ):
        return _third_party(x, scale, bias, axis, numAxes, scaleFromBlob)


class ScaleKernelSpec:
    golden = staticmethod(scale_golden)
    third_party = {"torch": _TorchScaleCompose}
    tolerance = _L0


class ScaleAclnnSpec:
    golden = staticmethod(aclnn_scale_golden)
    third_party = {"torch": _AclnnScaleCompose}
    tolerance = _L0
