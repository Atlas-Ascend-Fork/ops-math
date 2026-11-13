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
"""TestSpec definitions for Asinh on GEIR/Kernel, ACLNN, Torch, and TensorFlow E2E paths."""

import numpy
import torch

__spec__ = {
    "asinh": "AsinhKernelTestSpec",
    "aclnnAsinh": "AclnnAsinhTestSpec",
    "aclnnInplaceAsinh": "AclnnInplaceAsinhTestSpec",
    "torch.asinh": "TorchAsinhTestSpec",
    "tf.math.asinh": "AsinhTensorFlowTestSpec",
}


def _torch_to_kernel_numpy(value, dtype):
    if value.dtype == torch.bfloat16:
        return value.contiguous().view(torch.int16).numpy().view(dtype)
    return value.numpy()


def _cpu_asinh(x):
    """torch CPU asinh lacks Half/BFloat16 kernels; promote to float32 and round back."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.asinh(x.float()).to(x.dtype)
    return torch.asinh(x)


def asinh_kernel_golden(x, **kwargs):
    """CPU golden for the raw op; ``x`` follows the OpDef input order."""
    if x.dtype.name == "bfloat16":
        x_tensor = torch.from_numpy(numpy.ascontiguousarray(x).view(numpy.int16)).view(
            torch.bfloat16
        )
    else:
        x_tensor = torch.from_numpy(x)
    return [_torch_to_kernel_numpy(_cpu_asinh(x_tensor), x.dtype)]


def aclnn_asinh_golden(input, out, **kwargs):
    """CPU golden following aclnnAsinhGetWorkspaceSize(input, out, ...)."""
    return [_cpu_asinh(input).to(out.dtype)]


def aclnn_inplace_asinh_golden(inputRef, **kwargs):
    """CPU golden following aclnnInplaceAsinhGetWorkspaceSize(inputRef, ...)."""
    return [_cpu_asinh(inputRef)]


def torch_asinh_golden(input, *, out=None, **kwargs):
    """CPU golden following torch.asinh(input, *, out=None)."""
    result = _cpu_asinh(input)
    if out is not None:
        out.copy_(result)
        return [out]
    return [result]


class AsinhGraph(torch.nn.Module):
    """Graph module consumed by TTK E2E torch.compile modes."""

    def forward(self, input):
        return _cpu_asinh(input)


def _get_tf_input(args, kwargs):
    """从位置参数中取第一个 tf.Tensor，否则按别名取，均无则报错。"""
    import tensorflow as tf

    x = next((arg for arg in args if isinstance(arg, tf.Tensor)), None)
    if x is not None:
        return x
    for name in ("x", "self", "input", "inputRef"):
        x = kwargs.get(name)
        if x is not None:
            return x
    raise ValueError(
        "Missing tensor input; expected positional Tensor or one of: x, self, input, inputRef"
    )


class _TensorFlowAsinh:
    """TensorFlow 三方标杆：tf.math.asinh，原生 dtype 直接计算（不升精度）。"""

    def __call__(self, *args, **kwargs):
        import tensorflow as tf

        return [tf.math.asinh(_get_tf_input(args, kwargs))]


class AsinhKernelTestSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name``."""

    golden = asinh_kernel_golden
    third_party = {"torch": "torch.asinh", "tf": _TensorFlowAsinh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


class AclnnAsinhTestSpec:
    """ACLNN spec registered by CSV ``api_name``."""

    golden = aclnn_asinh_golden
    third_party = {"torch": "torch.asinh", "tf": _TensorFlowAsinh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float64": {"standard": "cross_check", "level": "L2"},
    }


class AclnnInplaceAsinhTestSpec:
    """Inplace ACLNN spec registered by CSV ``api_name``."""

    golden = aclnn_inplace_asinh_golden
    third_party = {"torch": "torch.asinh", "tf": _TensorFlowAsinh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "float64": {"standard": "cross_check", "level": "L2"},
    }


class TorchAsinhTestSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``."""

    golden = torch_asinh_golden
    third_party = {"torch": "torch.asinh", "tf": _TensorFlowAsinh}
    torch_graph = AsinhGraph
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


def tensorflow_asinh_golden(x, name=None, **kwargs):
    """CPU golden following tf.math.asinh(x, name=None); independent torch computation."""
    import numpy as np

    arr = np.asarray(x)
    if arr.dtype.name == "bfloat16":
        tensor = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
        return [_cpu_asinh(tensor).view(torch.int16).numpy().view(arr.dtype)]
    return [_cpu_asinh(torch.from_numpy(arr)).numpy()]


class AsinhTensorFlowTestSpec:
    """TensorFlow E2E spec registered by CSV ``api_name``."""

    golden = staticmethod(tensorflow_asinh_golden)
    third_party = {"tf": "tf.math.asinh"}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
