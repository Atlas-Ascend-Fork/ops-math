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
"""TestSpec definitions for Cosh on GEIR/Kernel, ACLNN, Torch Single/Graph, and TensorFlow E2E paths."""

import numpy as np
import torch

__spec__ = {
    "cosh": "CoshKernelTestSpec",
    "aclnnCosh": "AclnnCoshTestSpec",
    "aclnnInplaceCosh": "AclnnInplaceCoshTestSpec",
    "torch.cosh": "TorchCoshTestSpec",
    "tf.math.cosh": "CoshTensorFlowTestSpec",
}


class CoshGraph(torch.nn.Module):
    """Graph module consumed by TTK E2E torch.compile modes."""

    def forward(self, input):
        return torch.cosh(input)


def _as_numpy(x):
    """Torch/numpy -> numpy，供 CPU golden 统一计算。"""
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            # numpy 无原生 bf16，torch 也不支持 bf16 张量直接 .numpy()
            # （TypeError: Got unsupported ScalarType BFloat16）。
            # 经 fp32 无损中转（bf16->fp32 精确），再落到 ml_dtypes.bfloat16，
            # 保持 bf16 语义供 _cosh_numpy 统一按 fp32 计算并按原 dtype 回写。
            import ml_dtypes

            return x.detach().cpu().float().numpy().astype(ml_dtypes.bfloat16)
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _cosh_numpy(arr):
    """numpy cosh，保持原 dtype；bf16/fp16 无原生 numpy loop，抬 fp32 算完再转回。

    CPU golden 用 numpy(libm) 而非 torch.cosh：torch 的向量化 float64 cosh 在
    x>~709.78 就提前 exp 溢出（真实 cosh 溢出阈值 710.4758），大数组会给出错误的
    inf；scalar/0-dim/小数组与 numpy 均正确。
    """
    name = getattr(arr.dtype, "name", str(arr.dtype))
    if name in ("bfloat16", "float16"):
        return np.cosh(arr.astype(np.float32)).astype(arr.dtype)
    return np.cosh(arr)


def cosh_golden(x, **kwargs):
    """ACLNN inplace / Kernel / GEIR：单输入，numpy 计算。"""
    return [_cosh_numpy(_as_numpy(x))]


def aclnn_cosh_golden(self, out, **kwargs):
    """CPU golden following aclnnCoshGetWorkspaceSize(self, out, ...)."""
    result = _cosh_numpy(_as_numpy(self))
    if out is not None:
        result = result.astype(_as_numpy(out).dtype)
    return [result]


def torch_cosh_golden(input, **kwargs):
    """E2E torch single/graph golden（返回 torch 张量）。"""
    out = _cosh_numpy(_as_numpy(input))
    if getattr(out.dtype, "name", "") == "bfloat16":
        # torch.from_numpy 不支持 ml_dtypes.bfloat16，经 fp32 无损中转回 torch.bfloat16。
        return [torch.from_numpy(out.astype(np.float32)).to(torch.bfloat16)]
    return [torch.from_numpy(out)]


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


class _TensorFlowCosh:
    """TensorFlow 三方标杆：tf.math.cosh，原生 dtype 直接计算（不升精度）。"""

    def __call__(self, *args, **kwargs):
        import tensorflow as tf

        return [tf.math.cosh(_get_tf_input(args, kwargs))]


class CoshKernelTestSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name``."""

    golden = staticmethod(cosh_golden)
    third_party = {"torch": "torch.cosh", "tf": _TensorFlowCosh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


class AclnnCoshTestSpec:
    """ACLNN out-of-place spec registered by CSV ``api_name``."""

    golden = staticmethod(aclnn_cosh_golden)
    third_party = {"torch": "torch.cosh", "tf": _TensorFlowCosh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


class AclnnInplaceCoshTestSpec:
    """ACLNN inplace spec registered by CSV ``api_name``."""

    golden = staticmethod(cosh_golden)
    third_party = {"torch": "torch.cosh", "tf": _TensorFlowCosh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


class TorchCoshTestSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``."""

    golden = staticmethod(torch_cosh_golden)
    third_party = {"torch": "torch.cosh", "tf": _TensorFlowCosh}
    torch_graph = CoshGraph
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


def tensorflow_cosh_golden(x, name=None, **kwargs):
    """CPU golden following tf.math.cosh(x, name=None); independent numpy computation."""
    return [_cosh_numpy(_as_numpy(x))]


class CoshTensorFlowTestSpec:
    """TensorFlow E2E spec registered by CSV ``api_name``."""

    golden = staticmethod(tensorflow_cosh_golden)
    third_party = {"tf": "tf.math.cosh"}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
