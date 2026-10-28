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
"""Tan 算子独立 golden：raw、ACLNN、Torch、TensorFlow 通路。

口径说明:
- int32 输入按 fp32 计算链求值（aclnn 层 Cast→FP32→tan），输出按出参 dtype 回写；
- numpy↔torch 转换自包含（bf16 走 int16 位视图链），不依赖测试框架；
- 容差覆盖 AI Core 契约 dtype（fp16/bf16/fp32 三方比对 + int32 精确相等）；
  f64/complex64 出参由 TTK 默认判据处理（浮点族 mix_tolerance 回落档、
  complex isclose rtol=1e-4）。
"""

__spec__ = {
    "tan": "TanKernelSpec",
    "Tan": "TanKernelSpec",
    "aclnnTan": "TanAclnnSpec",
    "aclnnInplaceTan": "TanAclnnInplaceSpec",
    "torch.tan": "TanTorchSpec",
    "tf.math.tan": "TanTensorFlowSpec",
}

# 容差表覆盖 AI Core 契约 dtype（tan_def.cpp 声明 fp16/fp32/bf16/int32）：
# fp16/bf16/fp32 走 cross_check 三方比对（TTK resolve SMALL_VALUE 表仅支持此
# 三档），int32 精确整数语义 → binary_equal。f64/complex64 出参不特判，由 TTK
# dtype 轴默认判据处理：浮点族表外 dtype 回落 mix_tolerance fp32 档，complex
# 强制 isclose（rtol 1e-4）。
_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "int32": {"standard": "binary_equal"},
}

_L1 = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def _numpy_to_torch_tensor(np_array):
    """numpy → torch 转换（自包含，不依赖测试框架）。

    覆盖本算子涉及 dtype：fp16/fp32/f64/bf16/int32/c64/c128——bf16 经
    int16 位视图链还原 torch.bfloat16，其余直接 from_numpy；fp8/fp4 输入
    非本算子形态，显式不支持（complex32 在 numpy 侧以 float16 存储并由
    调用方标志区分，本算子无该形态，float16 一律按真实 fp16 处理）。
    """
    import numpy as np
    import torch

    if np_array is None:
        return None
    name = np_array.dtype.name
    if "bfloat16" in name:
        return torch.from_numpy(np_array.view(dtype=np.int16)).view(torch.bfloat16)
    if "float8" in name or "float4" in name or "int4" in name:
        raise RuntimeError(
            f"Dtype [{name}] is not supported by Tan golden (no fp8/fp4 input form)."
        )
    return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor):
    """torch → numpy 转换（自包含，不依赖测试框架），与 _numpy_to_torch_tensor 对偶。

    conj 张量先 resolve 再转；bf16 经 int16 位视图回 numpy 侧 bfloat16 表示
    （ml_dtypes 优先、缺失回落 tensorflow，均惰性导入）；fp8/fp4/complex32
    非本算子形态，显式不支持；其余直接 .numpy()。
    """
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    if torch_tensor.is_conj():
        torch_tensor = torch_tensor.resolve_conj()
    dtype_str = str(torch_tensor.dtype)
    if torch_tensor.dtype == torch.bfloat16:
        try:
            from ml_dtypes import bfloat16 as np_bfloat16
        except ModuleNotFoundError:
            try:
                from tensorflow import bfloat16 as tf_bfloat16

                np_bfloat16 = tf_bfloat16.as_numpy_dtype
            except ModuleNotFoundError:
                raise RuntimeError(
                    "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                    "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`"
                ) from None
        return torch_tensor.view(torch.int16).numpy().view(dtype=np_bfloat16)
    if (
        "complex32" in dtype_str
        or "float8" in dtype_str
        or "float4" in dtype_str
        or "int4" in dtype_str
    ):
        raise RuntimeError(
            f"Dtype [{dtype_str}] is not supported by Tan golden (no fp8/fp4/complex32 form)."
        )
    return torch_tensor.numpy()


def _compute(x):
    import numpy as np
    import torch

    is_torch = isinstance(x, torch.Tensor)
    if not is_torch and not isinstance(x, np.ndarray):
        x = np.asarray(x.numpy())
    tensor = x if is_torch else _numpy_to_torch_tensor(x)
    if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
        # 整型输入按 fp32 计算链口径显式提升（spec extreme_inputs
        # "INT32 → Cast→FP32→tan"；与 torch.tan 整型隐式提升 bit-exact）。
        tensor = tensor.to(torch.float32)
    result = torch.tan(tensor)
    return result if is_torch else _torch_to_numpy_tensor(result.detach().cpu())


def tan_golden(x, **kwargs):
    del kwargs
    return [_compute(x)]


def aclnn_tan_golden(self, out=None, **kwargs):
    del kwargs
    result = _compute(self)
    if out is not None:
        result = result.reshape(out.shape).to(out.dtype)
    return [result]


def aclnn_inplace_tan_golden(inputRef, **kwargs):
    """CPU golden following aclnnInplaceTanGetWorkspaceSize(inputRef, ...)."""
    del kwargs
    return [_compute(inputRef)]


def torch_tan_golden(input, out=None, **kwargs):
    del kwargs
    result = _compute(input)
    if out is not None:
        result = result.reshape(out.shape).to(out.dtype)
    return [result]


def _get_torch_input(args, kwargs):
    """从位置参数中取第一个 torch.Tensor, 否则按别名取, 均无则报错"""
    import torch

    x = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if x is not None:
        return x
    for name in ("x", "self", "input", "input_x"):
        x = kwargs.get(name)
        if x is not None:
            return x
    raise ValueError(
        "Missing tensor input; expected positional Tensor or one of: x, self, input, input_x"
    )


class _TorchTan:
    """torch 三方标杆。

    __call__ 兼容实例调用与类绑定调用两种形态（TTK 远端按类绑定调用
    third_party 时张量会落入 instance 槽），instance 槽为 Tensor 时采纳为输入。
    """

    def __call__(instance=None, *args, **kwargs):
        import torch

        if isinstance(instance, torch.Tensor):
            args = (instance, *args)
        x = _get_torch_input(args, kwargs)
        return [torch.tan(x)]


class _TensorFlowTan:
    """TensorFlow 三方标杆（类绑定兼容同 _TorchTan）。"""

    def __call__(instance=None, *args, **kwargs):
        import tensorflow as tf

        if instance is not None and tf.is_tensor(instance):
            args = (instance, *args)
        x = next((arg for arg in args if tf.is_tensor(arg)), None)
        if x is None:
            for name in ("x", "self", "input", "input_x"):
                x = kwargs.get(name)
                if x is not None:
                    break
        if x is None:
            raise ValueError(
                "Missing tensor input; expected positional Tensor or one of: x, self, input, input_x"
            )
        return [tf.math.tan(x)]


class _LazyTanGraph:
    """torch_graph 惰性描述符。

    首次类属性访问时才 import torch 并创建 _TanGraph(torch.nn.Module)。
    """

    def __get__(self, obj, objtype=None):
        import torch

        class _TanGraph(torch.nn.Module):
            def forward(self, input, **kwargs):
                del kwargs
                return torch.tan(input)

        setattr(objtype, "torch_graph", _TanGraph)
        return _TanGraph


class TanKernelSpec:
    golden = staticmethod(tan_golden)
    third_party = {
        "torch": _TorchTan,
        "tf": _TensorFlowTan,
    }
    tolerance = _L2


class TanAclnnSpec:
    golden = staticmethod(aclnn_tan_golden)
    third_party = {"torch": _TorchTan}
    tolerance = _L2


class TanAclnnInplaceSpec:
    """aclnnInplaceTan 通路 spec（显式注册，消除 TTK torch.tan 回退）。

    golden 遵循 aclnnInplaceTanGetWorkspaceSize(inputRef) 签名；in-place 与
    非 in-place 共用同一 Compute 数值链（op_api/aclnn_tan.cpp），tolerance 同 _L2。
    """

    golden = staticmethod(aclnn_inplace_tan_golden)
    third_party = {"torch": _TorchTan}
    tolerance = _L2


class TanTorchSpec:
    golden = staticmethod(torch_tan_golden)
    third_party = {"torch": "torch.tan"}
    torch_graph = _LazyTanGraph()
    tolerance = _L2


class TanTensorFlowSpec:
    golden = staticmethod(tan_golden)
    third_party = {"tf": "tf.math.tan"}
    tolerance = _L1
