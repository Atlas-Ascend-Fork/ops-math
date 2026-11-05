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
"""Addcmul 算子测试规范（TTK TestSpec）。

kernel/GEIR = addcmul（numpy 入/出）：out = input_data + value * (x1 * x2)；
aclnn = aclnnAddcmul / aclnnInplaceAddcmul（torch 入/出）：
out = self + value * (tensor1 * tensor2)；
e2e = torch.addcmul（torch 入/出）。
"""

import torch

__spec__ = {
    "addcmul": "AddcmulTestSpec",
    "aclnnAddcmul": "AclnnAddcmulTestSpec",
    "aclnnInplaceAddcmul": "AclnnInplaceAddcmulTestSpec",
    "torch.addcmul": "TorchAddcmulTestSpec",
}

_DTYPE_STR_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

_INT_DTYPES = {"int8", "uint8", "int16", "int32", "int64"}


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v
    if hasattr(v, "item"):
        return v.item()
    return v


def _to_fp32(t):
    if t.dtype in (
        torch.bfloat16,
        torch.float16,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return t.to(torch.float32)
    return t


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


def _third_party_inputs(
    args,
    kwargs,
    first_names=("self", "selfT", "input_data", "input", "x1"),
    second_names=("tensor1", "x1"),
    third_names=("tensor2", "x2"),
):
    """从三方标杆调用参数中提取三个输入张量（位置优先，kwargs 兜底）。

    addcmul 的 value 是标量，单独经 kwargs 读取。
    位置不足时按候选名从 kwargs 回退；仍缺失时抛含 args/kwargs 的明确异常。
    """

    def _pick(idx, names):
        if len(args) > idx:
            return args[idx]
        for name in names:
            if name in kwargs:
                return kwargs[name]
        return None

    first = _pick(0, first_names)
    second = _pick(1, second_names)
    third = _pick(2, third_names)
    missing = [
        n
        for n, v in (("input_data", first), ("tensor1", second), ("tensor2", third))
        if v is None
    ]
    if missing:
        raise TypeError(
            "addcmul third_party: input tensors %s not provided (args=%r, kwargs=%r)"
            % (missing, args, sorted(kwargs))
        )
    return first, second, third


def _value_from(kwargs, default=1.0):
    return _to_scalar(kwargs.get("value", default))


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：value 属性绑定，三个输入按位置接收（``self`` 为保留名）。"""

    def __init__(self, *, value=1.0, **kwargs):
        self.value = value

    def __call__(self, *args, **kwargs):
        a, b, c = _third_party_inputs(args, kwargs)
        v = kwargs.get("value", self.value)
        return [torch.addcmul(a, b, c, value=_to_scalar(v))]


class AddcmulTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(input_data, x1, x2, value, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        output_dtypes = tuple(kwargs.get("output_dtypes") or ())
        out_dtype = str(output_dtypes[0]).lower() if output_dtypes else None

        t_data = numpy_to_torch_tensor(input_data)
        t_x1 = numpy_to_torch_tensor(x1)
        t_x2 = numpy_to_torch_tensor(x2)
        v = _to_scalar(numpy_to_torch_tensor(value))

        if out_dtype in _INT_DTYPES:
            res = torch.addcmul(t_data, t_x1, t_x2, value=int(round(v)))
        else:
            target_dtype = _DTYPE_STR_TO_TORCH.get(out_dtype, t_data.dtype)
            t_data = _to_fp32(t_data)
            t_x1 = _to_fp32(t_x1)
            t_x2 = _to_fp32(t_x2)
            res = torch.addcmul(t_data, t_x1, t_x2, value=v)
            res = res.to(target_dtype)

        return [torch_to_numpy_tensor(res)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnAddcmulTestSpec:
    """aclnnAddcmul 流程：out = self + value * (tensor1 * tensor2)。"""

    def golden(self, tensor1, tensor2, value, out, **kwargs):
        """参数跟随 aclnnAddcmul：self、tensor1、tensor2 张量，value 标量，out 张量。"""
        tensor_dtypes = tuple(kwargs.get("tensor_dtypes") or ())
        out_dtype = tensor_dtypes[-1] if tensor_dtypes else None
        v = _to_scalar(value)
        if out_dtype in _INT_DTYPES:
            res = torch.addcmul(self, tensor1, tensor2, value=int(round(v)))
        else:
            res = torch.addcmul(
                _to_fp32(self), _to_fp32(tensor1), _to_fp32(tensor2), value=v
            )
        return [_cast_to_out(res, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceAddcmulTestSpec:
    """aclnnInplaceAddcmul 流程：selfRef += value * (tensor1 * tensor2)。"""

    def golden(selfRef, tensor1, tensor2, value, **kwargs):
        """参数跟随 aclnnInplaceAddcmul：selfRef、tensor1、tensor2 张量，value 标量。"""
        tensor_dtypes = tuple(kwargs.get("tensor_dtypes") or ())
        out_dtype = tensor_dtypes[0] if tensor_dtypes else None
        v = _to_scalar(value)
        if out_dtype in _INT_DTYPES:
            res = torch.addcmul(selfRef, tensor1, tensor2, value=int(round(v)))
        else:
            res = torch.addcmul(
                _to_fp32(selfRef), _to_fp32(tensor1), _to_fp32(tensor2), value=v
            )
        return [_cast_to_out(res, selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class TorchAddcmulTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, tensor1, tensor2, value=1.0, **kwargs):
        """参数跟随 torch.addcmul：input/tensor1/tensor2 位置张量，value 标量。"""
        return [torch.addcmul(input, tensor1, tensor2, value=_to_scalar(value))]

    third_party = {"torch": ThirdPartyImpl}
