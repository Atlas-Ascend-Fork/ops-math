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
"""RealDiv 算子测试规范（TTK TestSpec）。

kernel/GEIR = real_div（numpy 入/出）：真值 x1 / x2（float64 中间精度，按输出
dtype 回转，bfloat16 采用 round-to-nearest-even）；
aclnn = aclnnDiv / aclnnInplaceDiv / aclnnDivs / aclnnInplaceDivs /
aclnnDivMod / aclnnInplaceDivMod / aclnnDivMods / aclnnInplaceDivMods
（torch 入/出）：out = self / other（Divs/DivMods 的 other 为标量；DivMod 系列
带 mode 舍入：0 默认、1 trunc、2 floor；Inplace 写回 selfRef）；
e2e = torch.div（torch 入/出）：真值 torch.div(input, other)。
"""

import numpy
import torch

__spec__ = {
    "real_div": "RealDivTestSpec",
    "aclnnDiv": "AclnnDivTestSpec",
    "aclnnInplaceDiv": "AclnnInplaceDivTestSpec",
    "aclnnDivs": "AclnnDivTestSpec",
    "aclnnInplaceDivs": "AclnnInplaceDivTestSpec",
    "aclnnDivMod": "AclnnDivModTestSpec",
    "aclnnInplaceDivMod": "AclnnInplaceDivModTestSpec",
    "aclnnDivMods": "AclnnDivModTestSpec",
    "aclnnInplaceDivMods": "AclnnInplaceDivModTestSpec",
    "torch.div": "TorchRealDivTestSpec",
}

_ROUNDING_MODE_MAP = {0: None, 1: "trunc", 2: "floor"}

_DTYPE_STR_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bfloat16": torch.bfloat16,
}


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v
    return v


def _cast_output(value, dtype_name):
    """numpy 结果按声明 dtype 回转（kernel/GEIR 用）。"""
    dtype_name = str(dtype_name).lower() if dtype_name is not None else "float32"
    if dtype_name == "bfloat16":
        arr = numpy.asarray(value, dtype=numpy.float32)
        bits = arr.view(numpy.uint32)
        rounded = (
            bits + numpy.uint32(0x7FFF) + ((bits >> numpy.uint32(16)) & numpy.uint32(1))
        ) & numpy.uint32(0xFFFF0000)
        return rounded.view(numpy.float32)
    if dtype_name == "float16":
        return numpy.asarray(value, dtype=numpy.float32).astype(numpy.float16)
    if dtype_name == "float32":
        return numpy.asarray(value, dtype=numpy.float32)
    if dtype_name in ("float64", "double"):
        return numpy.asarray(value, dtype=numpy.float64)
    if dtype_name == "bool":
        return numpy.asarray(value).astype(numpy.bool_)
    if dtype_name == "int8":
        return numpy.asarray(value).astype(numpy.int8)
    if dtype_name == "uint8":
        return numpy.asarray(value).astype(numpy.uint8)
    if dtype_name == "int16":
        return numpy.asarray(value).astype(numpy.int16)
    if dtype_name == "int32":
        return numpy.asarray(value).astype(numpy.int32)
    if dtype_name == "int64":
        return numpy.asarray(value).astype(numpy.int64)
    if dtype_name == "complex64":
        return numpy.asarray(value, dtype=numpy.complex64)
    if dtype_name == "complex128":
        return numpy.asarray(value, dtype=numpy.complex128)
    return numpy.asarray(value)


def _cast_torch(result, out_dtype):
    """torch 结果按声明输出 dtype 回转（aclnn 用）。"""
    if out_dtype is None or result.dtype == out_dtype:
        return result
    try:
        return result.to(out_dtype)
    except (RuntimeError, TypeError):
        return result


def _out_torch_dtype(kwargs):
    tensor_dtypes = tuple(kwargs.get("tensor_dtypes") or ())
    if tensor_dtypes:
        return _DTYPE_STR_TO_TORCH.get(str(tensor_dtypes[-1]).lower())
    return None


def _div_round_support_bool(a, b, rounding):
    """按 mode 舍入做除法。

    bool 输入 + mode=1/2（rounding 非 None）时匹配 NPU 语义：
    bool → float32 → div → 结果再由调用方 _cast_torch 转回 bool。
    """
    if a.dtype == torch.bool and rounding is not None:
        a = a.float()
        b = b.float() if torch.is_tensor(b) else float(b)
    return torch.div(a, b, rounding_mode=rounding)


def _third_party_inputs(
    args,
    kwargs,
    first_names=("self", "selfT", "x1", "input"),
    second_names=("other", "x2"),
):
    """从三方标杆调用参数中提取两个输入张量（位置优先，kwargs 兜底）。

    位置不足时按候选名从 kwargs 回退（兼容关键字喂参如 self=/other=/x1=/x2=）；
    仍缺失时抛含 args/kwargs 的明确异常，避免裸 IndexError。
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
    if first is None or second is None:
        raise TypeError(
            "third_party: two input tensors not provided (args=%r, kwargs=%r)"
            % (args, sorted(kwargs))
        )
    return first, second


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：mode 舍入属性可选绑定（``self`` 为保留名）。"""

    def __init__(self, *, mode=None, **kwargs):
        self.mode = mode

    def __call__(self, *args, **kwargs):
        a, b = _third_party_inputs(args, kwargs)
        rounding = _ROUNDING_MODE_MAP.get(self.mode, None)
        res = _div_round_support_bool(a, b, rounding)
        if a.dtype == torch.bool and rounding is not None:
            res = res.to(torch.bool)  # bool → float32 → div → 转回 bool
        return [res]


class RealDivTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """kernel/GEIR golden：float64 中间精度相除后按输出 dtype 回转。"""
        out_dtypes = tuple(kwargs.get("output_dtypes") or ())
        out_dtype = str(out_dtypes[0]).lower() if out_dtypes else None

        lhs = numpy.asarray(x1)
        rhs = numpy.asarray(x2)
        if numpy.iscomplexobj(lhs) or numpy.iscomplexobj(rhs):
            res = lhs.astype(numpy.complex128) / rhs.astype(numpy.complex128)
        else:
            res = numpy.asarray(x1, dtype=numpy.float64) / numpy.asarray(
                x2, dtype=numpy.float64
            )
        return [_cast_output(res, out_dtype)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnDivTestSpec:
    """aclnnDiv / aclnnDivs 流程：out = self / other（other 张量或标量）。"""

    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnDiv/aclnnDivs：self 张量、other 张量/标量、out 张量。"""
        return [_cast_torch(torch.div(self, other), _out_torch_dtype(kwargs))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceDivTestSpec:
    """aclnnInplaceDiv / aclnnInplaceDivs 流程：selfRef /= other。"""

    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceDiv/aclnnInplaceDivs：selfRef 张量、other 张量/标量。"""
        return [_cast_torch(torch.div(selfRef, other), selfRef.dtype)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnDivModTestSpec:
    """aclnnDivMod / aclnnDivMods 流程：out = self / other（带 mode 舍入）。"""

    def golden(self, other, mode=0, out=None, **kwargs):
        """参数跟随 aclnnDivMod/aclnnDivMods：self 张量、other 张量/标量、mode 属性、out 张量。"""
        rounding = _ROUNDING_MODE_MAP.get(mode, None)
        res = _div_round_support_bool(self, other, rounding)
        return [_cast_torch(res, _out_torch_dtype(kwargs))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceDivModTestSpec:
    """aclnnInplaceDivMod / aclnnInplaceDivMods 流程：selfRef /= other（带 mode 舍入）。"""

    def golden(selfRef, other, mode=0, **kwargs):
        """参数跟随 aclnnInplaceDivMod/aclnnInplaceDivMods：selfRef 张量、other 张量/标量、mode 属性。"""
        rounding = _ROUNDING_MODE_MAP.get(mode, None)
        res = _div_round_support_bool(selfRef, other, rounding)
        return [_cast_torch(res, selfRef.dtype)]

    third_party = {"torch": ThirdPartyImpl}


class TorchRealDivTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, **kwargs):
        """参数跟随 torch.div：input/other 位置张量（真值除法）。"""
        return [torch.div(input, other)]

    third_party = {"torch": ThirdPartyImpl}
