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
"""Addcdiv 算子测试规范（TTK TestSpec）。

kernel/GEIR = addcdiv（numpy 入/出）：out = input_data + value * (x1 / x2)；
fp16/bf16 抬 fp32 中间精度（匹配 PrecisionReduceFlag）后回写；
aclnn = aclnnAddcdiv / aclnnInplaceAddcdiv（torch 入/出）；
e2e = torch.addcdiv（torch 入/出）。
"""

import torch

__spec__ = {
    "addcdiv": "AddcdivTestSpec",
    "aclnnAddcdiv": "AclnnAddcdivTestSpec",
    "aclnnInplaceAddcdiv": "AclnnInplaceAddcdivTestSpec",
    "torch.addcdiv": "TorchAddcdivTestSpec",
}

_FP_REDUCE = (torch.float16, torch.bfloat16)


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v
    if hasattr(v, "item"):
        return v.item()
    return v


def _to_fp32(t):
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.to(torch.float32)
    return t


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


def _third_party_inputs(args, kwargs):
    """从三方标杆调用参数中提取三个输入张量（位置优先，kwargs 兜底）。"""

    def _pick(idx, names):
        if len(args) > idx:
            return args[idx]
        for name in names:
            if name in kwargs:
                return kwargs[name]
        return None

    first = _pick(0, ("self", "selfRef", "selfT", "input_data", "input", "x1"))
    second = _pick(1, ("tensor1", "x1"))
    third = _pick(2, ("tensor2", "x2"))
    missing = [
        n
        for n, v in (("input_data", first), ("tensor1", second), ("tensor2", third))
        if v is None
    ]
    if missing:
        raise TypeError(
            "addcdiv third_party: input tensors %s not provided (args=%r, kwargs=%r)"
            % (missing, args, sorted(kwargs))
        )
    return first, second, third


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：value 属性绑定，三个输入按位置接收。"""

    def __init__(self, *, value=1.0, **kwargs):
        self.value = value

    def __call__(self, *args, **kwargs):
        a, b, c = _third_party_inputs(args, kwargs)
        v = kwargs.get("value", self.value)
        return [torch.addcdiv(a, b, c, value=_to_scalar(v))]


class AddcdivTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(input_data, x1, x2, value, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        t_data = numpy_to_torch_tensor(input_data)
        t_x1 = numpy_to_torch_tensor(x1)
        t_x2 = numpy_to_torch_tensor(x2)
        v = _to_scalar(numpy_to_torch_tensor(value))

        res = torch.addcdiv(_to_fp32(t_data), _to_fp32(t_x1), _to_fp32(t_x2), value=v)
        if t_data.dtype in _FP_REDUCE:
            res = res.to(t_data.dtype)
        return [torch_to_numpy_tensor(res)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnAddcdivTestSpec:
    """aclnnAddcdiv 流程：out = self + value * (tensor1 / tensor2)。"""

    def golden(self, tensor1, tensor2, value, out, **kwargs):
        """参数跟随 aclnnAddcdiv：self、tensor1、tensor2 张量，value 标量，out 张量。"""
        v = _to_scalar(value)
        res = torch.addcdiv(
            _to_fp32(self), _to_fp32(tensor1), _to_fp32(tensor2), value=v
        )
        if self.dtype in _FP_REDUCE:
            res = res.to(self.dtype)
        return [_cast_to_out(res, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceAddcdivTestSpec:
    """aclnnInplaceAddcdiv 流程：selfRef += value * (tensor1 / tensor2)。"""

    def golden(selfRef, tensor1, tensor2, value, **kwargs):
        """参数跟随 aclnnInplaceAddcdiv：selfRef、tensor1、tensor2 张量，value 标量。"""
        v = _to_scalar(value)
        res = torch.addcdiv(
            _to_fp32(selfRef), _to_fp32(tensor1), _to_fp32(tensor2), value=v
        )
        if selfRef.dtype in _FP_REDUCE:
            res = res.to(selfRef.dtype)
        return [res]

    third_party = {"torch": ThirdPartyImpl}


class TorchAddcdivTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, tensor1, tensor2, value=1.0, **kwargs):
        """参数跟随 torch.addcdiv：input/tensor1/tensor2 位置张量，value 标量。"""
        v = _to_scalar(value)
        res = torch.addcdiv(
            _to_fp32(input), _to_fp32(tensor1), _to_fp32(tensor2), value=v
        )
        if input.dtype in _FP_REDUCE:
            res = res.to(input.dtype)
        return [res]

    third_party = {"torch": ThirdPartyImpl}
