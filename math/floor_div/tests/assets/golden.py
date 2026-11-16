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
"""FloorDiv 算子测试规范（TTK TestSpec）。

kernel/GEIR = floor_div（numpy 入/出）：out = floor(x1 / x2)；fp16/bf16 抬 fp32
中间精度（匹配 PrecisionReduceFlag）后按 RINT 回写；
aclnn = aclnnFloorDivide / aclnnFloorDivides / aclnnInplaceFloorDivide /
aclnnInplaceFloorDivides（torch 入/出）；
e2e = torch.floor_divide（torch 入/出）。
"""

import torch

__spec__ = {
    "floor_div": "FloorDivTestSpec",
    "aclnnFloorDivide": "AclnnFloorDivideTestSpec",
    "aclnnFloorDivides": "AclnnFloorDividesTestSpec",
    "aclnnInplaceFloorDivide": "AclnnInplaceFloorDivideTestSpec",
    "aclnnInplaceFloorDivides": "AclnnInplaceFloorDividesTestSpec",
    "torch.floor_divide": "TorchFloorDivideTestSpec",
}

_FP_REDUCE = (torch.float16, torch.bfloat16)


def _to_tensor(v):
    """标量/0 维张量统一为张量，供 .to() 与广播。"""
    if torch.is_tensor(v):
        return v
    return torch.as_tensor(v)


def _floor_divide(t1, t2):
    """floor(x1 / x2)：bool 抬 float32（aclnn promote 规则）；fp16/bf16 抬 fp32
    中间精度后回写（匹配 PrecisionReduceFlag）。"""
    orig_dtype = t1.dtype
    if orig_dtype == torch.bool:
        res = torch.floor_divide(t1.to(torch.float32), _to_tensor(t2).to(torch.float32))
        return res.to(torch.bool)
    if orig_dtype in _FP_REDUCE:
        res = torch.floor_divide(t1.to(torch.float32), _to_tensor(t2).to(torch.float32))
        return res.to(orig_dtype)
    return torch.floor_divide(t1, _to_tensor(t2))


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


def _third_party_inputs(
    args,
    kwargs,
    first_names=("self", "selfRef", "selfT", "x1", "input"),
    second_names=("other", "x2"),
):
    """从三方标杆调用参数中提取两个输入张量（位置优先，kwargs 兜底）。"""

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
            "floor_div third_party: two input tensors not provided (args=%r, kwargs=%r)"
            % (args, sorted(kwargs))
        )
    return first, second


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：两个输入按位置接收（含标量 other）。"""

    def __call__(self, *args, **kwargs):
        a, b = _third_party_inputs(args, kwargs)
        return [_floor_divide(a, b)]


class FloorDivTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        x1_torch = numpy_to_torch_tensor(x1)
        x2_torch = numpy_to_torch_tensor(x2)
        return [torch_to_numpy_tensor(_floor_divide(x1_torch, x2_torch))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnFloorDivideTestSpec:
    """aclnnFloorDivide 流程：out = floor_divide(self, other)。"""

    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnFloorDivide：self、other 张量，out 张量。"""
        return [_cast_to_out(_floor_divide(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnFloorDividesTestSpec:
    """aclnnFloorDivides 流程：out = floor_divide(self, other 标量)。"""

    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnFloorDivides：self 张量、other 标量、out 张量。"""
        return [_cast_to_out(_floor_divide(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceFloorDivideTestSpec:
    """aclnnInplaceFloorDivide 流程：selfRef = floor_divide(selfRef, other)。"""

    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceFloorDivide：selfRef、other 张量。"""
        return [_floor_divide(selfRef, other)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceFloorDividesTestSpec:
    """aclnnInplaceFloorDivides 流程：selfRef = floor_divide(selfRef, other 标量)。"""

    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceFloorDivides：selfRef 张量、other 标量。"""
        return [_floor_divide(selfRef, other)]

    third_party = {"torch": ThirdPartyImpl}


class TorchFloorDivideTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, **kwargs):
        """参数跟随 torch.floor_divide：input/other 位置张量。"""
        return [_floor_divide(input, other)]

    third_party = {"torch": ThirdPartyImpl}
