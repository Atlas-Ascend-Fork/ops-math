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
"""Lerp 算子测试规范（TTK TestSpec）。

kernel/GEIR = lerp（numpy 入/出）：out = start + weight * (end - start)；
fp16/bf16 抬 fp32 中间精度（匹配 PrecisionReduceFlag）后回写；
aclnn = aclnnLerp / aclnnInplaceLerp / aclnnLerps / aclnnInplaceLerps（torch 入/出）；
e2e = torch.lerp（torch 入/出）。
"""

import torch

__spec__ = {
    "lerp": "LerpTestSpec",
    "aclnnLerp": "AclnnLerpTestSpec",
    "aclnnInplaceLerp": "AclnnInplaceLerpTestSpec",
    "aclnnLerps": "AclnnLerpsTestSpec",
    "aclnnInplaceLerps": "AclnnInplaceLerpsTestSpec",
    "torch.lerp": "TorchLerpTestSpec",
}


_FP_REDUCE = (torch.float16, torch.bfloat16)


def _to_fp32(v):
    if torch.is_tensor(v) and v.dtype in (torch.float16, torch.bfloat16):
        return v.to(torch.float32)
    return v


def _lerp(start, end, weight):
    """torch.lerp 包装：fp16/bf16 抬 fp32 中间精度（匹配 PrecisionReduceFlag）后回写
    start dtype；promote 模式下 weight 可能被抬到比 start 更高的精度（如 fp32→fp64
    vs fp16→fp32），需统一回 start dtype。"""
    orig_dtype = start.dtype
    if orig_dtype in _FP_REDUCE:
        res = torch.lerp(
            start.to(torch.float32), end.to(torch.float32), _to_fp32(weight)
        )
        return res.to(orig_dtype)
    if torch.is_tensor(weight) and weight.dtype != start.dtype:
        weight = weight.to(start.dtype)
    return torch.lerp(start, end, weight)


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


def _third_party_inputs(args, kwargs):
    """从三方标杆调用参数中提取三个输入（位置优先，kwargs 兜底）。"""

    def _pick(idx, names):
        if len(args) > idx:
            return args[idx]
        for name in names:
            if name in kwargs:
                return kwargs[name]
        return None

    start = _pick(0, ("self", "selfRef", "selfT", "start", "input"))
    end = _pick(1, ("end",))
    weight = _pick(2, ("weight",))
    missing = [
        n for n, v in (("start", start), ("end", end), ("weight", weight)) if v is None
    ]
    if missing:
        raise TypeError(
            "lerp third_party: input tensors %s not provided (args=%r, kwargs=%r)"
            % (missing, args, sorted(kwargs))
        )
    return start, end, weight


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：三个输入按位置接收（weight 可为标量）。"""

    def __call__(self, *args, **kwargs):
        start, end, weight = _third_party_inputs(args, kwargs)
        return [_lerp(start, end, weight)]


class LerpTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(start, end, weight, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        s = numpy_to_torch_tensor(start)
        e = numpy_to_torch_tensor(end)
        w = numpy_to_torch_tensor(weight)
        return [torch_to_numpy_tensor(_lerp(s, e, w))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnLerpTestSpec:
    """aclnnLerp 流程：out = lerp(self, end, weight 张量)。"""

    def golden(self, end, weight, out, **kwargs):
        """参数跟随 aclnnLerp：self、end、weight 张量，out 张量。"""
        return [_cast_to_out(_lerp(self, end, weight), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceLerpTestSpec:
    """aclnnInplaceLerp 流程：selfRef = lerp(selfRef, end, weight 张量)。"""

    def golden(selfRef, end, weight, **kwargs):
        """参数跟随 aclnnInplaceLerp：selfRef、end、weight 张量。"""
        return [_lerp(selfRef, end, weight)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnLerpsTestSpec:
    """aclnnLerps 流程：out = lerp(self, end, weight 标量)。"""

    def golden(self, end, weight, out, **kwargs):
        """参数跟随 aclnnLerps：self、end 张量，weight 标量，out 张量。"""
        return [_cast_to_out(_lerp(self, end, weight), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceLerpsTestSpec:
    """aclnnInplaceLerps 流程：selfRef = lerp(selfRef, end, weight 标量)。"""

    def golden(selfRef, end, weight, **kwargs):
        """参数跟随 aclnnInplaceLerps：selfRef、end 张量，weight 标量。"""
        return [_lerp(selfRef, end, weight)]

    third_party = {"torch": ThirdPartyImpl}


class TorchLerpTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, end, weight, **kwargs):
        """参数跟随 torch.lerp：input/end/weight 位置参数。"""
        return [_lerp(input, end, weight)]

    third_party = {"torch": ThirdPartyImpl}
