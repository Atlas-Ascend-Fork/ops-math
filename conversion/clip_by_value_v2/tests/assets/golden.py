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
"""ClipByValueV2 算子测试规范（TTK TestSpec）。

kernel/GEIR = clip_by_value_v2（numpy 入/出）：y = min(max(x, clip_value_min),
clip_value_max)（等价 torch.clamp）；
aclnn = aclnnClamp / aclnnClampMax / aclnnClampMaxTensor / aclnnClampMin /
aclnnClampMinTensor / aclnnClampTensor / aclnnHardtanh /
aclnnInplaceClampMax / aclnnInplaceClampMaxTensor /
aclnnInplaceClampMinTensor / aclnnInplaceHardtanh（torch 入/出）；
e2e = torch.clamp（torch 入/出）。
标量 min/max 按 ACLNN 规则与 self 计算 promote dtype 后在 CPU 经 torch.clamp 计算。
"""

import torch

__spec__ = {
    "clip_by_value_v2": "ClipByValueV2TestSpec",
    "aclnnClamp": "AclnnClampTestSpec",
    "aclnnClampMax": "AclnnClampMaxTestSpec",
    "aclnnClampMaxTensor": "AclnnClampMaxTensorTestSpec",
    "aclnnClampMin": "AclnnClampMinTestSpec",
    "aclnnClampMinTensor": "AclnnClampMinTensorTestSpec",
    "aclnnClampTensor": "AclnnClampTensorTestSpec",
    "aclnnHardtanh": "AclnnHardtanhTestSpec",
    "aclnnInplaceClampMax": "AclnnInplaceClampMaxTestSpec",
    "aclnnInplaceClampMaxTensor": "AclnnInplaceClampMaxTensorTestSpec",
    "aclnnInplaceClampMinTensor": "AclnnInplaceClampMinTensorTestSpec",
    "aclnnInplaceHardtanh": "AclnnInplaceHardtanhTestSpec",
    "torch.clamp": "TorchClampTestSpec",
}

_FP_REDUCE = (torch.float16, torch.bfloat16)


def _promote(a, b):
    """CANN PromoteType 的 torch 近似。"""
    return torch.promote_types(a, b)


def _scalar_default(dtype):
    """GetScalarDefaultDtype：浮点标量 → float32，整型/布尔保持原样。"""
    if dtype.is_floating_point:
        return torch.float32
    return dtype


def _combine_categories(higher, lower):
    """CombineCategoriesWithComplex 的简化实现（本算子域内无 complex）。"""
    if higher.is_floating_point:
        return higher
    if higher == torch.bool or lower.is_floating_point:
        return _promote(higher, lower)
    return higher


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


def _as_bound(v, dtype):
    """标量（0 维张量/数值）统一转为 promote dtype 的 0 维张量。"""
    if torch.is_tensor(v):
        return v.to(dtype)
    return v


def _clamp_scalar(x, clip_min, clip_max, out=None):
    """标量下界/上界 clamp：复刻 ClampPromoteType。"""
    promote = x.dtype
    if clip_min is not None and clip_max is not None:
        scalar_dtype = _promote(
            _scalar_default(_dtype_of(clip_min)), _scalar_default(_dtype_of(clip_max))
        )
        promote = _combine_categories(promote, scalar_dtype)
    elif clip_min is not None:
        promote = _combine_categories(promote, _scalar_default(_dtype_of(clip_min)))
    elif clip_max is not None:
        promote = _combine_categories(promote, _scalar_default(_dtype_of(clip_max)))

    x_p = x.to(promote)
    kwargs = {}
    if clip_min is not None:
        kwargs["min"] = _as_bound(clip_min, promote)
    if clip_max is not None:
        kwargs["max"] = _as_bound(clip_max, promote)
    return _cast_to_out(torch.clamp(x_p, **kwargs), out)


def _clamp_tensor(x, clip_min, clip_max, out=None):
    """张量下界/上界 clamp：复刻 ClampTensorPromoteType。"""
    promote = x.dtype
    if clip_min is not None:
        promote = _promote(promote, _dtype_of(clip_min))
    if clip_max is not None:
        promote = _promote(promote, _dtype_of(clip_max))

    x_p = x.to(promote)
    kwargs = {}
    if clip_min is not None:
        kwargs["min"] = _as_bound(clip_min, promote)
    if clip_max is not None:
        kwargs["max"] = _as_bound(clip_max, promote)
    return _cast_to_out(torch.clamp(x_p, **kwargs), out)


def _dtype_of(v):
    if torch.is_tensor(v):
        return v.dtype
    return torch.as_tensor(v).dtype


def _is_scalar_bound(v):
    return v is None or (torch.is_tensor(v) and v.dim() == 0) or not torch.is_tensor(v)


def _first(*values):
    """返回第一个非 None 的值（不做张量真值判断）。"""
    for v in values:
        if v is not None:
            return v
    return None


class ThirdPartyImpl:
    """torch 三方标杆：按其 min/max 为标量或张量调用对应 clamp（cross_check 用）。"""

    def __call__(self, *args, **kwargs):
        x = _first(kwargs.get("self"), kwargs.get("selfRef"), args[0] if args else None)
        clip_min = _first(
            kwargs.get("clipValueMin"), kwargs.get("clip_value_min"), kwargs.get("min")
        )
        clip_max = _first(
            kwargs.get("clipValueMax"), kwargs.get("clip_value_max"), kwargs.get("max")
        )
        if x is None:
            raise TypeError("clip_by_value_v2 third_party: input tensor not provided")
        # lower/upper 语义以参数名（pool key）为准：clipValueMin/clip_value_min/min
        # 为下界、clipValueMax/clip_value_max/max 为上界；不做按位置参数个数推断
        # min/max 的启发式（torch.clamp(x, v) 的第二位置参是 min，而 aclnnClampMax
        # 的唯一界是 max，位置猜测会把两者互相误判）。
        if _is_scalar_bound(clip_min) and _is_scalar_bound(clip_max):
            return [_clamp_scalar(x, clip_min, clip_max)]
        return [_clamp_tensor(x, clip_min, clip_max)]


class ClipByValueV2TestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x, clip_value_min, clip_value_max, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        x_t = numpy_to_torch_tensor(x)
        min_t = numpy_to_torch_tensor(clip_value_min)
        max_t = numpy_to_torch_tensor(clip_value_max)
        return [torch_to_numpy_tensor(torch.clamp(x_t, min=min_t, max=max_t))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnClampTestSpec:
    """aclnnClamp 流程：out = clamp(self, clipValueMin 标量, clipValueMax 标量)。"""

    def golden(self, clipValueMin, clipValueMax, out, **kwargs):
        return [_clamp_scalar(self, clipValueMin, clipValueMax, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnClampMaxTestSpec:
    """aclnnClampMax 流程：out = clamp_max(self, clipValueMax 标量)。"""

    def golden(self, clipValueMax, out, **kwargs):
        return [_clamp_scalar(self, None, clipValueMax, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnClampMinTestSpec:
    """aclnnClampMin 流程：out = clamp_min(self, clipValueMin 标量)。"""

    def golden(self, clipValueMin, out, **kwargs):
        return [_clamp_scalar(self, clipValueMin, None, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnHardtanhTestSpec:
    """aclnnHardtanh 流程：out = clamp(self, clipValueMin 标量, clipValueMax 标量)。"""

    def golden(self, clipValueMin, clipValueMax, out, **kwargs):
        return [_clamp_scalar(self, clipValueMin, clipValueMax, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnClampTensorTestSpec:
    """aclnnClampTensor 流程：out = clamp(self, clipValueMin 张量, clipValueMax 张量)。"""

    def golden(self, clipValueMin, clipValueMax, out, **kwargs):
        return [_clamp_tensor(self, clipValueMin, clipValueMax, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnClampMaxTensorTestSpec:
    """aclnnClampMaxTensor 流程：out = clamp_max(self, max 张量)。"""

    def golden(self, max, out, **kwargs):
        return [_clamp_tensor(self, None, max, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnClampMinTensorTestSpec:
    """aclnnClampMinTensor 流程：out = clamp_min(self, clipValueMin 张量)。"""

    def golden(self, clipValueMin, out, **kwargs):
        return [_clamp_tensor(self, clipValueMin, None, out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceClampMaxTestSpec:
    """aclnnInplaceClampMax 流程：selfRef = clamp_max(selfRef, clipValueMax 标量)。"""

    def golden(selfRef, clipValueMax, **kwargs):
        return [_clamp_scalar(selfRef, None, clipValueMax, selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceClampMaxTensorTestSpec:
    """aclnnInplaceClampMaxTensor 流程：selfRef = clamp_max(selfRef, max 张量)。"""

    def golden(selfRef, max, **kwargs):
        return [_clamp_tensor(selfRef, None, max, selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceClampMinTensorTestSpec:
    """aclnnInplaceClampMinTensor 流程：selfRef = clamp_min(selfRef, clipValueMin 张量)。"""

    def golden(selfRef, clipValueMin, **kwargs):
        return [_clamp_tensor(selfRef, clipValueMin, None, selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceHardtanhTestSpec:
    """aclnnInplaceHardtanh 流程：selfRef = clamp(selfRef, clipValueMin 标量, clipValueMax 标量)。"""

    def golden(selfRef, clipValueMin, clipValueMax, **kwargs):
        return [_clamp_scalar(selfRef, clipValueMin, clipValueMax, selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class TorchClampTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, min, max, **kwargs):
        """参数跟随 torch.clamp：input/min/max 位置张量。"""
        return [_clamp_tensor(input, min, max)]

    third_party = {"torch": ThirdPartyImpl}
