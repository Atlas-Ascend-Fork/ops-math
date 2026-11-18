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
"""Greater 算子测试规范（TTK TestSpec）。

通路覆盖：
  kernel/geir : greater（numpy 入/出）——out = x1 > x2，经 ttk 转 torch 后
                用 torch.gt 计算，再回转 numpy；
  aclnn       : aclnnGtScalar / aclnnInplaceGtScalar / aclnnGtTensor / aclnnInplaceGtTensor（torch 入/出）
                ——out = self > other（Scalar 的 other 为标量；Inplace 写回 selfRef）；
  e2e         : torch.gt（torch 入/出）——真值 torch.gt(input, other)。

三方标杆：各 Spec 均声明 third_party={"torch": ...}，以 torch 独立实现做交叉校验。
所有场景真值均由 torch 计算（无符号整型经保序 torch 位运算规避 torch 比较算子缺失）。
"""

import torch

__spec__ = {
    "greater": "GreaterTestSpec",
    "aclnnGtScalar": "AclnnGtTestSpec",
    "aclnnInplaceGtScalar": "AclnnInplaceGtTestSpec",
    "aclnnGtTensor": "AclnnGtTestSpec",
    "aclnnInplaceGtTensor": "AclnnInplaceGtTestSpec",
    "torch.gt": "TorchGtTestSpec",
}


def _to_torch(value):
    """numpy.ndarray / Python 标量 → torch.Tensor（bfloat16 等非原生 dtype 无损转换）。"""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (bool, int, float, complex)):
        return torch.as_tensor(value)
    from ttk.utilities import numpy_to_torch_tensor

    if not hasattr(value, "shape"):
        return torch.as_tensor(value)
    return numpy_to_torch_tensor(value)


def _cast_to_out(result, out):
    """按 out 张量 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if (
        out is not None
        and getattr(out, "dtype", None) is not None
        and result.dtype != out.dtype
    ):
        return result.to(out.dtype)
    return result


def _third_party_inputs(args, kwargs):
    """从三方标杆调用参数中提取两个输入（位置优先，关键字兜底）。"""

    def _pick(idx, names):
        if len(args) > idx:
            return args[idx]
        for name in names:
            if name in kwargs:
                return kwargs[name]
        return None

    lhs = _pick(0, ("self", "selfT", "selfRef", "x1", "input", "element"))
    rhs = _pick(1, ("other", "otherT", "x2", "testElement", "testElements"))
    if lhs is None or rhs is None:
        raise TypeError(
            "third_party: two inputs not provided (args=%r, kwargs=%r)"
            % (args, sorted(kwargs))
        )
    return lhs, rhs


_UNSIGNED_WIDE = (torch.uint16, torch.uint32)
_U64_SIGN_BIT = 1 << 63


def _to_u64(value, device=None):
    """标量/张量统一为 uint64 张量（无符号比较前的类型对齐）。"""
    if isinstance(value, torch.Tensor):
        return value if value.dtype == torch.uint64 else value.to(torch.uint64)
    return torch.as_tensor(value, dtype=torch.uint64, device=device)


def _orderable_pair(lhs, rhs):
    """无符号整型转 torch 可比较的保序表示（纯 torch 运算，逐值精确）。

    uint64：异或符号位后按 int64 解释，把 [0, 2^64) 单调映射到 int64 值域；
    uint16/uint32：值域可无损放进 int64。
    """
    lhs = _to_torch(lhs)
    rhs = _to_torch(rhs)
    dtypes = {getattr(t, "dtype", None) for t in (lhs, rhs)}
    if torch.uint64 in dtypes:
        lhs = _to_u64(lhs, getattr(rhs, "device", None))
        rhs = _to_u64(rhs, lhs.device)
        sign = torch.tensor(_U64_SIGN_BIT, dtype=torch.uint64, device=lhs.device)
        return (
            torch.bitwise_xor(lhs, sign).view(torch.int64),
            torch.bitwise_xor(rhs, sign).view(torch.int64),
        )
    if dtypes & set(_UNSIGNED_WIDE):
        return lhs.to(torch.int64), rhs.to(torch.int64)
    return lhs, rhs


def _gt(lhs, rhs):
    a, b = _orderable_pair(lhs, rhs)
    return torch.gt(a, b)


class ThirdPartyImpl:
    """torch 三方标杆：两个输入按位置接收，可带 out 张量。"""

    def __call__(self, *args, **kwargs):
        lhs, rhs = _third_party_inputs(args, kwargs)
        out = kwargs.get("out")
        if out is None and len(args) > 2 and isinstance(args[2], torch.Tensor):
            out = args[2]
        return [_cast_to_out(_gt(lhs, rhs), out)]


class GreaterTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """参数跟随 @greater_def.cpp：x1/x2 张量，输出 y（bool）。"""
        from ttk.utilities import torch_to_numpy_tensor

        return [torch_to_numpy_tensor(_gt(x1, x2))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnGtTestSpec:
    """aclnnGtScalar / aclnnGtTensor：out = self gt other。"""

    @staticmethod
    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnGtScalar/aclnnGtTensor：self、other（张量或标量）、out。"""
        return [_cast_to_out(_gt(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceGtTestSpec:
    """aclnnInplaceGtScalar / aclnnInplaceGtTensor：selfRef = selfRef gt other。"""

    @staticmethod
    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceGtScalar/aclnnInplaceGtTensor：selfRef、other。"""
        return [_cast_to_out(_gt(selfRef, other), selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class TorchGtTestSpec:
    """e2e 流程：torch 入/出。"""

    @staticmethod
    def golden(input, other, out=None, **kwargs):
        """参数跟随 torch.gt：input/other 位置张量，out 为可选输出张量。"""
        return [_cast_to_out(_gt(input, other), out)]

    third_party = {"torch": ThirdPartyImpl}
