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
"""NotEqual 算子测试规范（TTK TestSpec）。

通路覆盖：
  kernel/geir : not_equal（numpy 入/出）——out = x1 != x2，经 ttk 转 torch 后
                用 torch.ne 计算，再回转 numpy；
  aclnn       : aclnnNeScalar / aclnnInplaceNeScalar /
                aclnnNeTensor / aclnnInplaceNeTensor / aclnnLogicalXor（torch 入/出）；
  e2e         : torch.ne（torch 入/出）——真值 torch.ne(input, other)。

三方标杆：各 Spec 均声明 third_party={"torch": ...}，以 torch 独立实现做交叉校验。
所有场景真值均由 torch 计算（无符号整型经等价位映射规避 torch 混合类型 promote 限制）。
"""

import torch

__spec__ = {
    "not_equal": "NotEqualTestSpec",
    "aclnnNeScalar": "AclnnNeTestSpec",
    "aclnnInplaceNeScalar": "AclnnInplaceNeTestSpec",
    "aclnnNeTensor": "AclnnNeTestSpec",
    "aclnnInplaceNeTensor": "AclnnInplaceNeTestSpec",
    "aclnnLogicalXor": "AclnnLogicalXorTestSpec",
    "torch.ne": "TorchNeTestSpec",
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


_UNSIGNED_ALL = (torch.uint16, torch.uint32, torch.uint64)


def _not_equal(lhs, rhs):
    """torch 不等比较；无符号宽整型先按位等价映射到 int64（不等逐 bit 保持）。"""
    lhs = _to_torch(lhs)
    rhs = _to_torch(rhs)
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        if {lhs.dtype, rhs.dtype} & set(_UNSIGNED_ALL):
            lhs, rhs = lhs.to(torch.int64), rhs.to(torch.int64)
    return torch.ne(lhs, rhs)


class ThirdPartyImpl:
    """torch 三方标杆：两个输入按位置接收，可带 out 张量。"""

    def __call__(self, *args, **kwargs):
        lhs, rhs = _third_party_inputs(args, kwargs)
        out = kwargs.get("out")
        if out is None and len(args) > 2 and isinstance(args[2], torch.Tensor):
            out = args[2]
        return [_cast_to_out(_not_equal(lhs, rhs), out)]


class NotEqualTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """参数跟随 @not_equal_def.cpp：x1/x2 张量，输出 y（bool）。"""
        from ttk.utilities import torch_to_numpy_tensor

        return [torch_to_numpy_tensor(_not_equal(x1, x2))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnNeTestSpec:
    """aclnnNeScalar / aclnnNeTensor：out = self ne other。"""

    @staticmethod
    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnNeScalar/aclnnNeTensor：self、other（张量或标量）、out。"""
        return [_cast_to_out(_not_equal(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceNeTestSpec:
    """aclnnInplaceNeScalar / aclnnInplaceNeTensor：selfRef = selfRef ne other。"""

    @staticmethod
    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceNeScalar/aclnnInplaceNeTensor：selfRef、other。"""
        return [_cast_to_out(_not_equal(selfRef, other), selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class TorchNeTestSpec:
    """e2e 流程：torch 入/出。"""

    @staticmethod
    def golden(input, other, out=None, **kwargs):
        """参数跟随 torch.ne：input/other 位置张量，out 为可选输出张量。"""
        return [_cast_to_out(_not_equal(input, other), out)]

    third_party = {"torch": ThirdPartyImpl}


def _logical_xor(lhs, rhs):
    """aclnnLogicalXor 语义：非 0 视为 True，逐元素逻辑异或。"""
    lhs = _to_torch(lhs)
    rhs = _to_torch(rhs)
    return torch.logical_xor(_zero_semantics_pair(lhs), _zero_semantics_pair(rhs))


def _zero_semantics_pair(value):
    if isinstance(value, torch.Tensor) and value.dtype in _UNSIGNED_ALL:
        return value.to(torch.int64)
    return value


class AclnnLogicalXorTestSpec:
    """aclnnLogicalXor：out = logical_xor(self, other)。"""

    @staticmethod
    def golden(self, other, out, **kwargs):
        """参数跟随 aclnn_logical_xor.h：self、other 张量，out 张量。"""
        return [_cast_to_out(_logical_xor(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}
