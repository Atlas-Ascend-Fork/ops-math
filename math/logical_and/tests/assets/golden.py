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
"""LogicalAnd 算子测试规范（TTK TestSpec）。

通路覆盖：
  kernel/geir : logical_and（numpy 入/出）——out = x1 逻辑与 x2，经 ttk 转 torch 后用
                torch.logical_and 计算，再回转 numpy（输出 bool）；
  aclnn       : LogicalAnd / aclnnInplaceLogicalAnd（torch 入/出）——out = self 逻辑与 other，
                Inplace 写回 selfRef（输出按 selfRef dtype 回写）；
  e2e         : torch.logical_and（torch 入/出）——真值 torch.logical_and(input, other)。

三方标杆：各 Spec 均声明 third_party={"torch": ...}，以 torch 独立实现做交叉校验。
所有场景真值均由 torch 计算（非 bool 输入按 0=False/非 0=True 语义）。
"""

import torch

__spec__ = {
    "logical_and": "LogicalAndTestSpec",
    "aclnnLogicalAnd": "AclnnLogicalAndTestSpec",
    "aclnnInplaceLogicalAnd": "AclnnInplaceLogicalAndTestSpec",
    "torch.logical_and": "TorchLogicalAndTestSpec",
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


_UNSIGNED_WIDE = (torch.uint16, torch.uint32, torch.uint64)


def _zero_semantics(value):
    """torch 未实现无符号宽整型的 logical_*，转 int64 保持 0/非 0 语义。"""
    if isinstance(value, torch.Tensor) and value.dtype in _UNSIGNED_WIDE:
        return value.to(torch.int64)
    return value


def _logical_and(lhs, rhs):
    lhs = _zero_semantics(_to_torch(lhs))
    rhs = _zero_semantics(_to_torch(rhs))
    return torch.logical_and(lhs, rhs)


class ThirdPartyImpl:
    """torch 三方标杆：两个输入按位置接收，可带 out 张量。"""

    def __call__(self, *args, **kwargs):
        lhs, rhs = _third_party_inputs(args, kwargs)
        out = kwargs.get("out")
        if out is None and len(args) > 2 and isinstance(args[2], torch.Tensor):
            out = args[2]
        return [_cast_to_out(_logical_and(lhs, rhs), out)]


class LogicalAndTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """参数跟随 @logical_and_def.cpp：x1/x2 张量，输出 y（bool）。"""
        from ttk.utilities import torch_to_numpy_tensor

        return [torch_to_numpy_tensor(_logical_and(x1, x2))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnLogicalAndTestSpec:
    """aclnnLogicalAnd：out = self logical_and other。"""

    @staticmethod
    def golden(self, other, out, **kwargs):
        """参数跟随 aclnn_logical_and.h：self、other 张量，out 张量。"""
        return [_cast_to_out(_logical_and(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceLogicalAndTestSpec:
    """aclnnInplaceLogicalAnd：selfRef = selfRef logical_and other。"""

    @staticmethod
    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnn_logical_and.h：selfRef、other 张量（结果按 selfRef dtype 回写）。"""
        return [_cast_to_out(_logical_and(selfRef, other), selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class TorchLogicalAndTestSpec:
    """e2e 流程：torch 入/出。"""

    @staticmethod
    def golden(input, other, out=None, **kwargs):
        """参数跟随 torch.logical_and：input/other 位置张量，out 为可选输出张量。"""
        return [_cast_to_out(_logical_and(input, other), out)]

    third_party = {"torch": ThirdPartyImpl}
