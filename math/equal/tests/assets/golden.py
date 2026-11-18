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
"""Equal 算子测试规范（TTK TestSpec）。

通路覆盖：
  kernel/geir : equal（numpy 入/出）——out = x1 == x2，经 ttk 转 torch 后
                用 torch.eq 计算，再回转 numpy；
  aclnn       : aclnnEqScalar / aclnnInplaceEqScalar /
                aclnnEqTensor / aclnnInplaceEqTensor / aclnnIsInTensorScalar / aclnnIsInScalarTensor（torch 入/出）；
  e2e         : torch.eq（torch 入/出）——真值 torch.eq(input, other)。

三方标杆：各 Spec 均声明 third_party={"torch": ...}，以 torch 独立实现做交叉校验。
所有场景真值均由 torch 计算（无符号整型经等价位映射规避 torch 混合类型 promote 限制）。
"""

import torch

__spec__ = {
    "equal": "EqualTestSpec",
    "aclnnEqScalar": "AclnnEqTestSpec",
    "aclnnInplaceEqScalar": "AclnnInplaceEqTestSpec",
    "aclnnEqTensor": "AclnnEqTestSpec",
    "aclnnInplaceEqTensor": "AclnnInplaceEqTestSpec",
    "aclnnIsInTensorScalar": "AclnnIsInTensorScalarTestSpec",
    "aclnnIsInScalarTensor": "AclnnIsInScalarTensorTestSpec",
    "torch.eq": "TorchEqTestSpec",
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


def _equal(lhs, rhs):
    """torch 等值比较；无符号宽整型先按位等价映射到 int64（等值逐 bit 保持）。"""
    lhs = _to_torch(lhs)
    rhs = _to_torch(rhs)
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        if {lhs.dtype, rhs.dtype} & set(_UNSIGNED_ALL):
            lhs, rhs = lhs.to(torch.int64), rhs.to(torch.int64)
    return torch.eq(lhs, rhs)


class ThirdPartyImpl:
    """torch 三方标杆：两个输入按位置接收，可带 out 张量。"""

    def __call__(self, *args, **kwargs):
        lhs, rhs = _third_party_inputs(args, kwargs)
        out = kwargs.get("out")
        if out is None and len(args) > 2 and isinstance(args[2], torch.Tensor):
            out = args[2]
        return [_cast_to_out(_equal(lhs, rhs), out)]


class EqualTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """参数跟随 @equal_def.cpp：x1/x2 张量，输出 y（bool）。"""
        from ttk.utilities import torch_to_numpy_tensor

        return [torch_to_numpy_tensor(_equal(x1, x2))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnEqTestSpec:
    """aclnnEqScalar / aclnnEqTensor：out = self eq other。"""

    @staticmethod
    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnEqScalar/aclnnEqTensor：self、other（张量或标量）、out。"""
        return [_cast_to_out(_equal(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceEqTestSpec:
    """aclnnInplaceEqScalar / aclnnInplaceEqTensor：selfRef = selfRef eq other。"""

    @staticmethod
    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceEqScalar/aclnnInplaceEqTensor：selfRef、other。"""
        return [_cast_to_out(_equal(selfRef, other), selfRef)]

    third_party = {"torch": ThirdPartyImpl}


class TorchEqTestSpec:
    """e2e 流程：torch 入/出。"""

    @staticmethod
    def golden(input, other, out=None, **kwargs):
        """参数跟随 torch.eq：input/other 位置张量，out 为可选输出张量。"""
        return [_cast_to_out(_equal(input, other), out)]

    third_party = {"torch": ThirdPartyImpl}


def _is_in_tensor_scalar(element, testElement):
    """aclnnIsInTensorScalar 语义：逐元素判断 element 是否等于 testElement。"""
    return _equal(element, testElement)


def _is_in_scalar_tensor(element, testElements):
    """aclnnIsInScalarTensor 语义：判断标量 element 是否出现在 testElements 中。"""
    return _equal(testElements, element).any()


class IsInTensorScalarThirdPartyImpl:
    """aclnnIsInTensorScalar 的 torch 三方标杆。"""

    def __call__(self, *args, **kwargs):
        inp, scalar = _third_party_inputs(args, kwargs)
        invert = kwargs.get("invert", args[3] if len(args) > 3 else False)
        result = _is_in_tensor_scalar(inp, scalar)
        if invert:
            result = torch.logical_not(result)
        out = kwargs.get("out")
        if out is None and len(args) > 4 and isinstance(args[4], torch.Tensor):
            out = args[4]
        return [_cast_to_out(result, out)]


class IsInScalarTensorThirdPartyImpl:
    """aclnnIsInScalarTensor 的 torch 三方标杆。"""

    def __call__(self, *args, **kwargs):
        inp = args[0] if args else kwargs.get("element")
        tensor = args[1] if len(args) > 1 else kwargs.get("testElements")
        invert = kwargs.get("invert", args[3] if len(args) > 3 else False)
        result = _is_in_scalar_tensor(inp, tensor)
        if invert:
            result = torch.logical_not(result)
        out = kwargs.get("out")
        if out is None and len(args) > 4 and isinstance(args[4], torch.Tensor):
            out = args[4]
        return [_cast_to_out(result, out)]


class AclnnIsInTensorScalarTestSpec:
    """aclnnIsInTensorScalar：out = element == testElement（可选 invert）。"""

    @staticmethod
    def golden(element, testElement, assumeUnique, invert, out, **kwargs):
        """参数跟随 aclnn_isin_tensor_scalar.h：element 张量、testElement 标量、布尔属性、out。"""
        del assumeUnique
        result = _is_in_tensor_scalar(element, testElement)
        if invert:
            result = torch.logical_not(result)
        return [_cast_to_out(result, out)]

    third_party = {"torch": IsInTensorScalarThirdPartyImpl}


class AclnnIsInScalarTensorTestSpec:
    """aclnnIsInScalarTensor：out = element 是否存在于 testElements（0 维 bool）。"""

    @staticmethod
    def golden(element, testElements, assumeUnique, invert, out, **kwargs):
        """参数跟随 aclnn_isin.h：element 标量、testElements 张量、布尔属性、out。"""
        del assumeUnique
        result = _is_in_scalar_tensor(element, testElements)
        if invert:
            result = torch.logical_not(result)
        return [_cast_to_out(result, out)]

    third_party = {"torch": IsInScalarTensorThirdPartyImpl}
