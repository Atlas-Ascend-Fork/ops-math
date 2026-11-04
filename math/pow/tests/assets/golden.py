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
"""Pow 算子测试规范（TTK TestSpec）。

kernel/GEIR = pow（numpy 入/出）：真值 x1 ** x2（bfloat16 抬 fp32 计算后回转）；
aclnn = aclnnPowTensorScalar / aclnnInplacePowTensorScalar /
aclnnPowScalarTensor / aclnnPowTensorTensor / aclnnInplacePowTensorTensor /
aclnnExp2 / aclnnInplaceExp2（torch 入/出）：out = self ** exponent
（TensorScalar 的 exponent 为标量、ScalarTensor 的 self 为标量；Exp2 为 2**self；
Inplace 写回 selfRef）；
e2e = torch.pow（torch 入/出）：真值 torch.pow(input, exponent)。
"""

import torch

__spec__ = {
    "pow": "PowTestSpec",
    "aclnnPowTensorScalar": "AclnnPowTensorScalarTestSpec",
    "aclnnInplacePowTensorScalar": "AclnnInplacePowTensorScalarTestSpec",
    "aclnnPowScalarTensor": "AclnnPowScalarTensorTestSpec",
    "aclnnPowTensorTensor": "AclnnPowTensorTensorTestSpec",
    "aclnnInplacePowTensorTensor": "AclnnInplacePowTensorTensorTestSpec",
    "aclnnExp2": "AclnnExp2TestSpec",
    "aclnnInplaceExp2": "AclnnInplaceExp2TestSpec",
    "torch.pow": "TorchPowTestSpec",
}

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


def _out_torch_dtype(kwargs):
    tensor_dtypes = tuple(kwargs.get("tensor_dtypes") or ())
    if tensor_dtypes:
        return _DTYPE_STR_TO_TORCH.get(str(tensor_dtypes[-1]).lower())
    return None


def _cast_result(result, out_dtype):
    if out_dtype is None or result.dtype == out_dtype:
        return result
    try:
        return result.to(out_dtype)
    except (RuntimeError, TypeError):
        return result


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
    """torch 三方标杆（各模式共用）：两个输入按位置接收（``self`` 为保留名）。"""

    def __call__(self, *args, **kwargs):
        a, b = _third_party_inputs(
            args, kwargs, second_names=("exponent", "other", "x2")
        )
        return [torch.pow(a, b)]


class PowTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        t1 = numpy_to_torch_tensor(x1)
        t2 = numpy_to_torch_tensor(x2)
        return [torch_to_numpy_tensor(torch.pow(t1, t2))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnPowTensorScalarTestSpec:
    """aclnnPowTensorScalar 流程：out = self ** exponent（exponent 标量）。"""

    def golden(self, exponent, out, **kwargs):
        """参数跟随 aclnnPowTensorScalar：self 张量、exponent 标量、out 张量。"""
        return [
            _cast_result(
                torch.pow(self, _to_scalar(exponent)), _out_torch_dtype(kwargs)
            )
        ]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplacePowTensorScalarTestSpec:
    """aclnnInplacePowTensorScalar 流程：selfRef **= exponent（exponent 标量）。"""

    def golden(selfRef, exponent, **kwargs):
        """参数跟随 aclnnInplacePowTensorScalar：selfRef 张量、exponent 标量。"""
        return [_cast_result(torch.pow(selfRef, _to_scalar(exponent)), selfRef.dtype)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnPowScalarTensorTestSpec:
    """aclnnPowScalarTensor 流程：out = self ** exponent（self 标量）。"""

    def golden(self, exponent, out, **kwargs):
        """参数跟随 aclnnPowScalarTensor：self 标量、exponent 张量、out 张量。"""
        return [
            _cast_result(
                torch.pow(_to_scalar(self), exponent), _out_torch_dtype(kwargs)
            )
        ]

    third_party = {"torch": ThirdPartyImpl}


class AclnnPowTensorTensorTestSpec:
    """aclnnPowTensorTensor 流程：out = self ** exponent（exponent 张量）。"""

    def golden(self, exponent, out, **kwargs):
        """参数跟随 aclnnPowTensorTensor：self 张量、exponent 张量、out 张量。"""
        return [_cast_result(torch.pow(self, exponent), _out_torch_dtype(kwargs))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplacePowTensorTensorTestSpec:
    """aclnnInplacePowTensorTensor 流程：selfRef **= exponent（exponent 张量）。"""

    def golden(selfRef, exponent, **kwargs):
        """参数跟随 aclnnInplacePowTensorTensor：selfRef 张量、exponent 张量。"""
        return [_cast_result(torch.pow(selfRef, exponent), selfRef.dtype)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnExp2TestSpec:
    """aclnnExp2 流程：out = 2 ** self。"""

    def golden(self, out, **kwargs):
        """参数跟随 aclnnExp2：self 张量、out 张量。"""
        return [_cast_result(torch.exp2(self.float()), _out_torch_dtype(kwargs))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceExp2TestSpec:
    """aclnnInplaceExp2 流程：selfRef = 2 ** selfRef。"""

    def golden(selfRef, **kwargs):
        """参数跟随 aclnnInplaceExp2：selfRef 张量。"""
        return [_cast_result(torch.exp2(selfRef.float()), selfRef.dtype)]

    third_party = {"torch": ThirdPartyImpl}


class TorchPowTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, exponent, **kwargs):
        """参数跟随 torch.pow：input/exponent 位置张量。"""
        return [torch.pow(input, exponent)]

    third_party = {"torch": ThirdPartyImpl}
