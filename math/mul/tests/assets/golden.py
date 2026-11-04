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
"""Mul 算子测试规范（TTK TestSpec）。

kernel/GEIR = mul（numpy 入/出）：真值经 numpy_to_torch_tensor 转 torch 计算
torch.mul（bool 走 logical_and，complex32 拆实虚部分别计算再拼回），再经
torch_to_numpy_tensor 回转 numpy；
aclnn = aclnnMul / aclnnMuls / aclnnInplaceMul / aclnnInplaceMuls（torch 入/出）：
out = self * other（Muls/InplaceMuls 的 other 为标量；Inplace 写回 selfRef）；
e2e = torch.mul（torch 入/出）：真值 torch.mul(input, other)。
"""

import torch

__spec__ = {
    "mul": "MulTestSpec",
    "aclnnMul": "AclnnMulTestSpec",
    "aclnnMuls": "AclnnMulTestSpec",
    "aclnnInplaceMul": "AclnnInplaceMulTestSpec",
    "aclnnInplaceMuls": "AclnnInplaceMulTestSpec",
    "torch.multiply": "TorchMulTestSpec",
}


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
        a, b = _third_party_inputs(args, kwargs)
        return [torch.mul(a, b)]


class MulTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        out_dtypes = tuple(kwargs.get("output_dtypes") or ())
        out_dtype = str(out_dtypes[0]).lower() if out_dtypes else None

        if out_dtype == "complex32":
            import numpy

            xreal = x1[..., 0].astype(numpy.float32)
            yreal = x2[..., 0].astype(numpy.float32)
            ximag = x1[..., 1].astype(numpy.float32)
            yimag = x2[..., 1].astype(numpy.float32)
            zreal = xreal * yreal - ximag * yimag
            zimag = xreal * yimag + ximag * yreal
            return [
                numpy.stack(
                    (zreal.astype(numpy.float16), zimag.astype(numpy.float16)), axis=-1
                )
            ]

        t1 = numpy_to_torch_tensor(x1)
        t2 = numpy_to_torch_tensor(x2)
        if out_dtype == "bool":
            return [torch_to_numpy_tensor(torch.logical_and(t1, t2))]
        return [torch_to_numpy_tensor(torch.mul(t1, t2))]

    third_party = {"torch": ThirdPartyImpl}


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


class AclnnMulTestSpec:
    """aclnnMul / aclnnMuls 流程：out = self * other（other 张量或标量）。"""

    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnMul/aclnnMuls：self 张量、other 张量/标量、out 张量。"""
        return [_cast_to_out(torch.mul(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceMulTestSpec:
    """aclnnInplaceMul / aclnnInplaceMuls 流程：selfRef *= other。"""

    def golden(selfRef, other, **kwargs):
        """参数跟随 aclnnInplaceMul/aclnnInplaceMuls：selfRef 张量、other 张量/标量。"""
        return [torch.mul(selfRef, other)]

    third_party = {"torch": ThirdPartyImpl}


class TorchMulTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, **kwargs):
        """参数跟随 torch.mul：input/other 位置张量。"""
        return [torch.mul(input, other)]

    third_party = {"torch": ThirdPartyImpl}
