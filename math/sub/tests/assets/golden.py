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
"""Sub 算子测试规范（TTK TestSpec）。

kernel/GEIR = sub（numpy 入/出）：真值经 numpy_to_torch_tensor 转 torch 计算
torch.sub（bool 走 logical_xor，complex32 抬 complex64 后逐分量相减），再经
torch_to_numpy_tensor 回转 numpy；
aclnn = aclnnSub / aclnnSubs / aclnnInplaceSub / aclnnInplaceSubs /
aclnnRsub / aclnnRsubs（torch 入/出）：out = self - alpha * other
（Rsub 为 out = other - alpha * self；Subs/Rsubs 的 other 为标量；Inplace 写回
selfRef）；
e2e = torch.sub（torch 入/出）：真值 torch.sub(input, other, alpha)。
"""

import torch

__spec__ = {
    "sub": "SubTestSpec",
    "aclnnSub": "AclnnSubTestSpec",
    "aclnnSubs": "AclnnSubTestSpec",
    "aclnnInplaceSub": "AclnnInplaceSubTestSpec",
    "aclnnInplaceSubs": "AclnnInplaceSubTestSpec",
    "aclnnRsub": "AclnnRsubTestSpec",
    "aclnnRsubs": "AclnnRsubTestSpec",
    "torch.sub": "TorchSubTestSpec",
}


def _to_scalar(v):
    """标量归一化：torch 0-dim/1-element 张量取 item，Python 标量原样返回。"""
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v
    return v


def _bool_xor(a, b):
    """bool 减法（异或）：b 为张量或标量均可（标量广播为 0-dim bool 张量）。"""
    if not torch.is_tensor(b):
        b = torch.as_tensor(b, dtype=torch.bool)
    return torch.logical_xor(a, b)


def _torch_sub(t1, t2, alpha=1.0):
    """torch.sub 包装：整数输入时 alpha 必须为整数（torch 拒绝浮点 alpha）。"""
    if (
        not torch.is_floating_point(t1)
        and not torch.is_complex(t1)
        and isinstance(alpha, float)
        and alpha.is_integer()
    ):
        alpha = int(alpha)
    return torch.sub(t1, t2, alpha=alpha)


def _torch_rsub(t1, t2, alpha=1.0):
    """torch.rsub 包装：整数输入时 alpha 必须为整数。"""
    if (
        not torch.is_floating_point(t1)
        and not torch.is_complex(t1)
        and isinstance(alpha, float)
        and alpha.is_integer()
    ):
        alpha = int(alpha)
    return torch.rsub(t1, t2, alpha=alpha)


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
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
    """torch 三方标杆（各模式共用）：alpha 属性绑定，两个输入按位置接收。

    Sub 的入参名含 Python 保留名 ``self``，故 ``__call__`` 按位置取
    args[0]/args[1] 而非按名绑定。
    """

    def __init__(self, *, alpha=1.0, **kwargs):
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        x, y = _third_party_inputs(args, kwargs)
        if x.dtype == torch.bool:
            a = _to_scalar(self.alpha)
            if bool(a):
                return [_bool_xor(x, y)]
            return [x.clone()]
        return [_torch_sub(x, y, self.alpha)]


class SubTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, alpha=1.0, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        out_dtypes = tuple(kwargs.get("output_dtypes") or ())
        out_dtype = str(out_dtypes[0]).lower() if out_dtypes else None

        if out_dtype == "complex32":
            import numpy

            a = x1[..., 0].astype(numpy.float32) + 1j * x1[..., 1].astype(numpy.float32)
            b = x2[..., 0].astype(numpy.float32) + 1j * x2[..., 1].astype(numpy.float32)
            res = a - b
            return [
                numpy.stack(
                    (res.real.astype(numpy.float16), res.imag.astype(numpy.float16)),
                    axis=-1,
                )
            ]

        t1 = numpy_to_torch_tensor(x1)
        t2 = numpy_to_torch_tensor(x2)
        if out_dtype == "bool":
            return [torch_to_numpy_tensor(torch.logical_xor(t1, t2))]
        return [torch_to_numpy_tensor(_torch_sub(t1, t2, alpha))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnSubTestSpec:
    """aclnnSub / aclnnSubs 流程：out = self - alpha * other（other 张量或标量）。"""

    def golden(self, other, alpha, out, **kwargs):
        """参数跟随 aclnnSub/aclnnSubs：self 张量、other 张量/标量、alpha 标量、out 张量。"""
        if self.dtype == torch.bool:
            a = _to_scalar(alpha)
            if bool(a):
                return [_bool_xor(self, other)]
            return [self.clone()]
        return [_cast_to_out(_torch_sub(self, other, _to_scalar(alpha)), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceSubTestSpec:
    """aclnnInplaceSub / aclnnInplaceSubs 流程：selfRef -= alpha * other。"""

    def golden(selfRef, other, alpha, **kwargs):
        """参数跟随 aclnnInplaceSub/aclnnInplaceSubs：selfRef 张量、other 张量/标量、alpha 标量。"""
        if selfRef.dtype == torch.bool:
            a = _to_scalar(alpha)
            if bool(a):
                return [_bool_xor(selfRef, other)]
            return [selfRef.clone()]
        return [_torch_sub(selfRef, other, _to_scalar(alpha))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnRsubTestSpec:
    """aclnnRsub / aclnnRsubs 流程：out = other - alpha * self（other 张量或标量）。"""

    def golden(self, other, alpha, out, **kwargs):
        """参数跟随 aclnnRsub/aclnnRsubs：self 张量、other 张量/标量、alpha 标量、out 张量。"""
        if self.dtype == torch.bool:
            a = _to_scalar(alpha)
            if bool(a):
                return [_bool_xor(self, other)]
            if torch.is_tensor(other):
                return [other.clone()]
            return [torch.full_like(self, bool(other))]
        return [_cast_to_out(_torch_rsub(self, other, _to_scalar(alpha)), out)]

    third_party = {"torch": ThirdPartyImpl}


class TorchSubTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, alpha=1.0, **kwargs):
        """参数跟随 torch.sub：input/other 位置张量，alpha 标量属性。"""
        return [_torch_sub(input, other, alpha)]

    third_party = {"torch": ThirdPartyImpl}
