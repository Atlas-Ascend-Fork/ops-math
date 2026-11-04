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
"""Add 算子测试规范（TTK TestSpec）。

kernel/GEIR = add（numpy 入/出）：真值经 numpy_to_torch_tensor 转 torch 计算
torch.add（fp16/bf16 抬 fp32 中间精度以匹配 PrecisionReduceFlag），再经
torch_to_numpy_tensor 回转 numpy；浮点档走 cross_check 三方交叉校验（对比
torch 标杆）；
aclnn = aclnnAdd / aclnnAdds / aclnnInplaceAdd / aclnnInplaceAdds（torch 入/出）：
真值 out = self + alpha * other（Adds 的 other 为标量；Inplace 写回 selfRef）；
e2e = torch.add（torch 入/出）：真值 torch.add(input, other, alpha)。
"""

import torch

__spec__ = {
    "add": "AddTestSpec",
    "aclnnAdd": "AclnnAddTestSpec",
    "aclnnAdds": "AclnnAddTestSpec",
    "aclnnInplaceAdd": "AclnnInplaceAddTestSpec",
    "aclnnInplaceAdds": "AclnnInplaceAddTestSpec",
    "torch.add": "TorchAddTestSpec",
}


def _torch_add(t1, t2, alpha=1.0):
    """torch.add 包装：整数输入时 alpha 必须为整数（torch 拒绝浮点 alpha）。"""
    if (
        not torch.is_floating_point(t1)
        and isinstance(alpha, float)
        and alpha.is_integer()
    ):
        alpha = int(alpha)
    return torch.add(t1, t2, alpha=alpha)


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

    Add 的入参名含 Python 保留名 ``self``，故 ``__call__`` 按位置取
    args[0]/args[1] 而非按名绑定。
    """

    def __init__(self, *, alpha=1.0, **kwargs):
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        a, b = _third_party_inputs(args, kwargs)
        return [_torch_add(a, b, self.alpha)]


class AddTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, alpha=1.0, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        x1_torch = numpy_to_torch_tensor(x1)
        x2_torch = numpy_to_torch_tensor(x2)
        return [torch_to_numpy_tensor(_torch_add(x1_torch, x2_torch, alpha))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnAddTestSpec:
    """aclnnAdd / aclnnAdds 流程：out = self + alpha * other（other 张量或标量）。"""

    def golden(self, other, alpha, out, **kwargs):
        """参数跟随 aclnnAdd/aclnnAdds：self 张量、other 张量/标量、alpha 标量、out 张量。"""
        return [_cast_to_out(_torch_add(self, other, alpha), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnInplaceAddTestSpec:
    """aclnnInplaceAdd / aclnnInplaceAdds 流程：selfRef += alpha * other（other 张量或标量）。"""

    def golden(selfRef, other, alpha, **kwargs):
        """参数跟随 aclnnInplaceAdd/aclnnInplaceAdds：selfRef 张量、other 张量/标量、alpha 标量。"""
        return [_torch_add(selfRef, other, alpha)]

    third_party = {"torch": ThirdPartyImpl}


class TorchAddTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, alpha=1.0, **kwargs):
        """参数跟随 torch.add：input/other 位置张量，alpha 标量属性。"""
        return [_torch_add(input, other, alpha)]

    third_party = {"torch": ThirdPartyImpl}
