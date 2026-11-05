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
"""Gcd 算子测试规范（TTK TestSpec）。

kernel/GEIR = gcd（numpy 入/出）：out = gcd(x1, x2)；
aclnn = aclnnGcd（torch 入/出）：out = gcd(self, other)；
e2e = torch.gcd（torch 入/出）。
"""

import numpy as np
import torch

__spec__ = {
    "gcd": "GcdTestSpec",
    "aclnnGcd": "AclnnGcdTestSpec",
    "torch.gcd": "TorchGcdTestSpec",
}


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v
    if hasattr(v, "item"):
        return v.item()
    return v


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
    """从三方标杆调用参数中提取两个输入张量（位置优先，kwargs 兜底）。"""

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
            "gcd third_party: two input tensors not provided (args=%r, kwargs=%r)"
            % (args, sorted(kwargs))
        )
    return first, second


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：两个输入按位置接收（``self`` 为保留名）。"""

    def __call__(self, *args, **kwargs):
        a, b = _third_party_inputs(args, kwargs)
        return [torch.gcd(a, b)]


class GcdTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """kernel/GEIR golden：numpy 入/出，np.gcd 对负值正确返回非负 GCD。"""
        return [np.gcd(x1, x2)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnGcdTestSpec:
    """aclnnGcd 流程：out = gcd(self, other)。"""

    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnGcd：self、other 张量，out 张量。"""
        return [_cast_to_out(np.gcd(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class TorchGcdTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, **kwargs):
        """参数跟随 torch.gcd：input/other 位置张量。"""
        return [np.gcd(input, other)]

    third_party = {"torch": ThirdPartyImpl}
