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
"""Minimum 算子测试规范（TTK TestSpec）。

kernel/GEIR = minimum（numpy 入/出）：out = elementwise_min(x1, x2)；
aclnn = aclnnMinimum（self+other→out）/ aclnnMinN（TensorList→out）（torch 入/出）；
e2e = torch.minimum（torch 入/出）。
"""

import torch

__spec__ = {
    "minimum": "MinimumTestSpec",
    "aclnnMinimum": "AclnnMinimumTestSpec",
    "aclnnMinN": "AclnnMinNTestSpec",
    "torch.minimum": "TorchMinimumTestSpec",
}


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
            "minimum third_party: two input tensors not provided (args=%r, kwargs=%r)"
            % (args, sorted(kwargs))
        )
    return first, second


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：两个输入按位置接收。"""

    def __call__(self, *args, **kwargs):
        a, b = _third_party_inputs(args, kwargs)
        return [torch.minimum(a, b)]


class MinNThirdPartyImpl:
    """aclnnMinN 三方标杆：对 TensorList 逐元素 torch.minimum 归约。"""

    def __call__(self, *args, **kwargs):
        tensors = kwargs.get("tensors")
        if tensors is None and args:
            tensors = args[0]
        if not tensors:
            raise ValueError("aclnnMinN third_party: empty tensors list")
        result = tensors[0]
        for t in tensors[1:]:
            result = torch.minimum(result, t)
        return [result]


class MinimumTestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        x1_torch = numpy_to_torch_tensor(x1)
        x2_torch = numpy_to_torch_tensor(x2)
        return [torch_to_numpy_tensor(torch.minimum(x1_torch, x2_torch))]

    third_party = {"torch": ThirdPartyImpl}


class AclnnMinimumTestSpec:
    """aclnnMinimum 流程：out = minimum(self, other)。"""

    def golden(self, other, out, **kwargs):
        """参数跟随 aclnnMinimum：self、other 张量，out 张量。"""
        return [_cast_to_out(torch.minimum(self, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnMinNTestSpec:
    """aclnnMinN 流程：out = minimum over tensors list。"""

    def golden(tensors, out, **kwargs):
        """参数跟随 aclnnMinN：tensors 为 aclTensorList，out 张量。"""
        if not tensors:
            raise ValueError("aclnnMinN: empty tensors list")
        result = tensors[0]
        for t in tensors[1:]:
            result = torch.minimum(result, t)
        return [_cast_to_out(result, out)]

    third_party = {"torch": MinNThirdPartyImpl}


class TorchMinimumTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(input, other, **kwargs):
        """参数跟随 torch.minimum：input/other 位置张量。"""
        return [torch.minimum(input, other)]

    third_party = {"torch": ThirdPartyImpl}
