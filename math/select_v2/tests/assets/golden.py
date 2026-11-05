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
"""SelectV2 算子测试规范（TTK TestSpec）。

kernel/GEIR = select_v2（numpy 入/出）：out = condition ? x1 : x2；
aclnn = aclnnSWhere（torch 入/出）：out = condition ? self : other；
e2e = torch_npu.npu_select_v2 / torch.where（torch 入/出）。
"""

import numpy
import torch

__spec__ = {
    "select_v2": "SelectV2TestSpec",
    "aclnnSWhere": "AclnnSWhereTestSpec",
    "torch.where": "TorchWhereTestSpec",
}


def _to_fp32(t):
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.to(torch.float32)
    return t


def _cast_to_out(result, out):
    """按 out 张量的 dtype 回写结果（对齐 NPU 输出 dtype）。"""
    if out is not None and result.dtype != out.dtype:
        return result.to(out.dtype)
    return result


def _where(condition, x1, x2):
    """torch.where 包装，处理 uint32/uint64 等 torch 不支持的原生类型。"""
    cond_t = (
        condition.to(torch.bool)
        if isinstance(condition, torch.Tensor)
        else torch.as_tensor(numpy.asarray(condition).astype(bool))
    )
    x1_t = x1 if isinstance(x1, torch.Tensor) else torch.as_tensor(numpy.asarray(x1))
    x2_t = x2 if isinstance(x2, torch.Tensor) else torch.as_tensor(numpy.asarray(x2))
    x1_t = x1_t.to(cond_t.device)
    x2_t = x2_t.to(cond_t.device)
    promoted = torch.promote_types(x1_t.dtype, x2_t.dtype)
    x1_t = x1_t.to(promoted)
    x2_t = x2_t.to(promoted)
    if promoted == torch.uint32:
        return torch.where(cond_t, x1_t.to(torch.int64), x2_t.to(torch.int64)).to(
            torch.uint32
        )
    if promoted == torch.uint64:
        cn = cond_t.cpu().numpy()
        x1n = x1_t.cpu().numpy()
        x2n = x2_t.cpu().numpy()
        return torch.as_tensor(numpy.where(cn, x1n, x2n), dtype=torch.uint64)
    return torch.where(cond_t, x1_t, x2_t)


def _third_party_inputs(
    args,
    kwargs,
    first_names=("condition", "cond"),
    second_names=("self", "selfT", "x1", "input"),
    third_names=("other", "x2"),
):
    """从三方标杆调用参数中提取三个输入张量（位置优先，kwargs 兜底）。"""

    def _pick(idx, names):
        if len(args) > idx:
            return args[idx]
        for name in names:
            if name in kwargs:
                return kwargs[name]
        return None

    first = _pick(0, first_names)
    second = _pick(1, second_names)
    third = _pick(2, third_names)
    missing = [
        n for n, v in (("condition", first), ("x1", second), ("x2", third)) if v is None
    ]
    if missing:
        raise TypeError(
            "select_v2 third_party: input tensors %s not provided (args=%r, kwargs=%r)"
            % (missing, args, sorted(kwargs))
        )
    return first, second, third


class ThirdPartyImpl:
    """torch 三方标杆（各模式共用）：三个输入按位置接收（``self`` 为保留名）。"""

    def __call__(self, *args, **kwargs):
        c, a, b = _third_party_inputs(args, kwargs)
        return [_where(c, a, b)]


class SelectV2TestSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    @staticmethod
    def golden(condition, x1, x2, **kwargs):
        """kernel/GEIR golden：numpy → torch 计算 → numpy 回转。"""
        from ttk.utilities import numpy_to_torch_tensor, torch_to_numpy_tensor

        input_dtype = numpy.asarray(x1).dtype
        x1_t = numpy_to_torch_tensor(x1)
        x2_t = numpy_to_torch_tensor(x2)
        if x1_t.dtype not in (torch.uint32, torch.uint64):
            x1_t = _to_fp32(x1_t)
            x2_t = _to_fp32(x2_t)
        res = _where(torch.as_tensor(numpy.asarray(condition).astype(bool)), x1_t, x2_t)
        return [torch_to_numpy_tensor(res).astype(input_dtype, copy=False)]

    third_party = {"torch": ThirdPartyImpl}


class AclnnSWhereTestSpec:
    """aclnnSWhere 流程：out = condition ? self : other。"""

    def golden(condition, selfT, other, out, **kwargs):
        """参数跟随 aclnnSWhere：condition、self、other 张量，out 张量。"""
        return [_cast_to_out(_where(condition, selfT, other), out)]

    third_party = {"torch": ThirdPartyImpl}


class TorchWhereTestSpec:
    """e2e 流程：torch 入/出。"""

    def golden(condition, input, other, **kwargs):
        """参数跟随 torch.where：condition/input/other 位置张量。"""
        return [_where(condition, input, other)]

    third_party = {"torch": ThirdPartyImpl}
