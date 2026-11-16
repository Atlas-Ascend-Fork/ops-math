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
"""
AcosGrad TestSpec (golden.py)

通路: GEIR (kernel plugin 模式) + Torch 图模式 (GEIR 复用 kernel spec)
公式: z = -dy / sqrt(1 - y^2)，element-wise
输入: y (Acos 正向输入, 值域 [-1,1]), dy (上游梯度); 输出: z (与 y 同
shape/dtype 的梯度); dtype: FP16, FP32, BF16

纯函数契约: golden 不消费任何测试框架的模式字段, 到达槽 dtype 即计算
精度语义: FP16/BF16 升 FP32 计算后单次舍入 (RNE) 回投原生表达 —— FP16
回投 FP16, BF16 经 int16 位视图携带; FP32/FP64 原生计算原生返回。

执行径 = torch.compile 融合 kernel (Inductor); third_party 同径。
"""

__spec__ = {
    "acos_grad": "AcosGradTestSpec",
    "AcosGrad": "AcosGradTestSpec",  # GE op 类型名变体 (REG_OP(AcosGrad))
}

import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def _fused():
    """torch.compile 融合函数 (进程内单例)。dynamic=True 与 cache_size_limit
    抬高防止多 dtype 图变体超限后静默回落 eager。"""
    import torch

    if torch._dynamo.config.cache_size_limit < 64:
        torch._dynamo.config.cache_size_limit = 64
    return torch.compile(lambda a, b: b * -torch.rsqrt(1 + a * -a), dynamic=True)


def _compute(y, dy):
    """z = -dy / sqrt(1 - y^2)。flatten 为纯视图, 零数值影响。"""
    return _fused()(y.reshape(-1), dy.reshape(-1)).reshape(y.shape)


def _to_torch(arr):
    """numpy -> torch (纯格式转换，不涉及计算)。

    bfloat16 经 int16 位视图桥接 (位型携带)。
    """
    import torch

    if isinstance(arr, torch.Tensor):
        return arr
    if arr is None:
        return None
    dtype_name = arr.dtype.name if hasattr(arr.dtype, "name") else str(arr.dtype)
    if "bfloat16" in dtype_name:
        np_int16 = arr.view(dtype=np.int16)
        return torch.from_numpy(np_int16).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _from_torch(t):
    """torch -> numpy (纯格式转换,不涉及计算)。

    bfloat16 无 numpy 原生表达: 经 int16 位视图携带位型。
    """
    import torch

    if not isinstance(t, torch.Tensor):
        return t
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy()
    return t.numpy()


class ThirdPartyImpl:
    """三方标杆实现: torch.compile 融合执行, 与 golden 同径。"""

    def __call__(self, y, dy, **kwargs):
        return [_compute(y, dy)]


class AcosGradTestSpec:
    """AcosGrad 测试规范 (kernel/geir 流程，golden 收到 numpy.ndarray)。"""

    def golden(y, dy, **kwargs):
        """AcosGrad golden: z = -dy / sqrt(1 - y^2)；kernel/geir 入参为
        numpy.ndarray，返回 [numpy.ndarray]。"""
        import torch

        input_is_numpy = not isinstance(y, torch.Tensor)
        if input_is_numpy:
            y = _to_torch(y)
        if not isinstance(dy, torch.Tensor):
            dy = _to_torch(dy)

        z = _compute(y, dy)

        if input_is_numpy:
            return [_from_torch(z)]
        return [z]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L2"},
        "float16": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }
