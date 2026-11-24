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
"""AdjacentDifference golden（TestSpec 范式）。

通路支持表（本算子实际交付面）：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | 是   | op_kernel/arch35/adjacent_difference_kernel.h 有实现 |
| geir   | 是   | op_graph/adjacent_difference_proto.h 有 REG_OP(AdjacentDifference)，复用 kernel Spec |
| aclnn  | 否   | op_api 仅有 l0op::AdjacentDifference 内部入口，无公开 aclnnAdjacentDifference 两段式接口 |
| e2e    | 否   | torch_npu / op-plugin 无本算子绑定，无公开框架 API |

数学语义（op_graph/adjacent_difference_proto.h 与 op_kernel 的 CMPMODE::NE）：
相邻**不等标志**，不是数值差分。按展平顺序比较：y[0] = 0，y[i] = (x[i] != x[i-1]) ? 1 : 0，
输出形状恢复为输入形状。`torch.diff` 是相邻数值差，语义不同，不能作等价参考。
输出 dtype 由 y_dtype 属性决定：9（ge::DT_INT64）→ int64，否则（默认 3，ge::DT_INT32）→ int32。
"""

import numpy as np

# __spec__ 为 TestSpec 显式注册（loader 优先走这里）；__golden__ 保留给旧扫描器兜底。
__golden__ = {"kernel": {"adjacent_difference": "adjacent_difference_golden"}}

__spec__ = {"adjacent_difference": "AdjacentDifferenceKernelSpec"}


def _is_int64_output(y_dtype) -> bool:
    """把 y_dtype 属性归一为"输出是否为 int64"（ge::DT_INT64 == 9）。"""
    if y_dtype is None:
        return False
    if isinstance(y_dtype, str):
        return y_dtype.strip().lower() in ("9", "int64", "dt_int64")
    try:
        return int(y_dtype) == 9
    except (TypeError, ValueError):
        return False


def adjacent_difference_golden(x, y_dtype: int = 3, **kwargs):
    """
    Kernel golden for adjacent_difference（kernel 与 geir 共用，numpy 进 numpy 出）。
    All the parameters follow @adjacent_difference_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
             input_formats, output_formats, input_ori_formats, output_ori_formats,
             input_dtypes, output_dtypes.

    比较在输入原始 dtype 上做：算子支持 uint16/uint32/uint64，任何向浮点的
    中转都会把高位相邻值判成相等（例如 2**64-1 与 2**64-2 转 float64 后相等），
    因此这里不做数值类型转换，直接用 numpy 的同 dtype `!=`。
    """

    original_shape = x.shape
    x = x.flatten()
    total_size = np.prod(x.shape)
    ret = np.zeros(x.shape, np.int64 if _is_int64_output(y_dtype) else np.int32)

    if total_size == 0:
        return ret.reshape(original_shape)

    ret[0] = 0
    ret[1:] = (x[1:] != x[0:-1]).astype(np.int32)

    return ret.reshape(original_shape)


class _KernelThirdParty:
    """kernel/geir 通路的三方标杆：用 torch 的相邻 `!=` 独立组合实现。

    这是三方参考候选，不是本算子新增的框架 API。保持输入所在设备，按契约
    构造整型 0/1 输出并恢复输入形状。
    """

    def __init__(self, y_dtype: int = 3, **kwargs):
        self._int64_out = _is_int64_output(y_dtype)

    def __call__(self, x, **kwargs):
        import torch

        ori_shape = tuple(x.shape)
        flat = x.reshape(-1)
        out_dtype = torch.int64 if self._int64_out else torch.int32
        ret = torch.zeros(flat.numel(), dtype=out_dtype, device=x.device)
        if flat.numel() > 1:
            ret[1:] = (flat[1:] != flat[:-1]).to(out_dtype)
        return [ret.view(ori_shape)]


class AdjacentDifferenceKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。"""

    golden = adjacent_difference_golden
    third_party = {"torch": _KernelThirdParty}
    # 输出是 0/1 标志，差 1 即错：锁定逐 bit 相等，防默认混合容差把标志位漂移放过。
    tolerance = {
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }


# 本算子不交付 aclnn / e2e 通路：op_api 只有 l0op::AdjacentDifference 内部 L0 入口，
# torch_npu 无绑定。不构造占位 API，也不注册这两个通路的 Spec。
