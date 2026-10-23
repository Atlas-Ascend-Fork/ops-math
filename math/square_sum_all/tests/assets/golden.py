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
"""TTK TestSpec for SquareSumAll kernel and GEIR tests.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35/square_sum_all.cpp 有 arch35 实现 |
| geir   | ✅   | op_graph/square_sum_all_proto.h 有 REG_OP(SquareSumAll) |
| aclnn  | ❌   | CMakeLists.txt 配置 ACLNNTYPE aclnn_exclude，无 op_api 目录，无 aclnn 文档 |
| e2e    | 未注册 | 尚未验证能独立派发到本算子的框架 API；缺少 ACLNN 符号不能证明没有框架通路 |

kernel 与 geir 共用一个注册键（算子蛇形名），geir 不另写 Spec 类。

精度约定：默认 cross_check 由 TTK Promote 输入并执行 Torch 三方腿。
CPU golden 保持 TTK 下发的精度，不额外加宽或窄回；两方 FP32 输入按
NPU 的 FP32 精度计算。两方运行不代表三方验收通过，缺少 XPU 时应记录
三方验证未完成。两路输出均为 rank-0 标量，与 InferShape 的接口一致。

本 Spec 不实现 compare 钩子：
    TTK 的 try_custom_compare 只要拿到非 None 返回值就直接采信，**跳过标准判据**
    （ttk/core_modules/comparison/custom.py）。自写 compare 会把 cross_check 一起短路掉，
    三方数据永远进不了判定。特殊值（NaN / 同号 Inf）的语义比对 stat_rel_err 与
    cross_check 自身都已实现，不需要算子再包一层。
"""

import numpy as np
import torch

__spec__ = {
    # Kernel and GEIR share the snake-case registration and the same TestSpec.
    "square_sum_all": "SquareSumAllTestSpec",
}

# Retain the repository-facing legacy entry while consumers migrate to TestSpec.
__golden__ = {
    "kernel": {"square_sum_all": "square_sum_all_golden"},
}


def _reduce_one(x):
    """Use the supplied precision: FP32 normally, FP64 after TTK Promote."""
    tensor = torch.from_numpy(np.array(x, copy=True, order="C"))
    reduced = torch.sum(torch.square(tensor))
    return reduced.numpy().reshape(())


def square_sum_all_golden(x1, x2, **kwargs):
    """Compute both reductions without overriding TTK's input/output precision."""
    del kwargs
    return [_reduce_one(x1), _reduce_one(x2)]


class _Compose:
    """Competitor baseline built from a torch interface independent of the golden.

    torch.dot is a different high-level entry point than the golden's
    square-then-sum, so a misreading in one path does not reproduce in the other.
    Parameter names follow the REG_OP registration (x1, x2).

    ⚠️ 累加档位必须与【内核实现】一致，即 float32——不是与 golden 一致。
    内核在 float32 上做核内累加与跨核合并；竞品若像 golden 那样升 float64，它对
    golden 的相对误差会趋近 0。f419709 后 cross_check 按 dtype 的 small_value
    给比值分母夹底（FP32 为 2**-14）；这能消除旧版固定 1e-7 的一部分假红，但
    标量输出仍会把误差比量化成少数 ULP 的比例。因此三方 compose 仍须和内核
    处于同一精度档位。输出 dtype 同样 cast 回 NPU 的 float32。
    """

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        outputs = []
        for tensor in (x1, x2):
            flat = tensor.reshape(-1).to(torch.float32)
            outputs.append(torch.dot(flat, flat).to(torch.float32).reshape(()))
        return outputs


class SquareSumAllTestSpec:
    """SquareSumAll test contract shared by the TTK kernel and GEIR paths."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return square_sum_all_golden(x1, x2, **kwargs)

    third_party = {"torch": _Compose}
    # cross_check L1：mare<=5 / mere<=1.5 / rmse<=1.5（见 resolve.LEVEL_PRESETS）。
    # 取 L1 而非 L2：内核按分核结果合并，累加顺序与竞品的单核归约不同，比值天然有
    # 抖动；平方和全为非负项、无对消，条件数好，L1 足以拦住真实劣化。
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}


# 【不存在】aclnn 通路：CMakeLists.txt 配置 ACLNNTYPE aclnn_exclude，算子无 op_api 目录，
# 也不交付 docs/aclnnSquareSumAll.md，因此不注册 SquareSumAllAclnnSpec。
# E2E 暂未注册：本文件只验证 Kernel/GEIR 路径，没有确认独立框架 API 的派发绑定。
# 不以缺少同名 ACLNN 符号推断所有框架路径均不存在。
