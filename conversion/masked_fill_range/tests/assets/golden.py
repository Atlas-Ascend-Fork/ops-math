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

"""MaskedFillRange 三方 golden（third_party 三方标杆，kernel/geir 通路）。

本算子仅支持 kernel/geir 通路（aclnn/e2e 不适用）。PyTorch/TensorFlow
官方均无 masked_fill_range 现成竞品接口（SE 需求分析 req_analysis.md §5：
其语义需 masked_fill + arange 比较 + where 三算子组合才能表达），故三方
标杆采用类形式 ThirdPartyImpl（torch vendor）：以小算子拼接获取竞品
性能对照，入参 torch.Tensor（框架按 vendor 自动转换），可参与三方交叉
校验与性能比对。

拼接实现与 tests/assets/golden.py 的 TIK 语义基准（A2/A3 canndev
ops/built-in/tbe/impl/masked_fill_range.py）逐分支对齐：逐轮 i（后轮覆盖
前轮）构造命中掩码后 torch.where(mask, value[i], y)——

    1D             : k ∈ [max(s,0), min(e,D)) clamp 截断，仅取第 0 窗口 start[i,0]
                    （B>1 时其余窗口列忽略，不取并集）
    2D（恒最后一维）: k ∈ [max(s,0), min(e,D)) clamp 截断，逐行掩码
    3D axis 0      : n ∈ [s,e) 裸比较整片（无 clamp）→ y[n, :, :] = value[i]
    3D axis 1      : c ∈ [s,e) 裸比较掩码行       → y[n, c, :] = value[i]
    3D axis 2      : k ∈ [max(s,0), min(e,D)) clamp 截断 → y[n, :, k] = value[i]

同轮 i 内各行填充值恒为 value[i]，故逐行独立掩码（2D/3D）与基准的逐 j
顺序覆写严格等价；窗口列恒取行所属 batch 索引 start[i,n]，1D 基准仅读
start[i,0]（start_ub[0]）。行 n >= batches 为 TIK 未定义域（基准遍历上界
min(N, batches)），掩码补 INT_MAX 恒不命中。区间外元素经 torch.where 选择保留，比特级不动
（含 nan/inf/-0）。纯选择型算子，无浮点算术，三方实现与语义基准逐位
一致 → 全 dtype binary_equal（严于 cross_check，不因放宽阈值漏检填充错位）。
"""

__spec__ = {
    "MaskedFillRange": "MaskedFillRangeKernelSpec",
}

import numpy as np
import torch

_BINARY = {
    dt: {"standard": "binary_equal"} for dt in ("float16", "float32", "int8", "int32")
}

_INT32_MAX = torch.iinfo(torch.int32).max


def _rows_fit(start_i, end_i, rows, device):
    """裸比较域（3D axis 0/1）窗口行对齐：截断到 rows 或补 INT_MAX。

    基准遍历上界 min(N, batches)：batches > rows 时尾部窗口丢弃；
    batches < rows 时超界行为 TIK 未定义域，补 INT_MAX 使 (idx >= s)
    恒假、整行/整片不命中。
    """
    s, e = start_i[:rows], end_i[:rows]
    pad = rows - s.shape[0]
    if pad > 0:
        big = torch.full((pad,), _INT32_MAX, dtype=torch.int32, device=device)
        s = torch.cat([s, big])
        e = torch.cat([e, big])
    return s, e


def _mask_fit(hit, rows):
    """clamp 域（2D/3D axis2）行掩码对齐到 x 的行数。

    基准语义只填充前 min(B, batches) 行：batches >= B 截断，batches < B
    尾部补 False。
    """
    if hit.shape[0] >= rows:
        return hit[:rows]
    return torch.nn.functional.pad(hit, (0, 0, 0, rows - hit.shape[0]))


def _prep(x, start, axis):
    """预计算上下文：axis 归一化、执行分支、arange 列坐标与行对齐参数。

    仅依赖静态 shape/device 元数据（host 侧可见，无设备同步），可整体
    前置到 ThirdPartyImpl.__init__（__call__ 只留纯计算）。
    """
    dim = x.dim()
    a = dim + axis if axis < 0 else axis
    if not 0 <= a < dim:
        # 报错前半句与 NPU 侧 InferShapeMaskedFillRange 文案对齐（axis %ld is
        # out of range for input dim %ld），negative case 两端日志可按同一 err
        # 检索匹配；负轴仅归一化一次（axis += dim 不二次折叠，axis=-dim-1 级
        # 越界负轴归一化后仍 <0），合法范围 [-dim, dim-1]。
        raise ValueError(
            f"axis {axis} is out of range for input dim {dim} "
            f"(negative axis normalizes only once, valid range [{-dim}, {dim - 1}])"
        )
    ctx = {"dim": dim, "a": a, "device": x.device, "batches": start.shape[1]}
    if dim == 3 and a == 0:
        ctx["mode"] = "plane"  # n ∈ [s,e) 整片命中
        ctx["rows"] = x.shape[0]
        ctx["idx"] = torch.arange(x.shape[0], device=x.device)
    elif dim == 3 and a == 1:
        ctx["mode"] = "channel"  # c ∈ [s,e) 掩码行命中
        ctx["rows"] = x.shape[0]
        ctx["idx"] = torch.arange(x.shape[1], device=x.device)
    else:
        ctx["mode"] = "window"  # k ∈ [max(s,0), min(e,D)) clamp 窗口
        ctx["rows"] = 1 if dim == 1 else x.shape[0]
        ctx["idx"] = torch.arange(x.shape[-1], device=x.device)
    return ctx


def _apply(ctx, x, start, end, value):
    """纯计算：逐轮构造命中掩码 + torch.where（后轮覆盖前轮）。

    与基准 masked_fill_range_numpy 逐分支同构；无 .cpu()/.numpy()/
    .item() 设备同步，NPU 上全程 tensor 计算。
    """
    y = x.clone()
    # 空 tensor（numel==0）no-op 早退：与竞品 PyTorch 语义一致（req_analysis.md
    # §3.6.1 gpu_kernel 对 numel==0 直接 return，y 与 x 同形同值）；下方逐轮
    # 掩码/where/pad 对齐对零元素无意义，且 torch.nn.functional.pad 不接受
    # size-0 维补零（RuntimeError: negative output size），须在分支前返回。
    if y.numel() == 0:
        return y
    idx = ctx["idx"]
    for i in range(start.shape[0]):
        v = value[i]
        if ctx["mode"] == "plane":
            s, e = _rows_fit(start[i], end[i], ctx["rows"], ctx["device"])
            hit = (idx >= s) & (idx < e)
            y = torch.where(hit.view(ctx["rows"], 1, 1), v, y)
        elif ctx["mode"] == "channel":
            s, e = _rows_fit(start[i], end[i], ctx["rows"], ctx["device"])
            hit = (idx.unsqueeze(0) >= s.unsqueeze(1)) & (
                idx.unsqueeze(0) < e.unsqueeze(1)
            )
            y = torch.where(hit.unsqueeze(2), v, y)
        else:
            d = idx.shape[0]
            s = start[i].clamp_min(0)
            e = end[i].clamp_max(d)
            hit = (idx.unsqueeze(0) >= s.unsqueeze(1)) & (
                idx.unsqueeze(0) < e.unsqueeze(1)
            )
            if ctx["dim"] == 1:
                # TIK 基准 1D 每轮仅读 start[i,0]/end[i,0]（start_ub[0]），
                # B>1 时其余窗口列忽略——不取并集
                hit = hit[0]
            else:
                hit = _mask_fit(hit, ctx["rows"])
                if ctx["dim"] == 3:
                    hit = hit.unsqueeze(1)
            y = torch.where(hit, v, y)
    return y


class ThirdPartyImpl:
    """三方标杆（torch vendor）：arange 区间比较 + where 选择拼接。

    参数绑定契约：__init__ / __call__ 均按参数名从算子输入/属性池取值。
    __init__ 前置全部静态预计算（axis 归一化、分支决策、arange 列坐标），
    __call__ 只留逐轮掩码 + where 纯计算（计入性能比对时间）。
    """

    def __init__(self, x, start, end, value, *, axis=-1, **kwargs):
        self.ctx = _prep(x, start, axis)

    def __call__(self, x, start, end, value, **kwargs):
        return [_apply(self.ctx, x, start, end, value)]


def _golden_numpy_in(x, start, end, value, axis=-1, **kwargs):
    """kernel/geir golden：numpy.ndarray 入参 → torch 计算 → 转回 numpy。"""
    axis = kwargs.get("axisOptional", axis)
    x_t = torch.from_numpy(np.ascontiguousarray(x))
    start_t = torch.from_numpy(np.ascontiguousarray(start))
    end_t = torch.from_numpy(np.ascontiguousarray(end))
    value_t = torch.from_numpy(np.ascontiguousarray(value))
    y = _apply(_prep(x_t, start_t, axis), x_t, start_t, end_t, value_t)
    return [y.numpy()]


# TTK compatibility layer: __golden__ dict expected by ttk's scanner
# （geir 通路的 golden 走旧约定兼容层：custom_plugin_scanner 按 AST 扫模块级
# __golden__ 字典，Spec 类 golden 仅 kernel 模式识别；对齐 ttk/pdist.py 样板。
# 部署到 TTK 时随本文件一起拷入 ttk/masked_fill_range.py。）
__golden__ = {
    "kernel": {"masked_fill_range": "_golden_numpy_in"},
}


class MaskedFillRangeKernelSpec:
    """kernel / geir 通路（op_name=MaskedFillRange，golden 收 numpy.ndarray）。

    geir 复用本类（注册名仍为 op_name）。third_party 为 torch vendor 的
    ThirdPartyImpl 小算子拼接实现：无竞品现成接口，以拼接方式获取竞品
    性能对照与三方交叉校验。
    """

    golden = staticmethod(_golden_numpy_in)
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _BINARY
