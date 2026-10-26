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
"""TileWithAxis 独立 Golden：Kernel/GEIR。

沿指定 axis 将输入 x 复制 tiles 份并整块拼接（等价 concat([x]*tiles, dim=axis)），
仅 Caffe Tile 与之一一对应；PyTorch/TF 均无同名单算子 API
（torch.repeat / tf.tile 为全维度 multiples 形式，torch.repeat_interleave /
tf.repeat 为交错复制、语义不符）。
因此 golden 与三方标杆均以 torch 小算子拼接实现，并通过
torch_graph（torch.nn.Module）提供 TTK E2E torch.compile 模式的图封装。

TTK kernel 模式消费：
    python3 -m ttk kernel -i xxx.csv --plugin tile_with_axis_golden_torch.py

注册名 tile_with_axis 与 CSV 的 op_name 一致；
golden 入参为 numpy.ndarray（kernel 流程），返回 list（每个输出一个元素）。
"""

import numpy as np
import torch

__spec__ = {"tile_with_axis": "TileWithAxisTestSpec"}

# 纯拷贝算子容差口径：
#   - TTK tolerance 声明（本文件 _TOLERANCE）：整型 binary_equal；浮点同为 binary_equal
#     （拷贝语义下 golden 与 kernel 输出应逐位一致，为最严声明）。
#   - spec.yaml numerical_tolerance 与 CSV precision_tolerances 对浮点另声明 rtol/atol
#     （fp16 1e-3 / bf16 4e-3 / fp32 1e-5，stat_rel_err 口径），为三方精度报告的宽容差上限；
#     实际 run 收据按 stat_rel_err 记录。两者宽严不同但均为拷贝语义下的合法声明，不构成冲突。
_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
}


def _tile_with_axis_compute(x, axis, tiles):
    """核心计算（torch 小算子拼接）：y = cat([x]*tiles, dim=axis)。

    整块复制拼接、非交错；负索引先规约到 [0, rank)。
    rank=0 标量输入（人工裁决 2026-10-20：支持）：输出 [tiles]，axis 不生效。
    """
    axis = int(axis)
    tiles = int(tiles)
    if tiles <= 0:
        raise ValueError(f"tiles must be > 0, got {tiles}")
    if x.dim() == 0:
        return x.repeat(tiles)
    if axis < 0:
        axis += x.dim()
    return torch.cat([x] * tiles, dim=axis)


def _numpy_to_torch(np_array):
    if np_array is None:
        return None
    if "bfloat16" in str(np_array.dtype):
        return torch.from_numpy(np_array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np_array)


def _torch_to_numpy(torch_tensor, orig_dtype=None):
    if torch_tensor is None:
        return None
    if torch_tensor.dtype == torch.bfloat16:
        target = orig_dtype if orig_dtype is not None else np.dtype(np.int16)
        return torch_tensor.view(torch.int16).numpy().view(target)
    return torch_tensor.numpy()


def tile_with_axis_golden(x, *, axis, tiles, **kwargs):
    """Golden function for tile_with_axis (kernel mode, numpy in / numpy out)。"""
    del kwargs
    orig_dtype = x.dtype
    x_torch = _numpy_to_torch(x)
    result = _tile_with_axis_compute(x_torch, axis, tiles)
    return [_torch_to_numpy(result, orig_dtype)]


class _TorchTileWithAxis:
    """竞品 third_party：torch 小算子拼接实现（无一一对应竞品算子）。

    __init__ 吸收 CSV attributes（axis / tiles）；
    __call__ 直接使用输入张量计算，无多余 .to() 等数据转换，保证性能统计准确。
    输入支持位置参数 torch.Tensor 或按名 kwargs（x / input）。
    """

    def __init__(self, *, axis=1, tiles=1, **kwargs):
        del kwargs
        self.axis = axis
        self.tiles = tiles

    def __call__(self, *args, **kwargs):
        x = next((a for a in args if isinstance(a, torch.Tensor)), None)
        if x is None:
            x = kwargs.get("x", kwargs.get("input"))
        if x is None:
            raise ValueError(
                "Missing tensor input; expected positional Tensor or one of: x, input"
            )
        return [_tile_with_axis_compute(x, self.axis, self.tiles)]


class TileWithAxisGraph(torch.nn.Module):
    """Graph module consumed by TTK E2E torch.compile modes.

    与 third_party 同一核心拼接实现，属性经 __init__ 固化为图常量。
    """

    def __init__(self, *, axis=1, tiles=1, **kwargs):
        super().__init__()
        del kwargs
        self.axis = axis
        self.tiles = tiles

    def forward(self, x, **kwargs):
        del kwargs
        return _tile_with_axis_compute(x, self.axis, self.tiles)


class TileWithAxisTestSpec:
    """TileWithAxis operator golden spec (kernel flow: numpy in, numpy out).

    x:     input Tensor (numpy.ndarray)
    axis:  int, the axis along which to tile
    tiles: int, number of copies to concatenate along axis
    """

    golden = staticmethod(tile_with_axis_golden)
    third_party = {"torch": _TorchTileWithAxis}
    torch_graph = TileWithAxisGraph
    tolerance = _TOLERANCE
