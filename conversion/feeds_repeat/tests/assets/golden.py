#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""FeedsRepeat 独立 Golden：raw、Torch。

公式: 逐行重复 feeds[i] 共 feeds_repeat_times[i] 次，未填充部分清零。
      输出 y 的 dim0 = output_feeds_size，其余维度与 feeds 相同。

相对 operator-artifacts 原始 golden 的修正（经三轮上板验证）：
  1. 截断语义：sum(feeds_repeat_times) > output_feeds_size 时按 output_feeds_size
     截断（与算子/README 语义一致，原版切片越界直接报错）；
  2. __spec__ 双键注册（feeds_repeat 与 FeedsRepeat），兼容按接口名与算子注册名
     两种 TTK SpecResolver 查找路径；
  3. tolerance 采用 mix_tolerance（无 XPU/GPU 端点的比对标准），third_party（torch）
     保留供 cross-check / 三方通路使用。
"""

import torch

__spec__ = {
    # 同时注册算子注册名与接口名两种写法：
    # "feeds_repeat" 匹配安装元数据声明的 opInterface（Host 归一化后的执行 op_name），
    # "FeedsRepeat" 匹配按算子注册名（REG_OP 名）直接构造的 TTK SpecResolver 查找，
    # 避免任一路径因键大小写不一致而取不到 golden。
    "feeds_repeat": "FeedsRepeatKernelSpec",
    "FeedsRepeat": "FeedsRepeatKernelSpec",
}

_MIX = {
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
    "float32": {"standard": "mix_tolerance"},
}


def _compute(feeds, feeds_repeat_times, output_feeds_size):
    import torch
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    is_torch = isinstance(feeds, torch.Tensor)

    feeds_t = feeds if is_torch else numpy_to_torch_tensor(feeds)
    repeat_t = (
        feeds_repeat_times
        if isinstance(feeds_repeat_times, torch.Tensor)
        else numpy_to_torch_tensor(feeds_repeat_times)
    )

    orig_dtype = feeds_t.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        feeds_t = feeds_t.to(torch.float32)

    # 秩不限但元素总数须等于 feeds 第 0 维：统一按元素展平后线性读取，与 Kernel 行为一致
    repeat_t = repeat_t.to(torch.int64).reshape(-1)

    n = feeds_t.shape[0]
    elem_row_shape = feeds_t.shape[1:]

    y_t = torch.zeros(output_feeds_size, *elem_row_shape, dtype=feeds_t.dtype)

    y_start = 0
    for i in range(n):
        repeat_i = max(0, repeat_t[i].item())
        if repeat_i > 0:
            remain = output_feeds_size - y_start
            if remain <= 0:
                break
            copy_n = min(repeat_i, remain)
            repeat_dims = [copy_n] + [1] * len(elem_row_shape)
            y_t[y_start : y_start + copy_n] = feeds_t[i : i + 1].repeat(*repeat_dims)
        y_start += repeat_i

    y_t = y_t.to(orig_dtype)
    return y_t if is_torch else torch_to_numpy_tensor(y_t.detach().cpu())


def feeds_repeat_golden(feeds, feeds_repeat_times, *, output_feeds_size, **kwargs):
    del kwargs
    return [_compute(feeds, feeds_repeat_times, output_feeds_size)]


class _TorchFeedsRepeat:
    """Third-party reference using torch (native dtype, NO upcasting).

    Uses torch.repeat_interleave for an independent decomposed path
    matching the reference oracle in spec.yaml.
    """

    def __init__(self, *, output_feeds_size, **kwargs):
        del kwargs
        self.output_feeds_size = output_feeds_size

    def __call__(self, feeds, feeds_repeat_times, **kwargs):
        del kwargs
        feeds_dtype = feeds.dtype
        # 秩不限但元素总数须等于 feeds 第 0 维：展平后线性读取，与 Kernel 行为一致
        repeat_t = feeds_repeat_times.to(torch.int64).reshape(-1)
        repeat_list = [max(0, repeat_t[i].item()) for i in range(feeds.shape[0])]

        if sum(repeat_list) == 0:
            repeated = feeds.new_empty((0,) + tuple(feeds.shape[1:]))
        else:
            repeats = torch.tensor(repeat_list, dtype=torch.int64)
            repeated = torch.repeat_interleave(feeds, repeats, dim=0)

        elem_row_shape = feeds.shape[1:]
        # 输出跟随 feeds 的设备与布局，保证设备通路（NPU tensor 输入）上参考可执行
        y = torch.zeros(
            self.output_feeds_size,
            *elem_row_shape,
            dtype=feeds_dtype,
            device=feeds.device,
        )
        n = min(repeated.shape[0], self.output_feeds_size)
        if n > 0:
            y[:n] = repeated[:n].to(feeds.device)
        return [y]


class FeedsRepeatKernelSpec:
    golden = staticmethod(feeds_repeat_golden)
    third_party = {"torch": _TorchFeedsRepeat}
    tolerance = _MIX
