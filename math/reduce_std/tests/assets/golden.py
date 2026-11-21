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


__golden__ = {"kernel": {"reduce_std": "reduce_std_golden"}}


def _eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(
        set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis])
    )
    return axis


def reduce_std_golden(
    x, dim=None, unbiased: bool = True, keepdim: bool = False, **kwargs
):
    """
    Kernel golden for reduce_std.
    All the parameters follow @reduce_std_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    x_tensor = torch.from_numpy(x)

    axis = dim
    axis_d = []
    if not axis:
        for i, _ in enumerate(x_tensor.shape):
            axis_d.append(i)
    else:
        axis_d = axis
    axis_d = _eliminate_duplicate_axes(axis_d, x_tensor)

    std, mean = torch.std_mean(x_tensor, dim=axis_d, unbiased=unbiased, keepdim=keepdim)
    return [std.numpy(), mean.numpy()]
