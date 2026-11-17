#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ATK adapter for aclnnStridedSlice.

``function_strided_slice`` (BaseApi) implements the CPU golden in numpy
following the TensorFlow ``strided_slice`` semantics used by the operator:
``begin`` / ``end`` / ``strides`` plus the five masks (begin, end, ellipsis,
new_axis, shrink_axis).  The NPU side is driven by ``aclnn_name`` so no
pyaclnn wrapper class is required (single output, ND format).
"""

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _slice_axis(arr, axis, begin, end, stride, begin_mask, end_mask):
    """Slice ``arr`` along ``axis`` following TF strided_slice semantics."""
    dim = arr.shape[axis]
    stride = 1 if stride == 0 else stride

    if stride > 0:
        if begin_mask:
            b = 0
        else:
            b = begin + dim if begin < 0 else begin
            b = min(max(b, 0), dim)
        if end_mask:
            e = dim
        else:
            e = end + dim if end < 0 else end
            e = min(max(e, 0), dim)
    else:
        if begin_mask:
            b = dim - 1
        else:
            b = begin + dim if begin < 0 else begin
            b = min(max(b, 0), max(dim - 1, 0))
        if end_mask or end == -1 or end < -dim:
            e = None
        else:
            e = end + dim if end < 0 else end
            e = min(max(e, 0), dim)

    index = [slice(None)] * arr.ndim
    index[axis] = slice(b, e, stride)
    return arr[tuple(index)]


def _strided_slice(
    arr,
    begin,
    end,
    strides,
    begin_mask,
    end_mask,
    ellipsis_mask,
    new_axis_mask,
    shrink_mask,
):
    """NumPy implementation of TF ``strided_slice`` including all masks."""
    rank = arr.ndim
    length = len(begin)

    spec = []
    for i in range(length):
        if (new_axis_mask >> i) & 1:
            spec.append(("new", i))
        elif (ellipsis_mask >> i) & 1:
            spec.append(("ellipsis", i))
        else:
            spec.append(("slice", i))

    if any(item[0] == "ellipsis" for item in spec):
        consumed = sum(1 for item in spec if item[0] != "ellipsis")
        fill = max(rank - consumed, 0)
        expanded = []
        for item in spec:
            if item[0] == "ellipsis":
                expanded.extend([("full", item[1])] * fill)
            else:
                expanded.append(item)
    else:
        expanded = list(spec)

    out = arr
    axis = 0
    for kind, idx in expanded:
        if kind == "new":
            out = np.expand_dims(out, axis)
            axis += 1
        elif kind == "full":
            axis += 1
        elif (shrink_mask >> idx) & 1:
            dim = out.shape[axis]
            pos = int(begin[idx])
            if pos < 0:
                pos += dim
            out = np.take(out, pos, axis=axis)
        else:
            out = _slice_axis(
                out,
                axis,
                int(begin[idx]),
                int(end[idx]),
                int(strides[idx]),
                (begin_mask >> idx) & 1,
                (end_mask >> idx) & 1,
            )
            axis += 1
    return out


@register("function_strided_slice")
class FunctionStridedSliceApi(BaseApi):
    """CPU golden for aclnnStridedSlice."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return None
        kwargs = input_data.kwargs
        self_tensor = kwargs["self"].detach().cpu()
        dtype = self_tensor.dtype
        if dtype in (torch.float16, torch.bfloat16):
            src = self_tensor.to(torch.float32).numpy()
        else:
            src = self_tensor.numpy()

        result = _strided_slice(
            src,
            kwargs["begin"],
            kwargs["end"],
            kwargs["strides"],
            int(kwargs["beginMask"]),
            int(kwargs["endMask"]),
            int(kwargs["ellipsisMask"]),
            int(kwargs["newAxisMask"]),
            int(kwargs["shrinkAxisMask"]),
        )
        # numpy keeps negative strides for size-1 axes even after
        # ascontiguousarray(); an explicit C-order copy makes the array
        # acceptable to torch.from_numpy.
        result = np.array(result, dtype=result.dtype, copy=True, order="C")
        output = torch.from_numpy(result)
        return output.to(dtype)
