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
"""CPU golden for aclnnNpuFormatCast (ND -> FRACTAL_NZ on Ascend 950).

Only the ND -> FRACTAL_NZ direction (dstFormat = 29) is covered.  Following
the operator documentation the storage layout is, for the last two dims
``[..., M, N]`` of the ND tensor::

    storage shape: [..., ceil(N/C0), ceil(M/16), 16, C0]
    NZ[..., n1, m1, m0, n0] = ND[..., 16*m1+m0, C0*n1+n0]

with ``C0 = 32 / sizeof(additionalDtype)`` and zero filled padding.

Note: ATK only lets an executor set formats for input tensors, so the
FRACTAL_NZ destination tensor cannot be constructed for the device path.
These cases are therefore verified against the CPU golden only.
"""

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_C0_BY_DTYPE = {0: 8, 1: 16, 2: 32, 3: 8, 4: 32, 27: 16}  # f32, f16, i8, i32, u8, bf16


def _c0(additional_dtype):
    return _C0_BY_DTYPE.get(int(additional_dtype), 16)


def _nz_shape(shape, c0):
    shape = [int(d) for d in shape]
    lead, m, n = shape[:-2], shape[-2], shape[-1]
    return lead + [(n + c0 - 1) // c0, (m + 15) // 16, 16, c0]


def _nd_to_nz(array, c0):
    lead = array.shape[:-2]
    m, n = array.shape[-2], array.shape[-1]
    n1 = (n + c0 - 1) // c0
    m1 = (m + 15) // 16
    flat_lead = int(np.prod(lead)) if lead else 1
    src = array.reshape(flat_lead, m, n)
    padded = np.zeros((flat_lead, m1 * 16, n1 * c0), dtype=array.dtype)
    padded[:, :m, :n] = src
    nz = padded.reshape(flat_lead, m1, 16, n1, c0).transpose(0, 3, 1, 2, 4)
    nz = nz.reshape(tuple(lead) + (n1, m1, 16, c0))
    return np.array(nz, dtype=array.dtype, copy=True, order="C")


@register("function_npu_format_cast")
class FunctionNpuFormatCastApi(BaseApi):
    """CPU golden: ND -> FRACTAL_NZ."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return None
        kwargs = input_data.kwargs
        src = kwargs["srcTensor"].detach().cpu()
        dtype = src.dtype
        if dtype == torch.bfloat16:
            array = src.to(torch.float32).numpy()
        else:
            array = src.numpy()
        c0 = _c0(int(kwargs["additionalDtype"]))
        result = _nd_to_nz(array, c0)
        return torch.from_numpy(result).to(dtype)
