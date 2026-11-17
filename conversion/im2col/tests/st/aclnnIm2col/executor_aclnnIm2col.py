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
"""ATK adapter for aclnnIm2col.

``function_im2col`` implements the CPU golden with numpy (channel-major,
kernel-major ordering as produced by the arch35 NCHW kernel); the device call
is driven by ``aclnn_name`` (single ND output).
"""

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _im2col(x, kernel, dilation, padding, stride):
    squeeze = x.ndim == 3
    if squeeze:
        x = x[np.newaxis, ...]
    batch, channels, height, width = x.shape
    kh, kw = int(kernel[0]), int(kernel[1])
    dh, dw = int(dilation[0]), int(dilation[1])
    ph, pw = int(padding[0]), int(padding[1])
    sh, sw = int(stride[0]), int(stride[1])

    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = np.zeros((batch, channels * kh * kw, out_h * out_w), dtype=x.dtype)
    col = np.arange(channels) * (kh * kw)

    for oh in range(out_h):
        for ow in range(out_w):
            for i in range(kh):
                src_h = oh * sh - ph + i * dh
                if src_h < 0 or src_h >= height:
                    continue
                for j in range(kw):
                    src_w = ow * sw - pw + j * dw
                    if src_w < 0 or src_w >= width:
                        continue
                    out[:, col + i * kw + j, oh * out_w + ow] = x[:, :, src_h, src_w]
    if squeeze:
        out = out[0]
    return out


@register("function_im2col")
class FunctionIm2colApi(BaseApi):
    """CPU golden for aclnnIm2col."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return None
        kwargs = input_data.kwargs
        self_tensor = kwargs["self"].detach().cpu()
        dtype = self_tensor.dtype
        if dtype == torch.bfloat16:
            src = self_tensor.to(torch.float32).numpy()
        else:
            src = self_tensor.numpy()

        result = _im2col(
            src,
            kwargs["kernelSize"],
            kwargs["dilation"],
            kwargs["padding"],
            kwargs["stride"],
        )
        result = np.array(result, dtype=result.dtype, copy=True, order="C")
        output = torch.from_numpy(result)
        return output.to(dtype)
