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
"""ATK adapters for ``aclnnTransMatmulWeight`` (trans_data ND -> FRACTAL_NZ).

Registers two entry points referenced by ``atk_aclnnTransMatmulWeight.json``:

- ``function_trans_matmul_weight`` (BaseApi): CPU golden. It reproduces the NZ
  layout written by the arch35 kernel ``trans_data_with_simt.h``:

      H  = prod(shape[:-2])
      N1 = ceil(N / 16)          # N = shape[-2]
      C1 = ceil(C / C0)          # C = shape[-1], C0 = 32 for int8 else 16
      out[h, c1, n1, n0, c0] = ND[h, n1 * 16 + n0, c1 * C0 + c0] (0 in the padding area)

  so the flattened output has ``H * N1 * 16 * C1 * C0`` elements.

- ``aclnn_trans_matmul_weight`` (AclnnBaseApi): pyaclnn wrapper. The api is
  in-place (``mmWeightRef`` is both input and output) and, once padded, the NZ
  buffer is larger than ``numel``; therefore the input tensor is backed by a
  device buffer of ``weightSize`` elements and the whole buffer is copied back
  after the second-phase call.
"""

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

_N0 = 16


def _ceil_div(value, base):
    return (value + base - 1) // base


def _nz_params(shape, dtype):
    """Return (H, N, C, N0, C0, N1, C1, weight_size) for an ND weight tensor."""
    shape = [int(d) for d in shape]
    dim = len(shape)
    if dim == 2:
        h, n, c = 1, shape[0], shape[1]
    else:
        h = 1
        for d in shape[:-2]:
            h *= d
        n, c = shape[-2], shape[-1]
    c0 = 32 if dtype == torch.int8 else 16
    n1 = _ceil_div(n, _N0)
    c1 = _ceil_div(c, c0)
    weight_size = h * n1 * _N0 * c1 * c0
    return h, n, c, _N0, c0, n1, c1, weight_size


def _nz_transform(weight):
    """Golden ND -> FRACTAL_NZ conversion with zero padding."""
    h, n, c, n0, c0, n1, c1, weight_size = _nz_params(weight.shape, weight.dtype)
    if weight_size == 0:
        return torch.empty(0, dtype=weight.dtype)
    src = weight.reshape(h, n, c)
    padded = torch.zeros((h, n1 * n0, c1 * c0), dtype=weight.dtype)
    if n > 0 and c > 0:
        padded[:, :n, :c] = src
    nz = padded.view(h, n1, n0, c1, c0).permute(0, 3, 1, 2, 4).contiguous()
    return nz.reshape(-1)


def _get_weight(input_data: InputDataset):
    kwargs = getattr(input_data, "kwargs", None) or {}
    if "mmWeightRef" in kwargs:
        return kwargs["mmWeightRef"]
    return input_data.args[0]


@register("function_trans_matmul_weight")
class FunctionTransMatmulWeightApi(BaseApi):
    """CPU golden: ND weight -> padded FRACTAL_NZ weight (flattened)."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return _nz_transform(_get_weight(input_data).detach().cpu())


@register("aclnn_trans_matmul_weight")
class AclnnTransMatmulWeightApi(AclnnBaseApi):
    """pyaclnn wrapper for the in-place ``aclnnTransMatmulWeight`` api."""

    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self._weight_size = 0
        self._dtype = None
        self._buffer = None

    def init_by_input_data(self, input_data: InputDataset):
        weight = _get_weight(input_data)
        shape = list(weight.shape)
        dtype = weight.dtype
        weight_size = _nz_params(shape, dtype)[-1]

        device_weight = weight.detach()
        if device_weight.device.type != "npu":
            device_weight = device_weight.npu()

        numel = device_weight.numel()
        if weight_size > 0:
            # The NZ result (with padding) can be larger than numel, so the tensor
            # handed to the api must be backed by a buffer of weight_size elements.
            buffer = torch.zeros(weight_size, dtype=dtype, device=device_weight.device)
            if numel > 0:
                buffer[:numel].copy_(device_weight.reshape(-1))
            replaced = buffer[:numel].view(shape)
        else:
            buffer = device_weight.reshape(-1)
            replaced = buffer.view(shape)

        if "mmWeightRef" in (getattr(input_data, "kwargs", None) or {}):
            input_data.kwargs["mmWeightRef"] = replaced
        else:
            input_data.args[0] = replaced

        self._weight_size = weight_size
        self._dtype = dtype
        self._buffer = buffer

        input_args, output_packages = super().init_by_input_data(input_data)
        # In-place op: the api takes only mmWeightRef, so drop the auto-appended
        # output tensor and expose the input tensor as the output package.
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def after_call(self, output_packages):
        if self._weight_size == 0:
            return [torch.empty(0, dtype=self._dtype)]
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        return [self._buffer[: self._weight_size].detach().cpu()]
