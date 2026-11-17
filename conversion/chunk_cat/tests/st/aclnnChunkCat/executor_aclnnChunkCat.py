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
"""ATK adapter for aclnnChunkCat.

Each non-empty input is split along dim 0 into ``numChunks`` equal blocks of
``ceil(shape[0] / numChunks)`` rows (zero padded at the tail); the flattened
blocks with the same chunk index are concatenated in input order, producing a
2-D ``[numChunks, total_rows]`` output (dim is always 0).
"""

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _chunk_cat(tensors, num_chunks):
    rows = [[] for _ in range(num_chunks)]
    out_dtype = tensors[0].dtype
    for tensor in tensors:
        array = tensor
        if array.ndim == 0 or array.size == 0:
            continue
        height = array.shape[0]
        tail = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
        block = (height + num_chunks - 1) // num_chunks
        flat = array.reshape(height, tail)
        for index in range(num_chunks):
            start = index * block
            stop = min(start + block, height)
            if start >= height:
                chunk = np.zeros((0, tail), dtype=array.dtype)
            else:
                chunk = flat[start:stop]
            flat_chunk = chunk.reshape(-1)
            if flat_chunk.size < block * tail:
                padded = np.zeros(block * tail, dtype=array.dtype)
                padded[: flat_chunk.size] = flat_chunk
                flat_chunk = padded
            rows[index].append(flat_chunk)
    row_len = sum(part.size for part in rows[0]) if rows else 0
    out = np.zeros((num_chunks, row_len), dtype=out_dtype)
    for index in range(num_chunks):
        if rows[index]:
            out[index] = np.concatenate(rows[index])
    return out


@register("function_aclnnChunkCat")
class FunctionChunkCatApi(BaseApi):
    """CPU golden for aclnnChunkCat."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return None
        kwargs = input_data.kwargs
        tensors = [t.detach().cpu() for t in kwargs["tensors"]]
        num_chunks = int(kwargs["numChunks"])
        dtype = tensors[0].dtype

        arrays = []
        for tensor in tensors:
            if tensor.dtype == torch.bfloat16:
                arrays.append(tensor.to(torch.float32).numpy())
            else:
                arrays.append(tensor.numpy())
        result = _chunk_cat(arrays, num_chunks)
        result = np.array(result, dtype=result.dtype, copy=True, order="C")
        return torch.from_numpy(result).to(dtype)
