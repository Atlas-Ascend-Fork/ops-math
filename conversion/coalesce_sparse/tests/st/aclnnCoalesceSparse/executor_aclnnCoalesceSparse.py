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
"""ATK adapter for aclnnCoalesceSparse.

Inputs are regenerated deterministically (integer valued data) so that the
CPU golden and the device execution see exactly the same data, including
duplicate coordinate groups (the operator sums the values of one group):

- ``groups[i] = i % unique_len`` -> every group may receive several entries
- all entries of a group share the same coordinate row
- values are small integers, so the group sums are exact in fp16/fp32

Two outputs (``newIndices`` and ``newValues``) are declared as a tensor list.
"""

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


def _unique_len(kwargs):
    unique_len = kwargs["uniqueLen"]
    # ``unique_len`` is a [U] tensor (the operator infers U from shape[0]).
    return int(unique_len.reshape(-1).shape[0])


def _build_inputs(kwargs):
    """Deterministic values shared by the CPU golden and the device path."""
    unique_len = _unique_len(kwargs)
    unique_indices = kwargs["uniqueIndices"]
    indices = kwargs["indices"]
    values = kwargs["values"]

    nnz = int(unique_indices.reshape(-1).shape[0])
    dim = int(indices.shape[1]) if indices.dim() > 1 else 1
    tail = tuple(int(d) for d in values.shape[1:])

    groups = np.arange(nnz, dtype=np.int64) % max(unique_len, 1)
    coords = np.zeros((nnz, dim), dtype=np.int64)
    for k in range(dim):
        coords[:, k] = (groups * 3 + k * 5 + 7) % 97
    flat_values = ((np.arange(nnz, dtype=np.int64) * 7 + 3) % 17) - 8
    flat_values = np.broadcast_to(
        flat_values.reshape((nnz,) + (1,) * len(tail)), (nnz,) + tail
    ).copy()
    return unique_len, nnz, dim, tail, groups, coords, flat_values


def _apply_synthetic(kwargs):
    """Rewrite the ATK generated inputs with deterministic contents."""
    unique_len, _, _, _, groups, coords, flat_values = _build_inputs(kwargs)
    tensors = {
        "uniqueLen": np.full((unique_len,), unique_len, dtype=np.int64),
        "uniqueIndices": groups,
        "indices": coords,
        "values": flat_values,
    }
    for name, array in tensors.items():
        target = kwargs[name]
        source = torch.from_numpy(np.ascontiguousarray(array)).to(target.dtype)
        if target.device.type != "cpu":
            source = source.to(target.device)
        target.copy_(source)


def _golden(kwargs):
    unique_len, _, dim, tail, groups, coords, values = _build_inputs(kwargs)
    idx_dtype = kwargs["indices"].dtype
    val_dtype = kwargs["values"].dtype
    np_idx_dtype = np.int64 if idx_dtype == torch.int64 else np.int32
    np_val_dtype = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
    }[val_dtype]

    new_indices = np.zeros((unique_len, dim), dtype=np_idx_dtype)
    new_values = np.zeros((unique_len,) + tail, dtype=np_val_dtype)
    seen = np.zeros(unique_len, dtype=bool)
    for i, group in enumerate(groups):
        if not seen[group]:
            seen[group] = True
            new_indices[group] = coords[i]
    np.add.at(new_values, groups, values.astype(np_val_dtype))
    return new_indices, new_values


@register("function_aclnnCoalesceSparse")
class FunctionCoalesceSparseApi(BaseApi):
    """CPU golden for aclnnCoalesceSparse."""

    def __init__(self, task_result: TaskResult):
        super(FunctionCoalesceSparseApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return None
        _apply_synthetic(input_data.kwargs)
        new_indices, new_values = _golden(input_data.kwargs)
        return (torch.from_numpy(new_indices), torch.from_numpy(new_values))


@register("aclnn_function")
class AclnnCoalesceSparseExecutor(AclnnBaseApi):
    """Keeps the ATK generated inputs in sync with the CPU golden."""

    def init_by_input_data(self, input_data: InputDataset):
        _apply_synthetic(input_data.kwargs)
        return super().init_by_input_data(input_data)
