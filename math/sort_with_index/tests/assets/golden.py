#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SortWithIndex references for the Kernel/GEIR pathway (arch35 ascend950).

SortWithIndex sorts ``x`` along the last axis and permutes the caller supplied
``index`` tensor with the same permutation:

    y[k] = x[p[k]], sorted_index[k] = index[p[k]]

The reference uses stable ``torch.sort`` and gathers the caller supplied index
payload with that permutation.  For stable=False, comparison permits equal-value
reordering while checking the sorted values and the full value/payload pairs.
The repository does not deliver an aclnn pathway for this operator (l0op
only), so only the kernel registry is populated.
"""

from collections import Counter

import numpy as np
import torch

__spec__ = {
    "sort_with_index": "SortWithIndexKernelSpec",
}

__golden__ = {
    "kernel": {"sort_with_index": "sort_with_index_golden"},
}

__input__ = {"kernel": {"sort_with_index": "sort_with_index_input"}}

_TOLERANCE = {
    dtype: {
        "standard": "stat_rel_err"
        if dtype in ("float16", "float32", "bfloat16")
        else "binary_equal"
    }
    for dtype in (
        "float16",
        "float32",
        "bfloat16",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
}


def _from_array(x):
    """Convert the Kernel/GEIR array without changing its value bits."""
    if "bfloat16" in str(x.dtype):
        return torch.from_numpy(x.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(x)


def _to_array(x):
    x = x.detach().cpu().contiguous()
    if x.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return x.view(torch.int16).numpy().view(bfloat16)
    return x.numpy()


def _normalize_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is invalid for rank {ndim}")
    if axis != ndim - 1:
        raise ValueError("SortWithIndex ST covers the last-axis contract only")
    return axis


def sort_with_index_golden(
    x, index, *, axis=-1, descending=False, stable=True, output_dtypes=(), **kwargs
):
    """Return a valid sorted reference, including when stability is optional."""
    values = _from_array(x)
    indices = _from_array(index)
    if values.shape != indices.shape:
        raise ValueError(
            f"x.shape {tuple(values.shape)} != index.shape {tuple(indices.shape)}"
        )
    norm_axis = _normalize_axis(axis, values.ndim)
    sorted_values, perm = torch.sort(
        values, dim=norm_axis, descending=bool(descending), stable=True
    )
    sorted_index = torch.take_along_dim(indices, perm, dim=norm_axis)
    return [_to_array(sorted_values), _to_array(sorted_index)]


def _element_bits(array):
    array = np.ascontiguousarray(array)
    return array.view(np.uint8).reshape(array.size, array.dtype.itemsize)


def _exact_mismatches(actual, expected):
    if actual.size != expected.size or actual.dtype != expected.dtype:
        return max(actual.size, expected.size, 1)
    return int(
        np.count_nonzero(
            np.any(_element_bits(actual) != _element_bits(expected), axis=1)
        )
    )


def sort_with_index_compare(
    values, indices, golden_values, golden_indices, *, compare_context
):
    """Require exact stable output, or a legal permutation when stable=False."""
    values, indices = np.asarray(values), np.asarray(indices)
    golden_values, golden_indices = (
        np.asarray(golden_values),
        np.asarray(golden_indices),
    )
    value_total = max(values.size, golden_values.size, 1)
    index_total = max(indices.size, golden_indices.size, 1)
    if bool(compare_context.attributes.get("stable", False)):
        value_bad = _exact_mismatches(values, golden_values)
        index_bad = _exact_mismatches(indices, golden_indices)
        index_reason = "index payload mismatches"
    else:
        x, source_indices = (np.asarray(a) for a in compare_context.input_tensors)
        compatible = (
            values.size == golden_values.size == x.size
            and indices.size == golden_indices.size == source_indices.size == x.size
            and values.dtype == golden_values.dtype == x.dtype
            and indices.dtype == golden_indices.dtype == source_indices.dtype
            and x.shape == source_indices.shape
        )
        if not compatible:
            value_bad, index_bad = value_total, index_total
        elif x.size == 0:
            value_bad = index_bad = 0
        else:
            equal_values = values.reshape(-1) == golden_values.reshape(-1)
            if values.dtype.kind == "f" or str(values.dtype) == "bfloat16":
                equal_values |= np.isnan(values.reshape(-1)) & np.isnan(
                    golden_values.reshape(-1)
                )
            value_bad = int(np.count_nonzero(~equal_values))
            row_length = x.shape[-1]
            input_value_bits = _element_bits(x).reshape(
                -1, row_length, x.dtype.itemsize
            )
            input_index_bits = _element_bits(source_indices).reshape(
                -1, row_length, source_indices.dtype.itemsize
            )
            value_bits = _element_bits(values).reshape(input_value_bits.shape)
            index_bits = _element_bits(indices).reshape(input_index_bits.shape)
            index_bad = 0
            for xv, xi, yv, yi in zip(
                input_value_bits, input_index_bits, value_bits, index_bits
            ):
                expected = Counter(v.tobytes() + i.tobytes() for v, i in zip(xv, xi))
                actual = Counter(v.tobytes() + i.tobytes() for v, i in zip(yv, yi))
                index_bad += sum((expected - actual).values())
        index_reason = (
            "missing value/index payload pairs (equal-value reordering allowed)"
        )
    return [
        {
            "pass": value_bad == 0,
            "precision": 100.0 * max(0, value_total - value_bad) / value_total,
            "error_info": None
            if value_bad == 0
            else f"{value_bad}/{value_total} sorted value mismatches",
        },
        {
            "pass": index_bad == 0,
            "precision": 100.0 * max(0, index_total - index_bad) / index_total,
            "error_info": None
            if index_bad == 0
            else f"{index_bad}/{index_total} {index_reason}",
        },
    ]


def _fill_pattern_along_axis(tensor, values, axis):
    """Repeat a pattern inside each logical row of the sorted axis."""
    axis = _normalize_axis(axis, tensor.ndim)
    if not tensor.is_floating_point():
        values = [int(value) for value in values]
    pattern = torch.tensor(values, dtype=tensor.dtype)
    repeats = (tensor.numel() + pattern.numel() - 1) // pattern.numel()
    moved_shape = [tensor.shape[i] for i in range(tensor.ndim) if i != axis]
    moved_shape.append(tensor.shape[axis])
    data = pattern.repeat(repeats)[: tensor.numel()].reshape(moved_shape)
    tensor.copy_(data.movedim(-1, axis))


def sort_with_index_input(x, index, *, axis=-1, testcase_name="", **kwargs):
    """Inject deterministic tie patterns into x; index is kept as generated."""
    name = str(testcase_name)
    patterns = {
        "duplicate": [7.0, 1.0, 7.0, 3.0, 7.0, 2.0, 8.0, 8.0],
        "all_equal": [3.0],
    }
    for marker, values in patterns.items():
        if name.endswith(f"_{marker}"):
            tensor = _from_array(x)
            _fill_pattern_along_axis(tensor, values, axis)
            break
    return (x, index)


class SortWithIndexThirdParty:
    """Timed competitor: ``__call__`` contains only required Sort work."""

    def __init__(self, *, axis=-1, descending=False, stable=True, **kwargs):
        self.axis = int(axis)
        self.descending = bool(descending)
        self.stable = bool(stable)

    def __call__(self, x, index, **kwargs):
        if x.shape != index.shape:
            raise ValueError(
                f"x.shape {tuple(x.shape)} != index.shape {tuple(index.shape)}"
            )
        norm_axis = _normalize_axis(self.axis, x.ndim)
        values, perm = torch.sort(
            x, dim=norm_axis, descending=self.descending, stable=self.stable
        )
        return values, torch.take_along_dim(index, perm, dim=norm_axis)


class SortWithIndexKernelSpec:
    golden = staticmethod(sort_with_index_golden)
    compare = staticmethod(sort_with_index_compare)
    customize_inputs = staticmethod(sort_with_index_input)
    third_party = {"torch": SortWithIndexThirdParty}
    tolerance = dict(_TOLERANCE)
