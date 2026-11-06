# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU regressions for promoted weighted sums and Python heap use."""

import importlib.util
import tracemalloc
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location(
    "golden_residuals_" + _PATH.parents[2].name, _PATH
)
_G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_G)


def _one_bin_golden(weights):
    return _G.RaggedBinCountKernelSpec.golden(
        np.array([0, weights.size], np.int64),
        np.zeros(weights.size, np.int32),
        np.array([1], np.int32),
        weights,
        golden_mode="Promote",
    )[0]


@pytest.mark.parametrize(
    "terms",
    [
        [2.0**100, 1.0, -(2.0**100)],
        [2.0**60, 2.0**-100, -(2.0**60)],
        [
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            1.0,
        ],
        [np.finfo(np.float32).max, 2.0**-149, -np.finfo(np.float32).max],
    ],
)
def test_promoted_mixed_sign_bins_match_exact_rational_sum(terms):
    weights = np.asarray(terms, dtype=np.float64)
    exact = sum((Fraction.from_float(float(value)) for value in weights), Fraction())
    out = _one_bin_golden(weights)
    assert out.dtype == np.float64
    assert out[0, 0] == float(exact)


@pytest.mark.parametrize("terms", list(permutations((2.0**100, 1.0, -(2.0**100)))))
def test_promoted_cancellation_retains_residual_in_every_term_order(terms):
    out = _one_bin_golden(np.asarray(terms, dtype=np.float64))
    assert out.dtype == np.float64 and out[0, 0] == 1.0


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_promoted_rows_bins_and_ignored_weights_remain_independent(index_dtype):
    maximum = float(np.finfo(np.float32).max)
    weights = np.array(
        [2.0**100, 1, -(2.0**100), 3, -2, np.nan, maximum, -maximum, 2.0**-149],
        np.float64,
    )
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 6, 6, 9], np.int64),
        np.array([0, 0, 0, 1, 1, 7, 2, 2, 2], index_dtype),
        np.array([4], index_dtype),
        weights,
        golden_mode="Promote",
    )[0]
    expected = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 2.0**-149, 0]], np.float64)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float64
    assert not np.signbit(out[expected == 0]).any()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_bins_do_not_change_finite_or_empty_bins(dtype):
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 11], np.int64),
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], np.int32),
        np.array([6], np.int32),
        np.array(
            [np.inf, 2, -np.inf, -3, np.inf, -np.inf, np.nan, 5, 1, -1, 2.0**-20], dtype
        ),
        golden_mode="Promote",
    )[0]
    assert out.dtype == dtype
    assert np.isposinf(out[0, 0]) and np.isneginf(out[0, 1])
    assert np.isnan(out[0, 2:4]).all()
    assert out[0, 4] == 2.0**-20
    assert out[0, 5] == 0 and not np.signbit(out[0, 5])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_balanced_finite_weights_produce_positive_zero(dtype):
    out = _one_bin_golden(np.array([-1, 1, -0.0, 0.0], dtype))
    assert out.dtype == dtype and out[0, 0] == 0 and not np.signbit(out[0, 0])


@pytest.mark.parametrize("bins", [3, 1024])
def test_dense_and_sparse_bins_preserve_cancellation_residuals(bins):
    # A rounded weighted total of zero must not hide bin 0's exact residual.
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 5], np.int64),
        np.array([0, 0, 0, 2, 2], np.int32),
        np.array([bins], np.int32),
        np.array([2.0**100, 1, -(2.0**100), 2, -2], np.float64),
        golden_mode="Promote",
    )[0]
    expected = np.zeros((1, bins), np.float64)
    expected[0, 0] = 1
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, expected)
    assert not np.signbit(out[expected == 0]).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weighted_tensor_reference_preserves_dtype_and_cpu_device(dtype):
    indices = torch.tensor([0, 0, 0], dtype=torch.int64)
    weights = torch.tensor([1, -1, 2.0**-20], dtype=dtype)
    out = _G._weighted_bincount(indices, weights, 2)
    assert out.dtype == weights.dtype and out.device == weights.device
    assert out.tolist() == [2.0**-20, 0.0]
    assert not torch.signbit(out[1])


def test_promoted_mixed_sign_sum_keeps_python_heap_bounded():
    count = 65536
    splits = np.array([0, count], np.int64)
    values = np.zeros(count, np.int32)
    size = np.array([1], np.int32)
    weights = np.ones(count, np.float64)
    weights[1::2] = -1

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        initial, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        out = _G.RaggedBinCountKernelSpec.golden(
            splits, values, size, weights, golden_mode="Promote"
        )[0]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()

    assert out.dtype == np.float64 and out[0, 0] == 0 and not np.signbit(out[0, 0])
    # Inputs were prepared before tracing. This generous fixed allowance must
    # not accommodate one live scalar Tensor wrapper per input; native Torch
    # buffers and wall time are deliberately outside this Python-object guard.
    python_heap_budget = 1024 * 1024
    assert peak - initial <= python_heap_budget, (
        f"extra Python heap peak {peak - initial} exceeds {python_heap_budget} bytes"
    )
