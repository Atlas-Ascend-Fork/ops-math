# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only regression tests for the TTK golden contract (no device access)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location(
    "golden_contract_" + _PATH.parents[2].name, _PATH
)
_G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_G)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", ["ordinary", "rbc_k_precision_p0_twosum_fp64_gap_i64"])
def test_precision_follows_weights_not_case_name(dtype, name):
    weights = np.array([1, 2**-24], dtype=dtype)
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 2], np.int64),
        np.array([0, 0], np.int32),
        np.array([1], np.int32),
        weights,
        golden_mode="Promote",
        testcase_name=name,
    )[0]
    assert out.dtype == dtype
    assert out[0, 0] == dtype(1 + 2**-24)


def test_promoted_finite_overflow_is_not_narrowed():
    maximum = np.finfo(np.float32).max
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 2], np.int64),
        np.array([0, 0], np.int32),
        np.array([1], np.int32),
        np.array([maximum, maximum], np.float64),
        golden_mode="Promote",
    )[0]
    assert out.dtype == np.float64
    assert out[0, 0] == float(maximum) * 2


def test_invalid_splits_return_promoted_positive_zero():
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 3], np.int64),
        np.array([0, 0], np.int32),
        np.array([1], np.int32),
        np.array([np.nan, np.inf], np.float64),
        golden_mode="Promote",
    )[0]
    assert out.dtype == np.float64 and out[0, 0] == 0 and not np.signbit(out[0, 0])


def test_binary_output_does_not_read_nan_weights():
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 2], np.int64),
        np.array([0, 0], np.int32),
        np.array([1], np.int32),
        np.array([np.nan, np.nan], np.float32),
        binary_output=True,
    )[0]
    assert out.dtype == np.float32 and out[0, 0] == 1


def test_opposite_infinities_propagate_nan():
    out = _G.RaggedBinCountKernelSpec.golden(
        np.array([0, 2], np.int64),
        np.array([0, 0], np.int32),
        np.array([1], np.int32),
        np.array([np.inf, -np.inf], np.float64),
        golden_mode="Promote",
    )[0]
    assert out.dtype == np.float64 and np.isnan(out[0, 0])
