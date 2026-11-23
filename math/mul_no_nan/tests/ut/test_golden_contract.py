# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only regression tests for MulNoNan golden input conversion."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location("mul_no_nan_golden_contract", _PATH)
_G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_G)


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1, 2, 0], [0, 3, 4], [0, 6, 0]),
        ([1, 2, 0], 0, [0, 0, 0]),
        (2, [0, 3, 4], [0, 6, 8]),
        (2, 3, 6),
    ],
)
def test_tensorflow_tensors_and_scalars(dtype, x, y, expected):
    tf = pytest.importorskip("tensorflow")
    tf_dtype = getattr(tf, dtype)
    out = _G.MulNoNanTensorFlowSpec.golden(
        tf.constant(x, dtype=tf_dtype), tf.constant(y, dtype=tf_dtype)
    )
    assert len(out) == 1
    assert out[0].dtype == np.float64
    assert out[0].shape == np.shape(expected)
    np.testing.assert_array_equal(out[0], expected)


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
def test_tensorflow_zero_mask_and_nonfinite_values(dtype):
    tf = pytest.importorskip("tensorflow")
    tf_dtype = getattr(tf, dtype)
    x = tf.constant([float("nan"), float("inf"), 0], dtype=tf_dtype)
    y = tf.constant([0, -0.0, float("inf")], dtype=tf_dtype)
    out = _G.MulNoNanTensorFlowSpec.golden(x, y)[0]
    np.testing.assert_array_equal(out, [0, 0, float("nan")])


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16", "int32"])
def test_kernel_numpy_inputs_keep_output_dtype(dtype):
    if dtype == "bfloat16":
        numpy_dtype = pytest.importorskip("ml_dtypes").bfloat16
    else:
        numpy_dtype = np.dtype(dtype)
    x = np.array([1, -2, 0], dtype=numpy_dtype)
    y = np.array(3, dtype=numpy_dtype)
    out = _G.MulNoNanKernelSpec.golden(x, y)[0]
    assert out.dtype == x.dtype
    np.testing.assert_array_equal(out, [3, -6, 0])


def test_kernel_int32_wraparound():
    x = np.array([-2147483648], dtype=np.int32)
    y = np.array(-1, dtype=np.int32)
    out = _G.MulNoNanKernelSpec.golden(x, y)[0]
    np.testing.assert_array_equal(out, x)
