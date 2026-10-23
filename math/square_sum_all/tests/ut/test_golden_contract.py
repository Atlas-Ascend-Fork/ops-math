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
def test_scalar_shape_and_supplied_precision(dtype):
    x = np.array([3, 4], dtype=dtype)
    out = _G.SquareSumAllTestSpec.golden(x, x)
    for actual in out:
        assert actual.shape == () and actual.dtype == dtype and actual.item() == 25


def test_two_party_preserves_fp32_square_overflow():
    x = np.array([np.finfo(np.float32).max], np.float32)
    assert np.isposinf(_G.SquareSumAllTestSpec.golden(x, x)[0])
    promoted = x.astype(np.float64)
    result = _G.SquareSumAllTestSpec.golden(promoted, promoted, golden_mode="Promote")[
        0
    ]
    assert result.dtype == np.float64 and np.isfinite(result)


def test_competitor_returns_scalar_float32():
    import torch

    result = _G._Compose()(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0]))
    assert all(x.shape == torch.Size([]) and x.dtype == torch.float32 for x in result)
