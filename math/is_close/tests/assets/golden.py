#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {"kernel": {"is_close": "is_close_golden"}}


def _as_compute_input(value):
    array = np.asanyarray(value)
    if array.dtype.name in ("bfloat16", "float16", "float32"):
        # The supported floating kernel branches cast each input to FP32.
        return array.astype(np.float32, copy=False)
    return value


def is_close_golden(
    x1, x2, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False, **kwargs
):
    x1 = _as_compute_input(x1)
    x2 = _as_compute_input(x2)
    if np.asanyarray(x1).dtype == np.float32 and np.asanyarray(x2).dtype == np.float32:
        # np.isclose can promote zero-dimensional FP32 inputs internally.
        # Keep the kernel's separate FP32 subtract, multiply and add stages.
        with np.errstate(invalid="ignore", over="ignore"):
            difference = np.abs(x1 - x2)
            tolerance = np.float32(atol) + np.float32(rtol) * np.abs(x2)
            result = (difference <= tolerance) & np.isfinite(x2)
            result = result | (x1 == x2)
            if equal_nan:
                result = result | (np.isnan(x1) & np.isnan(x2))
            return result
    return np.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)
