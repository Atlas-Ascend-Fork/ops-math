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
"""TTK TestSpec golden for Diag (Kernel + ACLNN dual-path).

Independent CPU reference implementation — does NOT import or reuse any kernel product.
Uses numpy.diag as the reference oracle per spec.yaml reference_oracle:

    y = np.diag(x, k=diagonal)

Bidirectional diagonal operator (numpy.diag semantics):
  - 1D input (N,) → constructs 2D square matrix (N+|k|, N+|k|) with x on the
    k-th diagonal, zeros elsewhere.
  - 2D input (M, N) → extracts 1D diagonal vector.

Pure index movement + zero padding — no arithmetic, no rounding, bit-level
identical (spec.yaml numerical_tolerance: bitwise_equal for all dtypes;
determinism.bitwise_reproducible: true).

bf16 fidelity: numpy has no native bfloat16; ml_dtypes.bfloat16 is used
so that np.diag can process bf16 arrays directly after dtype registration.

Non-contiguous input: aclnn layer first calls Contiguous; golden side
uses np.ascontiguousarray before np.diag to match kernel ND contiguous-view.

Tolerance:
  - Kernel/GEIR (DiagV2TestSpec): binary_equal (self-contained, bitwise exact).
  - ACLNN (AclnnDiagSpec): binary_equal (bitwise exact); third_party=torch.diag used as CPU cross-check.
"""

__spec__ = {
    "diag_v2": "DiagV2TestSpec",
    "aclnnDiag": "AclnnDiagSpec",
}

import numpy as np
import torch
import ml_dtypes


def _np_to_torch(arr):
    """numpy ndarray -> torch tensor, preserving bf16 via int16 bit-bridge."""
    if isinstance(arr, torch.Tensor):
        return arr
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(arr)


def _torch_to_np(t):
    """torch tensor -> numpy ndarray, preserving bf16 via int16 bit-bridge."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _diag_core(x, diagonal):
    """Core compute: np.diag. Inputs are numpy or torch, returns numpy."""
    x_np = _torch_to_np(x) if isinstance(x, torch.Tensor) else np.asarray(x)
    x_np = np.ascontiguousarray(x_np)
    return np.diag(x_np, k=int(diagonal))


_BINARY_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "float64": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
    "complex64": {"standard": "binary_equal"},
}


class DiagV2TestSpec:
    """Kernel / GEIR flow — golden receives numpy.ndarray, returns list[numpy.ndarray].

    Parameters follow diag_v2_def.cpp (without output y):
      input: x(REQUIRED)
      attr:  diagonal(OPTIONAL Int, default 0)
    """

    def golden(x, diagonal=0, **kwargs):
        y = _diag_core(x, diagonal)
        return [y]

    third_party = {"torch": "torch.diag"}

    tolerance = _BINARY_TOLERANCE


class AclnnDiagSpec:
    """ACLNN flow — golden receives torch.Tensor, returns list[numpy.ndarray].

    Parameters follow aclnnDiagGetWorkspaceSize (without workspaceSize & executor):
      self, diagonal, out
    ``self`` is the input tensor; ``out`` is the output placeholder (ignored).
    """

    def golden(self, diagonal=0, out=None, **kwargs):
        y = _diag_core(self, diagonal)
        return [y]

    third_party = {"torch": "torch.diag"}

    tolerance = _BINARY_TOLERANCE
