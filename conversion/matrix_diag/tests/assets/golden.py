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

"""MatrixDiag TestSpec (kernel/geir golden).

matrix_diag = diag_embed: last dim n -> produces a trailing (n, n) block with
the input placed on the main diagonal and zeros elsewhere.

    out[..., i, j] = x[..., i] if i == j else 0
    out.shape = x.shape + (n,)  where n = x.shape[-1]

complex32 输入由框架展开为 float16 (..., n, 2)（尾维 [2] = real/imag）。该算子是
纯搬运（无复数运算），golden 以等位宽 int32 做位级承载：float16 (..., n, 2) →
int32 (..., n) → diag_embed → float16 (..., n, n, 2)，全程位模式不变。
"""

import numpy as np

__golden__ = {"kernel": {"matrix_diag": "matrix_diag_golden"}}


def matrix_diag_golden(x, **kwargs):
    """Kernel golden for matrix_diag.

    Args:
        x: input tensor (numpy.ndarray)
            - common dtypes: shape (..., n)
            - complex32: unpacked float16 (..., n, 2) (last dim [2] = real/imag)
              or complex64 (..., n) when promoted (judge ∈ PROMOTE_GOLDEN_TOKENS,
              e.g. mixed/mix_tolerance/cross_check)

    kwargs may contain: input_dtypes, output_dtypes, short_soc_version, etc.

    Returns:
        out (numpy.ndarray): batched diagonal tensor
            - common dtypes: shape (..., n, n)
            - complex32: float16 (..., n, n, 2) or complex64 (..., n, n)
              (framework repacks complex goldens via _normalize_goldens)
    """
    input_dtypes = kwargs.get("input_dtypes")
    if "complex32" in str(input_dtypes):
        if np.issubdtype(np.asarray(x).dtype, np.complexfloating):
            # Promote 形态：complex64 (..., n) 直接按复数语义展开，
            # 由框架 _normalize_goldens 拆回 float16 (..., n, n, 2) 比对。
            n = int(x.shape[-1])
            idx = np.arange(n)
            out = np.zeros(x.shape + (n,), dtype=x.dtype)
            out[..., idx, idx] = x
            return out
        # complex32 是纯搬运（diag_embed 不做复数运算），以等位宽 int32 做位级承载：
        # float16 (..., n, 2) → int32 (..., n) → diag_embed → float16 (..., n, n, 2)。
        # 全程位模式不变，等价于对 real/imag 分别搬运，且无需触碰复数语义。
        x_int = np.ascontiguousarray(x).view(np.int32).reshape(x.shape[:-1])
        n = int(x_int.shape[-1])
        idx = np.arange(n)
        out = np.zeros(x_int.shape + (n,), dtype=np.int32)
        out[..., idx, idx] = x_int
        return out.view(np.float16).reshape(x_int.shape + (n, 2))

    n = int(x.shape[-1])
    out = np.zeros(tuple(x.shape) + (n,), dtype=x.dtype)
    idx = np.arange(n)
    out[..., idx, idx] = x
    return out
