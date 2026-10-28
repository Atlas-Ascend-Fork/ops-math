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

__golden__ = {"kernel": {"diag_flat": "diag_flat_golden"}}

__spec__ = {"diag_flat": "DiagFlatKernelSpec", "aclnnDiagFlat": "AclnnDiagFlatSpec"}


def _bf16_numpy_dtype():
    """bf16 的 numpy dtype：优先 ml_dtypes，缺失时回退 TensorFlow 自带类型（两者皆无返回 None）。"""
    try:
        import ml_dtypes

        return ml_dtypes.bfloat16
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.bfloat16.as_numpy_dtype
    except ImportError:
        return None


def _as_numpy(x):
    """torch/numpy -> numpy。bf16 等 numpy 无原生 loop 的 dtype 经 fp32 无损中转。"""
    if hasattr(x, "detach"):
        t = x.detach().cpu()
        try:
            return t.numpy()
        except TypeError:
            bf16 = _bf16_numpy_dtype()
            if bf16 is None:
                raise
            return t.float().numpy().astype(bf16)
    return np.asarray(x)


def diag_flat_golden(x, **kwargs):
    diagonal = kwargs.get("diagonal", 0)
    return np.diagflat(x, k=diagonal)


class DiagFlatKernelSpec:
    @staticmethod
    def golden(x, **kwargs):
        return [diag_flat_golden(x, **kwargs)]

    tolerance = {
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


class AclnnDiagFlatSpec:
    """aclnnDiagFlat(self, diagonal, out) 的 ACLNN 通路参考实现。

    与 Kernel 通路同源（np.diagflat），仅按 ACLNN 调用形态取参：
    diagonal 由属性 kwargs 传入（缺省 0），输入先规整为 numpy（含 bf16 中转）。
    """

    @staticmethod
    def golden(*args, **kwargs):
        diagonal = kwargs.get("diagonal")
        if diagonal is None:
            diagonal = next(
                (a for a in args[1:] if isinstance(a, (int, np.integer))), 0
            )
        return [np.diagflat(_as_numpy(args[0]), k=int(diagonal))]

    tolerance = DiagFlatKernelSpec.tolerance
