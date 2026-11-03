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

__spec__ = {
    "mul_no_nan": "MulNoNanKernelSpec",
    "tf.raw_ops.MulNoNan": "MulNoNanTensorFlowSpec",
}

__golden__ = {"kernel": {"mul_no_nan": "mul_no_nan_golden"}}

_KERNEL_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}

_TENSORFLOW_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def mul_no_nan_golden(x1, x2, **kwargs):
    """
    Kernel golden for mul_no_nan.
    All the parameters follow @mul_no_nan_def.cpp without outputs.
    Inputs are array-like tensors.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Semantics: y = (x2 == 0) ? 0 : x1 * x2 (element-wise, NumPy broadcasting).
    The mask is on x2 ONLY, so wherever x2 == 0 the product is replaced by 0
    regardless of what that product would have been -- including NaN produced
    by x1 == inf (inf * 0) or by x1 == NaN. Conversely, where x2 != 0 the
    product is returned as is, so x1 == 0 with x2 == inf still yields NaN and
    NOT 0. This is the core differentiator vs plain Mul.

    TensorFlow's CPU MulNoNan lacks INT32 and some valid six-dimensional
    broadcasts. Use a Torch composition to cover the full CANN input domain
    and retain fp64 intermediate precision for the mathematical reference.
    """
    import torch

    output_dtype = np.asarray(x1).dtype
    is_integer = np.issubdtype(output_dtype, np.integer)
    compute_dtype = output_dtype if is_integer else np.float64
    x = torch.from_numpy(np.array(x1, dtype=compute_dtype, copy=True))
    y = torch.from_numpy(np.array(x2, dtype=compute_dtype, copy=True))
    # int32 multiplication intentionally retains native wraparound semantics.
    product = torch.mul(x, y)
    output = torch.where(y == 0, torch.zeros_like(product), product).numpy()
    if kwargs.get("golden_mode") == "Promote":
        return output
    # Use the framework cast for rounding; NumPy only carries the final bytes.
    if output_dtype.name == "bfloat16":
        return (
            torch.from_numpy(output)
            .to(torch.bfloat16)
            .view(torch.int16)
            .numpy()
            .view(output_dtype)
        )
    target_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
    }[output_dtype.name]
    return torch.from_numpy(output).to(target_dtype).numpy()


class _MulNoNanCompose:
    """Torch GPU composition, including INT32 which the TensorFlow API lacks."""

    def __call__(self, x1, x2, **kwargs):
        del kwargs
        import torch

        output_dtype = x1.dtype
        if output_dtype in (torch.float16, torch.bfloat16):
            x1, x2 = x1.to(torch.float32), x2.to(torch.float32)
        prod = torch.mul(x1, x2)
        output = torch.where(torch.eq(x2, 0), torch.zeros_like(prod), prod)
        return [output.to(output_dtype)]


class MulNoNanKernelSpec:
    """Kernel/GEIR: high-precision Torch CPU golden and a Torch GPU competitor."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return [mul_no_nan_golden(x1, x2, **kwargs)]

    third_party = {"torch": _MulNoNanCompose}
    tolerance = _KERNEL_TOLERANCE


class _MulNoNanE2ECompose:
    """Use the TF API's parameter names for TTK's name-based GPU dispatch."""

    def __call__(self, x, y, **kwargs):
        return _MulNoNanCompose()(x, y, **kwargs)


class MulNoNanTensorFlowSpec:
    """TensorFlow E2E spec registered by the CSV ``api_name``."""

    @staticmethod
    def golden(x, y, **kwargs):
        # Native CPU TF rejects some valid six-dimensional broadcasts. TTK
        # does not promote custom E2E inputs, so keep the reference in fp64.
        return [mul_no_nan_golden(x, y, golden_mode="Promote")]

    third_party = {"torch": _MulNoNanE2ECompose}
    # This TensorFlow TestSpec covers only float16/float32/bfloat16.
    # CANN's kernel/GEIR paths also support int32, which the TF API rejects.
    tolerance = _TENSORFLOW_TOLERANCE
