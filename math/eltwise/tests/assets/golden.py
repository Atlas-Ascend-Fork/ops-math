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
"""
Eltwise kernel-path golden (TestSpec).

Computation policy (aligned with the A2/A3 TBE baseline and the arch35 kernel):
all dtypes compute directly in the original input dtype, no upcast; coeff
scalars are rounded to the input dtype by torch tensor construction.
"""

import functools

import numpy as np
import torch

__spec__ = {"eltwise": "EltwiseTestSpec"}
__golden__ = {"kernel": {"eltwise": "eltwise_golden"}}


def _numpy_to_torch(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr)


def _torch_to_numpy(t):
    if t.dtype == torch.bfloat16:
        np_uint16 = t.view(torch.int16).numpy().astype(np.uint16)
        try:
            import ml_dtypes

            return np_uint16.view(ml_dtypes.bfloat16)
        except ImportError:
            return np_uint16
    return t.numpy()


def _eltwise_compute(tensors, mode, coeff):
    n = len(tensors)

    if coeff is not None and not isinstance(coeff, (list, tuple)):
        coeff = [coeff]
    if coeff is not None and len(coeff) == 0:
        coeff = None

    if mode == 1 and coeff is not None and len(coeff) != n:
        raise ValueError(f"coeff length {len(coeff)} != input count {n}")

    if n == 1:
        if mode == 1 and coeff is not None:
            c0 = torch.tensor(float(coeff[0]), dtype=tensors[0].dtype)
            return c0 * tensors[0]
        return tensors[0]

    if mode == 0:
        return functools.reduce(torch.mul, tensors)

    if mode == 1:
        if coeff is not None:
            dt = tensors[0].dtype
            scaled = [
                torch.tensor(float(c), dtype=dt) * t for c, t in zip(coeff, tensors)
            ]
            return functools.reduce(torch.add, scaled)
        return functools.reduce(torch.add, tensors)

    if mode == 2:
        return functools.reduce(torch.maximum, tensors)

    raise ValueError(f"Invalid mode: {mode}. Supported: 0=PRODUCT, 1=SUM, 2=MAX")


def _validate_inputs(inputs, mode, coeff):
    if len(inputs) == 0:
        raise ValueError("x must not be empty")
    if len(inputs) > 32:
        raise ValueError(f"input count {len(inputs)} exceeds the maximum of 32")

    shape0 = inputs[0].shape
    dtype0 = inputs[0].dtype
    for t in inputs[1:]:
        if t.shape != shape0:
            raise ValueError(f"shape_mismatch: {t.shape} != {shape0}")
        if t.dtype != dtype0:
            raise ValueError(f"dtype_mismatch: {t.dtype} != {dtype0}")

    if mode not in (0, 1, 2):
        raise ValueError(f"Invalid mode: {mode}. Supported: 0=PRODUCT, 1=SUM, 2=MAX")

    if mode == 1 and coeff is not None:
        if not isinstance(coeff, (list, tuple)):
            coeff = [coeff]
        if len(coeff) > 0 and len(coeff) != len(inputs):
            raise ValueError(f"coeff length {len(coeff)} != input count {len(inputs)}")


class EltwiseTestSpec:
    """Eltwise kernel-path test spec.

    Golden: torch ops chained left-to-right in the original input dtype
    (same accumulation order as the kernel).
    ThirdPartyImpl: independent torch composite reference, same dtype policy.
    """

    @staticmethod
    def golden(x, mode=1, coeff=[], **kwargs):
        """Eltwise golden.

        Args:
            x: input tensor or list of 1~32 numpy.ndarray with identical
               shape and dtype (no broadcast)
            mode: 0=PRODUCT, 1=SUM (default), 2=MAX
            coeff: coefficient list for SUM mode, defaults to all 1.0
        Returns:
            list[numpy.ndarray]: element-wise result in the input dtype
        """
        if x is None:
            raise ValueError("x must not be None")
        inputs = [x] if not isinstance(x, (list, tuple)) else list(x)
        _validate_inputs(inputs, mode, coeff)

        tensors = [_numpy_to_torch(t) for t in inputs]
        result = _eltwise_compute(tensors, mode, coeff)
        np_result = _torch_to_numpy(result)

        output_dtypes = kwargs.get("output_dtypes", [])
        if output_dtypes and len(output_dtypes) > 0:
            target = output_dtypes[0]
            if str(target) != str(np_result.dtype):
                np_result = np_result.astype(target)

        return [np_result]

    class ThirdPartyImpl:
        """Third-party reference: torch composite, original dtype."""

        def __init__(self, *, mode=1, coeff=[], **kwargs):
            self.mode = mode
            if coeff is not None and not isinstance(coeff, (list, tuple)):
                coeff = [coeff]
            if coeff is not None and len(coeff) == 0:
                coeff = None
            self.coeff = coeff

        def __call__(self, x, **kwargs):
            if x is None:
                raise ValueError("x must not be None")
            inputs = [x] if not isinstance(x, (list, tuple)) else list(x)
            _validate_inputs(inputs, self.mode, self.coeff)

            tensors = [_numpy_to_torch(t) for t in inputs]
            result = _eltwise_compute(tensors, self.mode, self.coeff)
            return [_torch_to_numpy(result)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L0"},
        "float16": {"standard": "cross_check", "level": "L0"},
        "bfloat16": {"standard": "cross_check", "level": "L0"},
    }


def eltwise_golden(x, mode=1, coeff=[], **kwargs):
    return EltwiseTestSpec.golden(x, mode, coeff, **kwargs)
