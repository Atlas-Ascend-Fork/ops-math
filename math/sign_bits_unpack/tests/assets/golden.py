#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================

"""SignBitsUnpack TestSpec for the GEIR and ACLNN paths.

SignBitsUnpack expands every bit of a one-dimensional uint8 tensor in
LSB-first order and maps ``0 -> -1`` and ``1 -> +1``. The result is reshaped
to ``(size, input_numel * 8 // size)``.

Precision policy for the primary Golden:
    The operator input itself is always uint8; FP16/FP32 describe the output
    dtype. Before any floating-point arithmetic, all unpacked bits are cast
    once to FP32. The complete expression is evaluated in FP32 and is cast to
    the requested output dtype only after the calculation. The helper also
    follows this policy for BF16, although BF16 is not part of the current
    SignBitsUnpack operator contract.

ThirdPartyImpl intentionally does not promote precision. It performs the same
composed Torch operations directly in the requested output dtype.
"""

from __future__ import annotations

import numpy as np
import torch

__spec__ = {
    "sign_bits_unpack": "SignBitsUnpackTestSpec",
    "SignBitsUnpack": "SignBitsUnpackTestSpec",
    "aclnnSignBitsUnpack": "AclnnSignBitsUnpackTestSpec",
    "torch_npu.npu_sign_bits_unpack": "AclnnSignBitsUnpackTestSpec",
}

_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _first_leaf(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def _dtype_name(dtype) -> str:
    """Normalize GE dtype objects, ACL dtype codes, and framework dtypes."""
    dtype = _first_leaf(dtype)
    if isinstance(dtype, (int, np.integer)):
        if int(dtype) == 0:
            return "float32"
        if int(dtype) == 1:
            return "float16"
        raise ValueError(f"unsupported SignBitsUnpack dtype code: {dtype}")

    name = str(dtype).lower()
    if "bfloat16" in name or "bf16" in name:
        return "bfloat16"
    if "float16" in name or name in {"half", "fp16"}:
        return "float16"
    if "float32" in name or name in {"float", "fp32"}:
        return "float32"
    raise ValueError(f"unsupported SignBitsUnpack dtype: {dtype}")


def _torch_output_dtype(dtype) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[_dtype_name(dtype)]


def _numpy_output_dtype(dtype):
    name = _dtype_name(dtype)
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)

    try:
        from ml_dtypes import bfloat16
    except ImportError as exc:
        raise RuntimeError("BF16 Golden requires the ml_dtypes package") from exc
    return np.dtype(bfloat16)


def _validate_input_and_size(input_numel: int, input_rank: int, size) -> int:
    if input_rank != 1:
        raise ValueError(f"SignBitsUnpack input must be 1-D, got rank {input_rank}")
    if input_numel < 0:
        raise ValueError("SignBitsUnpack input element count must not be negative")

    size = int(size)
    total_bits = input_numel * 8
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if total_bits % size != 0:
        raise ValueError(f"{total_bits} unpacked bits are not divisible by size={size}")
    return size


def _geir_golden(x, size=1, dtype=0, **kwargs):
    """GEIR Golden: numpy input and a list containing one numpy output."""
    x_array = np.asarray(x)
    if x_array.dtype != np.uint8:
        raise TypeError(
            f"SignBitsUnpack input dtype must be uint8, got {x_array.dtype}"
        )

    size = _validate_input_and_size(x_array.size, x_array.ndim, size)
    requested_dtype = kwargs.get("output_dtypes", dtype)
    output_dtype = _torch_output_dtype(requested_dtype)

    x_tensor = torch.from_numpy(np.ascontiguousarray(x_array))
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = ((x_tensor[:, None] >> shifts) & 1).reshape(-1)

    # Low-precision policy: finish every cast before the first float operation.
    compute_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    bits_compute = bits.to(compute_dtype)
    two = torch.tensor(2.0, dtype=compute_dtype)
    one = torch.tensor(1.0, dtype=compute_dtype)

    result_compute = bits_compute * two - one
    result = result_compute.to(output_dtype).reshape(size, x_tensor.numel() * 8 // size)
    if output_dtype == torch.bfloat16:
        numpy_dtype = _numpy_output_dtype(requested_dtype)
        result = result.to(torch.float32).numpy().astype(numpy_dtype, copy=False)
    else:
        result = result.numpy()
    return [result]


def _aclnn_golden(self_tensor, size, dtype, out=None, **kwargs):
    """ACLNN Golden; parameters follow ``self, size, dtype, out`` in the C header."""
    if not isinstance(self_tensor, torch.Tensor):
        raise TypeError(
            f"ACLNN Golden expects torch.Tensor, got {type(self_tensor).__name__}"
        )
    if self_tensor.dtype != torch.uint8:
        raise TypeError(
            f"SignBitsUnpack input dtype must be uint8, got {self_tensor.dtype}"
        )

    size = _validate_input_and_size(self_tensor.numel(), self_tensor.dim(), size)
    output_dtype = _torch_output_dtype(dtype)

    shifts = torch.arange(8, dtype=torch.uint8, device=self_tensor.device)
    bits = ((self_tensor[:, None] >> shifts) & 1).reshape(-1)

    # Low-precision policy: finish every cast before the first float operation.
    compute_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    bits_compute = bits.to(compute_dtype)
    two = torch.tensor(2.0, dtype=compute_dtype, device=self_tensor.device)
    one = torch.tensor(1.0, dtype=compute_dtype, device=self_tensor.device)

    result_compute = bits_compute * two - one
    result = result_compute.to(output_dtype)
    return [result.reshape(size, self_tensor.numel() * 8 // size)]


class ThirdPartyImpl:
    """Torch small-op reference that deliberately performs no promotion.

    ``*args`` replaces the usual explicit ``self`` parameter so a ``self``-
    named input (ACLNN contract) does not collide with the instance slot.
    TTK ``bind_params`` delivers a ``self`` pool entry positionally through
    ``*args`` and other inputs through ``**kwargs``, so the input tensor is
    looked up in kwargs first (GEIR ``x``) and then among positional args
    (ACLNN ``self``).
    """

    def __call__(*args, **kwargs):
        x = kwargs.get("x", kwargs.get("self"))
        if x is None:
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    x = arg
                    break
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"ThirdPartyImpl expects torch.Tensor, got {type(x).__name__}"
            )
        if x.dtype != torch.uint8:
            raise TypeError(f"SignBitsUnpack input dtype must be uint8, got {x.dtype}")

        size = _validate_input_and_size(x.numel(), x.dim(), kwargs.get("size", 1))
        output_dtype = _torch_output_dtype(kwargs.get("dtype", 0))
        bits = torch.empty(x.numel() * 8, dtype=torch.uint8, device=x.device)
        for bit_index in range(8):
            bits[bit_index::8] = (x >> bit_index) & 1

        # No promotion here: arithmetic runs directly in the requested dtype.
        bits_native = bits.to(output_dtype)
        two = torch.tensor(2.0, dtype=output_dtype, device=x.device)
        one = torch.tensor(1.0, dtype=output_dtype, device=x.device)
        result = bits_native * two - one
        return [result.reshape(size, x.numel() * 8 // size)]


class SignBitsUnpackTestSpec:
    """SignBitsUnpack GEIR TestSpec (GEIR reuses the kernel-style spec)."""

    golden = _geir_golden
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _L0


class AclnnSignBitsUnpackTestSpec:
    """aclnnSignBitsUnpack TestSpec."""

    golden = _aclnn_golden
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _L0
