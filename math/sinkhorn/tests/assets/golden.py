#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Repository ST companion aligned with the delivery Sinkhorn Golden contract."""

import hashlib
import math

import numpy as np

__spec__ = {
    "sinkhorn": "SinkhornKernelSpec",
    "Sinkhorn": "SinkhornGeirSpec",
    "aclnnSinkhorn": "AclnnSinkhornSpec",
}

__golden__ = {
    "kernel": {"sinkhorn": "sinkhorn_golden"},
    "geir": {"Sinkhorn": "sinkhorn_golden"},
}

__input__ = {
    "kernel": {"sinkhorn": "sinkhorn_input", "Sinkhorn": "sinkhorn_input"},
    "aclnn": {"aclnnSinkhorn": "aclnn_sinkhorn_input"},
}

EPS = 1.0e-8
MAX_ITER = 10000
DEFAULT_TOL = 0.0001
_TOLERANCE = {
    dtype: {"standard": "cross_check", "level": "L0"}
    for dtype in ("float16", "bfloat16", "float32")
}
_PATTERN_ALIASES = {
    "v1_uniform": "uniform",
    "v2_symmetric": "uniform",
    "v3_zero": "positive_zero",
    "v4_negative_zero": "negative_zero",
    "v5_subnormal": "subnormal",
    "v6_extremes": "extreme",
    "v7_nan": "nan",
    "v8_inf": "pos_inf",
    "v9_exp_boundary": "exp_boundary",
}


def _validate_tol(tol):
    return DEFAULT_TOL if tol is None else float(tol)


def _torch():
    import torch

    return torch


def _as_torch(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value
    array = np.ascontiguousarray(value)
    if "bfloat16" in str(array.dtype).lower():
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _as_numpy(value, dtype):
    torch = _torch()
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().view(dtype)
    return value.numpy().astype(dtype, copy=False)


def _golden_reference(cost, tol, output_dtype=None):
    torch = _torch()
    tol = _validate_tol(tol)
    output_dtype = output_dtype or cost.dtype
    if cost.numel() == 0:
        return cost.to(output_dtype).clone()
    work = cost.to(torch.float64)
    rows, cols = work.shape
    kernel = torch.exp(work)
    d0 = torch.ones(rows, dtype=torch.float64, device=work.device)
    d1 = torch.ones(cols, dtype=torch.float64, device=work.device)
    for _ in range(MAX_ITER):
        d0 = (1.0 / rows) / (torch.sum(d1 * kernel, dim=1) + EPS)
        d1_new = (1.0 / cols) / (torch.sum(d0[:, None] * kernel, dim=0) + EPS)
        error = torch.mean(torch.abs(d1 - d1_new))
        d1 = d1_new
        if not (error.item() > tol):
            break
    return (d1 * kernel * d0[:, None]).to(output_dtype)


class _SinkhornThirdParty:
    def __init__(self, *, tol=0.0001, **kwargs):
        del kwargs
        self.tol = _validate_tol(tol)

    def __call__(self, cost, **kwargs):
        torch = _torch()
        tol = _validate_tol(kwargs.get("tol", self.tol))
        p = kwargs.get("p")
        output_dtype = p.dtype if isinstance(p, torch.Tensor) else cost.dtype
        if cost.numel() == 0:
            return [cost.to(output_dtype).clone()]
        work = cost.to(torch.float32)
        rows, cols = work.shape
        kernel = torch.exp(work)
        left = torch.ones(rows, dtype=torch.float32, device=work.device)
        right = torch.ones(cols, dtype=torch.float32, device=work.device)
        for _ in range(MAX_ITER):
            left = torch.reciprocal(torch.sum(right * kernel, dim=1) + EPS) / rows
            next_right = (
                torch.reciprocal(torch.sum(left[:, None] * kernel, dim=0) + EPS) / cols
            )
            delta = torch.mean(torch.abs(right - next_right))
            right = next_right
            if not (delta.item() > tol):
                break
        return [(right * kernel * left[:, None]).to(output_dtype)]


def _stable_rng(testcase_name):
    seed = int(hashlib.sha256(str(testcase_name).encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def _normalize_pattern(value):
    pattern = str(value).lower()
    for alias, normalized in _PATTERN_ALIASES.items():
        if alias in pattern:
            pattern = pattern.replace(alias, normalized)
    return pattern


def sinkhorn_input(cost, *, testcase_name="", input_ranges=None, **kwargs):
    rng = _stable_rng(testcase_name)
    try:
        low, high = map(float, input_ranges[0])
    except (TypeError, IndexError):
        low, high = 0.0, 1.0
    explicit = kwargs.get("value_class") or kwargs.get("input_pattern")
    pattern = _normalize_pattern(explicit if explicit else testcase_name)
    data = (
        rng.normal((low + high) / 2.0, max((high - low) / 6.0, 1.0e-6), cost.shape)
        if "normal" in pattern
        else rng.uniform(low, high, cost.shape)
    )
    data = data.astype(cost.dtype, copy=False)
    flat = data.reshape(-1)
    if flat.size == 0:
        return (data,)
    target_dtype = str(kwargs.get("target_dtype", cost.dtype)).lower()
    if "outlier" in pattern:
        count = max(1, flat.size // 1000)
        indices = rng.choice(flat.size, count, replace=False)
        flat[indices] = (flat[indices] * 1000.0).astype(cost.dtype, copy=False)
    elif "negative_zero" in pattern:
        flat.fill(-0.0)
    elif "positive_zero" in pattern:
        flat.fill(0.0)
    elif "subnormal" in pattern:
        flat.fill(
            2.0**-133
            if "bfloat16" in target_dtype
            else np.nextafter(
                np.array(0, dtype=cost.dtype), np.array(1, dtype=cost.dtype)
            ).item()
        )
    elif "extreme" in pattern:
        torch = _torch()
        info = (
            torch.finfo(torch.bfloat16)
            if "bfloat16" in target_dtype
            else np.finfo(cost.dtype)
        )
        flat[:] = np.resize(
            np.array([info.min, -info.tiny, info.tiny, info.max], dtype=cost.dtype),
            flat.size,
        )
    elif "mixed" in pattern:
        values = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0], dtype=cost.dtype)
        flat[: min(flat.size, values.size)] = values[: min(flat.size, values.size)]
    elif "nan" in pattern:
        flat[0] = np.nan
    elif "pos_inf" in pattern:
        flat[0] = np.inf
    elif "neg_inf" in pattern:
        flat[0] = -np.inf
    elif "exp_boundary" in pattern:
        threshold = 88.72 - math.log(max(cost.shape[1], 1))
        flat[:] = np.resize(
            np.array([threshold - 0.25, threshold + 0.25], dtype=cost.dtype), flat.size
        )
    return (data,)


def aclnn_sinkhorn_input(cost, tol=None, p=None, **kwargs):
    torch = _torch()
    del tol, p
    if not isinstance(cost, torch.Tensor):
        return sinkhorn_input(cost, **kwargs)
    if cost.dtype == torch.bfloat16:
        template = np.empty(tuple(cost.shape), dtype=np.float32)
        generated = sinkhorn_input(template, target_dtype="bfloat16", **kwargs)[0]
    else:
        template = cost.detach().cpu().contiguous().numpy()
        generated = sinkhorn_input(template, **kwargs)[0]
    return (
        torch.from_numpy(np.ascontiguousarray(generated)).to(
            dtype=cost.dtype, device=cost.device
        ),
    )


def _float64_array(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float64).numpy()
    return np.asarray(value).astype(np.float64)


def sinkhorn_pre_compare(npu_output, golden_output, **kwargs):
    torch = _torch()
    del kwargs
    npu = _float64_array(npu_output)
    golden = _float64_array(golden_output)
    if npu.size == 0 or not np.all(np.isfinite(golden)):
        return None
    dtype_name = str(getattr(npu_output, "dtype", "float32")).lower()
    threshold = (
        2.0**-8
        if "bfloat16" in dtype_name
        else 2.0**-11
        if "float16" in dtype_name
        else 2.0**-14
    )
    if np.max(np.abs(golden)) >= threshold:
        return None
    if "bfloat16" in dtype_name:
        quantized_golden = (
            torch.from_numpy(np.ascontiguousarray(golden))
            .to(torch.bfloat16)
            .to(torch.float64)
            .numpy()
        )
    elif "float16" in dtype_name:
        quantized_golden = golden.astype(np.float16).astype(np.float64)
    else:
        quantized_golden = golden.astype(np.float32).astype(np.float64)
    if not np.any(quantized_golden):
        return None
    all_zero = not np.any(npu)
    golden_mass = abs(float(quantized_golden.sum()))
    mass_ratio = abs(float(npu.sum())) / golden_mass
    if all_zero or mass_ratio < 0.5 or mass_ratio > 2.0:
        if isinstance(npu_output, torch.Tensor):
            failed_output = torch.ones_like(npu_output)
        else:
            failed_output = np.ones_like(npu_output)
        return failed_output, golden_output
    return None


def sinkhorn_golden(cost, *, tol=0.0001, **kwargs):
    del kwargs
    array = np.ascontiguousarray(cost)
    return [_as_numpy(_golden_reference(_as_torch(array), tol), array.dtype)]


class SinkhornKernelSpec:
    golden = staticmethod(sinkhorn_golden)
    third_party = {"torch": _SinkhornThirdParty}
    tolerance = _TOLERANCE
    pre_compare = staticmethod(sinkhorn_pre_compare)


class SinkhornGeirSpec(SinkhornKernelSpec):
    pass


class AclnnSinkhornSpec:
    @staticmethod
    def golden(cost, tol=0.0001, p=None, **kwargs):
        del kwargs
        tensor = _as_torch(cost)
        output_dtype = _as_torch(p).dtype if p is not None else tensor.dtype
        result = _golden_reference(tensor, tol, output_dtype)
        if isinstance(cost, np.ndarray):
            return [
                _as_numpy(result, p.dtype if isinstance(p, np.ndarray) else cost.dtype)
            ]
        return [result]

    third_party = {"torch": _SinkhornThirdParty}
    tolerance = _TOLERANCE
    pre_compare = staticmethod(sinkhorn_pre_compare)
