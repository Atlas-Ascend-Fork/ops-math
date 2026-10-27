#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""RaggedBinCount multi-path golden in the TestSpec format.

Only one registration key is needed: kernel and GEIR both resolve the snake-case
operator name and share :class:`RaggedBinCountKernelSpec`.  The legacy
``__golden__`` entry stays available and calls the same Torch computation core.

The reference uses ``torch.bincount`` per ragged row in the dtype supplied
by TTK. All testcases use the same computation, including numerical corners;
the reference does not select a different algorithm from the testcase name.
``torch.bincount`` is semantically the per-row form of the TensorFlow
``RaggedBincount`` operator this IR is compatible with.

Frozen contract points mirrored from ``README.md``:
  * within the public rank-0-to-2 range, ``weights`` is judged by element count
    rather than matching ``values`` shape: zero elements (``[0]``, ``[0, 3]``,
    ``[2, 0]`` alike) accumulates 1.0 per hit, and a non-empty ``weights`` pairs
    with ``values`` in flattened order;
  * ``binary_output=True`` writes exactly 0.0/1.0 and ignores ``weights``
    entirely, including NaN/Inf/negative weights;
  * ``values >= M`` is legal and ignored;
  * if any split is illegal or any value is negative, the **whole** output is
    bitwise ``+0.0`` and no weight is ever read.
"""

import numpy as np
import torch


__spec__ = {
    # kernel + GEIR share this snake-case key and this one Spec class.
    "ragged_bin_count": "RaggedBinCountKernelSpec",
}

__golden__ = {
    "kernel": {"ragged_bin_count": "ragged_bin_count_golden"},
}


_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    # Integer input types are declared as well; the sole output is float32.
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
}


_FLOAT32_MAX = np.finfo(np.float32).max
_FLOAT32_MIN_SUBNORMAL = np.float32(2.0**-149)
_FLOAT32_MIN_NORMAL = np.float32(2.0**-126)
_FLOAT32_NEXT_MIN_NORMAL = np.asarray([0x00800001], dtype=np.uint32).view(np.float32)[0]
_FLOAT32_EXP24_MIN = np.asarray([0x0C000000], dtype=np.uint32).view(np.float32)[0]
_FLOAT32_NEG_EXP23_MAX = np.asarray([0x8BFFFFFF], dtype=np.uint32).view(np.float32)[0]


_PRECISION_SPECIAL_WEIGHTS = {
    "rbc_k_precision_special_pos_inf_i64": [float("inf"), 1.0, 2.0, 3.0],
    "rbc_k_precision_special_inf_cancel_i64": [
        float("inf"),
        -float("inf"),
        1.0,
        2.0,
    ],
    "rbc_k_precision_special_nan_i64": [float("nan"), 1.0, 2.0, 3.0],
    "rbc_k_precision_special_finite_overflow_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
    ],
    "rbc_k_precision_special_finite_residual_pos_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        1.0,
    ],
    "rbc_k_precision_special_finite_residual_neg_i64": [
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -1.0,
    ],
    "rbc_k_precision_special_subnormal_residual_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        _FLOAT32_MIN_SUBNORMAL,
    ],
    # Both inputs are normal, but their exact cancellation residual is the
    # minimum positive subnormal. This catches arithmetic FTZ before TwoSum
    # can observe and preserve the discarded bit.
    "rbc_k_precision_p0_normal_cancel_to_subnormal_i64": [
        _FLOAT32_NEXT_MIN_NORMAL,
        -_FLOAT32_MIN_NORMAL,
    ],
    # Keep the same minimum-subnormal result on a sparse, non-private ROW
    # output so the final direct-GM write is covered as well as accumulation.
    "rbc_k_precision_p0_direct_gm_subnormal_i64": [
        _FLOAT32_NEXT_MIN_NORMAL,
        -_FLOAT32_MIN_NORMAL,
    ],
    # Same-sign low-band normals can also create a subnormal TwoSum residual.
    # The exact result is 0x01400001; an FTZ expansion loses its final bit and
    # rounds to 0x01400000 unless the inputs select the exact accumulator.
    "rbc_k_precision_p0_same_sign_low_band_residual_i64": [
        _FLOAT32_MIN_NORMAL,
        _FLOAT32_NEXT_MIN_NORMAL,
        _FLOAT32_NEXT_MIN_NORMAL,
    ],
    # Lock the detector's inclusive raw-exponent upper boundary. The exp-24
    # operand itself does not require exact handling, while the exp-23 operand
    # must select it before their cancellation produces a subnormal result.
    "rbc_k_precision_p0_exp23_24_cancel_to_subnormal_i64": [
        _FLOAT32_EXP24_MIN,
        _FLOAT32_NEG_EXP23_MAX,
    ],
    "rbc_k_precision_special_max_boundary_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
    ],
    "rbc_k_precision_special_result_overflow_i64": [_FLOAT32_MAX, _FLOAT32_MAX],
    "rbc_k_precision_special_round_tie_even_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        1.0,
        2.0**-24,
    ],
    "rbc_k_precision_special_round_above_tie_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        1.0,
        2.0**-24,
        2.0**-25,
    ],
    # The first two terms land exactly halfway between adjacent FP32 values.
    # The final 1.0 is retained by FP64 golden accumulation but does not fit in
    # a two-component FP32 expansion without a third-order residual.
    "rbc_k_precision_special_third_order_residual_i64": [2.0**50, 2.0**26, 1.0],
    # One launch straddles the precise-work boundary: 64 * 64 == 4096 uses
    # the fixed-order path, while 65 * 64 falls back to the legacy scatter.
    "rbc_k_precision_p0_mixed_work_boundary_i32": (
        [2.0**50, 1.0, -(2.0**50)] + [0.0] * 61 + [1.0, -1.0 + 2.0**-23] + [0.0] * 63
    ),
    # Drive all 256 hits into one bin.  The leading overflow forces an exact
    # rescan and the balanced payload exercises the full scan: omitting the
    # final term would leave +FLT_MAX, while all 256 terms sum to bitwise +0.
    "rbc_k_precision_p0_exact_256_single_bin_i32": (
        [_FLOAT32_MAX] * 128 + [-_FLOAT32_MAX] * 128
    ),
    # Enter the exact rescan because of the first two finite terms, then make
    # that rescan account for both infinity signs.
    "rbc_k_precision_p0_exact_inf_cancel_after_overflow_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        float("inf"),
        -float("inf"),
    ],
    "rbc_k_precision_p0_invalid_before_precise_weights_i32": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        float("inf"),
        -float("inf"),
    ],
    "rbc_k_precision_p0_binary_nonfinite_ignored_i32": [
        float("nan"),
        float("inf"),
        -float("inf"),
        _FLOAT32_MAX,
    ],
    "rbc_k_precision_p0_negative_inf_i32": [-float("inf"), 1.0, 2.0, 3.0],
    # A single FP64 accumulator loses the unit term across this cancellation;
    # the promoted reference must retain the addition residual.
    "rbc_k_precision_p0_twosum_fp64_gap_i64": [
        2.0**100,
        1.0,
        -(2.0**100),
    ],
    # The leading overflow selects the exact rescan.  The final -4.0 has
    # exponent field 129, so its subtraction starts at accumulator word 2 and
    # propagates a borrow through the upper limbs.
    "rbc_k_precision_p0_exact_word2_borrow_i32": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        -4.0,
    ],
    # VALUE mapping with one hot bin uses the bounded output-owner path. The
    # leading overflow requires an exact rescan, and the final term proves that
    # the complete 1025-element row was consumed.
    "rbc_k_precision_p0_value_hot_bin_exact_i32": (
        [_FLOAT32_MAX, _FLOAT32_MAX]
        + [0.0] * 1020
        + [-_FLOAT32_MAX, -_FLOAT32_MAX, 1.0]
    ),
    # Exercise the inclusive VALUE-owner work boundary: 131072 values and two
    # bins require 262144 comparisons.  All hits target bin zero; the balanced
    # maximum-finite payload forces the exact rescan through all 131072 terms,
    # while bin one independently proves that a no-hit owner leaves +0.0.
    "rbc_k_precision_p0_value_max_work_exact_i64": (
        [_FLOAT32_MAX] * 65536 + [-_FLOAT32_MAX] * 65536
    ),
    # Exercise the maximum term count admitted by the same work bound.  With
    # one bin, all 262144 values are visited and the exact rescan must consume
    # the complete balanced maximum-finite payload to recover bitwise +0.0.
    "rbc_k_precision_p0_value_max_values_exact_i32": (
        [_FLOAT32_MAX] * 131072 + [-_FLOAT32_MAX] * 131072
    ),
    # N=4096 selects four VALUE cores and therefore 64 global fixed
    # partitions of 64 elements each.  Place each non-zero term at a distinct
    # partition head so the local partials are individually exact and only
    # the fixed-order core-zero merge discovers the third-order residual.
    "rbc_k_precision_p0_value_merge_residual_i64": (
        [2.0**50] + [0.0] * 63 + [2.0**26] + [0.0] * 63 + [1.0] + [0.0] * (4096 - 129)
    ),
    # The finite overflow is likewise created only while merging distinct
    # partition partials; the exact full-row rescan must recover +1.0.
    "rbc_k_precision_p0_value_merge_overflow_i64": (
        [_FLOAT32_MAX]
        + [0.0] * 63
        + [_FLOAT32_MAX]
        + [0.0] * 63
        + [-_FLOAT32_MAX]
        + [0.0] * 63
        + [-_FLOAT32_MAX]
        + [0.0] * 63
        + [1.0]
        + [0.0] * (4096 - 257)
    ),
    # Core one owns indices [1024, 2048).  Its first partition overflows
    # locally, proving a REQUIRES_EXACT flag produced away from core zero is
    # visible to the final owner and causes a complete exact rescan.
    "rbc_k_precision_p0_value_remote_local_exact_i64": (
        [0.0] * 1024
        + [_FLOAT32_MAX, _FLOAT32_MAX]
        + [0.0] * (2048 - 1026)
        + [-_FLOAT32_MAX, -_FLOAT32_MAX]
        + [0.0] * (4095 - 2050)
        + [1.0]
    ),
    # Valid non-finite inputs in different partitions must produce the
    # canonical quiet NaN emitted by the integer superaccumulator.
    "rbc_k_precision_p0_value_nonfinite_cross_partition_i64": (
        [float("inf")] + [0.0] * 63 + [-float("inf")] + [0.0] * (4096 - 65)
    ),
    # A non-zero subnormal is routed to the exact bit-field accumulator so the
    # result is independent of the SIMT denormal arithmetic mode.
    "rbc_k_precision_p0_value_subnormal_i64": ([_FLOAT32_MIN_SUBNORMAL] + [0.0] * 4095),
    # Put two normal operands in distinct fixed partitions. Their merge must
    # select the exact rescan and retain the minimum subnormal residual.
    "rbc_k_precision_p0_value_normal_cancel_to_subnormal_i32": (
        [_FLOAT32_NEXT_MIN_NORMAL]
        + [0.0] * 63
        + [-_FLOAT32_MIN_NORMAL]
        + [0.0] * (4096 - 65)
    ),
    # Four VALUE-owned rows include an empty row and uneven segment lengths.
    # Unit weights make every expected count exactly representable in FP32.
    "rbc_k_precision_p0_value_multirow_owner_i64": [1.0] * 8192,
    "rbc_g_contract_rank2_cancel_precision_32": [
        2.0**50,
        1.0,
        -(2.0**50),
        1.0,
        -1.0 + 2.0**-23,
        0.0,
    ],
}


_PRECISION_SPECIAL_VALUES = {
    "rbc_k_precision_p0_invalid_before_precise_weights_i32": [0, -1, 0, 0],
    "rbc_k_precision_p0_binary_nonfinite_ignored_i32": [0, 0, 1, 7],
    "rbc_g_contract_rank2_cancel_precision_32": [0, 0, 0, 1, 1, 1],
    "rbc_k_precision_p0_value_multirow_owner_i64": [index & 1 for index in range(8192)],
}


_PRECISION_SPECIAL_WEIGHTS.update(
    {
        f"rbc_k_promoted_cancellation_order_{index}": payload
        for index, payload in enumerate(
            [
                [2.0**100, 1.0, -(2.0**100)],
                [2.0**100, -(2.0**100), 1.0],
                [1.0, 2.0**100, -(2.0**100)],
                [1.0, -(2.0**100), 2.0**100],
                [-(2.0**100), 2.0**100, 1.0],
                [-(2.0**100), 1.0, 2.0**100],
                [-(2.0**100), -1.0, 2.0**100],
            ]
        )
    }
)


def _customize_precision_inputs(splits, values, size, weights, **kwargs):
    """Inject fixed numerical corner cases used by the precision ST."""
    testcase_name = str(kwargs.get("testcase_name", ""))
    weight_payload = _PRECISION_SPECIAL_WEIGHTS.get(testcase_name)
    value_payload = _PRECISION_SPECIAL_VALUES.get(testcase_name)
    if weight_payload is None and value_payload is None:
        return splits, values, size, weights
    fixed_values = (
        np.zeros_like(values)
        if value_payload is None
        else np.asarray(value_payload, dtype=values.dtype).reshape(values.shape)
    )
    fixed_weights = (
        weights
        if weight_payload is None
        else np.asarray(weight_payload, dtype=np.float32).reshape(weights.shape)
    )
    return splits, fixed_values, size, fixed_weights


def _attr(kwargs, name, default):
    """CSV attributes arrive as strings; coerce them to the default's type."""
    value = kwargs.get(name, default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false", "yes", "no", "1", "0"):
            return text in ("true", "yes", "1")
        try:
            return type(default)(value)
        except Exception:
            return default
    return value


def _as_tensor(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(np.ascontiguousarray(array))


def _contract_inputs(splits, values, size, weights):
    """Flatten to the shapes the frozen contract works in."""
    splits_flat = _as_tensor(splits).reshape(-1).to(torch.int64)
    values_flat = _as_tensor(values).reshape(-1).to(torch.int64)
    size_flat = _as_tensor(size).reshape(-1).to(torch.int64)
    weights_flat = _as_tensor(weights).reshape(-1)
    return splits_flat, values_flat, size_flat, weights_flat


def _runtime_invalid(splits_flat, values_flat):
    """Reproduce the whole-output +0.0 guard: illegal split or negative value."""
    value_count = values_flat.numel()
    if splits_flat.numel() < 2:
        return True
    if bool((splits_flat < 0).any()) or bool((splits_flat > value_count).any()):
        return True
    if int(splits_flat[0].item()) != 0:
        return True
    if int(splits_flat[-1].item()) != value_count:
        return True
    if bool((splits_flat[1:] < splits_flat[:-1]).any()):
        return True
    if bool((values_flat < 0).any()):
        return True
    return False


def _sum_with_residuals(values):
    """Sum finite promoted weights using nonoverlapping Torch partials.

    FastTwoSum retains each rounded addition's residual. This prevents a
    subsequent cancellation from discarding small terms, without changing the
    input/output dtype or depending on the testcase name.
    """
    partials = []
    for value in values.unbind():
        residuals = []
        for partial in partials:
            if bool(torch.abs(value) < torch.abs(partial)):
                value, partial = partial, value
            high = value + partial
            low = partial - (high - value)
            if bool(low != 0):
                residuals.append(low)
            value = high
        partials = residuals + [value]
    return torch.stack(partials).sum()


def _weighted_bincount(indices, weights, bins):
    result = torch.bincount(indices, weights=weights, minlength=bins)[:bins]
    if (
        weights.dtype != torch.float64
        or not bool((weights < 0).any())
        or not bool((weights > 0).any())
    ):
        return result
    # Only promoted, finite, mixed-sign bins need cancellation recovery.
    # Nonfinite bins retain the native Torch Inf/NaN behavior.
    for index in torch.unique(indices):
        terms = weights[indices == index]
        if (
            bool(torch.isfinite(terms).all())
            and bool((terms < 0).any())
            and bool((terms > 0).any())
        ):
            result[index] = _sum_with_residuals(terms)
    return result


def _compute(splits, values, size, weights, **kwargs):
    """Return ``[output]`` as a Torch tensor in def.cpp output order."""
    binary_output = _attr(kwargs, "binary_output", False)
    splits_flat, values_flat, size_flat, weights_flat = _contract_inputs(
        splits, values, size, weights
    )

    rows = max(splits_flat.numel() - 1, 0)
    bins = int(size_flat[0].item()) if size_flat.numel() > 0 else 0
    output = torch.zeros((rows, bins), dtype=weights_flat.dtype)

    # Invalid input returns bitwise +0.0 before any weight is read.
    if _runtime_invalid(splits_flat, values_flat) or rows == 0 or bins == 0:
        return [output]

    has_weights = weights_flat.numel() > 0
    # Float32 is the NPU accumulation type. Under Promote, TTK supplies
    # float64 weights; preserve that dtype without additional casts.
    dtype = weights_flat.dtype
    accumulator = torch.zeros((rows, bins), dtype=dtype)

    for row in range(rows):
        begin = int(splits_flat[row].item())
        end = int(splits_flat[row + 1].item())
        segment = values_flat[begin:end]
        # values >= M are legal and ignored.
        keep = segment < bins
        kept = segment[keep]
        if kept.numel() == 0:
            continue
        if binary_output:
            # Bitwise exact 0/1; weights are never read on this path.
            accumulator[row] = accumulator[row].index_fill(0, kept, 1.0)
        elif has_weights:
            segment_weights = weights_flat[begin:end][keep]
            accumulator[row] = _weighted_bincount(kept, segment_weights, bins)
        else:
            # Integer hit counts are assigned into the declared accumulator.
            accumulator[row] = torch.bincount(kept, minlength=bins)[:bins]

    return [accumulator]


def _kernel_golden(splits, values, size, weights, **kwargs):
    """Kernel/GEIR container adapter: NumPy in, NumPy list out."""
    outputs = _compute(splits, values, size, weights, **kwargs)
    return [np.ascontiguousarray(output.detach().cpu().numpy()) for output in outputs]


def _competitor_forward(splits_flat, values_flat, bins, weights_flat, binary_output):
    """Optimal single-pass competitor form (see :class:`_RaggedBinCountCompose`)."""
    device = values_flat.device
    rows = max(splits_flat.numel() - 1, 0)
    output = torch.zeros(rows * bins, dtype=torch.float32, device=device)

    # Row id per value, from the ragged offsets: one repeat_interleave instead
    # of a Python loop over rows.
    counts = splits_flat[1:] - splits_flat[:-1]
    row_id = torch.repeat_interleave(
        torch.arange(rows, device=device, dtype=torch.int64), counts
    )

    keep = values_flat < bins  # values >= M are legal and ignored
    flat_index = row_id * bins + values_flat
    flat_index = flat_index[keep]

    if binary_output:
        # Weights are never read on this path; writes are idempotent 1.0.
        output[flat_index] = 1.0
    elif weights_flat.numel() > 0:
        output.index_add_(0, flat_index, weights_flat.to(torch.float32)[keep])
    else:
        output.index_add_(
            0,
            flat_index,
            torch.ones(flat_index.numel(), dtype=torch.float32, device=device),
        )

    return output.view(rows, bins)


class _RaggedBinCountCompose:
    """Independent competitor composition, executed on the remote GPU.

    Two requirements pull in different directions and both are met here:

    * It must not share an implementation with :func:`_compute`, or the two
      reference paths cannot cross-validate each other.  ``_compute`` is a
      per-row ``torch.bincount`` in the TTK-supplied dtype; this is a single flat
      ``index_add_`` in FP32 over a ``repeat_interleave``-derived row index.
    * It must be the competitor's **optimal** form, or the G/N ratio is
      inflated and the comparison is not an honest one.  The whole ragged
      problem is expressed as one scatter over the flattened output, so torch
      launches a constant number of kernels regardless of the row count -- the
      earlier per-row Python loop cost 66 ms on an A100 for a case the fused
      form finishes in tens of microseconds.

    CUDA ``index_add_`` can use a nondeterministic atomic schedule.  Do not
    enable PyTorch's global deterministic mode in this performance provider:
    the same object supplies ``--xpu-perf``, and the deterministic scatter is a
    materially different, slower competitor.  A reproducibility experiment
    should use a separate precision-only provider or frozen competitor output.
    """

    def __init__(self, **kwargs):
        self.binary_output = _attr(kwargs, "binary_output", False)

    def __call__(self, splits, values, size, weights, **kwargs):
        del kwargs
        splits_flat, values_flat, size_flat, weights_flat = _contract_inputs(
            splits, values, size, weights
        )
        rows = max(splits_flat.numel() - 1, 0)
        bins = int(size_flat[0].item()) if size_flat.numel() > 0 else 0
        if _runtime_invalid(splits_flat, values_flat) or rows == 0 or bins == 0:
            return [
                torch.zeros(
                    (rows, bins), dtype=torch.float32, device=values_flat.device
                )
            ]

        # NPU output is always FP32; align the competitor dtype.
        return [
            _competitor_forward(
                splits_flat, values_flat, bins, weights_flat, self.binary_output
            )
        ]


class RaggedBinCountKernelSpec:
    """Shared kernel/GEIR TestSpec; parameter names follow def.cpp."""

    @staticmethod
    def golden(splits, values, size, weights, **kwargs):
        return _kernel_golden(splits, values, size, weights, **kwargs)

    @staticmethod
    def customize_inputs(splits, values, size, weights, **kwargs):
        return _customize_precision_inputs(splits, values, size, weights, **kwargs)

    third_party = {"torch": _RaggedBinCountCompose}
    tolerance = _TOL


def ragged_bin_count_golden(splits, values, size, weights, *args, **kwargs):
    """Compatibility entry for the historical ``__golden__`` kernel loader."""
    del args
    return _kernel_golden(splits, values, size, weights, **kwargs)[0]


# 【不存在】ACLNN 通路：CMakeLists.txt 显式配置 ``ACLNNTYPE aclnn_exclude``，
# 本算子无 op_api 目录，不交付 aclnnRaggedBinCount 接口。
# E2E 暂未注册：本文件验证 README 声明的 Graph IR 路径，没有确认独立框架
# API 的派发绑定；缺少同名 ACLNN 符号不能证明所有框架通路均不存在。
