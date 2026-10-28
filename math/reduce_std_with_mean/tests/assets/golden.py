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

"""ReduceStdWithMean Kernel/GEIR 参考实现（TTK TestSpec，``__spec__`` 显式注册）。

数学公式（与 README 一致）：
  invert=false : y = sqrt( sum((x-mean)^2, dim) / max(0, N - correction) )
  invert=true  : y = 1 / sqrt( sum((x-mean)^2, dim) / N + epsilon )
  其中 N = prod(shape[d] for d in dim)

边界语义（与 NPU kernel 对齐）：
  N==0                       → NaN（空张量归约）
  denom<=0 & invert=false    → ss 为 NaN → NaN; ss==0 → NaN(0/0); ss>0 → +Inf
  var==0 & eps==0 & invert=true → +Inf

精度策略（[FR-16] 修正）：FP16/BF16 → 升 FP32 计算；FP32 → 升 FP64 计算（消除大 N 累加精度损失）；
最后回写输入原始 dtype。customize_inputs 仅在输入无特殊值时替换 mean。
third_party 为 torch 原生 dtype 计算，不做升精度，用于 cross_check 对照。
"""

import numpy as np
import torch

__spec__ = {
    "reduce_std_with_mean": "ReduceStdWithMeanTestSpec",
    # [FR-15] 显式注册 aclnn 通路（aclnnStdMeanCorrection 经 L0 调用本算子）
    "aclnnStdMeanCorrection": "AclnnStdMeanCorrectionTestSpec",
}


def _normalize_dims(dim, ndim):
    """归一化归约轴：空/None → 全轴；负数索引 wrap；去重 + 排序。"""
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    dim = list(dim)
    if len(dim) == 0:
        return list(range(ndim))
    norm = []
    seen = set()
    for d in dim:
        nd = d + ndim if d < 0 else d
        if 0 <= nd < ndim and nd not in seen:
            seen.add(nd)
            norm.append(nd)
    return sorted(norm)


def _is_bf16_np(arr):
    """判断 numpy 数组是否为 bfloat16（ml_dtypes 表示为 kind='V', itemsize=2, name 含 'bfloat16'）。"""
    return (
        arr.dtype.kind == "V"
        and arr.dtype.itemsize == 2
        and "bfloat16" in arr.dtype.name
    )


def _np_to_torch(arr):
    """numpy → torch.Tensor，处理 FP32 / FP16 / BF16(ml_dtypes)。

    BF16 的 numpy 表示（ml_dtypes.bfloat16）无法直接 ``torch.from_numpy``，
    需经 int16 视图转换。
    """
    if isinstance(arr, torch.Tensor):
        return arr
    if _is_bf16_np(arr):
        np_i16 = arr.view(np.int16)
        return torch.from_numpy(np_i16).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _torch_to_np_bf16(t_bf16):
    """torch.bfloat16 → numpy bfloat16（ml_dtypes），经 int16 视图转换。"""
    np_i16 = t_bf16.view(torch.int16).numpy()
    try:
        from ml_dtypes import bfloat16 as bf16_np_dtype
    except ImportError:
        raise RuntimeError("ml_dtypes is required for bfloat16 support")
    return np_i16.view(np.dtype(bf16_np_dtype))


def _torch_to_np_like(t, ref_np_dtype):
    """torch.Tensor → numpy.ndarray，按参考 numpy dtype 转换。"""
    if ref_np_dtype.kind == "V" and ref_np_dtype.itemsize == 2:
        return _torch_to_np_bf16(t)
    return t.numpy()


def _has_special_values(*tensors):
    """检测输入张量是否含特殊值（±inf / NaN / -0.0），兼容 numpy / torch。

    特殊值边界用例（-0.0 / -inf 覆盖）依赖原始输入构造边界行为，customize_inputs
    不应重算 mean 覆盖这些精心构造的值；随机均匀数据实际不会产生这些值。
    """
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if not bool(torch.isfinite(t).all()):
                return True
            zeros = t[t == 0]
            if zeros.numel() and bool(torch.signbit(zeros).any()):
                return True
        elif isinstance(t, np.ndarray) and t.size:
            if not bool(np.isfinite(t).all()):
                return True
            zeros = t[t == 0]
            if zeros.size and bool(np.signbit(zeros).any()):
                return True
    return False


def _compute_std_core(x_t, mean_t, dims, keepdim, invert, epsilon, correction):
    """在给定 torch dtype 下完成 std/invstd 计算（不做 dtype 转换）。

    返回与 ``x_t`` 同 dtype 的输出张量。
    """
    mean_t = mean_t.expand_as(x_t)
    ndim = x_t.dim()
    if len(dims) == 0:
        dims = list(range(ndim))

    diff = x_t - mean_t
    sq = diff * diff

    # torch.sum 必须显式传 dim 参数（仅传 keepdim 在 PyTorch 中非法）
    ss = torch.sum(sq, dim=dims, keepdim=keepdim)
    N = 1
    for a in dims:
        N *= x_t.shape[a]

    if N == 0:
        out = torch.full_like(ss, float("nan"))
    elif invert:
        var = ss / float(N) + float(epsilon)
        out = torch.where(
            var == 0.0,
            torch.full_like(var, float("inf")),
            1.0 / torch.sqrt(var),
        )
    else:
        denom = N - correction
        if denom <= 0:
            out = torch.where(
                torch.isnan(ss) | (ss == 0.0),
                torch.full_like(ss, float("nan")),
                torch.full_like(ss, float("inf")),
            )
        else:
            out = torch.sqrt(ss / float(denom))
    return out


class ReduceStdWithMeanTestSpec:
    """ReduceStdWithMean kernel/GEIR 流程测试规范。

    golden 精度策略：FP16/BF16 升 FP32 计算，FP32 升 FP64 计算（消除大 N 累加精度损失），回写原 dtype。
    ThirdPartyImpl：不升精度，torch 原生 dtype 计算。
    """

    def golden(
        x,
        mean,
        dim=None,
        unbiased=True,
        keepdim=False,
        invert=False,
        epsilon=0.001,
        correction=1,
        **kwargs,
    ):
        """CPU golden（kernel/GEIR 流程，入参为 numpy.ndarray）。

        Parameters
        ----------
        x, mean : numpy.ndarray — 输入张量（FP32/FP16/BF16），参数名与 def.cpp 一致。
        dim : list[int] | None — 归约轴，空列表/None 表示全轴。
        unbiased : bool — legacy 字段，correction 优先。
        keepdim : bool
        invert : bool
        epsilon : float
        correction : int — 0=总体, 1=样本；invert=false 时生效。
        """
        # correction 优先于 unbiased
        if correction is None:
            correction = 1 if unbiased else 0

        # 1) numpy → torch，记录原始 numpy / torch dtype
        orig_np_dtype = x.dtype
        x_t = _np_to_torch(x)
        mean_t = _np_to_torch(mean)
        orig_torch_dtype = x_t.dtype

        # 2) 升精度：FP16/BF16 → FP32；FP32 → FP64（[FR-16] 消除大 N 累加精度损失）
        if orig_torch_dtype in (torch.float16, torch.bfloat16):
            x_t = x_t.to(torch.float32)
            mean_t = mean_t.to(torch.float32)
        elif orig_torch_dtype == torch.float32:
            x_t = x_t.to(torch.float64)
            mean_t = mean_t.to(torch.float64)

        # 3) 全部计算在升精度后的 dtype（FP32 或 FP64）下完成
        dims = _normalize_dims(dim, x_t.dim())
        out_fp32 = _compute_std_core(
            x_t, mean_t, dims, keepdim, invert, epsilon, correction
        )

        # 4) 回写原始 dtype
        out_t = out_fp32.to(orig_torch_dtype)
        return [_torch_to_np_like(out_t, orig_np_dtype)]

    def customize_inputs(x, mean, dim=None, **kwargs):
        """计算 x 沿 dim 的真实均值并广播到 x 形状，替换随机 mean。

        保证 (x-mean) 数值合理，避免 FP16 方差溢出；mean dtype 与 x 一致。
        [FR-16] 设计说明：仅当输入不含特殊值时替换。特殊值用例和调用方
        显式构造的极端 mean 不会被覆盖，保留对算子处理任意 mean 输入的验证能力。
        """
        if _has_special_values(x, mean):
            return (x, mean)
        orig_np_dtype = x.dtype
        x_t = _np_to_torch(x)

        # 在 FP32 下计算均值（精度安全）
        x_fp32 = x_t.to(torch.float32)
        dims = _normalize_dims(dim, x_fp32.dim())
        if len(dims) == 0:
            dims = list(range(x_fp32.dim()))

        # torch.mean 必须显式传 dim 参数（仅传 keepdim 在 PyTorch 中非法）
        mean_val = torch.mean(x_fp32, dim=dims, keepdim=True)
        mean_t = mean_val.expand_as(x_fp32).to(x_t.dtype)

        mean_np = _torch_to_np_like(mean_t, orig_np_dtype)
        return (x, mean_np)

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }

    class ThirdPartyImpl:
        """三方标杆实现：torch 原生 dtype 计算，不做升精度，用于 cross_check 对照。

        参数绑定：attrs → __init__，inputs → __call__。
        """

        def __init__(
            self,
            *,
            dim=None,
            unbiased=True,
            keepdim=False,
            invert=False,
            epsilon=0.001,
            correction=1,
            **kwargs,
        ):
            self.dim = dim
            self.unbiased = unbiased
            self.keepdim = keepdim
            self.invert = invert
            self.epsilon = epsilon
            self.correction = (
                correction if correction is not None else (1 if unbiased else 0)
            )

        def __call__(self, x, mean, **kwargs):
            """x, mean 为 torch.Tensor（由 numpy 转来），原生 dtype 计算。"""
            dims = _normalize_dims(self.dim, x.dim())
            out = _compute_std_core(
                x,
                mean,
                dims,
                self.keepdim,
                self.invert,
                self.epsilon,
                self.correction,
            )
            return [out]

    third_party = {"torch": ThirdPartyImpl}


class AclnnStdMeanCorrectionTestSpec:
    """aclnnStdMeanCorrection aclnn 通路参考实现。

    签名按 aclnnStdMeanCorrectionGetWorkspaceSize：
      (self, dim, correction, keepdim, stdOut, meanOut) → (std, mean)
    L2 侧 meanOut 也会输出，golden 需同时给出两个结果。
    """

    def golden(
        self, dim=None, correction=1, keepdim=False, stdOut=None, meanOut=None, **kwargs
    ):
        """aclnn golden（入参为 torch.Tensor，经 TTK aclnn 通路分发；stdOut/meanOut 为输出槽，golden 忽略）。"""
        import torch

        x_t = self.to(torch.float32)
        d = _normalize_dims(dim, x_t.dim())
        if len(d) == 0:
            d = list(range(x_t.dim()))

        std, mean = torch.std_mean(x_t, dim=d, correction=correction, keepdim=keepdim)
        return [std, mean]

    def customize_inputs(
        self, dim=None, correction=1, keepdim=False, stdOut=None, meanOut=None, **kwargs
    ):
        # 签名须与 aclnnStdMeanCorrection API 参数计划一致（TTK 以未绑定类属性按位置传参）；
        # 返回 None 表示不做输入定制（原地写法约定）。
        return None

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
