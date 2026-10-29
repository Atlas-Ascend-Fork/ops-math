#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""pdist 在 kernel/geir/aclnn 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 torch tensor 后按 p 分支：
  无 NaN 输入：调 torch.nn.functional.pdist（官方 CPU 实现，语义与契约一致）
  含 NaN 输入：用 torch 小算子（abs + count_nonzero / where-isnan + amax /
               pow + sum）向量化拼接 —— 官方 CPU 实现的 p=0（min(ceil) 路径）
               与 p=+inf 的 NaN 行为和 GPU/CUDA 契约不一致，不可直接使用
ACLNN 的 golden 直接收到 torch.Tensor，调同一核心实现，返回 torch.Tensor；
third_party 的 torch vendor 收到 torch.Tensor，复用核心实现，无需 numpy 转换。

p 分支语义（与 op_kernel/pdist.py 冻结的 SPEC 一致，GPU/CUDA 对齐）：
  p == 0      : y[idx] = sum_k (x[i,k] != x[j,k])               Hamming
                (NaN 维度计为不相等，输出不含 NaN)
  p == +inf   : y[idx] = max_k |x[i,k]-x[j,k]|                  Chebyshev
                (CUDA 比较取大语义: NaN 维度被忽略; 全 NaN 对 -> 0)
  0 < p < inf : y[idx] = ( sum_k |x[i,k]-x[j,k]|**p )**(1/p)    Minkowski
输出为长度 N*(N-1)/2 的上三角行优先向量, idx = i*(2*N-i-1)//2 + (j-i-1)。

pypto 约束：kernel 路径的 golden 与 TTK 在线编译同进程，golden 加载前
sys.modules["torch"] 可能已被 MagicMock 替换，因此所有入口先经
_real_torch() 恢复真实模块（含模块全局 torch 绑定的修正，见其 docstring）。
fp16 输入在入口拓宽到 fp32 计算，出口窄化回原 dtype（与 kernel 片上
fp32 累积链一致）。
"""

import math

import numpy as np

import torch

# TTK compatibility layer: __golden__ dict expected by ttk's scanner
__golden__ = {
    "kernel": {"pdist": "_kernel_golden_wrapper"},
    "aclnn": {
        "aclnnPdist": "_aclnn_golden_wrapper",
        "aclnnPdistForward": "_aclnn_forward_golden_wrapper",
    },
}


def _kernel_golden_wrapper(input_x, p=2.0, **kwargs):
    return PdistTestSpec.golden(input_x, p, **kwargs)


def _aclnn_golden_wrapper(x, p=2.0, out=None, **kwargs):
    return AclnnPdistTestSpec.golden(x, p=p, out=out, **kwargs)


def _aclnn_forward_golden_wrapper(x, pScalar=2.0, out=None, **kwargs):
    return AclnnPdistForwardTestSpec.golden(x, pScalar=pScalar, out=out, **kwargs)


__spec__ = {
    "pdist": "PdistTestSpec",
    "aclnnPdist": "AclnnPdistTestSpec",
    "aclnnPdistForward": "AclnnPdistForwardTestSpec",
}

# pdist_compare hook parameters (element rule mirrors TTK mix_tolerance).
_COMPARE_RTOL = 0.001
_COMPARE_ATOL = 0.001
_COMPARE_REQUIRED_RATIO = 0.99
_FIXED_ABS_CAP = {"float16": 2.0, "float32": 1.0}
_MANTISSA_BITS = {"float16": 10, "float32": 23}

_INF_STRINGS = ("inf", "+inf", "infinity", "+infinity")

# Work budgets for the chunked pair enumeration (performance knobs only).
_WORK_ELEMS = 128 * 1024 * 1024  # max elements of the (c, T, M) fp32 diff temporary
_DM_ELEMS = 16 * 1024 * 1024  # max elements of the (c, T) reduced distance matrix

_TORCH = None


def _normalize_p(p):
    """Normalize the p attribute: accepts int/float and the string "inf" forms."""
    if isinstance(p, str):
        s = p.strip().lower()
        if s in _INF_STRINGS:
            return float("inf")
        return float(s)
    return float(p)


def _real_torch():
    """Return the real torch module, recovering from the pypto build stub.

    The pypto opc dynamic-compile wrapper (custom_transformer_impl/dynamic/
    <op>.py) replaces sys.modules["torch"] with a MagicMock at import time so
    the DSL kernel can be loaded without a real torch. TTK's kernel mode
    compiles and computes the golden in the SAME process, so the module-global
    `torch` binding (and any later `import torch`) would silently compute on
    mocks. A fresh re-import is NOT possible either: torch._C is still loaded
    and re-executing torch's __init__ aborts on double TORCH_LIBRARY
    registration. Instead, the real module object is recovered from any
    torch.* submodule imported before the stub (their module globals keep the
    parent reference, e.g. torch._utils); sys.modules["torch"] and the
    module-global `torch` binding are both restored so the rest of the process
    (including the small-op core below, which uses the global name directly)
    gets the real module.
    """
    global _TORCH
    global torch
    if _TORCH is not None:
        if torch is not _TORCH:
            torch = _TORCH
        return _TORCH
    import sys

    mod = sys.modules.get("torch")
    if mod is not None and not type(mod).__module__.startswith("unittest.mock"):
        _TORCH = mod
        torch = mod
        return mod
    for name in list(sys.modules):
        if not name.startswith("torch."):
            continue
        cand = getattr(sys.modules[name], "__dict__", {}).get("torch", None)
        if (
            cand is not None
            and hasattr(cand, "from_numpy")
            and hasattr(cand, "__version__")
        ):
            _TORCH = cand
            sys.modules["torch"] = cand  # un-stub for the rest of the process
            torch = cand
            return cand
    raise RuntimeError(
        "pdist golden: real torch module not recoverable "
        "(pypto stub active and no pre-imported torch.* submodule)"
    )


def _validate(x, p):
    """共同校验 (kernel/aclnn 链路均拒绝): 2D / fp16|fp32 / p >= 0 非 NaN。

    非法参数抛 ValueError, 模拟算子的 bad params 行为 (TTK 将该用例按
    预期参数错误处理); 形状类 bad params 契约 kernel 与 aclnn 不同,
    分别见 _validate_kernel_shape 与 AclnnPdistTestSpec.golden。
    """
    if x.ndim != 2:
        raise ValueError(f"bad params: input_x must be 2D (N, M), got {x.shape}")
    dtype_str = str(x.dtype).replace("torch.", "")
    if dtype_str not in ("float16", "float32"):
        raise ValueError(
            f"bad params: input_x dtype must be float16/float32, got {x.dtype}"
        )
    if math.isnan(p) or p < 0:
        raise ValueError(f"bad params: p must be >= 0 or +inf, got {p}")
    if math.isinf(p) and p < 0:
        raise ValueError(f"bad params: p must be >= 0 or +inf, got {p}")


# A5 算子契约上限 (op_host/arch35/pdist_tiling.cpp): 2 <= N <= 65535, 1 <= M <= 65535.
_PDIST_MAX_ROWS = 65535
_PDIST_MAX_COLS = 65535


def _validate_kernel_shape(x):
    """kernel 路径形状契约 (950 tiling 全量校验): 2 <= N <= 65535, 1 <= M <= 65535."""
    n, m = x.shape
    if n < 2:
        raise ValueError(f"bad params: N must be >= 2 (at least one row pair), got {n}")
    if m < 1:
        raise ValueError(f"bad params: M must be >= 1, got {m}")
    if n > _PDIST_MAX_ROWS:
        raise ValueError(f"bad params: N must be <= {_PDIST_MAX_ROWS}, got {n}")
    if m > _PDIST_MAX_COLS:
        raise ValueError(f"bad params: M must be <= {_PDIST_MAX_COLS}, got {m}")


def _third_party_input(args, kwargs, names=("selfT", "self", "x")):
    """从 ThirdPartyImpl 的调用参数中提取输入张量。

    兜底路径: 当 __call__ 的原型名参数 (x/self) 未收到值时, 从 kwargs 的
    selfT/self/x 键中提取输入张量 (TTK 兜底别名/历史调用形态兼容)。
    """
    if args:
        return args[0]
    for name in names:
        if name in kwargs:
            return kwargs[name]
    raise TypeError(
        "pdist third_party: input tensor not provided (args=%r, kwargs=%r)"
        % (args, sorted(kwargs))
    )


def _golden_input(first, kwargs, *names):
    """golden 入口的输入张量归一化。

    ttk 的 _run_perf 按 api 原型名喂参 (aclnn 原型输入名为 'self'), 位置传参
    时落到首参; 关键字喂参时可能为 selfT/self/x 中的任一个, 依次兜底。
    """
    if first is not None:
        return first
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs.pop(name)
    raise ValueError("bad params: pdist golden input tensor not provided")


def _pdist_torch_native(xt, p):
    """官方 torch.nn.functional.pdist 路径 — 仅限无 NaN 输入。

    fp32 输入下官方 CPU 实现与本 golden 的 GPU/CUDA 契约逐分支一致
    （p=0 计数 / p=+inf 取大 / Minkowski 幂和）；含 NaN 输入时 CPU 的
    p=0（min(ceil) 路径）与 p=+inf 行为偏离契约，由 _pdist_dispatch 拦截。
    """
    _real_torch()
    return torch.nn.functional.pdist(xt, p)


def _pdist_torch_small(xt, p):
    """Chunked FP32 pairwise p-norm distance — torch 小算子拼接（契约完整实现）。

      - 0 < p < +inf : abs -> pow(p) -> sum(M) -> pow(1/p)
      - p == 0       : abs -> count_nonzero(M)   (NaN != 0, counted, no NaN out)
      - p == +inf    : abs -> where(isnan, 0) -> amax(M)  (NaN dims ignored,
                       all-NaN pair -> 0 — CUDA comparison-max semantics)
    输出为长度 N*(N-1)/2 的上三角行优先向量。
    """
    _real_torch()
    n, m = xt.shape
    length = n * (n - 1) // 2
    y = torch.empty(length, dtype=torch.float32)
    pos = 0
    i0 = 0
    with torch.no_grad():
        while i0 < n - 1:
            t_cnt = n - i0  # targets x[i0:] cover every j > i for rows in this chunk
            c = _WORK_ELEMS // (t_cnt * m)
            c = min(c, max(1, _DM_ELEMS // t_cnt))
            c = max(1, min(c, n - 1 - i0))
            rows = xt[i0 : i0 + c]  # (c, M)
            tgt = xt[i0:]  # (T, M)
            diff = (rows.unsqueeze(1) - tgt.unsqueeze(0)).abs()  # (c, T, M) fp32
            if p == 0.0:
                dm = torch.count_nonzero(diff, dim=-1).to(torch.float32)
            elif math.isinf(p):
                dm = torch.where(
                    torch.isnan(diff), torch.zeros((), dtype=diff.dtype), diff
                ).amax(dim=-1)
            else:
                dm = diff.pow(p).sum(dim=-1).pow(1.0 / p)
            # Pairs (r, t) with t > r in row-major order: exactly the global
            # upper-triangular order for rows [i0, i0+c).
            r = torch.arange(c).unsqueeze(1)
            t = torch.arange(t_cnt).unsqueeze(0)
            vals = torch.masked_select(dm, r < t)
            y[pos : pos + vals.numel()] = vals
            pos += vals.numel()
            i0 += c
    return y


def _pdist_dispatch(xt, p):
    """含 NaN 维度时走小算子契约实现；无 NaN 时官方 F.pdist 等价且更快。"""
    _real_torch()
    if bool(torch.isnan(xt).any()):
        return _pdist_torch_small(xt, p)
    return _pdist_torch_native(xt, p)


def _dtype_ulp(dtype_str, value):
    """One ulp of `value`'s binade in the golden's dtype (0 for non-positive/inf)."""
    if value <= 0.0 or not math.isfinite(value):
        return 0.0
    return 2.0 ** (math.floor(math.log2(value)) - _MANTISSA_BITS.get(dtype_str, 23))


def pdist_compare(*tensors, **kwargs):
    """TestSpec `compare` hook: mix_tolerance with an output-magnitude-scaled cap.

    Same element rule as TTK mix_tolerance (|a-g| <= atol + rtol*|g|, same-NaN /
    same-Inf counted equal, matched_ratio >= 0.99). The single deviation is the
    global absolute cap: instead of a fixed max_abs_error_limit it is
    max(fixed, 2 * ulp(max|golden|)), so the cap tracks the exit-rounding ulp of
    the largest actual output instead of an assumed small-magnitude output.
    Shared by the kernel / aclnn / geir pathways (outputs..., goldens...).
    """
    torch_mod = _real_torch()
    half = len(tensors) // 2
    results = []
    for output, golden in zip(tensors[:half], tensors[half:]):
        a = torch_mod.as_tensor(output)
        g = torch_mod.as_tensor(golden)
        dtype_str = str(g.dtype).replace("torch.", "")
        if a.numel() == 0 or g.numel() == 0:
            same_size = a.numel() == g.numel()
            results.append(
                {
                    "pass": same_size,
                    "precision": 100.0 if same_size else 0.0,
                    "metrics": {"standard": "pdist_scaled_mix", "note": "empty output"},
                }
            )
            continue
        work = torch_mod.promote_types(a.dtype, g.dtype)
        if work.itemsize < 4:
            work = torch_mod.float32  # at least fp32, matching the fp32 widen chain
        a = a.to(work)
        g = g.to(work)
        err = (a - g).abs()
        a_nan, g_nan = torch_mod.isnan(a), torch_mod.isnan(g)
        a_inf, g_inf = torch_mod.isinf(a), torch_mod.isinf(g)
        same_nan = a_nan & g_nan
        same_inf = a_inf & g_inf & (torch_mod.sign(a) == torch_mod.sign(g))
        finite = torch_mod.isfinite(a) & torch_mod.isfinite(g)
        elem_ok = (
            same_nan
            | same_inf
            | (finite & (err <= _COMPARE_ATOL + _COMPARE_RTOL * g.abs()))
        )
        matched_ratio = elem_ok.sum().item() / a.numel()
        mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
        if bool(mismatch.any()):
            max_abs_error = float("inf")
        else:
            finite_err = err[finite]
            max_abs_error = (
                float(finite_err.max().item()) if finite_err.numel() else 0.0
            )
        gmax = float(g.abs()[finite].max().item()) if bool(finite.any()) else 0.0
        cap = max(_FIXED_ABS_CAP.get(dtype_str, 1.0), 2.0 * _dtype_ulp(dtype_str, gmax))
        passed = matched_ratio >= _COMPARE_REQUIRED_RATIO and max_abs_error <= cap
        item = {
            "pass": bool(passed),
            "precision": matched_ratio * 100.0,
            "metrics": {
                "standard": "pdist_scaled_mix",
                "matched_ratio": matched_ratio,
                "max_abs_error": None if math.isinf(max_abs_error) else max_abs_error,
                "scaled_max_abs_error_limit": cap,
                "golden_max_abs": gmax,
            },
        }
        if not passed:
            item["diff_indices"] = (~elem_ok).nonzero(as_tuple=False).flatten().tolist()
        results.append(item)
    return results


class PdistTestSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray。"""

    def golden(input_x=None, p=2.0, **kwargs):
        """
        Golden function for pdist.
        All the parameters (names and order) follow @pdist_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            input_x: numpy array, shape (N, M), fp32/fp16 input
            p: int/float/str, p-norm order (p >= 0 or +inf; "inf" forms accepted)
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            [y]: list of numpy.ndarray, shape (N*(N-1)/2,), dtype follows input_x
        """
        _real_torch()
        if input_x is None:
            input_x = _golden_input(None, kwargs, "input_x", "x", "self")
        x_np = np.asarray(input_x)
        p = _normalize_p(p)
        _validate(x_np, p)
        _validate_kernel_shape(x_np)
        xt = torch.from_numpy(np.ascontiguousarray(x_np)).to(
            torch.float32
        )  # widen at entry
        y = _pdist_dispatch(xt, p)
        return [y.cpu().numpy().astype(x_np.dtype, copy=False)]  # narrow at exit

    class ThirdPartyImpl:
        """torch 竞品实现 — kernel 通路。

        ttk _run_perf 按 api 原型名喂参 (输入与属性同名两喂), 签名白名单
        仅接受输入名/属性名。__call__ 为 staticmethod: 参数名不再与 Python
        绑定冲突, 直接以原型名 (pdist_def.cpp: 输入 x / 属性 p) 作签名,
        selfT/x 关键字与位置喂参作兜底。
        """

        def __init__(self, *args, **kwargs):
            self.p = _normalize_p(kwargs.get("p", 2.0))

        @staticmethod
        def __call__(*args, **kwargs):
            x = _third_party_input(args, kwargs)
            y = _pdist_dispatch(
                x.detach().to(torch.float32), _normalize_p(kwargs.get("p", 2.0))
            )
            return [y.to(x.dtype)]

    third_party = {"torch": ThirdPartyImpl}
    # SPEC machine contract tolerance: atol = rtol = 1e-3 for both IO dtypes
    # (fp16 error is already contained by the on-chip fp32 accumulation chain).
    # The fixed max_abs_error_limit values are the SMALL-magnitude backstop of
    # the scaled cap enforced by the pdist_compare hook below (TestSpec
    # `compare`): pdist outputs are unbounded in magnitude, and the
    # op-vs-golden difference is exit-rounding noise of at most ~2 ulp of the
    # LARGEST output (fp16 CAST_RINT at the exit; fp32 direct write).
    tolerance = {
        "float16": {
            "standard": "mix_tolerance",
            "rtol": 0.001,
            "atol": 0.001,
            "max_abs_error_limit": 2.0,
        },
        "float32": {
            "standard": "mix_tolerance",
            "rtol": 0.001,
            "atol": 0.001,
            "max_abs_error_limit": 1.0,
        },
    }
    compare = staticmethod(pdist_compare)


class AclnnPdistTestSpec:
    """ACLNN 流程 — golden 收到 torch.Tensor（已在设备上）。"""

    class ThirdPartyImpl:
        """torch 竞品实现 — aclnn 通路。

        ttk _run_perf 按 api 原型名喂参 (输入与属性同名两喂), 签名白名单
        仅接受输入名/属性名。__call__ 为 staticmethod: 参数名不再与 Python
        绑定冲突, 直接以 aclnn 原型名 (aclnn_pdist.h: 输入 self / 属性 p)
        作签名; TTK 实例调用时 staticmethod 描述符已剥离实例绑定, self
        关键字可正常传入。
        """

        def __init__(self, *args, **kwargs):
            self.p = _normalize_p(kwargs.get("p", 2.0))

        @staticmethod
        def __call__(*args, **kwargs):
            x = _third_party_input(args, kwargs)
            y = _pdist_dispatch(
                x.detach().to(torch.float32), _normalize_p(kwargs.get("p", 2.0))
            )
            return [y.to(x.dtype)]

    third_party = {"torch": ThirdPartyImpl}

    def golden(selfT=None, p=2.0, out=None, **kwargs):
        """
        Aclnn golden for aclnnPdist.
        All the parameters (name & order) follow \
            function `aclnnPdistGetWorkspaceSize` in \
            @aclnn_pdist.h \
            without `workspaceSize` & `executor`.
        The Tensors in the parameters are torch.Tensor.

        Args:
            kwargs: tensor_{dtypes, formats}, short_soc_version, testcase_name

        Returns:
            [y]: list of torch.Tensor, shape (N*(N-1)/2,), dtype follows self
        """
        torch_mod = _real_torch()
        if selfT is None:
            selfT = _golden_input(None, kwargs, "selfT", "self", "x")
        p = _normalize_p(p)
        _validate(selfT.detach(), p)
        x_t = selfT.detach()
        n, m = x_t.shape
        # master aclnn 组合层特例 (aclnn_pdist.cpp): N <= 1 短路返回空输出
        if n <= 1:
            return [x_t.new_empty((0,))]
        # master aclnn 组合层特例: M == 0 走 Fill 0 (空向量距离为 0)
        if m == 0:
            return [x_t.new_zeros((n * (n - 1) // 2,))]
        # A5 tiling 契约: N/M 超 65535 报 bad params
        if n > _PDIST_MAX_ROWS or m > _PDIST_MAX_COLS:
            raise ValueError(
                f"bad params: N/M must be <= 65535, got N={n}, M={m} "
                "(2 <= N <= 65535, 1 <= M <= 65535 on Ascend 950)"
            )
        xt = x_t.to(torch_mod.float32)  # widen at entry
        y = _pdist_dispatch(xt, p)
        return [y.to(selfT.dtype)]  # narrow at exit

    tolerance = PdistTestSpec.tolerance
    compare = staticmethod(pdist_compare)


class AclnnPdistForwardTestSpec:
    """ACLNN 流程 — aclnnPdistForward（scalar-p 复合入口，同一数学）。"""

    def golden(selfT=None, pScalar=2.0, out=None, **kwargs):
        """
        Aclnn golden for aclnnPdistForward.
        All the parameters (name & order) follow \
            function `aclnnPdistForwardGetWorkspaceSize` in \
            @aclnn_pdist_forward.h \
            without `workspaceSize` & `executor`.
        The Tensors in the parameters are torch.Tensor.

        Args:
            kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

        Returns:
            [y]: list of torch.Tensor, shape (N*(N-1)/2,), dtype follows self
        """
        if selfT is None:
            selfT = _golden_input(None, kwargs, "selfT", "self", "x")
        return AclnnPdistTestSpec.golden(selfT, p=pScalar, out=out, **kwargs)

    class ThirdPartyImpl:
        """torch 竞品实现 — aclnn forward 通路。

        ttk _run_perf 按 api 原型名喂参 (输入与属性同名两喂), 签名白名单
        仅接受输入名/属性名。__call__ 为 staticmethod: 参数名不再与 Python
        绑定冲突, 直接以 aclnn 原型名 (aclnn_pdist_forward.h: 输入 self /
        属性 pScalar) 作签名, p 作别名兜底。
        """

        def __init__(self, *args, **kwargs):
            pv = kwargs.get("pScalar")
            if pv is None:
                pv = kwargs.get("p", 2.0)
            self.p = _normalize_p(pv)

        @staticmethod
        def __call__(*args, **kwargs):
            x = _third_party_input(args, kwargs)
            pv = kwargs.get("pScalar")
            if pv is None:
                pv = kwargs.get("p", 2.0)
            y = _pdist_dispatch(x.detach().to(torch.float32), _normalize_p(pv))
            return [y.to(x.dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = PdistTestSpec.tolerance
    compare = staticmethod(pdist_compare)
