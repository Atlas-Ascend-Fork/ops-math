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
"""Cast multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/cast_apt.cpp 有 arch35 实现 |
| geir   | ✅   | op_graph/cast_proto.h 有 REG_OP(Cast) |
| aclnn  | ✅   | op_api/aclnn_cast.cpp 暴露 aclnnCast 符号 |
| e2e    | ✅   | torch_npu 中 torch.Tensor.to / torch.cast 绑定到 aclnnCast |

kernel 与 geir 共用一个注册键（算子蛇形名），geir 不另写 Spec 类。
"""

import numpy as np

__spec__ = {
    "cast": "CastKernelSpec",
    "aclnnCast": "CastAclnnSpec",
    "torch.Tensor.to": "CastE2eSpec",
}

__golden__ = {
    "aclnn": {
        "aclnnCast": "aclnn_cast_golden",
    },
    "kernel": {"cast": "cast_golden"},
    "e2e": {"aclnnCast": "aclnn_cast_golden"},
}

_DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
}

_SPECIAL_DTYPES = (
    "bfloat16",
    "int4",
    "float8_e5m2",
    "float8_e4m3fn",
    "float4_e2m1",
    "float4_e1m2",
    "hifloat8",
)

# 判据声明: 覆盖 cast_def.cpp 注册的全部 dtype。
# 浮点默认 cross_check —— 只有 cross_check 才会让 TTK 取三方(GPU)输出并开 golden_mode=Promote;
# bfloat16 例外走 binary_equal: cast 是精确类型转换，而 cross_check 会触发 golden
# Promote，遇到输入为 e8m0(非 torch 原生) 时跨边界 promote 未实现会直接报错。
# 整型/布尔位精确，用 binary_equal; 复数走 stat_rel_err。
_KERNEL_TOLERANCE = {
    "float16": {"standard": "binary_equal", "level": "L1"},
    "float32": {"standard": "binary_equal", "level": "L1"},
    "float64": {"standard": "binary_equal", "level": "L1"},
    "bfloat16": {"standard": "binary_equal"},
    # E8M0 无尾数、按位精确，且 NaN 编码无法用浮点比较校验，故用 binary_equal。
    "float8_e8m0": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
    "complex64": {"standard": "stat_rel_err"},
    "complex128": {"standard": "stat_rel_err"},
}


def _resolve_custom_numpy_dtype(dtype_str):
    if dtype_str == "bfloat16":
        from ml_dtypes import bfloat16

        return bfloat16
    elif dtype_str == "int4":
        from ml_dtypes import int4

        return int4
    elif dtype_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    elif dtype_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    elif dtype_str == "float8_e8m0":
        from en_dtypes import float8_e8m0

        return float8_e8m0
    elif dtype_str == "hifloat8":
        from en_dtypes import hifloat8

        return hifloat8
    elif dtype_str == "float4_e2m1":
        from ml_dtypes import float4_e2m1

        return float4_e2m1
    elif dtype_str == "float4_e1m2":
        from ml_dtypes import float4_e1m2

        return float4_e1m2
    return None


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def _bf16_to_e8m0(x, e8m0_dtype):
    """BF16 -> E8M0，与 kernel/PyTorch fp8e8m0fnu_from_fp32_value 按位一致。

    ml_dtypes/en_dtypes 的 astype 会把 0 和负数编码成 NaN，与设备不符，故直接
    在 BF16 原始位上复现该语义：E8M0 无符号位（丢弃符号）、无 inf；指数字段
    255（inf/NaN 输入或幅值溢出）编码为 NaN(255)。
    """
    bits = np.ascontiguousarray(x).view(np.uint16).astype(np.uint32)
    exponent = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    # 规格化数按 guard 位四舍五入(逢 0.5 进位)；非规格化数隐含最低位为 0，
    # 恰为 0.5 时不进位。
    rounded = np.where(exponent != 0, mantissa + 0x40, mantissa + 0x3F)
    code = np.where(exponent == 0xFF, 0xFF, exponent + (rounded >> 7))
    return code.astype(np.uint8).view(e8m0_dtype)


def cast_golden(x, dst_type: int, **kwargs):
    """
    Kernel golden for cast.
    All the parameters follow @cast_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
              input_formats, output_formats, input_ori_formats, output_ori_formats,
              input_dtypes, output_dtypes.
    """
    dst_type_str = _DATA_TYPE_INT_TO_STR.get(dst_type, str(dst_type))
    if (x.dtype.name == "bfloat16" and dst_type_str == "hifloat8") or (
        x.dtype.name == "hifloat8" and dst_type_str == "bfloat16"
    ):
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np.float32).astype(np_dtype)
    elif dst_type_str == "float8_e8m0":
        # cast_def.cpp 仅注册 BF16 -> FLOAT8_E8M0。
        return _bf16_to_e8m0(x, _resolve_custom_numpy_dtype(dst_type_str))
    elif x.dtype.name in ("float8_e8m0", "float8_e8m0fnu"):
        # cast_def.cpp 仅注册 FLOAT8_E8M0 -> BF16；E8M0 经 float32 无损解码
        # (码值 c -> 2**(c-127)，c==255 -> NaN)。
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np.float32).astype(np_dtype)
    elif dst_type_str in _SPECIAL_DTYPES:
        np_dtype = _resolve_custom_numpy_dtype(dst_type_str)
        return x.astype(np_dtype)
    elif dst_type_str == "complex32":
        _shape = list(x.shape)
        x = x.reshape(_shape + [1])
        imag = np.zeros(_shape + [1], dtype=np.float16)
        res = np.concatenate((x, imag), axis=-1)
        return res
    elif dst_type_str == "bool":
        return x.astype(np.bool_)
    else:
        return x.astype(getattr(np, dst_type_str))


def aclnn_cast_golden(self, dtype=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnCast.
    Parameters follow @aclnnCastGetWorkspaceSize without workspaceSize & executor.

    Torch-native dtypes arrive as torch.Tensor; CANN-only target dtypes (e.g.
    float8_e8m0) are not representable by torch, so the framework keeps the
    input as numpy.ndarray (use_numpy=True) and the numpy golden is used.
    """
    import torch

    if isinstance(self, torch.Tensor):
        return self.to(dtype=_torch_dtype(dtype))
    return cast_golden(self, dtype, **kwargs)


# torch 侧命名差异与 torch 无法表示的 CANN 专有 dtype。
_TORCH_DTYPE_ALIASES = {"float8_e8m0": "float8_e8m0fnu"}
_TORCH_UNSUPPORTED_DTYPES = frozenset(
    (
        "hifloat8",
        "hifloat4",
        "int4",
        "int2",
        "uint1",
        "uint2",
        "float4_e2m1",
        "float4_e1m2",
    )
)


def _torch_dtype(dtype):
    """Resolve the cast target dtype to a torch.dtype.

    Accepts an aclnn dtype enum int, a torch.dtype, or a string ("torch.float16"
    / "float16"). Raises ValueError naming the dtype when torch cannot represent
    it, so an unsupported case fails readably instead of deep inside the table
    lookup.
    """
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, bool) or not isinstance(dtype, (int, str)):
        raise ValueError(f"cast: unsupported target dtype {dtype!r}")
    name = (
        _DATA_TYPE_INT_TO_STR.get(dtype, str(dtype))
        if isinstance(dtype, int)
        else dtype
    )
    name = _TORCH_DTYPE_ALIASES.get(name.rsplit(".", 1)[-1], name.rsplit(".", 1)[-1])
    if name in _TORCH_UNSUPPORTED_DTYPES:
        raise ValueError(f"cast: torch has no dtype for '{name}' (CANN-only dtype)")
    resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"cast: torch has no dtype for '{name}'")
    return resolved


def _third_party_tensor_and_dtype(args, kwargs):
    """Locate the input tensor and the target dtype in a bound third_party call.

    The aclnn input is named ``self`` (aclnn_cast.h) while the kernel pool is
    (x, dst_type). The compose declares ``self`` verbatim (renaming the instance
    slot to ``impl``) so it binds under both old and new XPU binders; kernel
    tensors still arrive as ``x``. Scan both channels by type.
    """
    import torch

    tensor = kwargs.get("x")
    tensor_types = (torch.Tensor, np.ndarray)
    if not isinstance(tensor, tensor_types):
        tensor = next(
            (
                v
                for v in list(args) + list(kwargs.values())
                if isinstance(v, tensor_types)
            ),
            None,
        )
    dtype = kwargs.get("dtype")
    if dtype is None:
        dtype = kwargs.get("dst_type")
    if dtype is None:
        dtype = next(
            (v for v in args if v is not None and not isinstance(v, tensor_types)), None
        )
    if tensor is None or dtype is None:
        raise ValueError(
            "cast third_party expects one input tensor and a target dtype "
            f"(args={[type(v).__name__ for v in args]}, kwargs={sorted(kwargs)})"
        )
    return tensor, dtype


class _CastCompose:
    """Third-party reference executed on the remote GPU server.

    torch.Tensor.to is the competitor interface; the kernel golden uses
    numpy.astype, so the two paths are independent implementations of the
    cast semantics and can cross-validate each other.
    """

    def __call__(impl, self=None, *args, **kwargs):
        tensor, dtype = _third_party_tensor_and_dtype((self, *args), kwargs)
        import torch

        if isinstance(tensor, torch.Tensor):
            return [tensor.to(dtype=_torch_dtype(dtype))]
        return [cast_golden(tensor, dtype, **kwargs)]


class CastKernelSpec:
    """kernel + geir shared spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, dst_type, **kwargs):
        result = cast_golden(x, dst_type, **kwargs)
        output_dtype = _output_dtype(kwargs, 0, str(result.dtype))
        np_dtype = _resolve_custom_numpy_dtype(output_dtype)
        if np_dtype is not None:
            return [result.astype(np_dtype, copy=False)]
        target = getattr(np, output_dtype, None)
        if target is not None:
            return [result.astype(target, copy=False)]
        return [result]

    third_party = {"torch": _CastCompose}
    tolerance = _KERNEL_TOLERANCE


class CastAclnnSpec:
    """aclnnCast spec. The golden entry receives torch tensors.

    The parameter name follows aclnn_cast.h, where the input is named self.
    """

    @staticmethod
    def golden(self, dtype=0, out=None, **kwargs):
        del out
        return [aclnn_cast_golden(self, dtype, **kwargs)]

    third_party = {"torch": _CastCompose}
    tolerance = _KERNEL_TOLERANCE


class _CastE2eCompose:
    """Third-party reference for the E2E path (api ``torch.Tensor.to``).

    torch.Tensor.to is a C builtin without an inspectable signature, so the
    server-side api mode fails with "no signature found for builtin". This
    compose re-declares the call in plain Python; tensors are recognized by
    instance type and the target dtype arrives under the 'dtype' attribute
    key (string form, e.g. "torch.float32").
    """

    def __call__(impl, self=None, *args, **kwargs):
        import torch

        tensor = None
        for v in (self, *args, *kwargs.values()):
            if isinstance(v, torch.Tensor):
                tensor = v
                break
        dtype = kwargs.get("dtype")
        if dtype is None:
            for v in (self, *args):
                if (
                    v is not None
                    and not isinstance(v, torch.Tensor)
                    and not isinstance(v, (list, tuple))
                ):
                    dtype = v
                    break
        if tensor is None or dtype is None:
            raise ValueError(
                "cast e2e compose expects one input tensor and a 'dtype' attribute"
            )
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.rsplit(".", 1)[-1])
        return [tensor.to(dtype=dtype)]


class CastE2eSpec:
    """E2E spec, keyed by the dotted api name ``torch.Tensor.to`` in __spec__.

    third_party is a compose class so the XPU dispatch runs in spec mode
    (server instantiates and binds params by name) instead of api mode.
    """

    third_party = {"torch": _CastE2eCompose}
