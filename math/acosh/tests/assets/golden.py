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
"""TTK golden plugin for the Acosh operator family.

Registered names (CSV op_name / api_name dialects):
  - "acosh" / "Acosh" : Kernel / GEIR path ("Acosh" = GE op type CamelCase)
  - "aclnnAcosh"      : ACLNN path
  - "torch.acosh"     : Torch E2E path
  - "tf.math.acosh"   : TensorFlow E2E path

Import discipline: torch / tensorflow are imported lazily inside function
bodies only -- TTK loads this file inside processes where aclrt may already
be initialized, and a module-level framework import risks SIGSEGV.

The Aclnn golden mirrors the aclnn cast chain (op_api/aclnn_acosh.cpp):
dtypes already in the output family compute natively; int/uint/bool inputs
promote to float64 when out=DOUBLE (INT64 low-bit preservation) else
float32; the result is cast back to the out dtype.

Tolerance is keyed by OUTPUT dtype (TTK convention): float16/float32/
bfloat16 cross_check L2 on Kernel/Aclnn/Torch specs, plus float64/
complex64/complex128 on Aclnn/Torch specs (wide input coverage); the TF
path is level L1.
"""

__spec__ = {
    "acosh": "AcoshKernelTestSpec",
    "Acosh": "AcoshKernelTestSpec",  # GE类型名双注册(REG_OP(Acosh))
    "aclnnAcosh": "AclnnAcoshTestSpec",
    "torch.acosh": "TorchAcoshTestSpec",
    "tf.math.acosh": "AcoshTensorFlowTestSpec",
}


def _get_tf_input(args, kwargs):
    """从位置参数中取第一个 tf.Tensor，否则按别名取，均无则报错。"""
    import tensorflow as tf

    x = next((arg for arg in args if isinstance(arg, tf.Tensor)), None)
    if x is not None:
        return x
    for name in ("x", "self", "input", "inputRef"):
        x = kwargs.get(name)
        if x is not None:
            return x
    raise ValueError(
        "Missing tensor input; expected positional Tensor or one of: x, self, input, inputRef"
    )


class _TensorFlowAcosh:
    """TensorFlow 三方标杆：tf.math.acosh，原生 dtype 直接计算（不升精度）。"""

    def __call__(self, *args, **kwargs):
        import tensorflow as tf

        return [tf.math.acosh(_get_tf_input(args, kwargs))]


def aclnn_acosh_golden(selfT, out, **kwargs):
    """CPU golden following aclnnAcoshGetWorkspaceSize(self, out, ...).

    Mirrors the aclnn cast chain (op_api/aclnn_acosh.cpp:170-183) --
    dtypes already in the output family (FLOAT/FLOAT16/DOUBLE/BF16/COMPLEX64/
    COMPLEX128) compute natively; int8-64/uint8/bool inputs promote to
    DT_DOUBLE when out=DOUBLE (INT64 |x|>2^24 low-bit preservation)
    else DT_FLOAT; the result is cast back to the out dtype (sibling Asinh
    aclnn golden convention: ``torch.asinh(input).to(out.dtype)``).

    TTK Promote interplay (ttk/core_modules/aclnn/golden.py::_promote_dtype,
    PROMOTE_GOLDEN_TOKENS = cross_check/mixed/mix_tolerance): under these
    comparison standards every FLOAT tensor slot — including ``out`` and the
    golden inputs — arrives pre-promoted (f16/bf16→f32, f32→f64, c64→c128),
    so e.g. an int8 case with a float32 CSV output sees out=float64 here and
    computes the f64 TRUE VALUE (5e-8-class rounding vs the f32 NPU output,
    absorbed by the output-dtype-keyed tolerance). Outside Promote the raw
    CSV out dtype drives the exact operator cast-chain mirror. Integers are
    absent from DTYPE_PROMOTE_MAP, so the int/uint/bool branch below is the
    only promotion those inputs ever receive.
    """
    import torch  # lazy import: TTK 进程内 aclrt 可能已初始化

    output_family = (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
    )
    if selfT.dtype in output_family:
        result = torch.acosh(selfT)
    else:
        compute_dtype = torch.float64 if out.dtype == torch.float64 else torch.float32
        result = torch.acosh(selfT.to(compute_dtype))
    return [result.to(out.dtype)]


class AcoshKernelTestSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name`` ("acosh"/"Acosh")."""

    golden = "torch.acosh"
    third_party = {"torch": "torch.acosh", "tf": _TensorFlowAcosh}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
    }


class AclnnAcoshTestSpec:
    """ACLNN spec registered by CSV ``api_name``.

    Function golden mirrors the aclnn cast chain (wide input set
    int8-64/uint8/bool/float64/complex64/128 per aclnn_acosh.cpp:34-50);
    tolerance covers the full OUTPUT dtype family (TTK keys tolerance by
    OUTPUT dtype).
    """

    golden = aclnn_acosh_golden
    third_party = {"torch": "torch.acosh"}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        # 宽集输入例的输出 dtype 补齐(计算类浮点口径); int/uint/bool 经算子
        # cast 链后输出必落输出 dtype 家族之一
        "float64": {"standard": "cross_check", "level": "L2"},
        "complex64": {"standard": "cross_check", "level": "L2"},
        "complex128": {"standard": "cross_check", "level": "L2"},
    }


class TorchAcoshTestSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``.

    E2E graph mode (torch.compile) uses TTK's generic GraphNetwork wrapping
    torch.acosh (TTK generic GraphNetwork).
    """

    golden = "torch.acosh"

    third_party = {"torch": "torch.acosh"}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        # torch e2e 用例含 float64/complex64 输入, 输出 dtype 补齐
        "float64": {"standard": "cross_check", "level": "L2"},
        "complex64": {"standard": "cross_check", "level": "L2"},
        "complex128": {"standard": "cross_check", "level": "L2"},
    }


class AcoshTensorFlowTestSpec:
    """TensorFlow E2E spec registered by CSV ``api_name``."""

    golden = "tf.math.acosh"
    third_party = {"tf": "tf.math.acosh"}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        # tf e2e 用例含 float64/complex64/complex128 输入, 输出 dtype 补齐
        "float64": {"standard": "cross_check", "level": "L1"},
        "complex64": {"standard": "cross_check", "level": "L1"},
        "complex128": {"standard": "cross_check", "level": "L1"},
    }
