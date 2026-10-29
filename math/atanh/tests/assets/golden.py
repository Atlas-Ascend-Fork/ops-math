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
"""Atanh 的原生 TTK TestSpec，供 --plugin 指向本文件。

Kernel/GEIR 使用 atanh（numpy 输入 x），ACLNN 使用 aclnnAtanh（input、out）
或 aclnnInplaceAtanh（inputRef），均返回单元素 list。
主计算保持真实 torch.atanh：FP16/BF16 -> FP32，其他输入精度保持。
最终整份结果按实际输出 dtype 完整 cast；无数值掩码、裁剪或 NaN/Inf 清洗。
Kernel/GEIR Enable 使用原生 output_dtypes；直接调用无元信息时按 x.dtype
推导（Atanh 的 y.dtype == x.dtype），在 BF16 容器转换之前保存目标 dtype。
Kernel/GEIR 输出类型仅支持 FP16、BF16、FP32。
Promote 会改写原生 output_dtypes，不能用其冒充实际输出；完整回归沿用 Enable。
ACLNN 使用 out.dtype 或 inputRef.dtype，cast 后再写回并返回。
BF16 在 Torch 中完成 cast，NumPy 仅按位承载同一 BF16 结果。
provider third_party 保持原样；此修正不依赖该 provider 启用。
"""

import numpy as np
import torch

__spec__ = {
    "atanh": "AtanhTestSpec",
    "aclnnAtanh": "AclnnAtanhTestSpec",
    "aclnnInplaceAtanh": "AclnnInplaceAtanhTestSpec",
}


def _atanh_reference(input):
    """只将 fp16/bf16 提升至 fp32，其余输入 dtype 保持。"""
    if input.dtype in (torch.float16, torch.bfloat16):
        input = input.to(dtype=torch.float32)
    return torch.atanh(input)


_OUTPUT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _output_dtype(dtype):
    name = getattr(dtype, "name", str(dtype)).removeprefix("torch.")
    return _OUTPUT_DTYPES[name]


def _numpy_result(result):
    result = result.detach().cpu()
    if result.dtype == torch.bfloat16:
        import ml_dtypes

        return result.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return result.numpy()


class AtanhTestSpec:
    """Kernel/GEIR 参数名以 atanh_def.cpp 的 Input(\"x\") 为准。"""

    def golden(x, **kwargs):
        if kwargs.get("golden_mode") == "Promote":
            raise ValueError(
                "Promote metadata is not the actual output dtype; use Enable for final-output reference"
            )
        output_dtypes = kwargs.get("output_dtypes")
        # Native Enable passes the declared output dtype before any BF16 carrier conversion.
        target_dtype = _output_dtype(
            output_dtypes[0] if output_dtypes is not None else x.dtype
        )
        # from_numpy 不支持扩展 bf16；仅此容器先转 fp32，后续映射不会再次提升。
        is_bfloat16 = x.dtype.name == "bfloat16"
        compute_x = x.astype(np.float32) if is_bfloat16 else x
        # 显式复制兼容负 stride/只读数组，且保留标量和空 Tensor 的 shape。
        tensor = torch.from_numpy(np.array(compute_x, copy=True, order="C"))
        result = _atanh_reference(tensor).to(dtype=target_dtype)
        return [_numpy_result(result)]

    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(self, x, **kwargs):
            # 直接执行 provider Tensor API，保留原始 dtype/device。
            return [torch.atanh(x)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L0"},
        "bfloat16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }


class AclnnAtanhTestSpec:
    """按真实 out dtype 完整 cast 后写回；返回同 dtype 的结果。"""

    def golden(input, out, **kwargs):
        target_dtype = out.dtype if out is not None else input.dtype
        result = _atanh_reference(input).to(dtype=target_dtype)
        if out is not None:
            out.copy_(result)
        return [result]


class AclnnInplaceAtanhTestSpec:
    """按原始 inputRef dtype 完整 cast，再执行原地写回。"""

    def golden(inputRef, **kwargs):
        result = _atanh_reference(inputRef).to(dtype=inputRef.dtype)
        inputRef.copy_(result)
        return [result]
