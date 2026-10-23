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

__spec__ = {
    "masked_fill": "MaskedFillTestSpec",
    "aclnnInplaceMaskedFillScalar": "AclnnInplaceMaskedFillScalarTestSpec",
    "aclnnInplaceMaskedFillTensor": "AclnnInplaceMaskedFillTensorTestSpec",
    "torch.masked_fill": "TorchMaskedFillTestSpec",
    "torch.Tensor.masked_fill_": "TorchMaskedFillInplaceTestSpec",
}

import torch

_MF_TOLERANCE = {
    "float16": {"standard": "cross_check"},
    "float32": {"standard": "cross_check"},
    "bfloat16": {"standard": "cross_check"},
    "bool": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
}


def _masked_fill_torch(x, mask, value):
    """Common torch implementation: value is a Python scalar or a 0-dim/1-elem Tensor."""
    value_scalar = value.item() if isinstance(value, torch.Tensor) else value
    return x.masked_fill(mask, value_scalar)


class _MaskedFillKernelThirdParty:
    """kernel third_party 适配器。

    kernel 流程 value 为 1 维单元素或 0 维 tensor（def.cpp 契约），公开接口
    torch.masked_fill 拒收 1 维 value（"only supports a 0-dimensional value
    tensor"），此处先 .item() 转标量再调用（自动走 masked_fill.Scalar 重载）。
    """

    def __call__(self, x, mask, value, **kwargs):
        v = value.item() if isinstance(value, torch.Tensor) else value
        return [x.masked_fill(mask, v)]


class MaskedFillTestSpec:
    """masked_fill 算子测试规范（kernel/geir 流程，numpy in/out）。"""

    def golden(x, mask, value, **kwargs):
        """
        All the parameters follow @masked_fill_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.
        kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
            input_formats, output_formats, input_ori_formats, output_ori_formats,
            input_dtypes, output_dtypes.
        """
        dtype = x.dtype
        if "bfloat16" in str(dtype):
            x = x.astype("float32")
            value = value.astype("float32")

        x_tensor = torch.from_numpy(x)
        mask_tensor = torch.from_numpy(mask)
        value_scalar = value[0]
        res = x_tensor.masked_fill(mask_tensor, value_scalar).numpy()

        if "bfloat16" in str(dtype):
            res = res.astype(dtype, copy=False)
        return res

    third_party = {"torch": _MaskedFillKernelThirdParty}

    tolerance = _MF_TOLERANCE


class AclnnInplaceMaskedFillScalarTestSpec:
    """aclnnInplaceMaskedFillScalar 测试规范（aclnn 流程，CPU torch.Tensor）。"""

    def golden(selfRef, mask, value, **kwargs):
        """
        All the parameters follow @aclnn_masked_fill_scalar.h without outputs.
        All the input Tensors are CPU torch.Tensor; value is a 0-dim torch.Tensor (aclScalar).
        Returns the filled selfRef (inplace output).
        """
        return [_masked_fill_torch(selfRef, mask, value)]

    third_party = {"torch": "torch.masked_fill"}

    tolerance = _MF_TOLERANCE


class AclnnInplaceMaskedFillTensorTestSpec:
    """aclnnInplaceMaskedFillTensor 测试规范（aclnn 流程，CPU torch.Tensor）。"""

    def golden(selfRef, mask, value, **kwargs):
        """
        All the parameters follow @aclnn_masked_fill_tensor.h without outputs.
        All the input Tensors are CPU torch.Tensor; value is a 0-dim torch.Tensor.
        Returns the filled selfRef (inplace output).
        """
        return [_masked_fill_torch(selfRef, mask, value)]

    third_party = {"torch": "torch.masked_fill"}

    tolerance = _MF_TOLERANCE


class TorchMaskedFillTestSpec:
    """torch.masked_fill 测试规范（e2e 流程，CPU torch.Tensor）。"""

    def golden(input, mask, value=None, **kwargs):
        """
        Parameters follow torch.masked_fill(input, mask, value) signature.
        value arrives as a positional tensor or via kwargs (attributes).
        """
        if value is None:
            value = kwargs.get("value")
        return [_masked_fill_torch(input, mask, value)]

    third_party = {"torch": "torch.masked_fill"}

    tolerance = _MF_TOLERANCE


class TorchMaskedFillInplaceTestSpec:
    """torch.Tensor.masked_fill_ 测试规范（e2e 流程，CPU torch.Tensor）。"""

    def golden(self, mask, value=None, **kwargs):
        """
        Parameters follow Tensor.masked_fill_(mask, value) signature.
        """
        if value is None:
            value = kwargs.get("value")
        return [_masked_fill_torch(self, mask, value)]

    third_party = {"torch": "torch.Tensor.masked_fill_"}

    tolerance = _MF_TOLERANCE
