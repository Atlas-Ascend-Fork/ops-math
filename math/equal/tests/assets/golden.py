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

__golden__ = {
    "aclnn": {
        "aclnnEqTensor": "aclnn_eq_tensor_golden",
        "aclnnEqScalar": "aclnn_eq_scalar_golden",
        "aclnnIsInScalarTensor": "aclnn_is_in_scalar_tensor_golden",
    },
    "e2e": {
        "aclnnIsInScalarTensor": "aclnn_is_in_scalar_tensor_golden",
    },
}


def aclnn_eq_tensor_golden(self, other, out, **kwargs):
    """
    Aclnn golden for aclnnEqTensor.
    All the parameters (name & order) follow \
        function `aclnnEqTensorGetWorkspaceSize` in @aclnn_eq_tensor.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import torch

    return torch.eq(self, other)


def aclnn_eq_scalar_golden(self, other, out, **kwargs):
    """
    Aclnn golden for aclnnEqScalar.
    All the parameters (name & order) follow \
        function `aclnnEqScalarGetWorkspaceSize` in @aclnn_eq_scalar.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import torch

    return torch.eq(self, other)


def aclnn_is_in_scalar_tensor_golden(
    element, testElements, assumeUnique, invert, out, **kwargs
):
    """
    Aclnn golden for aclnnIsInScalarTensor.
    All the parameters (name & order) follow \
        function `aclnnIsInScalarTensorGetWorkspaceSize` in @aclnn_isin.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import numpy as np
    import torch

    if not isinstance(element, torch.Tensor):
        if isinstance(element, np.generic):
            element = torch.from_numpy(np.array(element))
    return torch.isin(element, testElements, assume_unique=assumeUnique, invert=invert)
