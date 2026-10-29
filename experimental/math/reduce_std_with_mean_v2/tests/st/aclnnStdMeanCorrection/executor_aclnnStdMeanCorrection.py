#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("ascend_method_torch_tensor_std_mean")
class MethodTorchTensorStdMeanCorrectionApi(BaseApi):
    """CPU golden reference for aclnnStdMeanCorrectionV2.

    Computes std and mean along specified dimensions using PyTorch.
    Matches the aclnnStdMeanCorrectionV2 API semantics:
      - correction: Bessel's correction (0 = biased, >=1 = unbiased with delta)
      - keepdim: whether to retain reduced dimensions
    """

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input = input_data.kwargs["input"]

        if input.numel() == 0:
            # Empty tensor: return NaN
            return torch.tensor(float("nan")), torch.tensor(float("nan"))

        dim = input_data.kwargs.get("dim")
        correction = input_data.kwargs.get("correction", 1)
        keepdim = input_data.kwargs.get("keepdim", False)

        # dim can be a list, tuple, or None
        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
            dim = None  # reduce all dimensions

        result = torch.std_mean(input, dim=dim, correction=correction, keepdim=keepdim)

        return result[0], result[1]
