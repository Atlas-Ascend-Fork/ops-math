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
"""ATK adapter for aclnnRoll (torch.roll semantics)."""

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_roll")
class FunctionRollApi(BaseApi):
    """CPU golden for aclnnRoll."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            return None
        kwargs = input_data.kwargs
        x = kwargs["x"].detach().cpu()
        shifts = [int(s) for s in kwargs["shifts"]]
        dims = [int(d) for d in kwargs["dims"]]
        if len(dims) == 0:
            # Empty dims: roll the flattened tensor and restore the shape.
            flat = torch.roll(x.reshape(-1), shifts[0], 0)
            return flat.reshape(x.shape)
        return torch.roll(x, shifts, dims)
