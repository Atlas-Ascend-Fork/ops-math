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


def _amp_update_scale(
    current_scale,
    growth_tracker,
    found_inf,
    growth_factor,
    backoff_factor,
    growth_interval,
):
    """Adapt the official PyTorch operator to AmpUpdateScale's two outputs."""
    original_dtype = current_scale.dtype
    scale_fp32 = current_scale.to(torch.float32).clone()
    tracker = growth_tracker.clone()
    updated_scale = torch._amp_update_scale_(
        scale_fp32,
        tracker,
        found_inf.to(torch.float32),
        growth_factor,
        backoff_factor,
        growth_interval,
    )
    return updated_scale.to(original_dtype), tracker


@register("function_aclnnAmpUpdateScale")
class AmpUpdateScaleBenchmark(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return _amp_update_scale(
            input_data.kwargs["current_scale"],
            input_data.kwargs["growth_tracker"],
            input_data.kwargs["found_inf"],
            input_data.kwargs["growth_factor"],
            input_data.kwargs["backoff_factor"],
            input_data.kwargs["growth_interval"],
        )
