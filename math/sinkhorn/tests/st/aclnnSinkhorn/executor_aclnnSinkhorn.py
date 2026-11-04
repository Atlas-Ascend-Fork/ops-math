#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_sinkhorn")
class MethodTorchSinkhornApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchSinkhornApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        eps = 0.00000001
        error = 1e9
        cost = input_data.kwargs["cost"]
        tol_arg = input_data.kwargs.get("tol")
        tol = 0.0001 if tol_arg is None else float(tol_arg)
        if cost.numel() == 0:
            return cost.clone()
        output_dtype = cost.dtype
        if cost.dtype in (torch.float16, torch.bfloat16):
            cost = cost.to(torch.float32)
        cost = torch.exp(cost)
        cost0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        cost1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

        cost1_old = cost1
        for _ in range(10000):
            cost0 = (1 / cost0.size(0)) * 1 / (torch.sum(cost1 * cost, 1) + eps)
            cost1 = (
                (1 / cost1.size(0))
                * 1
                / (torch.sum(cost0.unsqueeze(1) * cost, 0) + eps)
            )
            error = torch.mean(torch.abs(cost1_old - cost1))
            cost1_old = cost1
            if not (error.item() > tol):
                break

        return (cost1 * cost * cost0.unsqueeze(1)).to(output_dtype)
