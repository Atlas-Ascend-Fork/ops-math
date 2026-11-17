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
"""ATK adapter for aclnnSplitTensor (torch.split semantics, tensor list out)."""

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("function_aclnnSplitTensor")
class FunctionSplitTensorApi(BaseApi):
    """CPU golden: split into chunks of ``splitSections`` along ``dim``."""

    def __init__(self, task_result: TaskResult):
        super(FunctionSplitTensorApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_tensor = input_data.kwargs["self"]
        split_sections = int(input_data.kwargs["splitSections"])
        dim = int(input_data.kwargs["dim"])
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        return torch.split(input_tensor.to(device), split_sections, dim)


@register("aclnn_function")
class AclnnFunctionExecutor(AclnnBaseApi):
    """Wrap the output list so the pyaclnn call receives an aclTensorList."""

    def init_by_input_data(self, input_data: InputDataset):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        return super().init_by_input_data(input_data)
