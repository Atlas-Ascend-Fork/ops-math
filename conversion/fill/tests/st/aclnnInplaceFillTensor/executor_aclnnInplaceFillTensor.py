#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


def prepare_unsigned_value(input_data, case_config):
    # Construct fixed unsigned boundary values as integers, avoiding float rounding
    # at UINT64_MAX in generic random data generation.
    value_config = next(item for item in case_config.inputs if item.name == "value")
    bounds = value_config.range_values
    if (
        value_config.dtype in ("uint16", "uint32", "uint64")
        and isinstance(bounds, list)
        and len(bounds) == 2
        and bounds[0] == bounds[1]
    ):
        value = int(bounds[0])
        tensor = input_data.kwargs["value"]
        input_data.kwargs["value"] = torch.from_numpy(
            np.full(tuple(tensor.shape), value, dtype=value_config.dtype)
        )
        assert input_data.kwargs["value"].item() == value


@register("ascend_method_torch_fill")
class MethodTorchFillApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchFillApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        prepare_unsigned_value(input_data, self.task_result.case_config)
        input_tensor = input_data.kwargs["self"]
        value = input_data.kwargs["value"]

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        input_tensor = input_tensor.to(device)

        return input_tensor.fill_(value.item())


@register("aclnn_inplace_fill_tensor")
class FillTensorAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        prepare_unsigned_value(input_data, self.task_result.case_config)
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output
