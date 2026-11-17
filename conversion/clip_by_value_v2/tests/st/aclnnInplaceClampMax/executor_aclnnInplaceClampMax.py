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

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_method_torch_inplace_clamp_max")
class MethodTorchInplaceClampMaxApi(BaseApi):
    """aclnnInplaceClampMax golden: the result overwrites the input tensor in place."""

    def __init__(self, task_result: TaskResult):
        super(MethodTorchInplaceClampMaxApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_tensor = input_data.kwargs["self"]
        clip_value_max = input_data.kwargs["clipValueMax"]
        return input_tensor.clamp_max_(clip_value_max)


@register("aclnn_inplace_clamp_max")
class InplaceClampMaxAclnnApi(AclnnBaseApi):
    """pyaclnn wrapper: aclnnInplaceClampMax returns no out tensor, the result aliases selfRef."""

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages
