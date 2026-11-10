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
"""ATK adapter for the aclnnHypot operator (torch.hypot).

Registers two entry points referenced by ``atk_aclnnHypot.json``:

- ``function_npu_hypot`` (BaseApi): CPU golden. ``torch.hypot`` is evaluated in
  float64 (no intermediate fp16/bf16 rounding) and rounded once to the
  type-promoted dtype, which is what a correct NPU kernel produces as well.
- ``aclnn_hypot`` (AclnnBaseApi): dispatches the two tensor inputs to the
  ``aclnnHypot`` binary op (ND format).
"""

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("function_npu_hypot")
class FunctionHypotApi(BaseApi):
    """CPU golden (benchmark) implementation."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if self.device == "cpu":
            x = input_data.kwargs["x"]
            y = input_data.kwargs["y"]
            out_dtype = torch.promote_types(x.dtype, y.dtype)
            output = torch.hypot(
                x.detach().to("cpu").to(torch.float64),
                y.detach().to("cpu").to(torch.float64),
            ).to(out_dtype)
        return output


@register("aclnn_hypot")
class AclnnHypotApi(AclnnBaseApi):
    """aclnnHypot binary-op wrapper (x1, x2 -> y), ND format only."""

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND
