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

import sys
import numpy as np
import torch


def gen_golden_data_simple(size, dtype):
    if dtype == "complex64":
        real = np.random.uniform(-100, 100, [int(size)]).astype(np.float32)
        imag = np.random.uniform(-100, 100, [int(size)]).astype(np.float32)
        input_x = real + imag * 1j
        inputX = torch.as_tensor(input_x).to(torch.complex64)
        dtype = np.float32
    elif dtype == "bfloat16":
        input_x = np.random.uniform(-100, 100, [int(size)]).astype(np.float32)
        inputX = torch.as_tensor(input_x).to(torch.bfloat16)
        golden = torch.angle(inputX.float()).to(torch.bfloat16)
        inputX.view(torch.int16).numpy().tofile("./input_x.bin")
        golden.view(torch.int16).numpy().tofile("./golden.bin")
        return
    else:
        input_x = np.random.uniform(-100, 100, [int(size)]).astype(dtype)
        inputX = torch.as_tensor(input_x)

    outputY = torch.angle(inputX)

    golden = outputY.numpy().astype(dtype)

    input_x.tofile("./input_x.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2])
