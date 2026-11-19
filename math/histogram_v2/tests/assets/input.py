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
__input__ = {"kernel": {"histogram_v2": "histogram_v2_input"}}


def histogram_v2_input(x, min, max, *, bins=100, y_dtype=3, **kwargs):
    # 仅归一 min>max(真实链路不可能出现, aclnn 层响亮报错); min==max 原样透传——
    # kernel ExpandMinMaxIfEqual(位级 nextafter/整数 ±1 展开)由 TTK 用例直测,
    # golden histogram_v2_golden 已同步实现同口径展开。
    if float(min[0]) > float(max[0]):
        min[0], max[0] = float(max[0]), float(min[0])

    return (x, min, max)
