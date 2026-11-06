#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {
    "xdivy": "XdivyTestSpec",
    "Xdivy": "XdivyTestSpec",
    "tf.math.xdivy": "XdivyTensorFlowSpec",
}


def xdivy_golden(x1, x2, **kwargs):
    """Kernel / GEIR / E2E(tf) 通用 golden — 竞品 tf.math.xdivy。

    输入支持 numpy.ndarray / tf.Tensor（e2e tf 路径 golden 在 CPU 执行，
    入参为 tf.EagerTensor）。直接用 TTK 传入的 dtype 计算（cross_check
    模式下 TTK golden_mode=Promote 已抬档：fp16/bf16→fp32, fp32→fp64），
    不额外 cast，保留高精度真值；cross_check 比对器会自动做类型提升。
    """
    del kwargs
    import tensorflow as tf

    x1_tf = tf.convert_to_tensor(x1)
    x2_tf = tf.convert_to_tensor(x2)
    result = tf.math.xdivy(x1_tf, x2_tf)
    return [result.numpy()]


class XdivyTestSpec:
    """Xdivy kernel test spec — zero-guarded element-wise binary division.

    Formula (spec.yaml math_semantics.formula):
        y = where(x1 == 0, 0, x1 / x2)

    Zero-guard: x1==0 -> 0 (0/0 -> 0, no NaN); x1!=0 & x2==0 -> ±inf.
    Broadcast: numpy broadcast rules, output shape = broadcast max.

    Reference oracle (spec.yaml math_semantics.reference_oracle):
        framework: tensorflow, api: tensorflow.math.xdivy
        available_for_dtype: [float16, float32, bfloat16]

    Pathway: kernel/geir + e2e(tf)。PyTorch 主线无原生 xdivy
    （aten 无 xdivy 条目），故 e2e 仅有 tf.math.xdivy 路径。

    Golden uses tf.math.xdivy (competitor API) with fp32 promotion for
    fp16/bf16 (matches kernel Cast->FP32 strategy, PrecisionReduceFlag=true).
    The result keeps the promoted dtype and is NOT cast back to the
    original dtype; the cross_check comparator handles type promotion.
    No numpy computation.

    ThirdPartyImpl delegates to tf.math.xdivy with native dtype (no
    conversion) for performance benchmarking.
    """

    golden = staticmethod(xdivy_golden)

    class ThirdPartyImpl:
        """Third-party reference implementation (spec.yaml reference_oracle).

        Called by TTK with tf.Tensor inputs. Delegates to tf.math.xdivy
        which natively implements zero-guard semantics.
        No dtype conversion — __call__ times only the competitor API.
        """

        def __init__(self, **kwargs):
            pass

        def __call__(self, x1, x2, **kwargs):
            import tensorflow as tf

            return [tf.math.xdivy(x1, x2)]

    third_party = {
        "tf": ThirdPartyImpl,
    }

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class XdivyTensorFlowSpec:
    """Xdivy E2E 流程 test spec — tf.math.xdivy（TF-adapter on NPU 端到端）。

    Formula (spec.yaml math_semantics.formula):
        y = where(x1 == 0, 0, x1 / x2)

    PyTorch 主线无原生 xdivy，e2e 仅有 tf 路径。golden 复用 xdivy_golden
    （竞品 tf.math.xdivy，零保护语义与 CANN 一致）；third_party 跟随被测框架
    （字符串形式）。
    """

    golden = staticmethod(xdivy_golden)
    third_party = {"tf": "tf.math.xdivy"}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
