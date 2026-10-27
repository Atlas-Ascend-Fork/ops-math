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
"""MinimumGrad 测试规范(numpy golden, kernel/geir 与 e2e 双 spec, 按查找名路由)。

golden 全 numpy 实现(独立 oracle, 不依赖 TF):
  - kernel/geir: mask + where 原生 dtype 直拼(透传逐位精确, IEEE 无 FTZ,
    次正规与 NPU 对齐) -> binary_equal; geir worker 无需 tensorflow
  - e2e: golden 已归 P1(CSV golden_api 指向 API 模块的 *_grad_np numpy
    组合函数), 升精度由框架 --golden-mode Promote 完成, 本插件不含 e2e golden

kernel / geir 通路(op_name "minimum_grad" -> MinimumGradSpec):
  golden: 广播大 shape 输出(算子契约), 精度标准 binary_equal。

e2e 通路(api_name "tensorflow.minimum_grad_e2e.minimum_grad" -> MinimumGradE2eSpec):
  golden: 各输入自身 shape(原生 tape 的 Sum+Reshape 契约);
  浮点 cross_check(三比值), int32 整型精确走 binary_equal。

third_party: GPU 侧 TF tape 反向(浮点)/手拼小算子(int32), 采集 device_us。
"""

import numpy as np


def _tf():
    """仅 third_party 需要 TF(golden 已全 numpy)。"""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    return tf


def _to_np(a):
    """tf.Tensor -> numpy(e2e P2 路径); numpy 原样返回。"""
    if isinstance(a, np.ndarray):
        return a
    return a.numpy()


def _golden_np(grads, x1, x2, grad_x=True, grad_y=True, cmp_ge=True):
    """numpy 小算子拼接 golden(kernel/geir): mask + where, 广播大 shape。

    原生 dtype 直拼(透传逐位精确, 原生比较无 FTZ)——binary_equal 逐位可比。
    e2e 通路的 golden 已归 P1(golden_api 指向 API 模块的 *_grad_np)。
    cmp_ge=True: mask = x1 >= x2(MaximumGrad); False: x1 <= x2(MinimumGrad)。
    """
    g, a, b = _to_np(grads), _to_np(x1), _to_np(x2)
    mask = (a >= b) if cmp_ge else (a <= b)
    zero = np.zeros_like(g)
    y1 = np.where(mask, g, zero)
    y2 = np.where(mask, zero, g)
    return [y1 if grad_x else None, y2 if grad_y else None]


def _golden_broadcast(grads, x1, x2, grad_x=True, grad_y=True, **kwargs):
    """kernel/geir 通路 golden: 广播大 shape(算子输出契约)。"""
    return _golden_np(grads, x1, x2, grad_x, grad_y, cmp_ge=False)


def _bcast_axes(in_shape, out_shape):
    padded = [1] * (len(out_shape) - len(in_shape)) + list(in_shape)
    return [i for i in range(len(out_shape)) if padded[i] == 1 and out_shape[i] != 1]


class MinimumGradTfImpl:
    """GPU 侧三方实现: 浮点走 GradientTape 反向; int32 与 NPU/golden 同口径的
    手拼小算子负载(GPU 原生 int32 kernel, 整型精确)。"""

    def __init__(self, grad_x=True, grad_y=True, **kwargs):
        self.grad_x = grad_x
        self.grad_y = grad_y

    def __call__(self, grads, x1, x2, **kwargs):
        tf = _tf()
        if not all(t.dtype.is_floating for t in (grads, x1, x2)):
            mask = tf.raw_ops.LessEqual(x=x1, y=x2)
            zeros = tf.raw_ops.ZerosLike(x=grads)
            y1 = tf.raw_ops.SelectV2(condition=mask, t=grads, e=zeros)
            y2 = tf.raw_ops.SelectV2(condition=mask, t=zeros, e=grads)

            def _red(t, in_shape):
                axes = _bcast_axes(list(in_shape), list(t.shape.as_list()))
                if not axes:
                    return t
                summed = tf.raw_ops.Sum(input=t, axis=axes, keep_dims=False)
                return tf.reshape(summed, in_shape)

            return [
                _red(y1, list(x1.shape.as_list())) if self.grad_x else None,
                _red(y2, list(x2.shape.as_list())) if self.grad_y else None,
            ]
        with tf.GradientTape() as tape:
            tape.watch([x1, x2])
            y = tf.math.minimum(x1, x2) * grads
        gs = tape.gradient(y, [x1, x2])
        return [gs[0] if self.grad_x else None, gs[1] if self.grad_y else None]


_BINARY = {
    dt: {"standard": "binary_equal"}
    for dt in ("float16", "float32", "bfloat16", "int32")
}


class MinimumGradSpec:
    """kernel/geir 通路: numpy golden(广播口径, IEEE 对齐 NPU) + 逐位判据。"""

    golden = staticmethod(_golden_broadcast)
    third_party = {"tf": MinimumGradTfImpl}
    tolerance = _BINARY


class MinimumGradE2eSpec:
    """e2e 通路: golden 走 P1(CSV golden_api 指向 API 模块内的 *_grad_np
    numpy 组合函数), --golden-mode Promote 下框架升精度输入后调用;
    本 spec 只提供 tolerance + third_party, 不提供 golden。
    浮点 cross_check(三比值), int32 整型精确走 binary_equal。"""

    third_party = {"tf": MinimumGradTfImpl}
    tolerance = {
        "float16": {"standard": "cross_check"},
        "float32": {"standard": "cross_check"},
        "bfloat16": {"standard": "cross_check"},
        "int32": {"standard": "binary_equal"},
    }


__spec__ = {
    "minimum_grad": "MinimumGradSpec",
    "MinimumGrad": "MinimumGradSpec",
    "tensorflow.minimum_grad_e2e.minimum_grad": "MinimumGradE2eSpec",
}
