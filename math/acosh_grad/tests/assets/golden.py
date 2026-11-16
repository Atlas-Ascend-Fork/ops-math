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
"""TensorFlow public Acosh autodiff reference, preserving supplied saved y."""

from typing import ClassVar
import numpy
import tensorflow as tf

__spec__ = {"acosh_grad": "AcoshGradKernelSpec"}
_L2 = {
    name: {"standard": "cross_check", "level": "L2"}
    for name in ("float16", "bfloat16", "float32")
}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            import torch
            from ml_dtypes import bfloat16

            return value.view(torch.int16).numpy().view(bfloat16)
        return value.numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return numpy.asarray(value)


def _compute(y, dy, device):
    y, dy = _numpy(y), _numpy(dy)
    if y.size != dy.size:
        raise ValueError("y and dy must have equal element counts")
    dy = dy.reshape(y.shape)
    graph = tf.Graph()
    with graph.as_default(), tf.device(device):
        original_x = tf.compat.v1.placeholder(tf.as_dtype(y.dtype), shape=y.shape)
        saved_y = tf.math.acosh(original_x)
        upstream = tf.compat.v1.placeholder(tf.as_dtype(dy.dtype), shape=dy.shape)
        result = tf.compat.v1.gradients(saved_y, original_x, grad_ys=upstream)[0]
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=False,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True
    if device == "/CPU:0":
        config.device_count["GPU"] = 0
    with tf.compat.v1.Session(graph=graph, config=config) as session:
        # Feed the actual saved forward output: no reconstruction or cast.
        return [session.run(result, feed_dict={saved_y: y, upstream: dy})]


def acosh_grad_golden(y, dy, **kwargs):
    return _compute(y, dy, "/CPU:0")


class _TensorFlowAcoshGrad:
    def __call__(self, y, dy, **kwargs):
        return _compute(y, dy, "/GPU:0")


class AcoshGradKernelSpec:
    golden = staticmethod(acosh_grad_golden)
    third_party: ClassVar = {"tf": _TensorFlowAcoshGrad}
    tolerance: ClassVar = _L2
