#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MaximumGrad E2E 被测 API（tensorflow 命名空间挂载, .pth 启动注入）。

api_name = "tensorflow.maximum_grad_e2e.maximum_grad"

原生 TF GradientTape 反向(生产代码形态):
  with tf.GradientTape() as tape:
      tape.watch([x1, x2])
      y = tf.maximum(x1, x2) * grads
  gs = tape.gradient(y, [x1, x2])

tf.function 内经 npu_device 转 GE 编译:
  - TF 适配器将 tape 反向图(gradient_tape/GreaterEqual + SelectV2*2 +
    zeros_like 常量折叠 + Sum/Reshape 广播归约)转为 GE 原生算子
    (SelectV2 映射由 custom_math 包 framework/tensorflow TF 插件注册);
  - MaximumMinimumGradFusionPass 命中 -> GreaterEqual+SelectV2*2 融合为
    MaximumGrad, Sum/Reshape 留在融合边界外;
  - te_maximumgrad + te_reducesum kernel 均在 NPU 执行。

int32: TF 无梯度定义(tape 返回 None), 手拼 math_grad 等价分解
  (Select v1 + Const 零 + gradient_tape scope, 同样可被融合)。
"""

import itertools

import tensorflow as tf

_TRACE_SEQ = itertools.count()


def _bcast_reduce(tensor, in_shape):
    """math_grad.py 广播归约: Sum(keepdims=False) + Reshape 到输入自身 shape。"""
    out_shape = tensor.shape.as_list()
    padded = [1] * (len(out_shape) - len(in_shape)) + list(in_shape)
    axes = [i for i in range(len(out_shape)) if padded[i] == 1 and out_shape[i] != 1]
    if not axes:
        return tensor
    summed = tf.raw_ops.Sum(input=tensor, axis=axes, keep_dims=False)
    return tf.reshape(summed, in_shape)


def _maximum_grad_int(grads, x1, x2):
    with tf.name_scope("gradient_tape"):
        mask = tf.raw_ops.GreaterEqual(x=x1, y=x2)
        zeros = tf.zeros(grads.shape, dtype=grads.dtype)
        y1 = tf.raw_ops.SelectV2(condition=mask, t=grads, e=zeros)
        y2 = tf.raw_ops.SelectV2(condition=mask, t=zeros, e=grads)
        return _bcast_reduce(y1, x1.shape.as_list()), _bcast_reduce(
            y2, x2.shape.as_list()
        )


def maximum_grad(grads, x1, x2):
    """y1 = grad wrt x1, y2 = grad wrt x2 (各自输入 shape, 含广播归约)。

    外层 name_scope 按调用唯一化: npu_device 动态图的 GE 会话模型缓存按
    算子节点名复用(graph_id 与 shape/dtype 无关), 复用进程批跑时不同用例
    的同名节点会命中彼此已编译模型导致常量串台; 唯一前缀迫使 graph_id
    错开(对齐 TTK graph_runner._unique_node_name 的修复机理, 覆盖 eager
    路径)。融合规则的 gradient_tape scope 检查按名字分段匹配, 前缀不影响。
    """
    with tf.name_scope(f"maximum_grad_e2e_{next(_TRACE_SEQ)}"):
        if grads.dtype in (tf.int32, tf.int64):
            return _maximum_grad_int(grads, x1, x2)
        with tf.GradientTape() as tape:
            tape.watch([x1, x2])
            y = tf.maximum(x1, x2) * grads
        gs = tape.gradient(y, [x1, x2])
        if any(g is None for g in gs):
            raise RuntimeError("tape.gradient returned None")
        return gs[0], gs[1]


def maximum_grad_np(grads, x1, x2):
    """numpy 小算子拼接 golden(独立 oracle, P1 golden_api 指向此函数)。

    输入可能是框架张量(P1 经 cpu_backend 转换)或 numpy 数组 —— np.asarray
    归一化。框架 --golden-mode Promote 已把浮点输入升精度(f16->f32/f32->f64),
    本函数只做纯计算: mask + where + 广播轴 Sum(f64 累加, 正确舍入)+Reshape。
    整型不升精度, Sum 保持原 dtype 模运算(与算子语义一致)。
    """
    import numpy as np

    g, a, b = np.asarray(grads), np.asarray(x1), np.asarray(x2)
    mask = a >= b
    zero = np.zeros_like(g)
    y1 = np.where(mask, g, zero)
    y2 = np.where(mask, zero, g)

    def _red(t, ref):
        padded = [1] * (len(t.shape) - len(ref.shape)) + list(ref.shape)
        axes = [i for i in range(len(t.shape)) if padded[i] == 1 and t.shape[i] != 1]
        if not axes:
            return t
        return np.sum(t, axis=tuple(axes), dtype=t.dtype).reshape(ref.shape)

    return [_red(y1, a), _red(y2, b)]
