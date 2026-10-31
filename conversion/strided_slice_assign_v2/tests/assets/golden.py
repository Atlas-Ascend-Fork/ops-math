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


"""strided_slice_assign_v2 在 kernel/geir 测试路径下的 golden（多通路 TestSpec 风格）。

Kernel/GEIR 的 golden 收到 numpy.ndarray：将 axes 稀疏切片 spec 稠密化（未列轴 = 整轴切片，
与算子 Host 侧 NormalizeIndices 同口径）后，转 tf tensor 调 tf.raw_ops.StridedSliceAssign 计算，
结果转回 numpy；GEIR 通路复用 Kernel 类。third_party 的 tf vendor 收到 tf.Tensor
（框架按 vendor 自动转换），直接调同一 ref 接口。
StridedSliceAssign 为 TF ref 算子（eager/tf.function 不支持），golden 与三方标杆统一用
独立 tf.Graph + compat.v1.Session + use_resource=False 旧式 ref Variable 承载 in-place 语义；
V2 无 mask 属性（对应 V1 的 begin/end/ellipsis/new_axis/shrink 掩码由 axes 输入替代），
稠密化后 TF 稠密 spec 与 V2 语义一致。
bfloat16 全程位级承载（uint16 view / ml_dtypes），不做 fp32 升降精度，保证 bit-exact。
"""

import numpy as np

__golden__ = {
    "kernel": {"strided_slice_assign_v2": "strided_slice_assign_v2_golden"},
    "aclnn": {"aclnnStridedSliceAssignV2": "aclnn_strided_slice_assign_v2_golden"},
}


def _dense_spec(var_shape, begin, end, strides, axes):
    """axes 稀疏切片 spec → TF 全秩稠密 spec。

    归一化与校验同算子 Host 侧 NormalizeIndices 口径：负轴 += ndim，越界 / 重复 /
    axes 与 begin 长度不等报 ValueError；未列轴取 [0, dim) 步长 1（整轴切片）。
    """
    ndim = len(var_shape)
    n = len(begin)
    if axes is None:
        axes = list(range(n))
    else:
        axes = np.asarray(axes).flatten().tolist()
    if len(axes) != n:
        raise ValueError(
            f"The element nums of axes ({len(axes)}) must be equal to that of begin ({n})."
        )
    spec = {}
    for i, axis in enumerate(axes):
        a = int(axis)
        if a < 0:
            a += ndim
        if a < 0 or a >= ndim:
            raise ValueError(
                f"axes[{i}] value {axis} is out of range [-{ndim}, {ndim})."
            )
        if a in spec:
            raise ValueError(f"axes[{i}] value {axis} is duplicated.")
        spec[a] = i
    dense_b, dense_e, dense_s = [], [], []
    for d in range(ndim):
        if d in spec:
            i = spec[d]
            dense_b.append(int(begin[i]))
            dense_e.append(int(end[i]))
            dense_s.append(int(strides[i]))
        else:
            dense_b.append(0)
            dense_e.append(int(var_shape[d]))
            dense_s.append(1)
    return dense_b, dense_e, dense_s


class StridedSliceAssignV2KernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 tf.Tensor"""

    def golden(self, var, input_value, begin, end, strides, axes=None, **kwargs):
        """
        计算方式: axes 稠密化后直接调用 TF 的 tf.raw_ops.StridedSliceAssign
                 （TF 兼容负索引 / clamp / 负 stride 语义，即算子对齐的权威标杆；
                   算子侧附加校验由 Host 承担，测试用例全部落在 TF 与 Ascend 语义交集内）

        参数:
            var:          numpy.ndarray，目标张量（ref/in-place 语义）
            input_value:  numpy.ndarray，待写入数据（shape 必须等于切片 final shape）
            begin/end/strides: 1 维 int64 数组（稀疏切片 spec，const 输入）
            axes:         1 维 int64 数组或 None（可选，切片轴；None = 前 len(begin) 轴）
            **kwargs:     {input,output}_{dtypes,ori_shapes,formats,ori_formats} 等框架元信息
        返回:
            [切片赋值后的 var]（列表，与输入 var 同 shape 同 dtype）
        """
        import tensorflow as tf

        var = np.asarray(var)
        input_value = np.asarray(input_value)
        in_dtype = var.dtype

        # ---- bfloat16 兼容处理（TF 原生支持 bf16，位级承载，无需升降精度）----
        is_bf16_u16 = in_dtype == np.uint16  # uint16 位级承载 bf16
        if in_dtype.name == "bfloat16" or is_bf16_u16:
            import ml_dtypes

            var_in = (
                np.ascontiguousarray(var).view(ml_dtypes.bfloat16)
                if is_bf16_u16
                else np.ascontiguousarray(var)
            )
            val_in = (
                np.ascontiguousarray(input_value).view(ml_dtypes.bfloat16)
                if input_value.dtype == np.uint16
                else np.ascontiguousarray(input_value)
            )
        else:
            var_in = np.ascontiguousarray(var)
            val_in = np.ascontiguousarray(input_value)

        dense_b, dense_e, dense_s = _dense_spec(
            var.shape,
            np.asarray(begin, dtype=np.int64).flatten().tolist(),
            np.asarray(end, dtype=np.int64).flatten().tolist(),
            np.asarray(strides, dtype=np.int64).flatten().tolist(),
            axes,
        )

        # ---- 独立 Graph + compat.v1.Session 执行 ref 算子（eager/tf.function 不支持）----
        graph = tf.Graph()
        with graph.as_default():
            tf_var = tf.compat.v1.Variable(var_in, use_resource=False)
            tf_value = tf.convert_to_tensor(val_in, dtype=tf_var.dtype.base_dtype)
            out = tf.raw_ops.StridedSliceAssign(
                ref=tf_var,
                begin=np.asarray(dense_b, dtype=np.int64),
                end=np.asarray(dense_e, dtype=np.int64),
                strides=np.asarray(dense_s, dtype=np.int64),
                value=tf_value,
            )
            with tf.compat.v1.Session() as sess:
                sess.run(tf_var.initializer)
                result = sess.run(out)

        # ---- 输出 dtype 还原 ----
        if is_bf16_u16:
            return [
                result.view(np.uint16)
            ]  # bf16 位级读回 uint16，由框架按 bf16 解释比对
        if result.dtype.name == "bfloat16":
            return [result]  # ml_dtypes bfloat16 原样返回
        return [np.asarray(result, dtype=in_dtype)]

    class ThirdPartyImpl:
        """tf vendor 三方标杆（入参 tf.Tensor，框架自动从 numpy 转换）。

        StridedSliceAssign 是 TF ref 算子，无法用字符串形式直接映射
        （第一输入必须是旧式 ref Variable，且 eager 不支持），故用类形式承载。

        职责分离（__call__ 计入性能比对时间）：
        - __init__：axes 稠密化、bf16 位级判定与 tf.bitcast 预处理、
          独立 Graph/ref Variable/算子图构建、Session 创建与变量初始化（全部前置）
        - __call__：仅 sess.run(out) 纯计算
        """

        def __init__(self, var, input_value, begin, end, strides, axes=None, **kwargs):
            import tensorflow as tf

            # bf16 位级承载：uint16 到达时按 bf16 解释（与 golden 的 uint16 view 逻辑一致）
            self.bf16_bit_carry = var.dtype == tf.uint16
            # eager tensor 不能被图内 Variable 直接捕获，先取 numpy 值，图内重建常量
            var_np = var.numpy()
            value_np = input_value.numpy()
            axes_np = (
                None if axes is None else np.asarray(axes.numpy()).flatten().tolist()
            )
            dense_b, dense_e, dense_s = _dense_spec(
                var_np.shape,
                np.asarray(begin.numpy(), dtype=np.int64).flatten().tolist(),
                np.asarray(end.numpy(), dtype=np.int64).flatten().tolist(),
                np.asarray(strides.numpy(), dtype=np.int64).flatten().tolist(),
                axes_np,
            )
            self.graph = tf.Graph()
            with self.graph.as_default():
                var_c = tf.constant(var_np)
                if self.bf16_bit_carry:
                    var_c = tf.bitcast(var_c, tf.bfloat16)
                value_c = tf.constant(value_np)
                if value_c.dtype != var_c.dtype:
                    value_c = (
                        tf.bitcast(value_c, tf.bfloat16)
                        if self.bf16_bit_carry and value_c.dtype == tf.uint16
                        else tf.cast(value_c, var_c.dtype)
                    )
                self.tf_var = tf.compat.v1.Variable(var_c, use_resource=False)
                begin_t = tf.reshape(
                    tf.cast(tf.constant(np.asarray(dense_b, np.int64)), tf.int64), [-1]
                )
                end_t = tf.reshape(
                    tf.cast(tf.constant(np.asarray(dense_e, np.int64)), tf.int64), [-1]
                )
                strides_t = tf.reshape(
                    tf.cast(tf.constant(np.asarray(dense_s, np.int64)), tf.int64), [-1]
                )
                self.out = tf.raw_ops.StridedSliceAssign(
                    ref=self.tf_var,
                    begin=begin_t,
                    end=end_t,
                    strides=strides_t,
                    value=value_c,
                )
                self.sess = tf.compat.v1.Session()
                self.sess.run(self.tf_var.initializer)

        def __call__(self, **kwargs):
            result = self.sess.run(self.out)  # graph 模式 sess.run 返回 numpy
            self.sess.close()
            # bf16 位级读回 uint16（与 golden 输出形态一致；result 为 ml_dtypes bfloat16 numpy）
            return [result.view(np.uint16) if self.bf16_bit_carry else result]

    third_party = {"tf": ThirdPartyImpl}

    tolerance = {
        # 纯数据搬运算子，位级精确，全部 dtype 统一 binary_equal
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "double": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
    }


def strided_slice_assign_v2_golden(
    var, input_value, begin, end, strides, axes=None, **kwargs
):
    """__golden__ kernel 通路兼容入口：转调 TestSpec.golden（返回裸数组，与既有契约一致）。"""
    return StridedSliceAssignV2KernelSpec().golden(
        var, input_value, begin, end, strides, axes, **kwargs
    )[0]


def aclnn_strided_slice_assign_v2_golden(
    var, inputValue, begin=None, end=None, strides=None, axesOptional=None, **kwargs
):
    """__golden__ aclnn 通路 golden（api_name = aclnnStridedSliceAssignV2）。

    契约（函数形式 golden）：按 C 形参序位置传参——
    (var, inputValue) 为 torch.Tensor，(begin, end, strides, axesOptional) 为
    aclIntArray 属性对应的 Python list（axesOptional 缺省为 None）。
    语义：var 原地赋值 var[begin:end:strides(axes)] = inputValue，返回 [新 var 张量]。
    接口约束（stride 全正、末维 stride==1）下用例均为在界切片，numpy 切片语义
    与 kernel 通路 TF ref 标杆一致；bfloat16 按 uint16 位级承载保证 binary_equal
    （框架 Promote 上下文若已升 fp32 则走普通分支，升精度无损，同样位级精确）。
    """
    import torch

    def _flat(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return np.asarray(v).flatten().tolist()

    is_bf16 = isinstance(var, torch.Tensor) and var.dtype == torch.bfloat16
    if is_bf16:
        var_np = np.array(var.detach().cpu().view(torch.uint16).numpy(), copy=True)
        val_np = np.array(
            inputValue.detach().cpu().view(torch.uint16).numpy(), copy=True
        )
    else:
        var_np = np.array(var.detach().cpu().numpy(), copy=True)
        val_np = np.array(inputValue.detach().cpu().numpy(), copy=True)

    ndim = var_np.ndim
    axes = list(range(len(begin))) if axesOptional is None else _flat(axesOptional)
    slices = [slice(None)] * ndim
    for i, axis in enumerate(axes):
        if axis < 0:
            axis += ndim
        slices[axis] = slice(
            int(_flat(begin)[i]), int(_flat(end)[i]), int(_flat(strides)[i])
        )
    var_np[tuple(slices)] = val_np

    if is_bf16:
        return [torch.from_numpy(np.ascontiguousarray(var_np)).view(torch.bfloat16)]
    return [torch.from_numpy(var_np)]
