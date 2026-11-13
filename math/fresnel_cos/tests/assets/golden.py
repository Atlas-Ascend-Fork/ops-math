#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""FresnelCos 独立 Golden：Kernel、GEIR 与 TensorFlow 通路。

Fresnel 余弦积分逐元素计算，语义对标 TensorFlow
tf.math.special.fresnel_cos：

    C(x) = integral_0^x cos(π/2 * t^2) dt

输入保真：bf16 经 uint16 位桥无损转 bfloat16（不经 fp16 近似）；
计算精度：fp16/bf16 在 golden 侧显式 cast fp32 计算、结果舍回原精度
（TF 该算子仅注册 fp32/fp64 内核；与 NPU kernel fp32 计算路径语义一致）。
"""

__spec__ = {
    "fresnel_cos": "FresnelCosKernelSpec",
    "tf.math.special.fresnel_cos": "FresnelCosTensorFlowSpec",
}

_L1 = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_tf_tensor(arr, dtype):
    """numpy array -> tf tensor，保留输入的原始精度。

    cross_check Promote 模式下 TTK 可能把 fp32 输入升为 fp64 传入，
    此处应保留 fp64 创建 tf tensor，不降回 fp32。
    bfloat16 numpy (ml_dtypes) 无法直接被 tf.constant 接受，需先转 float32 再指定 dtype。
    """
    import numpy as np
    import tensorflow as tf

    arr = np.ascontiguousarray(arr)
    if arr.dtype.name == "bfloat16":
        arr = arr.astype(np.float32)
        return tf.constant(arr, dtype=tf.bfloat16)
    else:
        return tf.constant(arr)


def _from_tf_tensor(t, dtype):
    """tf tensor -> numpy array，保留 TF 计算结果的原始精度。

    不强制转换精度——cross_check Promote 模式下 TTK 可能把 fp32 输入升为
    fp64 传给 golden，TF 内部用 fp64 计算，此处应保留 fp64 返回，不降回 fp32。
    bfloat16 是例外：TF bfloat16 无法直接 .numpy()，用 tf.cast 转 float32。
    """
    import tensorflow as tf

    if dtype == "bfloat16":
        return tf.cast(t, tf.float32).numpy()
    else:
        return t.numpy()


def _compute(x, dtype):
    import tensorflow as tf

    x_t = _to_tf_tensor(x, dtype)
    # TF 的 FresnelCos 仅注册 FLOAT/DOUBLE 内核，fp16/bf16 经 cast fp32 计算
    # 再舍回原精度——与 tensorflow_fresnel_cos_golden/_TensorFlowFresnelCos
    # 的桥接及 NPU kernel 的 fp32 内部计算路径语义一致。
    if x_t.dtype not in (tf.float32, tf.float64):
        result_t = tf.math.special.fresnel_cos(tf.cast(x_t, tf.float32))
        result_t = tf.cast(result_t, x_t.dtype)
    else:
        result_t = tf.math.special.fresnel_cos(x_t)
    return [_from_tf_tensor(result_t, dtype)]


def fresnel_cos_golden(x, **kwargs):
    del kwargs
    dtype = str(x.dtype) if not isinstance(x.dtype, str) else x.dtype
    if "float16" in dtype or "fp16" in dtype:
        dt = "float16"
    elif "bfloat16" in dtype or "bf16" in dtype:
        dt = "bfloat16"
    else:
        dt = dtype
    return _compute(x, dt)


def tensorflow_fresnel_cos_golden(x, name=None, **kwargs):
    del kwargs, name
    import tensorflow as tf
    from tensorflow.python.eager import context

    # npu_device 替换了 tf.device 的实现；直接使用 eager CPU scope。
    with context.device("/CPU:0"):
        x = tf.identity(tf.convert_to_tensor(x))
        # TF 的 FresnelCos 仅注册 FLOAT/DOUBLE 内核，fp16/bf16 经 cast fp32 计算
        # 再舍回原精度——与 NPU kernel 的 fp32 内部计算路径语义一致。
        if x.dtype not in (tf.float32, tf.float64):
            r = tf.math.special.fresnel_cos(tf.cast(x, tf.float32))
            return [tf.cast(r, x.dtype)]
        return [tf.math.special.fresnel_cos(x)]


def _install_fresnel_cos_cast_wrapper():
    """给 tf.math.special.fresnel_cos 安装 cast 包装（幂等）。

    TF 的 FresnelCos 算子仅注册 FLOAT/DOUBLE 内核（CPU/GPU 同），fp16/bf16
    直接调用报 NotFoundError。e2e 被测执行按 api_name 原样解析调用（无 dtype
    桥接钩子），故在输入生成阶段（先于 resolve_api 与 eager/graph 执行）
    安装包装：fp16/bf16 -> cast fp32 -> 原函数计算 -> cast 回原精度；
    fp32/fp64 直通原函数。与 Kernel 通路 _TensorFlowFresnelCos 的 cast 处理
    及 NPU kernel 的 fp32 内部计算路径语义一致。
    """
    import tensorflow as tf

    original = tf.math.special.fresnel_cos
    if getattr(original, "_ttk_fresnel_cos_cast_wrapped", False):
        return

    def fresnel_cos_cast(x, name=None):
        if x.dtype not in (tf.float32, tf.float64):
            r = original(tf.cast(x, tf.float32), name=name)
            return tf.cast(r, x.dtype)
        return original(x, name=name)

    fresnel_cos_cast._ttk_fresnel_cos_cast_wrapped = True
    tf.math.special.fresnel_cos = fresnel_cos_cast


def fresnel_cos_customize_inputs(x, name=None, **kwargs):
    """e2e input 钩子：安装 cast 包装后输入原样返回。

    e2e 的 input 钩子按原地约定工作（返回值被忽略，仅数据/精度保持 CSV
    声明），此处的唯一作用是在 resolve_api 解析 tf.math.special.fresnel_cos
    之前完成包装安装。kernel 语义下返回与输入结构一致的 tuple。
    """
    del name, kwargs
    _install_fresnel_cos_cast_wrapper()
    return (x,)


class _TensorFlowFresnelCos:
    """竞品 API 隔离入口（cross_check 子进程调用，与被测 kernel 隔离）。

    直接调用 tf.math.special.fresnel_cos，不经 _compute 中转。
    TF 的 FresnelCos 算子仅注册 FLOAT/DOUBLE 内核（CPU/GPU 同），fp16/bf16
    直接调用报 NotFoundError。fp16/bf16 输入经 cast 到 fp32 计算、再舍回
    原精度——与 NPU kernel 自身的 fp32 内部计算路径语义一致。
    返回 tf tensor，由服务端 _to_numpy_pair 统一转 numpy（bf16 走
    as_numpy_dtype 位保真通道，避免本地 .numpy() 对 bf16 的版本差异）。
    """

    def __call__(self, x, **kwargs):
        del kwargs
        import tensorflow as tf

        x_t = tf.constant(x)
        if x_t.dtype not in (tf.float32, tf.float64):
            r = tf.math.special.fresnel_cos(tf.cast(x_t, tf.float32))
            out = tf.cast(r, x_t.dtype)
        else:
            out = tf.math.special.fresnel_cos(x_t)
        if out.dtype == tf.bfloat16 and out.shape == ():
            # 0-d bf16: 服务端 _to_numpy_pair 用 np.asarray 走 TF __array__ 协议，
            # 在 0-d 上抛 "object __array__ method not producing an array"（1-d
            # 正常）。先 .numpy()（服务端可产 bf16 numpy）；失败则退化 fp32——
            # bf16 值在 fp32 中精确表示，比对器统一 promote，数值等价。
            try:
                return [out.numpy()]
            except Exception:
                return [tf.cast(out, tf.float32).numpy()]
        return [out]


class FresnelCosKernelSpec:
    golden = staticmethod(fresnel_cos_golden)
    third_party = {"tf": _TensorFlowFresnelCos}
    tolerance = _L1


class FresnelCosTensorFlowSpec:
    golden = staticmethod(tensorflow_fresnel_cos_golden)
    customize_inputs = staticmethod(fresnel_cos_customize_inputs)
    third_party = {"tf": _TensorFlowFresnelCos}
    tolerance = _L1
