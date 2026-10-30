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
"""Inv 算子 golden（Kernel / GEIR plugin 模式 + Torch / TF E2E）。

数学语义：y = 1 / x（逐元素），与 op_kernel/arch35/inv.h 的实现策略一致：
- int32：整数倒数三值映射（1 -> 1，-1 -> -1，其余含 0/INT_MIN/INT_MAX -> 0），
  纯整型比较实现，不经 float32 中转。
- float32：1 / x，IEEE 特殊值语义（±0 -> ±Inf，±Inf -> ±0，NaN -> NaN）。
- float16 / bfloat16：计算前统一升精度到 float32，计算完成后再舍回原 dtype
  （对应 kernel 的 Cast -> Div(0 ULP, 非 FTZ) -> Cast(RINT)）。

参考实现政策（GOLDEN.05）：参考结果的数学计算（算术、逻辑选择、语义类型转换）全部由
Torch 承担——float 路径调用直接竞品 torch.reciprocal，int32 三值映射用 torch.where 组合；
NumPy 仅承担数组/Tensor 数据桥接（from_numpy / to_numpy / 位视图），不参与参考数学。
bfloat16 的 numpy 形态（ml_dtypes）经 uint16 位视图桥接（仅重解释位型，不改数值与语义）。

- 三方标杆（third_party）为 TensorFlow 与 PyTorch，三方部分不做升精度，按原生 dtype 计算。
- E2E 注册：TF 通路注册 tf.raw_ops.Inv（对齐 framework/inv_tf_plugin.cpp 的
  OriginOpType("Inv")，经插件映射执行交付算子；tf.math.reciprocal 为 Reciprocal
  原生路径，仅作三方标杆）。Torch E2E 不注册：torch.reciprocal 在 NPU 侧分发到
  内置 aclnnReciprocal 而非交付算子（无 per-operator torch 扩展），注册会误导路由。
- 精度判据按 README 约束说明的两方混合容差：FLOAT rtol/atol = 1e-4，FLOAT16 = 1e-3，
  BFLOAT16 = 4e-3，INT32 精确匹配（binary_equal）。cross_check 为三方判据，仅在配置
  GPU 三方比对时适用，不在默认两方比较中声明。
"""

import numpy as np

# 延迟导入 torch：纯 TF NPU 环境（TF + npu_device，无 torch）可加载本模块完成 __spec__ 注册；
# torch 仅在 golden 实际计算时按需导入，避免模块加载时 ImportError 或库初始化冲突。
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch

        torch = _torch
    return torch


__spec__ = {
    "inv": "InvTestSpec",
    "tf.raw_ops.Inv": "InvTensorFlowTestSpec",
}


def _inv_reference_torch(t):
    _ensure_torch()
    """Torch 承担全部参考计算。入参/返回均为 torch.Tensor，dtype 语义保持。

    - int32：三值映射 {+1 -> +1, -1 -> -1, 其余 -> 0}。torch.reciprocal 对 int32 输入
      返回 float32 浮点倒数（0 -> inf、|x| >= 2 -> 小数），与本算子公开的三值映射语义
      不符（配 int32 out 张量时另报 Float->Int 类型转换错误），故按 GOLDEN.05 以
      torch.where 基础算子组合表达三值映射。
    - float：升精度到 float32 -> torch.reciprocal -> 舍回原 dtype。
    """
    if t.dtype == torch.int32:
        return torch.where(
            t == 1,
            torch.ones_like(t),
            torch.where(t == -1, -torch.ones_like(t), torch.zeros_like(t)),
        )
    orig_dtype = t.dtype
    t_f32 = t.to(torch.float32) if orig_dtype != torch.float32 else t
    result = torch.reciprocal(t_f32)
    return result if orig_dtype == torch.float32 else result.to(orig_dtype)


# torch.from_numpy 原生支持的 numpy dtype（含 TTK Promote 上下文可能传入的 float64）
_TORCH_NATIVE_DTYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool_,
)


def _torch_from_numpy(x):
    _ensure_torch()
    """NumPy 数组桥接到 torch（仅数据搬运）。

    原生 dtype 直接 from_numpy；ml_dtypes bfloat16（torch.from_numpy 不支持）经
    uint16 位视图桥接——仅重解释位型，不改数值与语义。未知 dtype 不猜测。
    """
    if x.dtype in _TORCH_NATIVE_DTYPES:
        return torch.from_numpy(x)
    if x.dtype.name == "bfloat16" and x.dtype.itemsize == 2:
        return torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16)
    raise TypeError(f"unsupported golden input dtype: {x.dtype!r}")


def _numpy_from_torch(t, orig_dtype):
    _ensure_torch()
    """torch 张量桥接回 NumPy 数组（仅数据搬运）。bfloat16 经位视图还原为入参 dtype。"""
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(orig_dtype)
    return t.numpy()


class InvTestSpec:
    """Inv 算子测试规范（Kernel / GEIR 流程，注册名对齐安装包 opInterface.value=inv）。"""

    def golden(x, **kwargs):
        _ensure_torch()
        """Kernel golden for Inv. Parameters follow @inv_def.cpp without outputs.

        All the input Tensors are numpy.ndarray.
        kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
                 input_formats, output_ori_formats, input_dtypes, output_dtypes.
        """
        t = _torch_from_numpy(x)
        return [_numpy_from_torch(_inv_reference_torch(t), x.dtype)]

    class ThirdPartyImpl:
        """TensorFlow 三方标杆：tf.math.reciprocal，原生 dtype 计算（不升精度）。"""

        def __call__(self, x, **kwargs):
            import tensorflow as tf

            if x.dtype == tf.int32:
                return [
                    tf.where(
                        tf.equal(x, 1),
                        tf.ones_like(x),
                        tf.where(tf.equal(x, -1), -tf.ones_like(x), tf.zeros_like(x)),
                    )
                ]
            return [tf.math.reciprocal(x)]

    class TorchImpl:
        """PyTorch 三方标杆：torch.reciprocal，原生 dtype 计算（不升精度）。"""

        def __call__(self, x, **kwargs):
            _ensure_torch()
            if x.dtype == torch.int32:
                return [
                    torch.where(
                        x == 1,
                        torch.ones_like(x),
                        torch.where(x == -1, -torch.ones_like(x), torch.zeros_like(x)),
                    )
                ]
            return [torch.reciprocal(x)]

    third_party = {"tf": ThirdPartyImpl, "torch": TorchImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance", "rtol": 1e-3, "atol": 1e-3},
        "bfloat16": {"standard": "mix_tolerance", "rtol": 4e-3, "atol": 4e-3},
        "float32": {"standard": "mix_tolerance", "rtol": 1e-4, "atol": 1e-4},
        "int32": {"standard": "binary_equal"},
    }


def tensorflow_inv_golden(x, name=None, **kwargs):
    _ensure_torch()
    """CPU golden for tf.raw_ops.Inv(x, name=None); torch computation on bridged tensors.

    注册键对齐 framework/inv_tf_plugin.cpp 的 OriginOpType("Inv")：tf.raw_ops.Inv 在
    TF 图中产生 op type "Inv"，经 CANN TF 适配插件映射到交付的自定义 Inv 算子；
    tf.math.reciprocal 产生 op type "Reciprocal"，走框架原生路径，不经过交付算子。
    TF EagerTensor 先桥接到 numpy 再入 torch（EagerTensor 无 .to() 等 torch 接口），
    桥接仅做数据搬运，参考计算仍由 torch 承担；结果经 _numpy_from_torch 还原为
    numpy（torch bfloat16 .numpy() 不支持，需经位视图桥接回）。
    bfloat16 在无 ml_dtypes 的环境下 np.asarray 会抛 ValueError（TF __array__ 转换
    失败），此时经 tf.cast(x, tf.float32) 桥接，参考计算仍在 float32 域完成。
    """
    if isinstance(x, torch.Tensor):
        return [_inv_reference_torch(x)]
    try:
        arr = np.asarray(x)
        orig_dtype = arr.dtype
    except (ValueError, TypeError):
        import tensorflow as tf

        arr = np.asarray(tf.cast(x, tf.float32))
        orig_dtype = None
    t = _torch_from_numpy(arr)
    result = _inv_reference_torch(t)
    if orig_dtype is not None:
        return [_numpy_from_torch(result, orig_dtype)]
    return [result]


class InvTensorFlowTestSpec:
    """TensorFlow E2E spec registered by CSV ``api_name`` (tf.raw_ops.Inv).

    third_party 保留 tf.math.reciprocal 作为独立三方标杆（框架原生 API）；
    NPU 侧执行经插件 OriginOpType("Inv") 映射到交付算子。
    """

    golden = staticmethod(tensorflow_inv_golden)
    third_party = {"tf": "tf.math.reciprocal"}
    tolerance = {
        "float16": {"standard": "mix_tolerance", "rtol": 1e-3, "atol": 1e-3},
        "bfloat16": {"standard": "mix_tolerance", "rtol": 4e-3, "atol": 4e-3},
        "float32": {"standard": "mix_tolerance", "rtol": 1e-4, "atol": 1e-4},
        "int32": {"standard": "binary_equal"},
    }
