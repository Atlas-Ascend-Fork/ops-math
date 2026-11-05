# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""AsinhGrad Golden/TestSpec，兼容 TTK 两方与三方测试。

计算公式：
  y = forward asinh 的输出（op.outputs[0]）
  z = dy / cosh(y)

CPU Golden 使用 TensorFlow 小算子按 z = dy / cosh(y) 计算。
参考值升精度由 TTK Promote 完成；golden 保持传入 dtype。
"""

__spec__ = {
    "AsinhGrad": "AsinhGradTestSpec",
}


def asinh_grad_golden(y, dy, **kwargs):
    """Kernel 两方 Golden，参数顺序与 AsinhGrad OpDef 一致。

    参考值升精度由 TTK Promote 完成；golden 保持传入 dtype。
    使用 z = dy / cosh(y)，与 A2 实现及三方参考保持一致。
    """
    del kwargs
    import tensorflow as tf
    from ttk.utilities.dtypes import normalize_from_tf_dtype, normalize_to_tf_dtype

    y_tensor = tf.convert_to_tensor(normalize_to_tf_dtype(y))
    dy_tensor = tf.convert_to_tensor(normalize_to_tf_dtype(dy))
    result = tf.math.divide(dy_tensor, tf.math.cosh(y_tensor))
    return [normalize_from_tf_dtype(result.numpy())]


class _TorchAsinhGrad:
    """PyTorch 三方实现：直接计算 dy / cosh(y)，不升精度。"""

    def __call__(self, y, dy, **kwargs):
        del kwargs
        import torch

        return [torch.div(dy, torch.cosh(y))]


class AsinhGradTestSpec:
    golden = staticmethod(asinh_grad_golden)
    third_party = {"torch": _TorchAsinhGrad}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
