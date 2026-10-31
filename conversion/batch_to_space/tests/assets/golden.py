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

import numpy as np

__spec__ = {
    "batch_to_space": "BatchToSpaceTestSpec",
    "tf.raw_ops.BatchToSpace": "BatchToSpaceTfE2eTestSpec",
}


class BatchToSpaceTestSpec:
    """BatchToSpace operator golden spec (kernel flow: numpy in, numpy out).

    x:          4D NHWC [N * bs * bs, H_in, W_in, C]
    crops:      1D [crop_top, crop_bottom, crop_left, crop_right]
    block_size: int

    Returns [output] where output is [N, H_out, W_out, C] numpy array.
    """

    def golden(x, crops, *, block_size, **kwargs):
        import tensorflow as tf

        crops = np.asarray(crops, dtype=np.int32).reshape(2, 2)
        # tf.raw_ops.BatchToSpace 要求 block_size >= 2，而算子公开契约声明 block_size >= 1
        # （spec.yaml/README/proto 一致）；bs=1 的合法域改走等价的 BatchToSpaceND
        # （bs=1 时即纯裁剪），仍为 TF 三方参考。
        if block_size >= 2:
            out = tf.raw_ops.BatchToSpace(input=x, crops=crops, block_size=block_size)
        else:
            out = tf.raw_ops.BatchToSpaceND(
                input=x, block_shape=[block_size, block_size], crops=crops
            )
        return [out.numpy()]

    third_party = "tf.raw_ops.BatchToSpace"

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "float64": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
    }


class BatchToSpaceTfE2eTestSpec:
    """BatchToSpace operator golden spec (e2e flow: tf.Tensor in, tf.Tensor out).

    CSV api_name: tf.raw_ops.BatchToSpace
    input:      4D NHWC [N * bs * bs, H_in, W_in, C] tf.Tensor
    crops:      2D [[crop_top, crop_bottom], [crop_left, crop_right]] int32 tf.Tensor
                (value usually pinned via CSV attributes, e.g. 'crops': [[0, 0], [0, 0]])
    block_size: int, required, provided via CSV attributes

    tf.raw_ops params are passed by keyword, so golden receives
    input/crops/block_size as keyword arguments.

    Returns [output] where output is [N, H_out, W_out, C] tf.Tensor.
    Golden computation is pinned to CPU via tf.device: after npu_device.open()
    the NPU is TF's default device, and the reference must not silently run on it.
    """

    def golden(input, crops, block_size, **kwargs):
        import tensorflow as tf

        with tf.device("/CPU:0"):
            crops = tf.cast(tf.reshape(crops, (2, 2)), tf.int32)
            # 同 BatchToSpaceTestSpec：bs=1 的合法域走等价 BatchToSpaceND（TF v1 接口要求 bs>=2）
            if block_size >= 2:
                out = tf.raw_ops.BatchToSpace(
                    input=input, crops=crops, block_size=block_size
                )
            else:
                out = tf.raw_ops.BatchToSpaceND(
                    input=input, block_shape=[block_size, block_size], crops=crops
                )
        return [out]

    third_party = "tf.raw_ops.BatchToSpace"

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "float64": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
    }
