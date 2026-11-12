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
"""SpaceToBatch golden reference implementations.

The operator only accepts dtypes that TensorFlow supports, so golden always
uses TensorFlow (``tf.space_to_batch_nd`` with ``block_shape=[bs, bs]``),
including empty tensors. bool/complex32/complex64/complex128 are NOT operator
dtypes, so no dtype fallback is provided here.

A torch small-operator composition (``F.pad`` + ``reshape`` + ``permute``) is
kept only as an independent third_party reference for cross_check, not as a
golden path.

The e2e(tf) pathway is registered as ``tf.raw_ops.SpaceToBatch``, matching
``framework/space_to_batch_tf_plugin.cpp`` (OriginOpType("SpaceToBatch")):
the TF front-end API drives the GE graph, and the golden stays the operator
semantics, pinned to CPU.

Kernel tests must set ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` so importing PyTorch
does not pull in the PyTorch NPU plugin/CANN protobuf, whose symbols conflict
with TensorFlow in the same process.
"""

import numpy as np

__spec__ = {
    "space_to_batch": "SpaceToBatchTestSpec",
    "tf.raw_ops.SpaceToBatch": "SpaceToBatchTfE2eTestSpec",
}


def _normalize_paddings(paddings):
    """Normalize paddings to an int64 [2, 2] array (accepts flat 4 elements)."""
    arr = np.asarray(paddings)
    if arr.size != 4:
        raise ValueError(f"paddings must have 4 elements, got shape {arr.shape}")
    return arr.astype(np.int64).reshape(2, 2)


def _resolve_block_size(block_size):
    """Normalize the block_size attribute to a Python int."""
    if block_size is None:
        raise ValueError("block_size is required")
    return int(np.asarray(block_size).reshape(-1)[0])


def _to_numpy(x):
    """Convert a value to numpy."""
    return np.asarray(x)


def _numpy_bfloat16():
    import ml_dtypes

    return ml_dtypes.bfloat16


def _to_torch(x):
    """Convert a numpy array / torch tensor to a CPU torch tensor."""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    arr = np.ascontiguousarray(np.asarray(x))
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _from_torch(tensor):
    """Convert a torch tensor back to numpy, preserving bfloat16 lanes."""
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_numpy_bfloat16())
    return tensor.numpy()


def _space_to_batch_torch(x, paddings, block_size):
    """SpaceToBatch reference composed from torch small operators.

    torch has no native space_to_batch, so it is built from ``F.pad`` +
    ``reshape`` + ``permute``.  Mirrors ``tf.space_to_batch_nd`` with
    ``block_shape=[bs, bs]``: zero-pad H/W, split each padded spatial dim into
    (out, block), move the block dims in front of the batch dim, then merge
    them back into the batch dimension.  complex32 is stored as float16 with a
    trailing lane dimension, which is simply part of ``C``/tail here, so the
    real ndim is used throughout.  Returns numpy (bfloat16 kept as ml_dtypes).
    """
    import torch

    x_t = _to_torch(x)
    bs = _resolve_block_size(block_size)
    pads = _normalize_paddings(paddings)
    block = [bs, bs]
    M = len(block)
    rank = x_t.dim()
    batch = int(x_t.shape[0])

    out_spatial = [
        (int(x_t.shape[1 + i]) + int(pads[i, 0]) + int(pads[i, 1])) // block[i]
        for i in range(M)
    ]
    tail = [int(v) for v in x_t.shape[1 + M :]]
    out_shape = [batch * int(np.prod(block))] + out_spatial + tail

    if x_t.numel() == 0:
        return _from_torch(torch.zeros(out_shape, dtype=x_t.dtype))

    # torch F.pad lists pads from the last dim backwards; pad spatial dims only.
    pad = [0] * (2 * rank)
    for i in range(M):
        pos = rank - 2 - i
        pad[2 * pos] = int(pads[i, 0])
        pad[2 * pos + 1] = int(pads[i, 1])
    if any(pad):
        x_t = torch.nn.functional.pad(x_t, pad, mode="constant", value=0)

    # [B, outS_1, b_1, ..., outS_M, b_M, tail...]
    interleaved = [batch]
    for i in range(M):
        interleaved.extend([out_spatial[i], block[i]])
    interleaved.extend(tail)
    x_t = x_t.reshape(interleaved)

    # [b_1..b_M, B, outS_1..outS_M, tail...]; output batch is block_index * B + b.
    perm = [2 * i + 2 for i in range(M)]
    perm.append(0)
    perm.extend(2 * i + 1 for i in range(M))
    perm.extend(range(1 + 2 * M, 1 + 2 * M + len(tail)))
    x_t = x_t.permute(perm)

    return _from_torch(x_t.reshape(out_shape).contiguous())


def _space_to_batch_tf(x, paddings, block_size):
    """SpaceToBatch via TensorFlow (in-process); returns numpy."""
    import tensorflow as tf

    x_np = np.ascontiguousarray(_to_numpy(x))
    bs = _resolve_block_size(block_size)
    pads_np = paddings.astype(np.int32)

    if x_np.dtype.name == "bfloat16":
        tensor = tf.bitcast(tf.constant(x_np.view(np.uint16)), tf.bfloat16)
    else:
        tensor = tf.convert_to_tensor(x_np)

    out = tf.space_to_batch_nd(
        input=tensor,
        block_shape=tf.constant([bs, bs], dtype=tf.int32),
        paddings=tf.constant(pads_np, dtype=tf.int32),
    )

    if x_np.dtype.name == "bfloat16":
        import ml_dtypes

        return tf.bitcast(out, tf.uint16).numpy().view(ml_dtypes.bfloat16)
    return out.numpy()


def space_to_batch_golden(x, paddings, *, block_size, **kwargs):
    """
    Kernel golden for space_to_batch.

    Args:
        x: 4D NHWC tensor [N, H_in, W_in, C].
        paddings: int tensor [[pad_top, pad_bottom], [pad_left, pad_right]].
            Shape may be [2, 2] or a flat 4-element tensor ([4]/[1, 4]); the
            flat order is [pad_top, pad_bottom, pad_left, pad_right].
        block_size: int, the spatial block size.

    Returns:
        numpy result with shape [N * bs * bs, H_out, W_out, C].
    """
    pads_2x2 = _normalize_paddings(paddings)
    return _space_to_batch_tf(x, pads_2x2, block_size)


class SpaceToBatchTFImpl:
    def __init__(self, *, block_size=None, **kwargs):
        self.block_size = block_size

    def __call__(self, x, paddings, block_size=None, **kwargs):
        if block_size is None:
            block_size = self.block_size
        pads_2x2 = _normalize_paddings(paddings)
        return [_space_to_batch_tf(x, pads_2x2, block_size)]


class SpaceToBatchTorchImpl:
    """torch 三方实现：PyTorch 无原生算子，用 torch 小算子拼接。"""

    def __init__(self, *, block_size=None, **kwargs):
        self.block_size = block_size

    def __call__(self, x, paddings, block_size=None, **kwargs):
        if block_size is None:
            block_size = self.block_size
        pads_2x2 = _normalize_paddings(paddings)
        return [_space_to_batch_torch(x, pads_2x2, block_size)]


_BINARY_EQUAL_TOLERANCE = {
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "float64": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
}


class SpaceToBatchTestSpec:
    @staticmethod
    def golden(x, paddings, *, block_size, **kwargs):
        return [space_to_batch_golden(x, paddings, block_size=block_size, **kwargs)]

    third_party = {"torch": SpaceToBatchTorchImpl, "tf": SpaceToBatchTFImpl}

    tolerance = _BINARY_EQUAL_TOLERANCE


class SpaceToBatchTfE2eTestSpec:
    """E2E (tf) 通路：api_name ``tf.raw_ops.SpaceToBatch`` —— 驱动接口，非标杆。

    与 ``framework/space_to_batch_tf_plugin.cpp`` 的 OriginOpType("SpaceToBatch")
    对齐：tf 前端 API 经 TF adapter 入 GE 图后映射到本仓库 SpaceToBatch 算子。
    注意不能写成 ``tf.raw_ops.SpaceToBatchND``——那是 SpaceToBatchND 算子的
    起源算子，本算子没有对应的插件映射。

    input:      4D NHWC [N, H_in, W_in, C] tf.Tensor
    paddings:   2D [[pad_top, pad_bottom], [pad_left, pad_right]] int32/int64
                tf.Tensor，取值通常由 CSV attributes 同名字段钉住
    block_size: int 属性，由 CSV attributes 提供

    tf.raw_ops 参数按签名顺序传入，golden 以位置参数接收
    input/paddings/block_size，返回 [output]（tf.Tensor）。

    golden 固定在 CPU 执行：npu_device.open() 后 NPU 是 TF 默认设备，参考实现
    不能静默跑到被测设备上。

    精度标杆对标算子语义（零填充 + 块搬移），与 kernel 通路一致。参考实现用公开
    接口 ``tf.space_to_batch_nd``（``block_shape=[bs, bs]``），等价于 v1 的
    ``SpaceToBatch``，且没有 ``block_size >= 2`` 限制。
    """

    def golden(input, paddings, block_size, **kwargs):
        import tensorflow as tf

        bs = _resolve_block_size(block_size)
        with tf.device("/CPU:0"):
            pads = tf.cast(tf.reshape(paddings, (2, 2)), tf.int32)
            out = tf.space_to_batch_nd(input=input, block_shape=[bs, bs], paddings=pads)
        return [out]

    third_party = "tf.raw_ops.SpaceToBatch"

    tolerance = _BINARY_EQUAL_TOLERANCE
