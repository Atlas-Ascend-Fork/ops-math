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
"""SpaceToBatchND golden reference implementations.

TensorFlow is the in-process reference for the dtypes it supports. Operator
dtypes that TensorFlow does not support (bool/complex32/complex64)
and empty tensors use a reference built from torch small operators
(``F.pad`` + ``reshape`` + ``permute``) instead. This is a dtype-based routing
decision, not a runtime fallback: supported non-empty inputs always go through
TensorFlow.

third_party 优先 torch（FX：远端 XPU 默认 provider 为 torch）。

The e2e(tf) pathway is registered as ``tf.raw_ops.SpaceToBatchND``, matching
``framework/space_to_batch_nd_tf_plugin.cpp`` (OriginOpType("SpaceToBatchND")).
It reuses the same dtype routing as the kernel golden: TF for the dtypes it
supports, torch small-operator composition for bool/complex/empty inputs, with
the reference pinned to CPU.

Kernel tests must set ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` so importing PyTorch
does not pull in the PyTorch NPU plugin/CANN protobuf, whose symbols conflict
with TensorFlow in the same process.
"""

import numpy as np

__input__ = {"kernel": {"space_to_batch_nd": "space_to_batch_nd_input"}}

__spec__ = {
    "space_to_batch_nd": "SpaceToBatchNDTestSpec",
    "tf.raw_ops.SpaceToBatchND": "SpaceToBatchNDTfE2eTestSpec",
}


# Operator dtypes that TensorFlow has no (or only experimental) support for.
# complex128 is not an operator dtype, so it is intentionally excluded.
_TF_UNSUPPORTED_DTYPES = {
    "bool",
    "complex32",
    "complex64",
}


def space_to_batch_nd_input(x, block_shape, paddings, **kwargs):
    """Fill a default input only when TTK did not already provide one.

    TTK generates ``x`` from the CSV ranges/attributes before invoking this
    hook.  Overwriting it here would discard the declared value ranges
    (negatives, fractions) and overflow float16, so a provided input is
    returned unchanged.
    """
    if x is not None:
        return x, block_shape, paddings

    shapes = kwargs.get("input_ori_shapes") or kwargs.get("input_shapes") or ()
    dtypes = kwargs.get("input_dtypes") or ()
    if not shapes or not dtypes:
        raise ValueError("space_to_batch_nd_input requires x or input shapes/dtypes")

    # 异常用例故意省略必选输入时保持 x 缺失，不伪造数据；只有真正畸形的 shape 才报错。
    if shapes[0] is None or dtypes[0] is None:
        return None, block_shape, paddings

    try:
        shape = tuple(int(v) for v in shapes[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "space_to_batch_nd_input expects input_ori_shapes[0] to be a "
            f"sequence, got {shapes[0]!r}"
        ) from exc

    dtype_name = str(dtypes[0]).lower()
    count = int(np.prod(shape))
    if dtype_name == "bfloat16":
        import ml_dtypes

        x = np.arange(count, dtype=np.float32).reshape(shape).astype(ml_dtypes.bfloat16)
    elif dtype_name == "complex32":
        x = (
            np.arange(count * 2, dtype=np.float64)
            .reshape(shape + (2,))
            .astype(np.float16)
        )
    else:
        try:
            np_dtype = np.dtype(dtype_name)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"space_to_batch_nd_input does not support dtype {dtypes[0]!r}"
            ) from exc
        x = np.arange(count, dtype=np.float64).reshape(shape).astype(np_dtype)
    return x, block_shape, paddings


def _is_empty(x):
    """Return True when x holds no elements (any dimension is 0)."""
    return int(np.prod(getattr(x, "shape", ()))) == 0


def _dtype_name(x, kwargs):
    """Resolve the logical dtype name, preferring the declared input dtype."""
    dtypes = kwargs.get("input_dtypes")
    if isinstance(dtypes, str):
        candidate = dtypes
    elif dtypes is not None and len(dtypes) > 0:
        candidate = dtypes[0]
    else:
        candidate = getattr(x, "dtype", None)
    if not isinstance(candidate, str):
        candidate = getattr(candidate, "name", str(candidate))
    name = str(candidate).lower().rsplit(".", 1)[-1]
    return name


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


def _space_to_batch_nd_torch(x, block_shape, paddings):
    """SpaceToBatchND reference composed from torch small operators.

    torch has no native space_to_batch_nd, so it is built from ``F.pad`` +
    ``reshape`` + ``permute``.  Mirrors ``tf.space_to_batch_nd``: pad
    spatial dims 1..M with zeros, split each padded spatial dim into
    (out, block), move the block dims in front of the batch dim, then merge
    them back into the batch dimension.  complex32 is stored as float16 with a
    trailing lane dimension, which is simply part of ``T...`` here, so the real
    ndim is used throughout.  Returns numpy (bfloat16 kept as ml_dtypes).
    """
    import torch

    x_t = _to_torch(x)
    block = [int(v) for v in _to_numpy(block_shape).reshape(-1)]
    pads = _to_numpy(paddings).astype(np.int64).reshape(len(block), 2)
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


def _space_to_batch_nd_tf(x, block_shape, paddings):
    """SpaceToBatchND via TensorFlow (in-process); returns numpy."""
    import tensorflow as tf

    x_np = np.ascontiguousarray(_to_numpy(x))
    block_np = _to_numpy(block_shape).astype(np.int32).reshape(-1)
    pads_np = _to_numpy(paddings).astype(np.int32).reshape(-1, 2)

    if x_np.dtype.name == "bfloat16":
        import ml_dtypes

        tensor = tf.bitcast(tf.constant(x_np.view(np.uint16)), tf.bfloat16)
    else:
        tensor = tf.convert_to_tensor(x_np)

    out = tf.space_to_batch_nd(
        input=tensor,
        block_shape=tf.constant(block_np, dtype=tf.int32),
        paddings=tf.constant(pads_np, dtype=tf.int32),
    )

    if x_np.dtype.name == "bfloat16":
        import ml_dtypes

        return tf.bitcast(out, tf.uint16).numpy().view(ml_dtypes.bfloat16)
    return out.numpy()


def _space_to_batch_nd_tf_e2e(x, block_shape, paddings, dtype_name):
    """SpaceToBatchND golden for the e2e(tf) pathway; returns a tf.Tensor.

    Inputs arrive as tf.Tensor (the TF front-end is the tested API).  The same
    dtype routing as the kernel golden applies: TF handles dtypes it supports,
    while bool/complex and empty tensors fall back to the torch composition and
    are converted back to tf.Tensor.
    """
    import tensorflow as tf

    if _is_empty(x) or dtype_name in _TF_UNSUPPORTED_DTYPES:
        out_np = np.asarray(_space_to_batch_nd_torch(x, block_shape, paddings))
        if out_np.dtype.name == "bfloat16":
            return tf.bitcast(tf.constant(out_np.view(np.uint16)), tf.bfloat16)
        return tf.convert_to_tensor(out_np)

    out = tf.space_to_batch_nd(
        input=x,
        block_shape=tf.cast(tf.reshape(block_shape, (-1,)), tf.int32),
        paddings=tf.cast(tf.reshape(paddings, (-1, 2)), tf.int32),
    )
    return out


def space_to_batch_nd_golden(x, block_shape, paddings, **kwargs):
    """
    Kernel golden for space_to_batch_nd.

    Args:
        x: input ND tensor, shape [B, S_1, ..., S_M, T...].
        block_shape: 1-D integer tensor of length M.
        paddings: 2-D integer tensor of shape [M, 2].
        kwargs may contain input_dtypes etc.

    Returns:
        numpy result; bfloat16 uses ml_dtypes.bfloat16, complex32 uses
        interleaved float16 lanes.
    """
    dtype_name = _dtype_name(x, kwargs)
    if not _is_empty(x) and dtype_name not in _TF_UNSUPPORTED_DTYPES:
        return _space_to_batch_nd_tf(x, block_shape, paddings)
    return _space_to_batch_nd_torch(x, block_shape, paddings)


class SpaceToBatchNDTFImpl:
    def __call__(self, x, block_shape, paddings, **kwargs):
        dtype_name = _dtype_name(x, kwargs)
        if not _is_empty(x) and dtype_name not in _TF_UNSUPPORTED_DTYPES:
            return [_space_to_batch_nd_tf(x, block_shape, paddings)]
        return [_space_to_batch_nd_torch(x, block_shape, paddings)]


class SpaceToBatchNDTorchImpl:
    """torch 三方实现：PyTorch 无原生算子，用 torch 小算子拼接。"""

    def __call__(self, x, block_shape, paddings, **kwargs):
        return [_space_to_batch_nd_torch(x, block_shape, paddings)]


_TOLERANCE = {
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
    "bool": {"standard": "binary_equal"},
    "complex32": {"standard": "isclose"},
    "complex64": {"standard": "isclose"},
}


class SpaceToBatchNDTestSpec:
    @staticmethod
    def golden(x, block_shape, paddings, **kwargs):
        return [space_to_batch_nd_golden(x, block_shape, paddings, **kwargs)]

    third_party = {"torch": SpaceToBatchNDTorchImpl, "tf": SpaceToBatchNDTFImpl}

    tolerance = _TOLERANCE


class SpaceToBatchNDTfE2eTestSpec:
    """E2E (tf) 通路：api_name ``tf.raw_ops.SpaceToBatchND`` —— 驱动接口，非标杆。

    与 ``framework/space_to_batch_nd_tf_plugin.cpp`` 的
    OriginOpType("SpaceToBatchND") 对齐：tf 前端 API 经 TF adapter 入 GE 图后
    映射到本仓库 SpaceToBatchND 算子。

    input:       输入 ND tensor [B, S_1, ..., S_M, T...] tf.Tensor
    block_shape: 1-D int32/int64 tf.Tensor，长度 M，取值通常由 CSV attributes
                 同名字段钉住
    paddings:    2-D int32/int64 tf.Tensor，形状 [M, 2]，取值同上钉住

    tf.raw_ops 参数按签名顺序传入，golden 以位置参数接收
    input/block_shape/paddings，返回 [output]（tf.Tensor）。

    golden 固定在 CPU 执行：npu_device.open() 后 NPU 是 TF 默认设备，参考实现
    不能静默跑到被测设备上。dtype 路由与 kernel golden 一致：TF 不支持的
    bool/complex 及空 tensor 改走 torch 小算子拼接，再转回 tf.Tensor。
    """

    def golden(input, block_shape, paddings, **kwargs):
        import tensorflow as tf

        dtype_name = _dtype_name(input, kwargs)
        with tf.device("/CPU:0"):
            out = _space_to_batch_nd_tf_e2e(input, block_shape, paddings, dtype_name)
        return [out]

    third_party = "tf.space_to_batch_nd"

    tolerance = _TOLERANCE
