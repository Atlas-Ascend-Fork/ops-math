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

TensorFlow is the in-process reference for the dtypes it supports. Dtypes that
TensorFlow does not support (bool/uint32/uint64/complex32/complex64/complex128)
and empty tensors use a pure numpy reference implementation instead. This is a
dtype-based routing decision, not a runtime fallback: supported non-empty
inputs always go through TensorFlow.

Kernel tests must set ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` so importing PyTorch
does not pull in the PyTorch NPU plugin/CANN protobuf, whose symbols conflict
with TensorFlow in the same process.
"""

import numpy as np

__golden__ = {"kernel": {"space_to_batch_nd": "space_to_batch_nd_golden"}}

__input__ = {"kernel": {"space_to_batch_nd": "space_to_batch_nd_input"}}

__spec__ = {"space_to_batch_nd": "SpaceToBatchNDTestSpec"}


# TensorFlow has no (or only experimental) support for these dtypes.
_TF_UNSUPPORTED_DTYPES = {
    "bool",
    "uint32",
    "uint64",
    "complex32",
    "complex64",
    "complex128",
}

_DTYPE_ALIASES = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
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
    return _DTYPE_ALIASES.get(name, name)


def _to_numpy(x):
    """Convert a value to numpy."""
    return np.asarray(x)


def _space_to_batch_nd_numpy(x, block_shape, paddings):
    """SpaceToBatchND reference implemented with numpy primitives.

    Mirrors ``tf.raw_ops.SpaceToBatchND``: pad spatial dims 1..M with zeros,
    split each padded spatial dim into (out, block), move the block dims in
    front of the batch dim, then merge them back into the batch dimension.
    complex32 is stored as float16 with a trailing lane dimension, which is
    simply part of ``T...`` here, so the real ndim is used throughout.
    """
    x_np = _to_numpy(x)
    block = [int(v) for v in _to_numpy(block_shape).reshape(-1)]
    pads = _to_numpy(paddings).astype(np.int64).reshape(len(block), 2)
    M = len(block)
    rank = x_np.ndim
    batch = int(x_np.shape[0])

    if x_np.size == 0:
        out_spatial = [
            (int(x_np.shape[1 + i]) + int(pads[i, 0]) + int(pads[i, 1])) // block[i]
            for i in range(M)
        ]
        out_shape = (
            [batch * int(np.prod(block))]
            + out_spatial
            + [int(v) for v in x_np.shape[1 + M :]]
        )
        return np.zeros(out_shape, dtype=x_np.dtype)

    # Pad only spatial dimensions 1..M; remaining dimensions are untouched.
    pad_width = [(0, 0)]
    for i in range(M):
        pad_width.append((int(pads[i, 0]), int(pads[i, 1])))
    pad_width.extend([(0, 0)] * (rank - 1 - M))
    if any(any(pair) for pair in pad_width):
        x_np = np.pad(x_np, pad_width, mode="constant", constant_values=0)

    out_spatial = [int(x_np.shape[1 + i]) // block[i] for i in range(M)]
    tail = [int(v) for v in x_np.shape[1 + M :]]

    # [B, outS_1, b_1, ..., outS_M, b_M, tail...]
    interleaved = [batch]
    for i in range(M):
        interleaved.extend([out_spatial[i], block[i]])
    interleaved.extend(tail)
    x_np = x_np.reshape(interleaved)

    # [b_1..b_M, B, outS_1..outS_M, tail...]; output batch is block_index * B + b.
    perm = [2 * i + 2 for i in range(M)]
    perm.append(0)
    perm.extend(2 * i + 1 for i in range(M))
    perm.extend(range(1 + 2 * M, 1 + 2 * M + len(tail)))
    x_np = x_np.transpose(perm)

    out_shape = [batch * int(np.prod(block))] + out_spatial + tail
    return np.ascontiguousarray(x_np.reshape(out_shape))


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

    out = tf.raw_ops.SpaceToBatchND(
        input=tensor,
        block_shape=tf.constant(block_np, dtype=tf.int32),
        paddings=tf.constant(pads_np, dtype=tf.int32),
    )

    if x_np.dtype.name == "bfloat16":
        import ml_dtypes

        return tf.bitcast(out, tf.uint16).numpy().view(ml_dtypes.bfloat16)
    return out.numpy()


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
    return _space_to_batch_nd_numpy(x, block_shape, paddings)


class SpaceToBatchNDTFImpl:
    def __call__(self, x, block_shape, paddings, **kwargs):
        dtype_name = _dtype_name(x, kwargs)
        if not _is_empty(x) and dtype_name not in _TF_UNSUPPORTED_DTYPES:
            return [_space_to_batch_nd_tf(x, block_shape, paddings)]
        return [_space_to_batch_nd_numpy(x, block_shape, paddings)]


class SpaceToBatchNDTestSpec:
    @staticmethod
    def golden(x, block_shape, paddings, **kwargs):
        return [space_to_batch_nd_golden(x, block_shape, paddings, **kwargs)]

    third_party = {"tf": SpaceToBatchNDTFImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
