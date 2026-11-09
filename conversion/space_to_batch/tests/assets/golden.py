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

TensorFlow is the in-process reference for the dtypes it supports. Dtypes that
TensorFlow does not support (bool/uint32/uint64/complex32/complex64/complex128)
and empty tensors use a pure numpy reference implementation instead. This is a
dtype-based routing decision, not a runtime fallback: supported non-empty
inputs always go through TensorFlow. For this 4D NHWC operator
``tf.raw_ops.SpaceToBatchND`` is called with ``block_shape=[bs, bs]``.

Kernel tests must set ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` so importing PyTorch
does not pull in the PyTorch NPU plugin/CANN protobuf, whose symbols conflict
with TensorFlow in the same process.
"""

import numpy as np

__golden__ = {"kernel": {"space_to_batch": "space_to_batch_golden"}}

__spec__ = {"space_to_batch": "SpaceToBatchTestSpec"}


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


def _is_empty(x):
    return any(int(dim) == 0 for dim in getattr(x, "shape", ()))


def _to_numpy(x):
    """Convert a value to numpy."""
    return np.asarray(x)


def _space_to_batch_numpy(x, paddings, block_size):
    """SpaceToBatch reference implemented with numpy primitives.

    Mirrors ``tf.raw_ops.SpaceToBatchND`` for 4D NHWC input: zero-pad H/W,
    split each padded spatial dim into (out, block), move the block dims in
    front of the batch dim, then merge them back into the batch dimension.
    complex32 is stored as float16 with a trailing lane dimension, which is
    simply part of ``C`` here.
    """
    x_np = _to_numpy(x)
    bs = _resolve_block_size(block_size)

    batch, h_in, w_in, channels = (int(dim) for dim in x_np.shape)
    pad_top = int(paddings[0, 0])
    pad_bottom = int(paddings[0, 1])
    pad_left = int(paddings[1, 0])
    pad_right = int(paddings[1, 1])

    h_padded = h_in + pad_top + pad_bottom
    w_padded = w_in + pad_left + pad_right
    h_out = h_padded // bs
    w_out = w_padded // bs
    out_shape = (batch * bs * bs, h_out, w_out, channels)

    if x_np.size == 0:
        return np.zeros(out_shape, dtype=x_np.dtype)

    pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    if any(any(pair) for pair in pad_width):
        x_np = np.pad(x_np, pad_width, mode="constant", constant_values=0)

    # [N, H_out, bs_h, W_out, bs_w, C] -> [bs_h, bs_w, N, H_out, W_out, C]
    # -> [N * bs * bs, H_out, W_out, C] (block-major, batch index fastest)
    out = x_np.reshape(batch, h_out, bs, w_out, bs, channels)
    out = out.transpose(2, 4, 0, 1, 3, 5)
    return np.ascontiguousarray(out.reshape(out_shape))


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

    out = tf.raw_ops.SpaceToBatchND(
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
    dtype_name = _dtype_name(x, kwargs)
    pads_2x2 = _normalize_paddings(paddings)
    if not _is_empty(x) and dtype_name not in _TF_UNSUPPORTED_DTYPES:
        return _space_to_batch_tf(x, pads_2x2, block_size)
    return _space_to_batch_numpy(x, pads_2x2, block_size)


class SpaceToBatchTFImpl:
    def __init__(self, *, block_size=None, **kwargs):
        self.block_size = block_size

    def __call__(self, x, paddings, block_size=None, **kwargs):
        if block_size is None:
            block_size = self.block_size
        dtype_name = _dtype_name(x, kwargs)
        pads_2x2 = _normalize_paddings(paddings)
        if not _is_empty(x) and dtype_name not in _TF_UNSUPPORTED_DTYPES:
            return [_space_to_batch_tf(x, pads_2x2, block_size)]
        return [_space_to_batch_numpy(x, pads_2x2, block_size)]


class SpaceToBatchTestSpec:
    @staticmethod
    def golden(x, paddings, *, block_size, **kwargs):
        return [space_to_batch_golden(x, paddings, block_size=block_size, **kwargs)]

    third_party = {"tf": SpaceToBatchTFImpl}

    tolerance = {
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "float64": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
