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

"""Kernel golden for sparse_bincount using the TTK TestSpec format."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import functools
import numpy as np

__spec__ = {
    "sparse_bincount": "SparseBincountSpec",
    "SparseBincount": "SparseBincountSpec",
    "tf.raw_ops.SparseBincount": "SparseBincountSpec",
}

__golden__ = {
    "kernel": {"sparse_bincount": "SparseBincountTfGolden"},
}

_TF_PARAM_NAMES = ("indices", "values", "dense_shape", "size", "weights")

_ORIG_TF_SPARSE_BINCOUNT = None


def _to_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _normalized_size(size, values):
    import tensorflow as tf

    if size.shape.rank == 0 and size.dtype == values.dtype:
        return size
    tf_dtype = (
        tf.int64
        if getattr(values.dtype, "as_numpy_dtype", None) == np.int64
        else tf.int32
    )
    return tf.cast(tf.reshape(size, []), tf_dtype)


def _tf_sparse_bincount_compat(*args, **kwargs):
    if args:
        kwargs = {**dict(zip(_TF_PARAM_NAMES, args)), **kwargs}
    binary_output = kwargs.pop("binary_output", False)
    kwargs.pop("name", "None")
    kwargs["size"] = _normalized_size(kwargs["size"], kwargs["values"])
    return _ORIG_TF_SPARSE_BINCOUNT(binary_output=bool(binary_output), **kwargs)


def _install_tf_compat():
    global _ORIG_TF_SPARSE_BINCOUNT
    try:
        import tensorflow as tf
    except ImportError:
        return
    _ORIG_TF_SPARSE_BINCOUNT = tf.raw_ops.SparseBincount
    tf.raw_ops.SparseBincount = functools.wraps(_ORIG_TF_SPARSE_BINCOUNT)(
        _tf_sparse_bincount_compat
    )


_install_tf_compat()


def _as_torch(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "numpy"):
        value = value.numpy()
    return torch.from_numpy(np.ascontiguousarray(value))


class SparseBincountTfGolden:
    def __init__(
        self, indices, values, dense_shape, size, weights, binary_output=False, **kwargs
    ):
        import tensorflow as tf

        del kwargs
        values_t = tf.constant(_to_numpy(values))
        self.values = values_t
        self.indices = tf.constant(_to_numpy(indices))
        self.dense_shape = tf.constant(_to_numpy(dense_shape))
        self.size = _normalized_size(tf.constant(_to_numpy(size)), values_t)
        self.weights = tf.constant(_to_numpy(weights))
        self.binary_output = binary_output

    def __call__(self, **kwargs):
        global _ORIG_TF_SPARSE_BINCOUNT
        if _ORIG_TF_SPARSE_BINCOUNT is None:
            import tensorflow as tf

            _ORIG_TF_SPARSE_BINCOUNT = tf.raw_ops.SparseBincount
        out = _ORIG_TF_SPARSE_BINCOUNT(
            indices=self.indices,
            values=self.values,
            dense_shape=self.dense_shape,
            size=self.size,
            weights=self.weights,
            binary_output=bool(self.binary_output),
        )
        return out.numpy()


class SparseBincountTorchImpl:
    def __init__(
        self, indices, values, dense_shape, size, weights, binary_output=False, **kwargs
    ):
        import torch

        del kwargs
        self.binary_output = bool(binary_output)
        dense_shape_t = _as_torch(dense_shape)
        self.multi_dim = dense_shape_t.numel() > 1
        self.size = int(_as_torch(size).reshape(-1)[0].item())
        self.batch = int(dense_shape_t.reshape(-1)[0].item()) if self.multi_dim else 0
        self.values = _as_torch(values).to(torch.int64)
        self.rows = _as_torch(indices)[:, 0].to(torch.int64) if self.multi_dim else None
        self.weights = _as_torch(weights)

    def __call__(self, **kwargs):
        import torch

        bins = self.values
        if self.multi_dim:
            valid = (
                (bins >= 0)
                & (bins < self.size)
                & (self.rows >= 0)
                & (self.rows < self.batch)
            )
            linear = (self.rows * self.size + bins)[valid]
            length = self.batch * self.size
            out_shape = (self.batch, self.size)
        else:
            valid = (bins >= 0) & (bins < self.size)
            linear = bins[valid]
            length = self.size
            out_shape = (self.size,)

        if self.binary_output or self.weights.numel() == 0:
            acc = torch.ones(
                linear.numel(), dtype=self.weights.dtype, device=linear.device
            )
        else:
            acc = self.weights.reshape(-1)[valid]
        out = torch.bincount(linear, weights=acc, minlength=length).reshape(out_shape)
        if self.binary_output:
            out = (out != 0).to(out.dtype)
        return out if out.is_cuda else out.numpy()


class SparseBincountSpec:
    golden = SparseBincountTfGolden

    third_party = {"torch": SparseBincountTorchImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }
