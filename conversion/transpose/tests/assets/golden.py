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

"""Transpose 算子多通路 TTK golden（Kernel + GEIR + ACLNN 合一）。

同一文件按注册名区分通路（__spec__ key 全通路共用一个命名空间）：

  - "transpose"           → Kernel 通路：CSV op_name，golden 收 numpy.ndarray，
                            参数跟随 op_host/transpose_def.cpp（x, perm）。
  - "Transpose"           → GEIR 通路：GE 类型名（REG_OP(Transpose)），
                            复用 TransposeKernelSpec（GEIR op_name 为 Transpose 时命中）。
  - "Permute"             → GEIR 通路：GE 类型名（REG_OP(Permute)），
                            参数跟随 op_graph/permute_proto.h（x + order 属性）。
  - "aclnnPermute"        → ACLNN 通路：CSV api_name，golden 收 torch.Tensor，
                            参数跟随 op_api/aclnn_permute.h（self, dims, out）。
  - "aclnnChannelShuffle" → ACLNN 通路：CSV api_name，golden 收 torch.Tensor，
                            参数跟随 op_api/aclnn_channel_shuffle.h（self, groups, out）。

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | 是   | op_host/transpose_def.cpp（arch35 策略族） |
| geir   | 是   | op_graph/transpose_proto.h、op_graph/permute_proto.h |
| aclnn  | 是   | op_api/aclnn_permute.h、op_api/aclnn_channel_shuffle.h |

纯数据重排，任意 dtype（fp16/fp32/bf16/整型/复数/布尔等）均逐 bit 一致，
故 tolerance 统一 binary_equal，无需浮点阈值。
"""

import numpy as np
import torch

__spec__ = {
    # Kernel 通路
    "transpose": "TransposeKernelSpec",
    # GEIR 通路：GE 类型名（REG_OP(Transpose) / REG_OP(Permute)）
    "Transpose": "TransposeKernelSpec",
    "Permute": "PermuteKernelSpec",
    # ACLNN 通路
    "aclnnPermute": "AclnnPermuteSpec",
    "aclnnChannelShuffle": "AclnnChannelShuffleSpec",
}


def _parse_perm(perm):
    """perm / dims / order 统一为 python int 列表。

    兼容 list / tuple / np.ndarray / torch.Tensor / python 标量；
    空序列（[] / 空 ndarray）与 None 统一归一为 None，由调用方按缺省逆序置换处理。
    """
    if perm is None:
        return None
    if hasattr(perm, "tolist"):
        perm = perm.tolist()
    if isinstance(perm, (int, np.integer)):
        return [int(perm)]
    perm = [int(axis) for axis in perm]
    return perm or None


def _reverse_perm(ndim):
    """缺省/空 perm 的逆序置换 [ndim-1, ..., 0]，与 numpy 缺省 np.transpose(a) 一致。"""
    return list(range(ndim - 1, -1, -1))


def _bf16_numpy_dtype():
    """bf16 的 numpy dtype：优先 ml_dtypes，缺失时回退 TensorFlow 自带类型。"""
    try:
        import ml_dtypes

        return ml_dtypes.bfloat16
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.bfloat16.as_numpy_dtype
    except ImportError:
        return None


def _as_numpy(x):
    """torch.Tensor（含 bf16）/ numpy → numpy.ndarray。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            return x.view(torch.int16).numpy().view(_bf16_numpy_dtype())
        return x.numpy()
    return np.asarray(x)


def _to_torch(x):
    """numpy（含 bf16）/ torch → torch.Tensor。"""
    if isinstance(x, torch.Tensor):
        return x
    arr = np.asarray(x)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _transpose_numpy(x, perm):
    """numpy 维度重排；perm 为空时按缺省逆序置换（ONNX Transpose 语义）。"""
    arr = _as_numpy(x)
    perm_val = _parse_perm(perm)
    if perm_val is None:
        perm_val = _reverse_perm(arr.ndim)
    return np.ascontiguousarray(np.transpose(arr, perm_val))


def _write_out(out, result):
    """把 golden 结果写回预分配的 out（torch / numpy 两种承载），返回 out。"""
    if isinstance(out, torch.Tensor) and isinstance(result, torch.Tensor):
        out.copy_(result)
        return out
    np.copyto(np.asarray(out), np.asarray(result))
    return out


def _channel_shuffle_torch(x, groups):
    """torch 版 ChannelShuffle：通道维(dim=1)拆成 g 组后交换第 1/2 维。"""
    groups = int(groups)
    n, c = x.shape[0], x.shape[1]
    rest = tuple(x.shape[2:])
    shuffled = (
        x.reshape((n, groups, c // groups) + rest).transpose(1, 2).reshape(x.shape)
    )
    return shuffled.contiguous()


def _channel_shuffle_numpy(x, groups):
    """numpy 版 ChannelShuffle（非 torch 承载时使用）。"""
    groups = int(groups)
    n, c = x.shape[0], x.shape[1]
    rest = tuple(x.shape[2:])
    reshaped = x.reshape((n, groups, c // groups) + rest)
    return np.ascontiguousarray(np.swapaxes(reshaped, 1, 2).reshape(x.shape))


class TransposeImpl:
    """三方（torch）Transpose 参考：Kernel / GEIR 通路。"""

    def __call__(self, x, perm, **kwargs):
        tensor = _to_torch(x)
        perm_val = _parse_perm(perm)
        if perm_val is None:
            perm_val = _reverse_perm(tensor.dim())
        return torch.permute(tensor, perm_val).contiguous()


class PermuteImpl:
    """三方（torch）Permute 参考：GEIR 通路（order 属性，缺省/为空时逆序置换）。"""

    def __call__(self, x, order=None, **kwargs):
        tensor = _to_torch(x)
        perm = _parse_perm(order)
        if perm is None:
            perm = _reverse_perm(tensor.dim())
        return torch.permute(tensor, perm).contiguous()


class AclnnPermuteImpl:
    """三方（torch）aclnnPermute 参考：ACLNN 通路。

    ACLNN 输入张量名为 "self"，远端/类形态绑定会经 *inputs 按位置送达，
    避免与 Python 实例参数冲突。
    """

    def __call__(self, *inputs, dims=0, out=None, input_formats=None, **kwargs):
        if len(inputs) != 1:
            raise ValueError("aclnnPermute expects exactly one input tensor")
        return TransposeImpl()(inputs[0], dims)


class AclnnChannelShuffleImpl:
    """三方（torch）aclnnChannelShuffle 参考：ACLNN 通路。"""

    def __call__(self, *inputs, groups=1, out=None, input_formats=None, **kwargs):
        if len(inputs) != 1:
            raise ValueError("aclnnChannelShuffle expects exactly one input tensor")
        return _channel_shuffle_torch(_to_torch(inputs[0]), groups)


class TransposeTFImpl:
    """三方（tf）Transpose 参考：覆盖 eager / graph 两种执行模式。"""

    def __call__(self, x, perm, **kwargs):
        import tensorflow as tf

        x_np = _as_numpy(x)
        perm_val = _parse_perm(perm)
        if perm_val is None:
            perm_val = _reverse_perm(x_np.ndim)
        perm = [axis + x_np.ndim if axis < 0 else axis for axis in perm_val]
        if tf.executing_eagerly():
            out_np = tf.transpose(tf.convert_to_tensor(x_np), perm=perm).numpy()
        else:
            graph = tf.Graph()
            with graph.as_default():
                out_g = tf.transpose(tf.convert_to_tensor(x_np), perm=perm)
            with tf.compat.v1.Session(graph=graph) as sess:
                out_np = sess.run(out_g)
        return np.asarray(out_np)


_TOLERANCE = {
    "float32": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
    "hifloat8": {"standard": "binary_equal"},
    "float8_e5m2": {"standard": "binary_equal"},
    "float8_e4m3fn": {"standard": "binary_equal"},
}


class TransposeKernelSpec:
    """Kernel / GEIR Transpose：numpy 输入，(x, perm)。"""

    def golden(x, perm, **kwargs):
        return [_transpose_numpy(x, perm)]

    third_party = {"torch": TransposeImpl, "tf": TransposeTFImpl}
    tolerance = _TOLERANCE


class PermuteKernelSpec:
    """GEIR Permute：numpy 输入，(x, order 属性)，order 缺省/为空时逆序置换。"""

    def golden(x, order=None, **kwargs):
        return [_transpose_numpy(x, order)]

    third_party = {"torch": PermuteImpl}
    tolerance = _TOLERANCE


class AclnnPermuteSpec:
    """ACLNN aclnnPermute：torch 输入，(selfT, dims, out)；dims 为空时逆序置换。"""

    def golden(selfT, dims=None, out=None, **kwargs):
        if isinstance(selfT, torch.Tensor):
            perm = _parse_perm(dims)
            if perm is None:
                perm = _reverse_perm(selfT.dim())
            result = torch.permute(selfT, perm).contiguous()
        else:
            result = _transpose_numpy(selfT, dims)
        if out is not None:
            return [_write_out(out, result)]
        return [result]

    third_party = {"torch": AclnnPermuteImpl}
    tolerance = _TOLERANCE


class AclnnChannelShuffleSpec:
    """ACLNN aclnnChannelShuffle：torch 输入，(selfT, groups, out)。"""

    def golden(selfT, groups=1, out=None, **kwargs):
        if isinstance(selfT, torch.Tensor):
            result = _channel_shuffle_torch(selfT, groups)
        else:
            result = _channel_shuffle_numpy(_as_numpy(selfT), groups)
        if out is not None:
            return [_write_out(out, result)]
        return [result]

    third_party = {"torch": AclnnChannelShuffleImpl}
    tolerance = _TOLERANCE
