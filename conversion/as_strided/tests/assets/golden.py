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
import numpy
import torch
import ml_dtypes

__spec__ = {
    "as_strided": "AsStridedTestSpec",
    "aclnnAsStrided": "AclnnAsStridedTestSpec",
    "torch_npu.npu_stride_copy": "NpuStrideCopyTestSpec",
    # 兼容写法
    "torch.ops.npu.npu_stride_copy.default": "NpuStrideCopyTestSpec",
}


def _as_int_seq(value):
    """size/stride 归一为 int 列表（aclnn 路径来自属性为 list；kernel 路径为张量/ndarray）。"""
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    elif isinstance(value, numpy.ndarray):
        value = value.reshape(-1).tolist()
    return [int(v) for v in value]


def _as_int_scalar(value):
    """storageOffset/storage_offset 归一为 int（兼容 None/空值/list/ndarray/张量/标量）。

    storage_offset 在 as_strided_def.cpp 中为 OPTIONAL 输入，缺省（None）按偏移 0。
    """
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        return int(flat[0].item()) if flat.numel() else 0
    if isinstance(value, numpy.ndarray):
        flat = value.reshape(-1)
        return int(flat[0]) if flat.size else 0
    if isinstance(value, (list, tuple)):
        return int(value[0]) if len(value) else 0
    return int(value)


def _torch_out_to_numpy(out, dtype_name):
    """torch 输出转 numpy；bf16/fp8 无 numpy 原生 dtype，按位载体转回 ml_dtypes。"""
    out = out.clone(memory_format=torch.contiguous_format)
    if dtype_name == "bfloat16":
        return out.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    if dtype_name == "float8_e4m3fn":
        return out.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
    if dtype_name == "float8_e5m2":
        return out.view(torch.uint8).numpy().view(ml_dtypes.float8_e5m2)
    return out.numpy()


class AclnnAsStridedTestSpec:
    """aclnn 路径：本地 golden。"""

    @staticmethod
    def golden(x, size, stride, storageOffset=0, y=None, **kwargs):
        offset = _as_int_scalar(storageOffset)
        size_list = _as_int_seq(size)
        stride_list = _as_int_seq(stride)
        if isinstance(x, torch.Tensor):
            out = torch.as_strided(
                x, size_list, stride_list, offset + x.storage_offset()
            )
            # numel<=1 或零 stride 的 strided 视图 contiguous() 为空操作，需显式连续拷贝
            return [out.clone(memory_format=torch.contiguous_format)]
        # hifloat8 等无 torch 原生 dtype：numpy 载体按位 gather（纯数据搬运，字节语义）。
        # numpy as_strided 无 offset 形参（第4位是 subok），offset 用切片平移实现
        max_idx = (
            offset + sum((s - 1) * d for s, d in zip(size_list, stride_list))
            if size_list
            else offset
        )
        flat = numpy.lib.stride_tricks.as_strided(
            x, (max(max_idx + 1, 1),), (x.itemsize,)
        )
        out = numpy.lib.stride_tricks.as_strided(
            flat[offset:], size_list, [s * x.itemsize for s in stride_list]
        )
        return [numpy.array(out, order="C", copy=True)]


class AsStridedTestSpec:
    """kernel 路径：参数序与 as_strided_def.cpp 一致（不含输出），输入为 numpy.ndarray。"""

    @staticmethod
    def golden(x, size, stride, storage_offset=None, **kwargs):
        offset = _as_int_scalar(storage_offset)
        size_list = _as_int_seq(size)
        stride_list = _as_int_seq(stride)
        # kernel 模式输入为底层存储的平坦视图；多维输入按扁平存储偏移生效
        flat = x.reshape(-1)

        dtype_name = str(kwargs.get("input_dtypes", ["float32"])[0])
        if dtype_name == "complex32":
            # complex32 载体为 fp16 交错对（尾维 [2]），按位还原为 torch.complex32
            t_x = torch.from_numpy(flat).view(torch.complex32)
            out = torch.as_strided(t_x, size_list, stride_list, offset)
            out = torch.view_as_real(out).clone(memory_format=torch.contiguous_format)
            return out.numpy()
        if dtype_name == "bfloat16":
            t_x = torch.from_numpy(flat.view(numpy.int16)).view(torch.bfloat16)
        elif dtype_name == "float8_e4m3fn":
            t_x = torch.from_numpy(flat.view(numpy.uint8)).view(torch.float8_e4m3fn)
        elif dtype_name == "float8_e5m2":
            t_x = torch.from_numpy(flat.view(numpy.uint8)).view(torch.float8_e5m2)
        else:
            t_x = torch.from_numpy(flat)
        out = torch.as_strided(t_x, size_list, stride_list, offset)
        return _torch_out_to_numpy(out, dtype_name)


class NpuStrideCopyTestSpec:
    """E2E 路径：对标 torch_npu 的 npu_stride_copy。

    签名（op_plugin_functions.yaml）：
        npu_stride_copy(Tensor self, int[] shape, int[] stride, Scalar storage_offset) -> Tensor
    """

    @staticmethod
    def golden(self, shape, stride, storage_offset, **kwargs):
        size_list = _as_int_seq(shape)
        stride_list = _as_int_seq(stride)
        offset = _as_int_scalar(storage_offset)
        out = torch.as_strided(self, size_list, stride_list, offset)
        return [out.clone(memory_format=torch.contiguous_format)]
