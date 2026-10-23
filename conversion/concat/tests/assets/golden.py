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
"""concat 算子多通路 TTK golden（kernel + aclnn 合一）。

同一文件按注册名区分通路（__spec__ key 全通路共用一个命名空间）：
  - "concat"    → kernel/GEIR 通路：op_name，golden 收 numpy.ndarray，
                  参数名/顺序跟随 concat_def.cpp（concat_dim, x, N）；
  - "aclnnCat" / "aclnnStack" → aclnn 通路：api_name，golden 收
                  torch.Tensor，参数名/顺序跟随 aclnn_cat.h / aclnn_stack.h
                  （tensors, dim, out）。

两通路入参类型与签名不同，各自独立实现，不能共用同一个函数。
"""

import numpy
import torch

__spec__ = {
    "concat": "ConcatKernelSpec",
    "aclnnCat": "CatStackSpec",
    "aclnnStack": "CatStackSpec",
}


def update_axis_for_hw_inner_format(
    ori_shape, axis, input_format, ori_format, reduce_mode=False
):
    if input_format in ("NDC1HWC0", "NC1HWC0"):
        ori_shape_len = len(ori_shape) if -2 not in ori_shape else len(ori_format)
        axis = axis % ori_shape_len
        offset_6hd = 1 if input_format == "NDC1HWC0" else 0
        format_c_axis = (
            1 + offset_6hd if not reduce_mode else [1 + offset_6hd, 4 + offset_6hd]
        )
        format_axis_map = {
            "N": 0,
            "C": format_c_axis,
            "H": 2 + offset_6hd,
            "W": 3 + offset_6hd,
            "D": 1,
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    if input_format in ("FRACTAL_NZ",):
        axis = axis % len(ori_shape)
        if axis == len(ori_shape) - 1:
            axis = (
                len(ori_shape) - 2
                if not reduce_mode
                else [len(ori_shape) - 2, len(ori_shape) + 1]
            )
        elif axis == len(ori_shape) - 2:
            axis = (
                len(ori_shape) - 1
                if not reduce_mode
                else [len(ori_shape) - 1, len(ori_shape) + 0]
            )

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        axis = axis % len(ori_shape)
        offset_3d = 1 if input_format == "FRACTAL_Z_3D" else 0
        format_c_axis = (
            0 + offset_3d if not reduce_mode else [0 + offset_3d, 5 + offset_3d]
        )
        format_n_axis = (
            3 + offset_3d if not reduce_mode else [3 + offset_3d, 4 + offset_3d]
        )
        format_axis_map = {
            "N": format_n_axis,
            "C": format_c_axis,
            "H": 1 + offset_3d,
            "W": 2 + offset_3d,
            "D": 0,
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    return axis


class ConcatKernelSpec:
    """golden for concat (kernel / GEIR pathway)。入参 numpy.ndarray，
    参数名/顺序跟随 concat_def.cpp；5HD/NZ/Z 排布下需把用户语义 axis
    映射到物理排布 axis。"""

    def golden(concat_dim, x, *, N=1, **kwargs):
        """
        All the parameters (names and order) follow @concat_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor
        """
        x_arrays = list(x)

        ori_shape = kwargs.get("input_ori_shapes", [x[0].shape])[0]
        input_formats = kwargs.get("input_formats", ["ND"])
        input_ori_formats = kwargs.get("input_ori_formats", ["ND"])

        concat_dim = update_axis_for_hw_inner_format(
            ori_shape, concat_dim, input_formats[0], input_ori_formats[0]
        )
        return numpy.concatenate(x_arrays, axis=concat_dim)


class CatStackSpec:
    """golden for aclnnCat / aclnnStack (aclnn pathway). Pure data-movement
    ops require bit-exact results.

    入参类型随 dtype 二分（由 TTK is_torch_dtype_support 决定）：
      - torch 原生 dtype（含 float8_e5m2 / float8_e4m3fn）→ torch.Tensor；
      - 非 torch 原生 dtype（hifloat8 / float8_e8m0 / float4_e2m1 /
        float4_e1m2）→ numpy.ndarray（en_dtypes，unpacked，1 byte/元素），
        此时 out 也为 ndarray，按位搬运用 numpy.concatenate/stack 同样
        bit 级精确（fp4 在 aclnn 侧的 packed 存储由 TTK 比较链路 unpack
        对齐，golden 无需感知 packing）。
    """

    tolerance = {
        d: {"standard": "binary_equal"}
        for d in (
            "float32",
            "float64",
            "float16",
            "bfloat16",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "bool",
            "hifloat8",
            "float8_e5m2",
            "float8_e4m3fn",
            "float8_e8m0",
            "float4_e2m1",
            "float4_e1m2",
        )
    }

    # 不声明 third_party：静态 "torch.cat" 对 stack 用例是错误对标，且
    # hifloat8/fp4 三方侧 torch 无对应 dtype；未声明时服务端按 api_name
    # 自动推导（aclnnCat→torch.cat / aclnnStack→torch.stack）。

    @staticmethod
    def golden(tensors, dim=0, out=None, **kwargs):
        tensors = list(tensors)
        # 老版本 CANN 中 dim 为 aclScalar（0-dim tensor），统一归一为 int
        if hasattr(dim, "item"):
            dim = int(dim.item())
        # cat/stack 派发判据, 优先级从高到低:
        # 1) out 秩判据(确定性): stack 输出恒比输入多一轴, cat 与输入同秩;
        #    TTK 会把 API 签名中的 out 以 Tensor/ndarray(含输出 shape) 传入
        # 2) dim 合法域判据: cat ∈ [-R, R-1], stack ∈ [-R-1, R] (R=输入秩);
        #    dim 落在 cat 域外(==R 或 ==-(R+1)) 时必为 stack
        # 3) 用例名启发(兜底, 仅前两者不可用时)
        # 注: .ndim 对 torch.Tensor 与 numpy.ndarray 均成立
        in_rank = tensors[0].ndim if tensors else 0
        if out is not None and hasattr(out, "ndim"):
            is_stack = out.ndim == in_rank + 1
        else:
            if dim == in_rank or dim == -(in_rank + 1):
                is_stack = True
            else:
                name = kwargs.get("testcase_name", "")
                is_stack = (
                    "stack" in name
                    or "_st_" in name
                    or "stneg" in name
                    or "stnum" in name
                    or name.startswith("st_")
                )
        if tensors and isinstance(tensors[0], numpy.ndarray):
            fn = numpy.stack if is_stack else numpy.concatenate
            return [fn(tensors, axis=dim)]
        fn = torch.stack if is_stack else torch.cat
        return [fn(tensors, dim=dim)]
