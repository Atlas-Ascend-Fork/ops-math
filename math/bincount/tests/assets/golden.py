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
"""Bincount 多通路 golden（TestSpec 范式）。

通路支持表（本算子实际交付面）：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | 是   | op_kernel/bincount.cpp 有实现（ascend950/350/960dt/960pr 注册） |
| geir   | 是   | op_graph/bincount_proto.h 有 REG_OP(Bincount)，复用 kernel Spec |
| aclnn  | 是   | op_api/aclnn_bincount.h 的公开两段式接口 aclnnBincount |
| e2e    | 是   | torch.bincount → torch_npu/op-plugin → aclnnBincount |

三个入口的契约**不同**，不能互相套用：

kernel / geir（op_host/bincount_def.cpp）
    输入 array(int32, 1D) / size(int32, ValueDepend 标量) / weights(float32, 1D)，
    输出 bins(float32)。输出长度固定为 size 的值，与有无权重无关；
    array 的每个元素必须满足 0 <= array[i] < size，由调用方保证。

aclnn（op_api/aclnn_bincount.h）
    aclnnBincountGetWorkspaceSize(self, weights, minlength, out)。
    out 由调用方分配，长度为 max(max(self) + 1, minlength)，dtype 支持
    INT32/INT64/FLOAT/DOUBLE；weights 可为空指针，非空时 shape 必须与 self 一致。

e2e（torch.bincount(input, weights=None, minlength=0)）
    输出长度 max(max(input) + 1, minlength)，由数据决定；
    输出 dtype 由 torch 决定，**不是** weights 的 dtype：无 weights 为 int64，
    weights 为浮点时跟随 weights dtype，weights 为整型/bool 时为 float64
    （torch 2.10 实测）。weights 非 None 时长度必须与 input 一致，
    "非空 input + 空 weights" 是非法调用，由 torch 报错，参考实现不得改写成"未提供"。

TF 支持结论：仓内只有 framework/bincount_tf_plugin.cpp 这个图解析插件（把 TF 图
节点映射到本算子 IR），没有可运行的 TF E2E 绑定，本文件不注册 TF 通路。
"""

import numpy as np

# __spec__ 为 TestSpec 显式注册（loader 优先走这里）；__golden__ 保留给旧扫描器兜底。
__golden__ = {
    "kernel": {"bincount": "bincount_golden"},
    "aclnn": {"aclnnBincount": "aclnn_bincount_golden"},
}

__spec__ = {
    "bincount": "BincountKernelSpec",
    "aclnnBincount": "AclnnBincountSpec",
    "torch.bincount": "TorchBincountSpec",
}


def numpy_to_torch_tensor(np_array):
    import torch

    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np.ascontiguousarray(np_array).view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    else:
        return torch.from_numpy(np.ascontiguousarray(np_array))


def torch_to_numpy_tensor(torch_tensor):
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=np.bfloat16)
    else:
        return torch_tensor.numpy()


def _align_to_length(tensor, length):
    """把 1-D bincount 结果对齐到声明的输出长度。

    torch.bincount 只保证 max(max(array) + 1, minlength) 个 bin，而 kernel 的
    输出长度固定为 size、aclnn 的输出长度由调用方分配的 out 决定，二者都可能更长，
    所以不足时补零。结果**长于** length 时直接报错而不是截断：那只可能是输入违反了
    0 <= array[i] < length，截断会把非法输入域悄悄抹掉。
    """
    import torch

    if length is None or tensor.numel() == length:
        return tensor
    if tensor.numel() > length:
        raise ValueError(
            f"bincount golden produced {tensor.numel()} bins but the declared "
            f"output length is {length}; the input violates 0 <= value < {length}"
        )
    return torch.nn.functional.pad(tensor, (0, length - tensor.numel()))


def _first_output_dtype(kwargs, default="float32"):
    """取第一个输出的 dtype 名。

    TTK 下发的 output_dtypes 是按输出分布嵌套的序列（如 ('float32',) 或
    (('float32', 'float32'),)）。这里显式处理"整体就是一个 dtype 字符串"的形态：
    对字符串做下标会取到首字符（'float32'[0] == 'f'），随后 getattr(torch, 'f')
    取不到任何 dtype，输出转换被静默跳过。
    """
    od = kwargs.get("output_dtypes")
    if not od:
        return default
    first = od if isinstance(od, str) else od[0]
    while not isinstance(first, str) and isinstance(first, (list, tuple)):
        if not first:
            return default
        first = first[0]
    if first is None:
        return default
    return str(first)


def _to_torch_dtype(dtype_name):
    """dtype 名 → torch dtype；torch 没有同名 dtype 时报错而不是静默跳过转换。"""
    import torch

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"declared output dtype {dtype_name!r} has no matching torch dtype"
        )
    return dtype


def bincount_golden(array, size, weights, **kwargs):
    """
    Kernel golden for bincount（kernel 与 geir 共用，numpy 进 numpy 出）。
    All the parameters (names and order) follow @bincount_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    输出长度是 size 这个 ValueDepend 输入的值（CSV 里同时经 attributes 列下发，
    如 {'size': 11}），有权重和无权重都一样；array 的每个元素须满足
    0 <= array[i] < size。

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        bins: numpy.ndarray，长度 = size，dtype 跟随声明的输出 dtype（float32）
    """
    import torch

    array_tensor = numpy_to_torch_tensor(array)
    weight_tensor = numpy_to_torch_tensor(weights)

    out_length = int(size.item()) if size is not None else None
    minlength = out_length if out_length is not None else 0

    # 无权重分支同样遵守 size：原实现直接 torch.bincount(array) 并固定成 int32，
    # 既丢了 size 又违背 bins 的声明 dtype。
    if weight_tensor is None or weight_tensor.numel() == 0:
        res = torch.bincount(array_tensor.to(torch.int64), minlength=minlength)
    else:
        res = torch.bincount(
            array_tensor.to(torch.int64), weights=weight_tensor, minlength=minlength
        )

    res = _align_to_length(res, out_length)
    res = res.to(dtype=_to_torch_dtype(_first_output_dtype(kwargs)))

    return torch_to_numpy_tensor(res)


def aclnn_bincount_golden(self, weights, minlength, out, **kwargs):
    """
    Aclnn golden for aclnnBincount.
    All the parameters (name & order) follow \
        function `aclnnBincountGetWorkspaceSize` in @aclnn_bincount.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    out 的长度和 dtype 由调用方声明：结果短于 out 时补零（minlength 大于实际
    bin 数的正常情形），长于 out 时报错（self 越出了 out 能表示的 bin 范围）。
    weights 为空指针时 TTK 传 None；非空 weights 与 self 长度不一致是非法调用，
    这里原样交给 torch 报错，不改写成"未提供"。

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        out: torch.Tensor，长度 = out.numel()，dtype 跟随 out
    """
    import torch

    minlength = int(minlength) if minlength is not None else 0
    # self 支持 INT8/INT16/INT32/INT64/UINT8，CPU torch.bincount 只吃部分整型；
    # 转 int64 对合法的非负值完全无损，负值仍会被 torch 判为非法。
    res = torch.bincount(self.to(torch.int64), weights=weights, minlength=minlength)
    res = _align_to_length(res, out.numel())
    return res.to(out.dtype)


def torch_bincount_golden(input, weights=None, minlength=0, **kwargs):
    """
    E2E golden for torch.bincount：按真实 schema 在 CPU 上独立计算，不经 NPU 入口。
    入参为 CPU 侧 torch.Tensor（weights 可为 None）。

    输出长度 max(max(input) + 1, minlength)，由数据决定；dtype 由 torch 决定
    （无 weights → int64；浮点 weights → weights dtype；整型/bool weights → float64）。
    这里不改写任何入参：weights 与 input 长度不一致等非法组合由 torch 报错。

    Returns:
        [result]: 单元素 list
    """
    import torch

    minlength = int(minlength) if minlength is not None else 0
    return [torch.bincount(input, weights=weights, minlength=minlength)]


class _KernelThirdParty:
    """kernel/geir 通路的三方标杆（远端 provider 上的 torch 执行）。

    参数名取自 op_host/bincount_def.cpp 的输入名（array/size/weights），由框架按名绑定。
    kernel 的输出长度固定为 size，所以用 minlength=size 对齐，并按 bins 的声明 dtype
    返回 float32。
    """

    def __call__(self, array, size, weights, **kwargs):
        import torch

        # size 既是输入也是属性，而两条装配路径的池优先级相反：golden 侧
        # {**inputs, **attributes} 属性赢（收到 int），third_party 侧
        # merged.update(named) 输入赢（收到张量）。两种形态都要能收。
        out_length = None
        if size is not None:
            out_length = int(size.item() if hasattr(size, "item") else size)
        w = weights if (weights is not None and weights.numel() > 0) else None
        res = torch.bincount(
            array.to(torch.int64), weights=w, minlength=out_length or 0
        )
        res = _align_to_length(res, out_length)
        return [res.to(torch.float32)]


def _aclnn_third_party(self, weights, minlength, out, **kwargs):
    """aclnn 通路的三方标杆（远端 provider 上的 torch 执行）。

    写成普通函数而不是类：aclnnBincount 的第一个输入名就叫 `self`，而类形态的
    `__call__` 是绑定方法，`self` 位被实例占用，框架无法把名为 `self` 的输入按名送进来
    （把形参改名成 selfT 之类更糟——池里没有这个名字，绑定直接报 UnknownParamError）。
    普通函数没有实例位，`self` 就是 API 的真实形参名，按名绑定成立。
    """
    import torch

    minlength = int(minlength) if minlength is not None else 0
    res = torch.bincount(self.to(torch.int64), weights=weights, minlength=minlength)
    res = _align_to_length(res, out.numel())
    return [res.to(out.dtype)]


class _TorchThirdParty:
    """e2e 通路的三方标杆：直接对标真实 torch.bincount（远端 provider 上的 torch）。

    参数名取自 torch.bincount 的 schema（input/weights/minlength）。入参原样传入，
    非法组合由 torch 报错。
    """

    def __call__(self, input, weights=None, minlength=0, **kwargs):
        import torch

        minlength = int(minlength) if minlength is not None else 0
        return [torch.bincount(input, weights=weights, minlength=minlength)]


class BincountKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。"""

    golden = bincount_golden
    third_party = {"torch": _KernelThirdParty}
    # bins 是 float32 累加结果，CPU 与 NPU 的累加顺序不同会有微小差异，走统计相对误差。
    tolerance = {"float32": {"standard": "stat_rel_err"}}


class AclnnBincountSpec:
    """aclnn 流程。golden 收 CPU 侧 torch.Tensor，返 torch.Tensor。"""

    golden = aclnn_bincount_golden
    third_party = {"torch": _aclnn_third_party}
    tolerance = {
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "float32": {"standard": "stat_rel_err"},
        "float64": {"standard": "stat_rel_err"},
    }


class TorchBincountSpec:
    """e2e 流程（torch.bincount）。golden 收 CPU 侧 torch.Tensor，返 list[torch.Tensor]。"""

    golden = torch_bincount_golden
    third_party = {"torch": _TorchThirdParty}
    tolerance = {
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "float32": {"standard": "stat_rel_err"},
        "float64": {"standard": "stat_rel_err"},
    }
