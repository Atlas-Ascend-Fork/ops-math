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
"""Pack 多通路测试规范（TestSpec 格式）。

pack 是纯数据搬运算子（kernel Pack 等价 torch.stack），输出即输入按 axis 拼接的
逐 bit 拷贝，任意 dtype（含浮点）均二进制一致，因此：

- golden 统一对标 torch 接口（kernel 流程 numpy 输入内部转 torch.stack 计算）；
- 精度标准统一为 binary_equal；
- third_party 仅供远程 XPU 通路消费（--xpu-perf 三方性能采集 / cross_check 交叉
  校验），精度跑批不读它（binary_equal 路由不经过 third_party）。纯拷贝场景无需
  cross_check；声明 third_party 是为 --xpu-perf 对标 GPU 性能时，让远程 XPU 知道
  执行哪个标杆 API（torch.stack）。

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | 是   | op_host/pack_def.cpp 定义 Pack |
| geir   | 是   | op_graph/pack_proto.h |
| aclnn  | 是   | op_api/aclnn_stack.h 暴露 aclnnStack |
"""

import numpy as np
import torch

__spec__ = {
    "pack": "PackTestSpec",
    "aclnnStack": "AclnnStackTestSpec",
    "torch.stack": "StackTorchTestSpec",
}

__golden__ = {
    "aclnn": {"aclnnStack": "aclnn_stack_golden"},
    "kernel": {"pack": "pack_golden"},
}

# torch.from_numpy 只认 numpy 原生 dtype；这些类型在 kernel 流程以 ml_dtypes/en_dtypes
# 扩展 numpy 数组到达 golden，需走 numpy bit 路径（np.stack 与 torch.stack 逐 bit 等价）
_NUMPY_ONLY_DTYPES = ("float8_e5m2", "float8_e4m3fn", "hifloat8", "bfloat16")

_BINARY_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "float64": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
    "bool": {"standard": "binary_equal"},
    "complex32": {"standard": "binary_equal"},
    "complex64": {"standard": "binary_equal"},
    "complex128": {"standard": "binary_equal"},
    "hifloat8": {"standard": "binary_equal"},
    "float8_e5m2": {"standard": "binary_equal"},
    "float8_e4m3fn": {"standard": "binary_equal"},
}


def _flatten_strings(container):
    out = []

    def _walk(node):
        if isinstance(node, (tuple, list)):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            out.append(node)

    _walk(container)
    return out


def _is_complex32_storage(arrays, kwargs):
    """complex32 以 float16 + 尾维 [2] 存储（real/imag 交错），stack 轴须插在尾维之前。"""
    declared = _flatten_strings(kwargs.get("input_dtypes", ()))
    if "complex32" not in declared:
        return False
    first = arrays[0]
    return (
        hasattr(first, "dtype")
        and first.dtype.name == "float16"
        and first.ndim >= 1
        and first.shape[-1] == 2
    )


def pack_golden(x, axis: int = 0, N: int = 1, **kwargs):
    """Kernel golden for pack，对标 torch.stack。

    参数名与顺序跟随 pack_def.cpp（不含输出）。输入 x 为 numpy.ndarray 列表；
    kwargs 可能包含 input/output 的 dtypes、shapes、formats 等元信息。

    fp8/bf16 的 numpy 数组为 ml_dtypes/en_dtypes 扩展 dtype，torch.from_numpy
    不识别，走 np.stack（纯搬运，与 torch.stack 逐 bit 等价）；
    complex32 为 float16+尾维[2] 存储，负 axis 需归一化到尾维之前。
    """
    del N
    arrays = list(x)
    if not arrays:
        return [np.empty(0)]
    if _is_complex32_storage(arrays, kwargs):
        view_ndim = arrays[0].ndim - 1
        if axis < 0:
            axis = axis + view_ndim + 1
        return [np.stack(arrays, axis=axis)]
    if any(a.dtype.name in _NUMPY_ONLY_DTYPES for a in arrays):
        return [np.stack(arrays, axis=axis)]
    return [torch.stack([torch.from_numpy(t) for t in arrays], dim=axis).numpy()]


_ACLNN_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float8_e5m2": torch.float8_e5m2,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def _stack_output_dtype(tensor_dtypes):
    """从嵌套 tensor_dtypes 提取输出 dtype 名（输出恒位于结构末尾）。"""
    if not tensor_dtypes:
        return None
    out = tensor_dtypes[-1]
    if isinstance(out, (tuple, list)):
        out = out[0] if out else None
    return out


def _resolve_np_dtype(dtype_name):
    """dtype 名 -> numpy dtype 对象（fp8/bf16 解析为 ml_dtypes/en_dtypes 扩展 dtype）。"""
    name = str(dtype_name).split(".")[-1]
    if name in ("double", "fp64", "f64"):
        name = "float64"
    if name == "bfloat16":
        from ml_dtypes import bfloat16

        return np.dtype(bfloat16)
    if name == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return np.dtype(float8_e5m2)
    if name == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return np.dtype(float8_e4m3fn)
    if name == "hifloat8":
        from en_dtypes import hifloat8

        return np.dtype(hifloat8)
    return np.dtype(name)


_INT_WIDTH = {"int8": 8, "int16": 16, "int32": 32, "int64": 64}
_FP8_NAMES = ("float8_e5m2", "float8_e4m3fn", "hifloat8")


def _cann_promote(dtype_a, dtype_b):
    """复刻 CANN op::PromoteType（kPromoteTypesLookup）的关键提升规则。"""
    a = str(dtype_a).split(".")[-1]
    b = str(dtype_b).split(".")[-1]
    if a == b:
        return a
    pair = (a, b)
    if "complex128" in pair:
        return "complex128"
    if "complex64" in pair:
        return "complex128" if "float64" in pair else "complex64"
    if a in _FP8_NAMES or b in _FP8_NAMES:
        return "float32"  # CANN 查找表 fp8 行零值填充，与任意异类提升为 float
    if "float64" in pair:
        return "float64"
    if "bfloat16" in pair:
        other = b if a == "bfloat16" else a
        if other in ("float32", "float16"):
            return "float32"
        if other == "bool" or other in _INT_WIDTH or other == "uint8":
            return "bfloat16"
        raise RuntimeError(f"PromoteType({a}, {b}) is undefined in CANN.")
    if "float32" in pair:
        other = b if a == "float32" else a
        if other == "bool" or other in _INT_WIDTH or other == "uint8":
            return "float32"
        raise RuntimeError(f"PromoteType({a}, {b}) is undefined in CANN.")
    if "float16" in pair:
        other = b if a == "float16" else a
        if other == "bool" or other in _INT_WIDTH or other == "uint8":
            return "float16"
        raise RuntimeError(f"PromoteType({a}, {b}) is undefined in CANN.")
    if "uint8" in pair:
        other = b if a == "uint8" else a
        if other == "bool":
            return "uint8"
        if other in _INT_WIDTH:
            return "int16" if _INT_WIDTH[other] < 16 else other
        raise RuntimeError(f"PromoteType({a}, {b}) is undefined in CANN.")
    if a in _INT_WIDTH and b in _INT_WIDTH:
        return a if _INT_WIDTH[a] >= _INT_WIDTH[b] else b
    if "bool" in pair:
        other = b if a == "bool" else a
        if other in _INT_WIDTH:
            return other
    raise RuntimeError(f"PromoteType({a}, {b}) is undefined in CANN.")


def _np_value_cast(arr, np_dtype):
    """numpy 值转换；ml_dtypes/en_dtypes 跨库不能直转时经 float32 桥接（无损等价）。"""
    if arr.dtype == np_dtype:
        return arr
    try:
        return arr.astype(np_dtype)
    except (TypeError, ValueError):
        return arr.astype(np.float32).astype(np_dtype)


def aclnn_stack_golden(tensors, dim, out, **kwargs):
    """aclnn golden for aclnnStack，对标 torch.stack。

    aclnnStack 实现见 op_api/aclnn_stack.cpp：输入 cast 到 promoteType 后打包，
    输出侧再 `l0op::Cast(packResult, out dtype)`（与 stack 后对结果 .to(out dtype)
    等价）。因此 golden 在 stack 之后把输出转换为 CSV 声明的 out dtype。

    输入按类型分发（不再依赖 ttk 的 use_numpy 私有 kwarg）：ttk 侧同一开关
    （is_torch_dtype_support）同时决定 use_numpy 取值与输入以 numpy 还是 torch
    张量下发，两者严格一一对应，isinstance 判断与之等价。hifloat8 等非 torch
    原生 dtype 以 numpy 数组（ml_dtypes/en_dtypes 扩展 dtype）到达，分两种情况：
    - 输入与输出同 dtype：无值转换、纯逐 bit 搬移，把输入数组原始字节按 uint8
      载体张量做 stack，字节序列与设备输出一致（比对侧 requant 将双方 view 成
      int8 按字节比对）。
    - 输入与输出 dtype 不同（如混合 int16/bf16 输入转 hifloat8 输出）：为值
      转换，按算子 cast 链走 numpy：各输入 astype(promoteType) -> np.stack ->
      astype(out dtype)，与 NPU Cast 的 RNE 舍入按位一致。
    """
    del out
    if hasattr(dim, "item"):
        dim = dim.item()
    if tensors and isinstance(tensors[0], np.ndarray):
        out_name = _stack_output_dtype(kwargs.get("tensor_dtypes"))
        in_names = {t.dtype.name for t in tensors}
        if in_names == {out_name}:
            itemsize = tensors[0].dtype.itemsize
            if dim < 0:
                dim += tensors[0].ndim + 1
            carriers = [
                torch.frombuffer(t.tobytes(), dtype=torch.uint8).reshape(
                    t.shape + (itemsize,)
                )
                for t in tensors
            ]
            return [torch.stack(carriers, dim=dim)]
        promote = tensors[0].dtype.name
        for t in tensors[1:]:
            promote = _cann_promote(t.dtype.name, promote)
        stacked = np.stack(
            [_np_value_cast(t, _resolve_np_dtype(promote)) for t in tensors], axis=dim
        )
        return [_np_value_cast(stacked, _resolve_np_dtype(out_name))]
    result = torch.stack(tensors, dim=dim)
    target = _ACLNN_TORCH_DTYPES.get(_stack_output_dtype(kwargs.get("tensor_dtypes")))
    if target is not None and result.dtype != target:
        result = result.to(target)
    return [result]


class _PackTorchStack:
    """kernel pack 的三方标杆（远程 XPU 上执行）。

    pack 参数名（x/axis/N，随 pack_def.cpp）与 torch.stack（tensors/dim）不一致，
    字符串形式无法按名绑定，用类形式适配：参数名对齐算子定义，服务端把输入转成
    目标设备上的 torch 张量后实例化调用（契约：参数名 ⊆ inputs∪attrs）。
    """

    def __call__(self, x, axis=0, N=1, **kwargs):
        del N
        return [torch.stack(list(x), dim=axis)]


class PackTestSpec:
    """kernel/geir 共用 spec。golden 入口接收 numpy 数组。"""

    @staticmethod
    def golden(x, axis=0, N=1, **kwargs):
        return pack_golden(x, axis=axis, N=N, **kwargs)

    third_party = {"torch": _PackTorchStack}

    tolerance = _BINARY_TOLERANCE


class AclnnStackTestSpec:
    """aclnnStack spec。golden 入口接收 torch 张量。"""

    @staticmethod
    def golden(tensors, dim, out, **kwargs):
        return aclnn_stack_golden(tensors, dim, out, **kwargs)

    # aclnnStack 输入名（tensors/dim）与 torch.stack 参数名一致，字符串形式可按名绑定
    third_party = {"torch": "torch.stack"}

    tolerance = _BINARY_TOLERANCE


def stack_torch_golden(tensors, dim=0, out=None, **kwargs):
    """e2e (framework-api) golden for torch.stack。

    e2e golden 三级回退（framework_api/golden_generation.py）：CSV golden_api 列 →
    自定义插件（本函数，经 __spec__["torch.stack"] 注册）→ 同 API CPU 复算。
    torch 原生 dtype 原本第三级即可覆盖，注册本 spec 后统一由其接管：

    - torch 张量输入：直通 torch.stack（含 out= 用例），与第三级 CPU 复算同语义；
    - numpy 输入（hifloat8 等非 torch 原生 dtype；第三级 is_dtype_support=False
      会返回 UNSUPPORTED，必须由本插件兜底）：同 dtype 纯位搬运，np.stack 与
      torch.stack 逐 bit 等价；混合 dtype 不支持——torch 的 promote 语义无法在
      numpy 侧对非原生 dtype 复刻，设备侧 torch_npu 对 hifloat8 的 promote 行为
      也无标杆，直接报 NotImplementedError 拒绝而非静默给错值。

    与 aclnnStack golden 的本质差异：e2e 对标框架 API 语义，无 aclnn 的
    promote→stack→cast 链，输出 dtype 恒等于输入 dtype；torch.stack 的 out 为
    keyword-only 且返回值型用例不传，签名须带默认值（e2e 还会额外下发 backend
    等 kwarg，由 **kwargs 吸收）。
    """
    if hasattr(dim, "item"):
        dim = dim.item()
    if tensors and isinstance(tensors[0], np.ndarray):
        in_names = {t.dtype.name for t in tensors}
        if len(in_names) != 1:
            raise NotImplementedError(
                f"torch.stack e2e golden 不支持混合 numpy dtype {sorted(in_names)}："
                "非 torch 原生 dtype 的 promote 语义无法在 numpy 侧复刻"
            )
        # torch_npu 对 hifloat8 无真实 torch.dtype：设备侧 npu_dtype_cast 返回
        # uint8 字节载体（torch_npu.hifloat8 为整型枚举 290）。e2e 比对按实际
        # 输出 dtype（uint8）走 binary_equal，golden 须返回同形态字节视图。
        return [np.ascontiguousarray(np.stack(list(tensors), axis=dim)).view(np.uint8)]
    if out is None:
        return [torch.stack(tensors, dim=dim)]
    return [torch.stack(tensors, dim=dim, out=out)]


class StackTorchTestSpec:
    """e2e spec for torch.stack。golden 入口接收 torch 张量或 numpy 数组。"""

    @staticmethod
    def golden(tensors, dim=0, out=None, **kwargs):
        return stack_torch_golden(tensors, dim=dim, out=out, **kwargs)

    third_party = {"torch": "torch.stack"}

    tolerance = _BINARY_TOLERANCE
