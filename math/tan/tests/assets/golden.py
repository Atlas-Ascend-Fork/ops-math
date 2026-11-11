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
"""Tan 算子独立 golden：raw、ACLNN、Torch、TensorFlow 通路。

口径说明:
- _compute 为纯计算函数：直调 torch.tan，零精度管理、零值域后处理——
  整型输入（int8/int16/int32/int64/uint8/bool）经 torch 提升语义在 f32
  域求值（与显式升 f32 计算链位级一致），f16/bf16 在 src dtype 域计算
  （内部升 f32 求值 + RNE 回写 src），complex 域内直算（std/thrust）；
- 值域后处理按通路分层：输出 dtype 与输入同型的通路（kernel/GEIR，及
  无 out 容器的 aclnn/inplace 调用）将整型的 f32 提升计算结果向零截断
  回原整型（toward zero，镜像设备侧 static_cast 回写链）；带 out 容器
  的 aclnn 通路由 out dtype 决定转换（out 与 self 同为 int32 时
  .to(int32) 即向零截断、与 kernel 回写链位级一致；整型 self 配浮点
  out 的参考值不被整型截断破坏）；torch 通路为行为镜像——无 out 容器
  时纯直调（整型提升出 f32，与被测 torch.tan 同 API 同链），带 out
  容器时直用 torch.tan 原生 out 语义；
- f16/bf16 输入配异 dtype 浮点/复数 out：golden 为纯直调——torch.tan 在
  src dtype 域计算（内部升 f32 求值 + RNE 回写 src）后按 out dtype 扩
  位；设备 aclnn 实现链与 PyTorch/GPU 完全同款（升 f32 计算 + RNE 回写
  src + cast out），third_party 原生 out 语义（torch.tan(x, out=o)）同
  经 src 级舍入；mixed 两方比对下框架将 golden 输入提升至 f32（真值口
  径），与实现口径（src 级 RNE，同 GPU）在大 |tan| 值处存在 src ulp
  级固有差——属框架真值口径与实现口径的固有张力而非数值缺陷，紧容差
  组合（如复数 isclose 1e-4）可显差，用例值域宜避让或按三方比值制
  （cross_check）判读；
- numpy↔torch 转换自包含（bf16 走 int16 位视图链，输出回投输入携带的
  bfloat16 dtype 对象），不依赖测试框架；入口另收框架张量（TF EagerTensor
  等鸭子类型 .numpy()）第三态，惰性转换后与 numpy 态同构；
- 容差覆盖 AI Core 契约 dtype（fp16/bf16/fp32 三方比对 + int32 精确相等）；
  f64 显式 isclose（判据默认参数 rtol=1e-4/atol=1e-8/ptol=1e-4）；
  complex64/complex128 出参由 TTK 默认判据处理（complex isclose rtol=1e-4）。
"""

__spec__ = {
    "tan": "TanKernelSpec",
    "Tan": "TanKernelSpec",
    "aclnnTan": "TanAclnnSpec",
    "aclnnInplaceTan": "TanAclnnInplaceSpec",
    "torch.tan": "TanTorchSpec",
    "tf.math.tan": "TanTensorFlowSpec",
}

# 容差表覆盖 AI Core 契约 dtype（tan_def.cpp 声明 fp16/fp32/bf16/int32）：
# fp16/bf16/fp32 走 cross_check 三方比对（TTK resolve SMALL_VALUE 表仅支持此
# 三档），int32 精确整数语义 → binary_equal。f64 显式 isclose（判据默认参数
# rtol=1e-4/atol=1e-8/ptol=1e-4）；complex64/complex128 出参不特判，由 TTK
# dtype 轴默认判据处理：complex 强制 isclose（rtol 1e-4）。
_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "float64": {"standard": "isclose"},
    "int32": {"standard": "binary_equal"},
}

_L1 = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "float64": {"standard": "isclose"},
    "int32": {"standard": "binary_equal"},
}


def _numpy_to_torch_tensor(np_array):
    """numpy → torch 转换（自包含，不依赖测试框架）。

    覆盖本算子涉及 dtype：fp16/fp32/f64/bf16/int32/c64/c128——bf16 经
    int16 位视图链还原 torch.bfloat16，其余直接 from_numpy；fp8/fp4 输入
    非本算子形态，显式不支持（complex32 在 numpy 侧以 float16 存储并由
    调用方标志区分，本算子无该形态，float16 一律按真实 fp16 处理）。
    """
    import numpy as np
    import torch

    if np_array is None:
        return None
    name = np_array.dtype.name
    if "bfloat16" in name:
        return torch.from_numpy(np_array.view(dtype=np.int16)).view(torch.bfloat16)
    if "float8" in name or "float4" in name or "int4" in name:
        raise RuntimeError(
            f"Dtype [{name}] is not supported by Tan golden (no fp8/fp4 input form)."
        )
    return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor, np_dtype=None):
    """torch → numpy 转换（自包含，不依赖测试框架），与 _numpy_to_torch_tensor 对偶。

    conj 张量先 resolve 再转；bf16 经 int16 位视图回 numpy 侧 bfloat16 表示
    （目标 dtype 取自输入数组携带的 bfloat16 dtype 对象，本文件不引入第三方
    dtype 库）；fp8/fp4/complex32 非本算子形态，显式不支持；其余直接 .numpy()。
    """
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    if torch_tensor.is_conj():
        torch_tensor = torch_tensor.resolve_conj()
    dtype_str = str(torch_tensor.dtype)
    if torch_tensor.dtype == torch.bfloat16:
        if np_dtype is None or "bfloat16" not in np_dtype.name:
            raise RuntimeError(
                "bfloat16 output must be paired with the bfloat16 dtype object "
                "carried by the input array; pass the original input dtype."
            )
        return torch_tensor.view(torch.int16).numpy().view(dtype=np_dtype)
    if (
        "complex32" in dtype_str
        or "float8" in dtype_str
        or "float4" in dtype_str
        or "int4" in dtype_str
    ):
        raise RuntimeError(
            f"Dtype [{dtype_str}] is not supported by Tan golden (no fp8/fp4/complex32 form)."
        )
    return torch_tensor.numpy()


def _compute(x):
    """纯计算：直调 torch.tan——整型经 torch 提升语义自动 f32；f16/bf16 域内；
    complex 域内（std/thrust）。输入/输出 numpy↔torch 双态适配，零精度管理
    （值域后处理由各通路承载）。"""
    import numpy as np
    import torch

    is_torch = isinstance(x, torch.Tensor)
    if not is_torch and not isinstance(x, np.ndarray):
        x = np.asarray(x.numpy())
    np_dtype = None if is_torch else x.dtype
    tensor = x if is_torch else _numpy_to_torch_tensor(x)
    result = torch.tan(tensor)
    return (
        result if is_torch else _torch_to_numpy_tensor(result.detach().cpu(), np_dtype)
    )


def _trunc_to_src(result, x):
    """整型（含 bool）输入的值域后处理：tan 的 f32 提升计算结果向零截断回
    原整型（numpy astype / torch to 均为向零截断，镜像设备侧 static_cast
    回写链）；浮点/复数输入直出（域内计算，无后处理）。输入三态适配：
    torch.Tensor / numpy ndarray / 框架张量（TF EagerTensor 等）——框架
    张量经鸭子类型 .numpy() 惰性转换后与 numpy 态同构（值位级一致）。"""
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if x.dtype.is_floating_point or x.dtype.is_complex:
            return result
        return result.to(x.dtype)
    if not isinstance(x, np.ndarray) and hasattr(x, "numpy"):
        # 框架张量第三态（TF EagerTensor 等鸭子类型）：判型与回写 dtype
        # 统一经 numpy 惰性转换——与 _compute 的入参转换壳同款，零顶层
        # 框架 import。
        x = np.asarray(x.numpy())
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        return result
    return result.astype(x.dtype)


def tan_golden(x, **kwargs):
    # kernel/GEIR 通路：输出与输入同型——整型的 f32 提升计算结果向零截断回原整型。
    return [_trunc_to_src(_compute(x), x)]


def aclnn_tan_golden(self, out=None, **kwargs):
    if out is None:
        # 无 out 容器：同型契约（整型截断回 self dtype），与 kernel/GEIR 通路一致。
        return tan_golden(self)
    # out 容器：纯计算结果按 out dtype 转换（out 与 self 同为 int32 时 .to(int32)
    # 即向零截断、与 kernel 回写链位级一致；整型 self 配浮点 out 不被整型截断破坏）。
    return [_compute(self).reshape(out.shape).to(out.dtype)]


def aclnn_inplace_tan_golden(selfRef, **kwargs):
    """CPU golden following aclnnInplaceTanGetWorkspaceSize(selfRef, ...)."""
    # 就地语义：结果写回输入张量（dtype 不变）——整型截断回 selfRef dtype。
    return [_trunc_to_src(_compute(selfRef), selfRef)]


def torch_tan_golden(input, out=None, **kwargs):
    import torch

    if out is None:
        # 行为镜像：纯直调零后处理（整型经 torch 提升语义出 f32，与被测 torch.tan 同链）。
        return [_compute(input)]
    # 原生 out 语义：计算与 out 写入由 torch.tan 原生承载（与被测同 API 同链）。
    return [torch.tan(input, out=out)]


class TanKernelTorchRef:
    """kernel/GEIR 路 torch 三方参考: 原生域转调原生 API;int32 按算子契约拼接截断链。"""

    def __call__(self, x):
        import torch

        if torch.is_tensor(x) and x.dtype == torch.int32:
            # int32 入出同型契约: 原生 tan 为提升语义出 f32, 拼接截断链矫正
            # (.to(int32) 向零截断, 与 kernel static_cast 回写链一致)
            return torch.tan(x.to(torch.float32)).to(torch.int32)
        return torch.tan(x)


class TanKernelTfRef:
    """kernel/GEIR 路 tf 三方参考: 同上分派。"""

    def __call__(self, x):
        import tensorflow as tf

        if x.dtype == tf.int32:
            # tf.cast f32->int32 向零截断, 与 kernel static_cast 回写链一致
            return tf.cast(tf.math.tan(tf.cast(x, tf.float32)), tf.int32)
        return tf.math.tan(x)


class TanKernelSpec:
    golden = staticmethod(tan_golden)
    # kernel/GEIR 路 third_party 挂双框架参考: 原生 dtype 域直接转调原生
    # API(torch.tan / tf.math.tan);int32 域原生 API 为提升语义出 f32, 与
    # 本通路 int32 入出同型契约错配, 由拼接类按契约截断矫正(f32 计算链 +
    # 向零截断回 int32, 与 kernel static_cast 回写一致)。类形态经远端
    # spec 协议同步执行(spec 类参考与原生 API 字符串同为可达形态)。
    third_party = {"torch": TanKernelTorchRef, "tf": TanKernelTfRef}
    tolerance = _L2


class TanAclnnTorchRef:
    """aclnn 路 torch 三方参考: 原生 out 语义——计算后按 out 数据类型转换(与 aclnn 输出 Cast 链同构);整型输入经提升计算后写入 out。"""

    # 形参名对位 aclnnTan(self, out, ...) 的 C 接口参数序——装配层按名绑定;
    # staticmethod 使 __call__ 保持未绑定的普通函数形态, self 作为真实 API
    # 形参名参与具名对位(实例调用不再占用首参位), out 缺省时退化为直调。
    @staticmethod
    def __call__(self, out=None):
        import torch

        if out is not None:
            return torch.tan(self, out=out)
        return torch.tan(self)


class TanAclnnInplaceTorchRef:
    """aclnnInplaceTan 路 torch 三方参考: 就地语义(结果写回输入张量)。"""

    def __call__(self, selfRef):
        import torch

        return torch.tan(selfRef, out=selfRef)


class TanAclnnSpec:
    golden = staticmethod(aclnn_tan_golden)
    # aclnn 路 third_party 挂类参考: 原生 out 参数语义与 aclnn 输出 Cast 链
    # 同构(计算后按 out 数据类型转换, 整型 self 经提升计算写入 out), 文档域
    # self 全 dtype × out 六 dtype 由原生 out 容器承载; 类形态经远端 spec
    # 协议执行, 形参按 C 接口参数名对位装配。
    third_party = {"torch": TanAclnnTorchRef}
    tolerance = _L2


class TanAclnnInplaceSpec:
    """aclnnInplaceTan 通路 spec（显式注册，消除 TTK torch.tan 回退）。

    golden 遵循 aclnnInplaceTanGetWorkspaceSize(selfRef) 签名；in-place 与
    非 in-place 共用同一 Compute 数值链（op_api/aclnn_tan.cpp），tolerance 同 _L2。
    """

    golden = staticmethod(aclnn_inplace_tan_golden)
    # 就地语义类参考: 结果写回输入张量(dtype 不变), 与 aclnnInplaceTan 的
    # 写回契约对齐; 形参名对位 aclnnInplaceTan(selfRef, ...) 的 C 接口参数序。
    third_party = {"torch": TanAclnnInplaceTorchRef}
    tolerance = _L2


class TanTorchSpec:
    golden = staticmethod(torch_tan_golden)
    third_party = {"torch": "torch.tan"}
    tolerance = _L2


class TanTensorFlowSpec:
    golden = staticmethod(tan_golden)
    third_party = {"tf": TanKernelTfRef}
    tolerance = _L1
