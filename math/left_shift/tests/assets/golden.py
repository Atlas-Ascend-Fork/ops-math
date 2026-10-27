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
"""left_shift multi-pathway golden in the TestSpec format.

通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35/left_shift_dag.h 有 arch35 实现 |
| geir   | ✅   | op_graph/left_shift_proto.h 有 REG_OP(LeftShift) |
| aclnn  | ✅   | op_api/aclnn_left_shift.cpp 暴露 aclnnLeftShift / aclnnLeftShifts 符号 |
| e2e    | ✅   | 两条 e2e 通路：torch（经 torch_npu 派发）与 tf（经 TF 插件入图） |

各通路注册键 / api_name、入出参类型与路由（见 ttk-how-write-plugin）：
  - kernel      : op_name  ``left_shift``                          → numpy 入/出；
                  直调 op_kernel/arch35/left_shift_dag.h 的 arch35 kernel
  - geir        : op_name  ``left_shift``（与 kernel 共用同名注册）→ numpy 入/出；
                  由 op_graph/left_shift_proto.h 的 REG_OP(LeftShift) 构图
  - aclnn       : api_name ``aclnnLeftShift`` / ``aclnnLeftShifts``→ torch 入/出；
                  aclnn → Cast/BroadcastTo → l0op::LeftShift → kernel
  - e2e (torch) : 驱动接口 api_name ``torch.bitwise_left_shift``   → torch 入/出；
                  torch_npu 派发 → aclnnLeftShift → kernel
  - e2e (tf)    : 驱动接口 api_name ``tf.bitwise.left_shift``     → torch 入/出；
                  TF 插件 framework/left_shift_tf_plugin.cpp（ImplyType::AI_CPU）
                  → GE 图 → LeftShift 算子

e2e 有两条**驱动接口**：torch 与 tf，接入同一个 LeftShift 算子（torch→torch_npu→aclnn；
tf→TF 插件→GE 图）。务必区分「驱动接口」与「精度标杆」：
  - 接口（torch / tf）只决定用什么前端把 NPU 驱动起来；
  - 精度标杆对标的始终是**算子语义** = ``torch.bitwise_left_shift`` 的期望结果，
    两条 e2e 通路共用同一套 torch 真值（``_left_shift``），均返回 torch.Tensor。
tf 只是换了个前端接口，不改标杆；负/越界等边界同样按算子（torch）语义验收。
kernel 与 geir 共用一个注册键（算子蛇形名），geir 不另写 Spec 类。
golden 对标 torch.bitwise_left_shift：按位左移，位移量为负或不小于“参与运算的
数据类型（提升后）位宽”时输出 0（与 PyTorch CPU/CUDA kernel 的钳位语义一致）。
因 PyTorch 仅完整支持 uint8，uint16/uint32/uint64 这几种无符号类型 torch 无法
承载（from_numpy 与 bitwise_left_shift 均不可用），只要入参涉及它们就改用 numpy
等价实现，避免 golden 报错。
类型提升遵循 CANN 互推导规则（docs/zh/context/deduction_relationship.md）：aclnn 侧
op::PromoteType 决定计算位宽，且 uint16/uint32/uint64 只能与同 dtype 组合（异 dtype
组合返回 DT_UNDEFINED → ACLNN_ERR_PARAM_INVALID）；kernel/geir(AI_CPU)/tf 侧则要求
x/y 同 dtype（left_shift_aicpu.cpp 校验 input0_type == input1_type）。
注意：其间用到的「同宽 signed .view() 中转」依赖较新版本 torch 对 uint16/uint32/uint64
limited dtype 及 .view() 的支持；若目标 torch 版本不支持需另加 fallback。

三方(GPU)标杆：kernel/geir、e2e(torch)、e2e(tf) 三个 Spec 内嵌 ``ThirdPartyImpl``
（写法见 ttk-how-write-plugin）并声明
``third_party = {"torch": ThirdPartyImpl}``，由 TTK 远端 XPU 以设备侧 torch.Tensor 执行，
与 golden(CPU) 独立、互为印证(三方: NPU / golden / third_party)。
是否纳入判定由 Spec 的 ``tolerance`` 决定：TTK 里只有 standard=cross_check 会取三方输出
判精度，其余标准下 third_party 照样执行但不进判定——而 cross_check 当前只支持
float16/float32/bfloat16。本算子 tolerance 全为 binary_equal，故 third_party 对全部整型
(含 uint16/uint32/uint64)仅作声明、不进判定。
竞品实现须与 golden 独立，故在设备侧直调 ``torch.bitwise_left_shift``，不搬回 CPU、也不
复用 golden 的 numpy 路径。uint16/uint32/uint64 属 torch limited dtype、竞品 torch 无法承载，
且不参与三方比对，third_party 不为它们兜底(与同族 bitwise_and 口径一致)。
aclnn 通路不声明 third_party：首参名为 ``self``、与 Python 保留名冲突，TTK 类形式绑定会丢弃
该条目，三方实现拿不到 self 无法计算(详见 LeftShiftAclnnSpec)。
"""

import numpy as np
import torch

__spec__ = {
    # kernel 与 geir 共用蛇形名注册
    "left_shift": "LeftShiftKernelSpec",
    "aclnnLeftShift": "LeftShiftAclnnSpec",
    "aclnnLeftShifts": "LeftShiftAclnnSpec",
    # e2e(torch) 通路
    "torch.bitwise_left_shift": "LeftShiftE2eSpec",
    # e2e(tf) 通路（tf 仅为驱动接口；精度标杆为 torch 算子语义）
    "tf.bitwise.left_shift": "LeftShiftTfSpec",
}

# legacy 注册(消费方迁移到 TestSpec 期间保留)
__golden__ = {
    "kernel": {"left_shift": "left_shift_golden"},
    "aclnn": {
        "aclnnLeftShift": "aclnn_left_shift_golden",
        "aclnnLeftShifts": "aclnn_left_shifts_golden",
    },
    "e2e": {
        "torch.bitwise_left_shift": "torch_left_shift_golden",
        "tf.bitwise.left_shift": "tf_left_shift_golden",
    },
}

# 契约: left_shift_def.cpp 的 8 个整型(无 bool/浮点)
_INT_DTYPE_STR = (
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
)
_KERNEL_INT_DTYPES = frozenset(map(np.dtype, _INT_DTYPE_STR))

# 按位运算无舍入路径, 判据统一 binary_equal
_KERNEL_TOLERANCE = {dtype: {"standard": "binary_equal"} for dtype in _INT_DTYPE_STR}
# aclnn/e2e 同为按位运算, 判据与 kernel 一致(单列以对齐同族 golden 命名)
_API_TOLERANCE = dict(_KERNEL_TOLERANCE)


# torch 原生不承载的无符号整型: PyTorch 仅 uint8 完整支持, uint16/uint32/uint64
# 属 limited support, torch.from_numpy / .numpy() / torch.bitwise_left_shift 均不可用。
_TORCH_UNSUPPORTED_DTYPES = frozenset(("uint16", "uint32", "uint64"))

# CANN 互推导中只能与同 dtype 组合的无符号整型: u16/u32/u64 与任何异 dtype 组合时
# op::PromoteType 返回 DT_UNDEFINED, aclnn 报 ACLNN_ERR_PARAM_INVALID。集合与
# _TORCH_UNSUPPORTED_DTYPES 相同但语义不同 (前者是算子契约, 后者是 torch 能力), 故单列。
_UNSIGNED_SAME_DTYPE_ONLY = frozenset(("uint16", "uint32", "uint64"))

# limited dtype 与同宽有符号类型互转 (位宽相同, view 只换 dtype 标签, 位模式不变)。
_UNSIGNED_TO_SIGNED = {"uint16": "int16", "uint32": "int32", "uint64": "int64"}


def _to_torch(array):
    """numpy → torch (无损; torch 为统一标杆).

    uint16/uint32/uint64 属 torch limited dtype, from_numpy 不可用, 经同宽有符号
    类型 view 中转 (位模式不变)。依赖较新 torch 对 limited dtype 及 .view() 的支持。
    """
    array = np.asarray(array)
    if array.ndim != 0:
        # ascontiguousarray 会把 0 维数组升成 (1,), 破坏标量 shape, 故仅规整非标量
        array = np.ascontiguousarray(array)
    signed = _UNSIGNED_TO_SIGNED.get(array.dtype.name)
    if signed is None:
        return torch.from_numpy(array)
    return torch.from_numpy(array.view(signed)).view(getattr(torch, array.dtype.name))


def _as_numpy(tensor):
    """入参归一为 numpy.ndarray (numpy 原样返回; python 标量/asarray; torch 转回 numpy).

    uint16/uint32/uint64 属 torch limited dtype, .numpy() 不可用, 经同宽有符号
    类型 view 中转 (位模式不变)。
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if np.isscalar(tensor):
        # aclnn 标量通路(aclnnLeftShifts)的 shiftBits 可能以 python 标量/numpy 标量传入:
        # np.isscalar 已覆盖 python int/float/bool 与 numpy 标量(np.generic)
        return np.asarray(tensor)
    tensor = tensor.detach().cpu()
    name = str(tensor.dtype).split(".")[-1]
    signed = _UNSIGNED_TO_SIGNED.get(name)
    if signed is None:
        return tensor.numpy()
    return tensor.contiguous().view(getattr(torch, signed)).numpy().view(np.dtype(name))


def _dtype_name(value):
    """取入参的 dtype 名 (如 "int8"/"uint64"), 兼容 numpy / torch / python 标量。"""
    if isinstance(value, np.ndarray):
        return value.dtype.name
    if hasattr(value, "dtype"):
        return str(value.dtype).split(".")[-1]
    return np.asarray(value).dtype.name


def _promote_int_dtype(a, b):
    """按 CANN 互推导规则推导两个整型 dtype (对齐 op::PromoteType), 供 numpy 等价实现取
    计算位宽与结果类型。

    - u16/u32/u64 只能与同 dtype 组合; 异 dtype 组合 op::PromoteType 返回 DT_UNDEFINED
      -> aclnn 报 ACLNN_ERR_PARAM_INVALID, 此处抛 ValueError 以同样拒绝 (与同族
      bitwise_and golden 口径一致);
    - {int8, uint8, int16, int32, int64} 按标准整型提升 (与 CANN 表逐格一致): 同符号取
      较宽; 有符号+无符号时较宽有符号可容纳则取该有符号, 否则升双倍宽有符号
      (实到仅 int8+uint8 -> int16, 与互推导表 s8×u8=s16 一致)。
    """
    a = np.dtype(a)
    b = np.dtype(b)
    if a == b:
        return a
    if a.name in _UNSIGNED_SAME_DTYPE_ONLY or b.name in _UNSIGNED_SAME_DTYPE_ONLY:
        raise ValueError(
            f"op::PromoteType yields DT_UNDEFINED for {a} x {b}: uint16/uint32/uint64 can "
            f"only combine with the same dtype (aclnn host check -> ACLNN_ERR_PARAM_INVALID)"
        )
    if a.kind not in ("i", "u") or b.kind not in ("i", "u"):
        raise TypeError(f"left_shift expects integer dtypes, got {a} x {b}")
    if a.kind == b.kind:
        return a if a.itemsize >= b.itemsize else b
    signed, unsigned = (a, b) if a.kind == "i" else (b, a)
    if signed.itemsize > unsigned.itemsize:
        return signed
    return np.dtype("i{}".format(2 * max(signed.itemsize, unsigned.itemsize)))


def _left_shift_numpy(x, y, promote):
    """numpy 等价实现: 位移量为负或不小于“提升后类型位宽”时输出 0 (对齐 torch 语义).

    promote 为 _promote_int_dtype 推导出的 CANN 计算类型 (op::PromoteType 结果), 其位宽
    即钳位阈值。位移以无符号位模式进行 (有符号数按补码 reinterpret, 结果再按提升类型
    还原), 使判据统一为 "无符号计数值 < 位宽", 同时规避 C/numpy 上有符号溢出与越界位移
    的未定义行为。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    bits = promote.itemsize * 8
    uint_dtype = np.dtype("u{}".format(promote.itemsize))
    work = x.astype(promote, copy=False).astype(uint_dtype, copy=False)
    # 无符号位模式: 有符号负值变成大无符号数, 统一被 "cnt < 位宽" 判为无效
    cnt = y.astype(promote, copy=False).astype(uint_dtype, copy=False)
    valid = cnt < uint_dtype.type(bits)
    safe_cnt = np.where(valid, cnt, uint_dtype.type(0)).astype(uint_dtype)
    shifted = np.left_shift(work, safe_cnt)
    result = np.where(valid, shifted, uint_dtype.type(0)).astype(uint_dtype)
    return result.astype(promote, copy=False)


def _left_shift(x, y, force_promote=False):
    """统一标杆: 位移量为负或越界时输出 0.

    先按 CANN 互推导 (_promote_int_dtype, 对齐 op::PromoteType) 校验/推导计算类型:
    u16/u32/u64 与异 dtype 组合会被拒绝 (DT_UNDEFINED -> ACLNN_ERR_PARAM_INVALID)。
    再: 任一入参为 torch 不承载的 uint16/uint32/uint64 -> numpy 等价实现 (返回 numpy);
    其余 dtype -> torch.bitwise_left_shift (返回 torch)。

    force_promote=True: 按 CANN op::PromoteType 结果显式提升两个操作数再位移 (aclnn
    通路需要——本机 torch 对 int8×uint8 的隐式提升结果是 uint8, 与 CANN 互推导表
    int16 不一致, 若依赖 torch 隐式提升会在位移量 >= 8 时截断为 0);
    force_promote=False: 沿用 torch 原生提升 (e2e torch 通路口径, 与 torch 设备侧一致)。
    """
    x_np = _as_numpy(x)
    y_np = _as_numpy(y)
    promote = _promote_int_dtype(x_np.dtype, y_np.dtype)
    if (
        x_np.dtype.name in _TORCH_UNSUPPORTED_DTYPES
        or y_np.dtype.name in _TORCH_UNSUPPORTED_DTYPES
    ):
        return _left_shift_numpy(x_np, y_np, promote)
    x_t, y_t = _to_torch(x_np), _to_torch(y_np)
    if force_promote:
        pt = getattr(torch, promote.name)
        x_t, y_t = x_t.to(pt), y_t.to(pt)
    return torch.bitwise_left_shift(x_t, y_t)


def _write_into_out(result, out):
    """把标杆结果写入 out 并返回 out (兼容 numpy / torch 两类入参).

    out 可能是非连续/广播(0 stride)视图: torch 的 copy_ 在目标内存重叠
    (如全 0 stride 广播 out) 时会直接报错; 此类情况改用逐元素 C-order 写回
    (与 tensor 共享存储, 末次写生效), 与单线程参考语义一致; 其余情况正常
    copy_, 让真实错误照常抛出, 不吞异常。
    """
    value = _as_numpy(result)
    if isinstance(out, np.ndarray):
        out[...] = value
        return out
    if any(size > 1 and stride == 0 for stride, size in zip(out.stride(), out.shape)):
        _write_overlapping(out, value)
    else:
        out.copy_(_to_torch(value))
    return out


def _write_overlapping(out, value):
    """0-stride(广播)视图写回: copy_ 因目标内存重叠报错, 改逐元素 C-order 写回
    (共享存储, 末次写生效, 与单线程参考语义一致)。

    普通 CPU 整型经 numpy 直接写回; 受限 dtype(uint16/uint32/uint64, .numpy()
    不可用)或非 CPU 场景, 由于 0-stride 张量无法 .view 换 dtype, 退化为 torch
    逐元素赋值(每步只写 0 维目标, 不触发重叠检查)。
    """
    target = out.detach()
    name = str(target.dtype).split(".")[-1]
    if target.device.type == "cpu" and name not in _TORCH_UNSUPPORTED_DTYPES:
        target.numpy()[...] = value
        return
    flat = _to_torch(np.ascontiguousarray(value).reshape(-1))
    for i, index in enumerate(np.ndindex(*target.shape)):
        target[index] = flat[i]


def _to_out_torch(result):
    """把标杆结果统一归一为 torch.Tensor（e2e / aclnn 出参约定）。

    result 已是 torch.Tensor 时直接返回；uint16/uint32/uint64 走 numpy 分支得到
    numpy，经同宽 signed view 中转转 torch（位模式不变）。真值保持在计算设备
    （通常 CPU/竞品侧），不按入参设备搬移。
    """
    return result if isinstance(result, torch.Tensor) else _to_torch(np.asarray(result))


# 三方(GPU)标杆：由 TTK 远端 XPU 以设备侧 torch.Tensor 执行，与 golden(CPU) 独立、
# 互为印证。竞品实现内嵌到各 Spec（``ThirdPartyImpl``），
# 设备侧直调 ``torch.bitwise_left_shift``(其 CPU/CUDA 钳位语义即算子语义)，不搬回 CPU、
# 也不复用 golden 的 numpy 路径。形参名按各入口对位：kernel/geir=x/y，
# e2e(torch)=input/other，e2e(tf)=x/y；aclnn 首参名为 self、与 Python 保留名冲突，
# 不提供三方（见 LeftShiftAclnnSpec）。
# 说明：third_party 是否纳入判定由 tolerance 决定——只有 cross_check 取三方输出，且
# cross_check 只支持 float16/float32/bfloat16；本算子为整型(binary_equal)，故 third_party
# 仅作声明、不进判定。


def _third_party_left_shift(x, y):
    """三方(GPU)竞品：设备侧直接调 ``torch.bitwise_left_shift`` (与 golden 独立)。

    uint16/uint32/uint64 属 torch limited dtype、竞品 torch 无法承载，且不参与三方比对
    (cross_check 只覆盖浮点)，故不做兜底(与同族 bitwise_and 的 third_party 口径一致)。
    """
    return torch.bitwise_left_shift(x, y)


class LeftShiftKernelSpec:
    """Kernel/GEIR: numpy 入参, 契约源 left_shift_def.cpp (同 dtype, 8 整型)。
    校验后经 ``_left_shift`` 计算 (torch 支持的 dtype 走 torch, u16/u32/u64 走 numpy),
    再归一为 numpy 出参。"""

    @staticmethod
    def golden(x, y, **kwargs):
        del kwargs
        x = np.asarray(x)
        y = np.asarray(y)
        for name, arr in (("x", x), ("y", y)):
            if arr.dtype not in _KERNEL_INT_DTYPES:
                raise ValueError(
                    f"left_shift kernel supports the 8 integer dtypes of "
                    f"left_shift_def.cpp (no bool/float), got {name}.dtype={arr.dtype}"
                )
        if x.dtype != y.dtype:
            raise ValueError(
                f"left_shift kernel requires identical input dtypes "
                f"(left_shift_def.cpp pairs x/y dtypes), got x={x.dtype}, y={y.dtype}"
            )
        # numpy 式广播 == BroadcastBaseTiling 规则, torch 逐位一致
        return [_as_numpy(_left_shift(x, y))]

    class ThirdPartyImpl:
        """三方(GPU)标杆: kernel/geir(x/y 同 dtype, 8 整型), 设备侧直调竞品。

        kernel/geir 契约要求 x/y 同 dtype, 竞品结果 dtype 即 x.dtype, 无需再 cast。
        """

        def __call__(self, x, y, **kwargs):
            del kwargs
            return [_third_party_left_shift(x, y)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = _KERNEL_TOLERANCE


class LeftShiftAclnnSpec:
    """ACLNN: 签名随 aclnnLeftShiftGetWorkspaceSize(self, shiftBits, out) /
    aclnnLeftShiftsGetWorkspaceSize(self, shiftBits, out)。
    流水建模: PromoteType → Cast → LeftShift → Cast 入 out; 位移位宽 = PromoteType 结果
    的位宽 (非 self 位宽)。

    提升口径: 两条路径在 C++ 中都用同一个 ``op::PromoteType(self, shiftBits)``
    (aclnn_left_shift.cpp), scalar 版并未做 self 优先的弱标量特化 (对比 bitwise_and 的
    CombineCategoriesWithComplex)——即 scalar 被当作「其自身 dtype 的张量」参与提升。
    故 shiftBits 需带真实 dtype 传入 (numpy 标量 / 0 维 tensor); 若传入裸 Python int
    会被视作 int64。
    shiftBits 既可为 Tensor(aclnnLeftShift) 也可为 Scalar(aclnnLeftShifts)。"""

    @staticmethod
    def golden(self, shiftBits, out=None, **kwargs):
        del kwargs
        result = _left_shift(self, shiftBits, force_promote=True)
        if out is not None:
            return [_write_into_out(result, out)]
        return [_to_out_torch(result)]

    # 不声明 third_party：ACLNN 首参名为 ``self``，与 Python 保留名冲突——TTK 类形式
    # 绑定(bind_params)会丢弃池中名为 self 的条目，三方实现拿不到 self，无法正确计算；
    # 且本算子为整型、cross_check 只覆盖浮点，三方输出本就进不了判定，故 aclnn 通路不
    # 提供三方标杆（kernel/geir 与 e2e 已覆盖三方声明）。
    tolerance = _API_TOLERANCE


class LeftShiftE2eSpec:
    """E2E (torch.bitwise_left_shift): 契约即算子语义(torch.bitwise_left_shift), 真值同
    kernel/aclnn 共用 ``_left_shift``, 不叠加额外的数值口径; CPU 真值与设备侧(经
    torch_npu 派发到 aclnnLeftShift)的实现差异属被测对象。

    注意: torch 通路实际落到 aclnnLeftShift, 故类型提升仍受 aclnn 约束——位宽取
    op::PromoteType(互推导表)后的 dtype, 且 u16/u32/u64 异 dtype 组合会被拒绝
    (op::PromoteType 返回 DT_UNDEFINED)。此约束与 ``_left_shift`` 一致, 非 golden 额外叠加。
    """

    @staticmethod
    def golden(input, other, **kwargs):
        del kwargs
        return [_to_out_torch(_left_shift(input, other))]

    class ThirdPartyImpl:
        """三方(GPU)标杆: ``torch.bitwise_left_shift(input, other)`` 函数形式(e2e)。"""

        def __call__(self, input, other, **kwargs):
            del kwargs
            return [_third_party_left_shift(input, other)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = _API_TOLERANCE


class LeftShiftTfSpec:
    """E2E (tf) 通路：api_name ``tf.bitwise.left_shift`` —— 注意这是**驱动接口**，不是标杆。

    路由：TF 插件 framework/left_shift_tf_plugin.cpp（ImplyType::AI_CPU）→ GE 图
    → LeftShift(AI_CPU) 算子。输入 torch.Tensor，返回 torch.Tensor。

    精度标杆对标**算子语义（torch.bitwise_left_shift）**，而非 tf 接口：tf 只负责把
    NPU 驱动起来，期望输出始终是 LeftShift 算子的 torch 语义结果，故真值与
    kernel/aclnn 共用同一套 torch 标杆 ``_left_shift``（由此也天然覆盖负/越界等边界，
    与算子实现一致）。

    与 ``LeftShiftE2eSpec`` 的差异：torch 通路经 aclnn 走类型提升（op::PromoteType），
    而 tf 通路落到 AI_CPU LeftShift，x/y **必须同 dtype**（left_shift_aicpu.cpp 校验
    input0_type == input1_type，tf.bitwise.left_shift 亦要求同型），故此处按同 dtype 校验。
    """

    @staticmethod
    def golden(x, y, **kwargs):
        del kwargs
        for name, arr in (("x", x), ("y", y)):
            if _dtype_name(arr) not in _INT_DTYPE_STR:
                raise ValueError(
                    f"left_shift(AI_CPU) supports the 8 integer dtypes of "
                    f"left_shift_aicpu_def.cpp (no bool/float), got {name}.dtype={_dtype_name(arr)}"
                )
        if _dtype_name(x) != _dtype_name(y):
            raise ValueError(
                f"left_shift(AI_CPU) requires identical input dtypes "
                f"(left_shift_aicpu.cpp checks input0_type == input1_type), "
                f"got x={_dtype_name(x)}, y={_dtype_name(y)}"
            )
        return [_to_out_torch(_left_shift(x, y))]

    class ThirdPartyImpl:
        """三方(GPU)标杆: tf 驱动接口(x/y 同 dtype, 语义同 torch.bitwise_left_shift)。

        tf 只是驱动接口, 三方标杆仍取 torch 算子语义; x/y 同 dtype, 结果 dtype 即 x.dtype。
        """

        def __call__(self, x, y, **kwargs):
            del kwargs
            return [_third_party_left_shift(x, y)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = _API_TOLERANCE


# legacy 函数入口(TestSpec 迁移期保留); 参数名随对应 C/torch 签名, 按位置绑定,
# 统一返回列表(每输出一个元素).


def left_shift_golden(x, y, **kwargs):
    """legacy kernel golden (numpy 入参)."""
    return LeftShiftKernelSpec.golden(x, y, **kwargs)


def aclnn_left_shift_golden(self, shiftBits, out=None, **kwargs):
    """legacy aclnn golden: aclnnLeftShift (shiftBits 为 Tensor)."""
    return LeftShiftAclnnSpec.golden(self, shiftBits, out=out, **kwargs)


def aclnn_left_shifts_golden(self, shiftBits, out=None, **kwargs):
    """legacy aclnn golden: aclnnLeftShifts (shiftBits 为 Scalar)."""
    return LeftShiftAclnnSpec.golden(self, shiftBits, out=out, **kwargs)


def torch_left_shift_golden(input, other, **kwargs):
    """legacy e2e golden: torch.bitwise_left_shift."""
    return LeftShiftE2eSpec.golden(input, other, **kwargs)


def tf_left_shift_golden(x, y, **kwargs):
    """legacy e2e golden: tf.bitwise.left_shift."""
    return LeftShiftTfSpec.golden(x, y, **kwargs)
