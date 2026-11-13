#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""bernoulli golden + XPU 性能路由（kernel 单算子 + aclnn 标量/张量接口）

精度（__golden__）golden 来源由环境变量决定：
  * 未配置：CPU 本地 numpy 复刻随机流（无 GPU 依赖）
  * 配置 BERNOULLI_GOLDEN_HOST=<IP>（可选 PORT，默认 9999）：调用同目录
    bernoulli_remote_golden.py --serve 的 GPU 服务取真值（两方 bit 级对拍，配 --compare binary）

性能（__spec__）：third_party impl 类供 --xpu-perf 计时（__init__ 预分配，
__call__ 仅含 bernoulli_，device_us 只统计真实 device kernel）。

随机流语义（probMode 决定）：
  * probMode=1（标量）：交织映射 + u < p，T = 256·min(⌈n/256⌉, SM·8)；
    n 超出 32 位索引域（numel > 2³¹-1 或 1+(numel-1)·elemSize > 2³¹-1）时
    按竞品 TensorIterator 递归切分：每块独立 T_j，counter 起点级联
    （外层先按总 numel 预扣 CO_total，再逐块累加 co_j）
  * probMode=0（张量，缺省）：连续映射 + u <= p[li]，prob 按广播 stride 寻址，
    T = 512·⌈n/2048⌉（cap 2³¹−1 块；竞品 CUDA_tensor_apply2 单 kernel 不切分）
  * u = x·2⁻³² + 2⁻³³（(0,1]）；counter 起点 = ⌈offset/4⌉
"""

import os
import pickle
import socket
import struct

import numpy as np
import torch

__golden__ = {
    "kernel": {"stateless_bernoulli": "stateless_bernoulli_golden"},
    "aclnn": {
        "aclnnBernoulli": "aclnn_bernoulli_golden",
        "aclnnInplaceBernoulli": "aclnn_inplace_bernoulli_golden",
        "aclnnBernoulliTensor": "aclnn_bernoulli_tensor_golden",
        "aclnnInplaceBernoulliTensor": "aclnn_inplace_bernoulli_tensor_golden",
    },
    # e2e 通路（api_name = torch framework API 路径）：精度 golden 走本层注册，
    # 与 __spec__（三方/性能路由）无关。签名契约与 plan.build_args 装配结果一致：
    # (self, p, generator)：generator spec 按 ops-test-kit PR #578 原样传给 golden。
    "e2e": {
        # torch.ops.aten 具名 overload：p 为 float（标量）/ Tensor（张量）两种，
        # 由 e2e_bernoulli_golden 内按 p 类型自动分流。
        "torch.ops.aten.bernoulli_.float": "e2e_bernoulli_golden",
        "torch.ops.aten.bernoulli_.Tensor": "e2e_bernoulli_golden",
        # "torch.Tensor.bernoulli_": "e2e_bernoulli_golden",
    },
}

# XPU 性能采集路由（--xpu-perf / cross_check 的 third_party 适配）：
# kernel 通路 op_name 与 aclnn 张量 API 在竞品无同名 API，server 自动推导 miss，
# 须由 TestSpec.third_party 显式路由到 impl 类。impl 的 __init__ 预分配输出/预搬 prob/
# 预建 generator（不计时），__call__ 仅含被测 bernoulli_ —— device_us 只累加真实 device
# kernel 条目，prob 的 H2D 拷贝与 dtype cast 若留在 __call__ 会污染统计。
# 本注册不声明 golden 属性，golden 查找自动回落到上方 __golden__。
#
# e2e 的 generator 物化/逐轮复位由框架负责；算子侧只注册 golden，不再重设默认 RNG。
__spec__ = {
    "stateless_bernoulli": "StatelessBernoulliTestSpec",
    "aclnnBernoulli": "AclnnBernoulliTestSpec",
    "aclnnInplaceBernoulli": "AclnnInplaceBernoulliTestSpec",
    "aclnnBernoulliTensor": "AclnnBernoulliTensorTestSpec",
    "aclnnInplaceBernoulliTensor": "AclnnInplaceBernoulliTensorTestSpec",
}

_DT_ENUM = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    12: "bool",
    27: "bfloat16",
}
_TORCH_DT = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "bool": "bool",
    "uint8": "uint8",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
}


def _xpu_np(x):
    """torch tensor（含 device 侧）转 host numpy（bf16 经 fp32 承载——numpy 无 bfloat16）。"""
    import numpy as _np

    if hasattr(x, "detach"):
        t = x.detach().cpu()
        if t.dtype is torch.bfloat16:
            return t.float().numpy()
        return t.numpy()
    return _np.asarray(x)


def _seed_generator(gen, seed, offset):
    """按 (seed, offset) 设置 generator：manual_seed 归零 offset 后，CUDA
    generator 的 Philox offset 需写入 state（16 字节 = seed[0:8] +
    offset[8:16]，与 bernoulli_remote_golden 的全局状态 patch 同构）；CPU
    MT19937 无 offset 语义，跳过。offset 容忍标量/tensor/numpy（kernel 通路
    传 numpy 数组、aclnn 通路传 tensor 或标量）。"""
    offset_v = int(offset.item() if hasattr(offset, "item") else offset)
    gen.manual_seed(int(seed))
    if gen.device.type != "cuda":
        return
    st = gen.get_state()
    st[8:16] = torch.tensor(
        list(struct.pack("<Q", offset_v & 0xFFFFFFFFFFFFFFFF)),
        dtype=torch.uint8,
    )
    gen.set_state(st)


def _kernel_shape_and_prob(shape, prob, input_ori_shapes=None):
    """Recover logical scalars when TTK promotes () to a one-element array."""
    shape_arr = _xpu_np(shape)
    prob_arr = _xpu_np(prob)
    declared = input_ori_shapes or ()
    rank_zero_shape = shape_arr.ndim == 0 or (len(declared) > 0 and declared[0] == ())
    output_shape = [] if rank_zero_shape else [int(s) for s in shape_arr.reshape(-1)]
    if len(declared) > 1 and declared[1] == ():
        prob_arr = prob_arr.reshape(())
    return output_shape, prob_arr


def _check_writable_tensor(tensor):
    """CUDA assert_no_internal_overlap semantics, before any layout materialization.

    Non-zero-stride layouts with indeterminate overlap are not rejected here,
    matching PyTorch's MemOverlap::TooHard behavior. Empty tensors are no-ops.
    """
    if tensor is None or any(int(size) == 0 for size in tensor.shape):
        return
    strides = tensor.stride() if isinstance(tensor, torch.Tensor) else tensor.strides
    if any(
        int(size) > 1 and stride == 0 for size, stride in zip(tensor.shape, strides)
    ):
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location. Please clone() the tensor before "
            "performing the operation."
        )


def _reference_output(self_t, out, device):
    """Preserve a supplied output view instead of hiding it behind empty(shape)."""
    _check_writable_tensor(out)
    if isinstance(out, torch.Tensor):
        storage_size = out.untyped_storage().nbytes() // out.element_size()
        storage = torch.empty(storage_size, dtype=out.dtype, device=device)
        return storage.as_strided(out.shape, out.stride(), out.storage_offset())
    dtype = self_t.dtype if isinstance(self_t, torch.Tensor) else torch.float32
    return torch.empty(list(self_t.shape), dtype=dtype, device=device)


class BernoulliTorchImpl:
    """kernel 通路（参数 shape/prob/seed/offset/dtype）：prob 单元素走标量重载。"""

    def __init__(self, shape, prob, seed, offset, dtype=0, device=None, **kwargs):
        import numpy as _np
        import torch

        self.shape, prob_arr = _kernel_shape_and_prob(
            shape, prob, kwargs.get("input_ori_shapes")
        )
        self.seed = int(_xpu_np(seed).reshape(-1)[0])
        self.is_scalar = prob_arr.size == 1
        dt = _TORCH_DT.get(
            _DT_ENUM.get(int(dtype) if dtype is not None else 0, "float32"), "float32"
        )
        # 预分配（按目标 dtype，免 cast kernel）/ 预建 generator / prob 预上卡（免 H2D 计时）
        self.out = torch.empty(self.shape, device=device, dtype=getattr(torch, dt))
        self.gen = torch.Generator(device=device)
        _seed_generator(self.gen, self.seed, offset)
        if self.is_scalar:
            self.p = float(prob_arr.reshape(-1)[0])
            self.prob_t = None
        else:
            self.p = None
            # 保留 prob 原 shape: bernoulli_(p=) 要求 p 可广播到输出 shape, 展平后不可广播
            prob_t = torch.from_numpy(
                _np.ascontiguousarray(prob_arr, dtype=_np.float32)
            ).to(device)
            self.prob_t = prob_t.reshape(prob_arr.shape)

    def __call__(self, shape, prob, seed, offset, device=None, **kwargs):
        if self.is_scalar:
            self.out.bernoulli_(self.p, generator=self.gen)
        else:
            self.out.bernoulli_(p=self.prob_t, generator=self.gen)
        return [self.out]


class AclnnBernoulliImpl:
    """aclnn 标量 API（参数 self/prob/seed/offset）：self 为 shape/dtype 载体。"""

    def __init__(
        self,
        self_t=None,
        prob=None,
        seed=0,
        offset=0,
        out=None,
        *args,
        device=None,
        **kwargs,
    ):
        import torch

        # TTK delivers the API parameter named "self" via *args to avoid the
        # Python instance parameter of the same name. Direct callers use self_t.
        self_t = self_t if self_t is not None else args[0]
        self.shape = list(self_t.shape)
        self.seed = int(seed.item() if hasattr(seed, "item") else seed)
        self.p = float(prob.item() if hasattr(prob, "item") else prob)
        self.out = _reference_output(self_t, out, device)
        self.gen = torch.Generator(device=device)
        _seed_generator(self.gen, self.seed, offset)

    def __call__(
        self,
        self_t=None,
        prob=None,
        seed=0,
        offset=0,
        out=None,
        *args,
        device=None,
        **kwargs,
    ):
        self.out.bernoulli_(self.p, generator=self.gen)
        return [self.out]


class AclnnBernoulliTensorImpl:
    """aclnn 张量 API：prob 为概率张量（可广播），__init__ 预搬上卡。"""

    def __init__(
        self,
        self_t=None,
        prob=None,
        seed=0,
        offset=0,
        out=None,
        *args,
        device=None,
        **kwargs,
    ):
        import torch

        self_t = self_t if self_t is not None else args[0]
        self.shape = list(self_t.shape)
        self.seed = int(seed.item() if hasattr(seed, "item") else seed)
        self.out = _reference_output(self_t, out, device)
        self.gen = torch.Generator(device=device)
        _seed_generator(self.gen, self.seed, offset)
        prob_t = (
            prob if isinstance(prob, torch.Tensor) else torch.from_numpy(_xpu_np(prob))
        )
        if prob_t.dtype is not torch.float32:
            prob_t = prob_t.to(torch.float32)
        self.prob_t = prob_t.to(device)

    def __call__(
        self,
        self_t=None,
        prob=None,
        seed=0,
        offset=0,
        out=None,
        *args,
        device=None,
        **kwargs,
    ):
        self.out.bernoulli_(p=self.prob_t, generator=self.gen)
        return [self.out]


class AclnnInplaceBernoulliImpl(AclnnBernoulliImpl):
    def __init__(self, selfRef, prob, seed=0, offset=0, device=None, **kwargs):
        super().__init__(
            selfRef, prob, seed, offset, out=selfRef, device=device, **kwargs
        )

    def __call__(self, selfRef, prob, seed=0, offset=0, device=None, **kwargs):
        return super().__call__(selfRef, prob, seed, offset, device=device, **kwargs)


class AclnnInplaceBernoulliTensorImpl(AclnnBernoulliTensorImpl):
    def __init__(self, selfRef, prob, seed=0, offset=0, device=None, **kwargs):
        super().__init__(
            selfRef, prob, seed, offset, out=selfRef, device=device, **kwargs
        )

    def __call__(self, selfRef, prob, seed=0, offset=0, device=None, **kwargs):
        return super().__call__(selfRef, prob, seed, offset, device=device, **kwargs)


class StatelessBernoulliTestSpec:
    third_party = {"torch": BernoulliTorchImpl}


class AclnnBernoulliTestSpec:
    third_party = {"torch": AclnnBernoulliImpl}


class AclnnBernoulliTensorTestSpec:
    third_party = {"torch": AclnnBernoulliTensorImpl}


class AclnnInplaceBernoulliTestSpec:
    third_party = {"torch": AclnnInplaceBernoulliImpl}


class AclnnInplaceBernoulliTensorTestSpec:
    third_party = {"torch": AclnnInplaceBernoulliTensorImpl}


# ============================================================
# CPU 本地随机流复刻（numpy 向量化）
# ============================================================

_M4X32_A = np.uint64(0xD2511F53)
_M4X32_B = np.uint64(0xCD9E8D57)
_W32_A = np.uint64(0x9E3779B9)
_W32_B = np.uint64(0xBB67AE85)
_MASK32 = np.uint64(0xFFFFFFFF)
_R2P32INV = np.float32(2.3283064e-10)
_R2P32INV_HALF = np.float32(_R2P32INV / np.float32(2.0))

SM_COUNT = 78  # 对拍 GPU 的 SM 数（标量 T 封顶 = SM·8·256）
THREADS_PER_GROUP_SCALAR = 256
THREADS_PER_AIC = 2048
BLOCK_THREAD_TENSOR = 512
MAX_THREADS_PER_PROCESSOR = 2048
GPU_GRID_CAP = 2**31 - 1

# 标量语义 32 位索引域切分依赖输出元素宽度（字节）：竞品 can_use_32bit_indexing
# 检查 1+(numel-1)·stride_bytes ≤ 2³¹-1，不同 dtype 的切分边界不同
_ELEM_SIZE = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}


def _philox10(k0, k1, c0, c1, c2, c3):
    for _ in range(10):
        p0 = c0 * _M4X32_A
        p1 = c2 * _M4X32_B
        n0 = ((p1 >> np.uint64(32)) & _MASK32) ^ c1 ^ k0
        n1 = p1 & _MASK32
        n2 = ((p0 >> np.uint64(32)) & _MASK32) ^ c3 ^ k1
        n3 = p0 & _MASK32
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 = (k0 + _W32_A) & _MASK32
        k1 = (k1 + _W32_B) & _MASK32
    return c0, c1, c2, c3


def _words(seed, counters_lo, counters_hi):
    """对一组 counter (lo, hi) 执行 philox4x32-10，返回 4 个分量的 u 值。"""
    k0 = np.uint64(seed & 0xFFFFFFFF)
    k1 = np.uint64((seed >> 32) & 0xFFFFFFFF)
    w0, w1, w2, w3 = _philox10(
        k0,
        k1,
        counters_lo & _MASK32,
        counters_lo >> np.uint64(32),
        counters_hi & _MASK32,
        counters_hi >> np.uint64(32),
    )
    return np.stack(
        [w.astype(np.float32) * _R2P32INV + _R2P32INV_HALF for w in (w0, w1, w2, w3)],
        axis=1,
    )


def _scalar_T(numel):
    """标量语义单次 kernel 的虚拟线程数 T = 256·min(⌈n/256⌉, SM·8)。"""
    return THREADS_PER_GROUP_SCALAR * min(
        (numel + 255) // 256, SM_COUNT * (THREADS_PER_AIC // 256)
    )


def _scalar_counter_offset(numel):
    """单次 kernel 启动消耗的 philox offset（element 单位）= ⌈n/(T·4)⌉·4。

    竞品 calc_execution_policy 的 max_generator_offsets_per_curand_call=4 同式。
    """
    t = _scalar_T(numel)
    return ((numel + t * 4 - 1) // (t * 4)) * 4


def _split_blocks(n, elem_size):
    """32 位索引域切分，完全对标竞品 TensorIterator（连续输出 coalesce 后等价 1-D 二分）。

    - 可索引判定: numel ≤ 2³¹-1 且 1+(numel-1)·elem_size ≤ 2³¹-1
      （can_use_32bit_indexing / CANN Is32bitIndexable 同式）
    - 切分: 对半分、前半(floor)先处理（split 的 copy_size=shape/2、SplitUntil32Bit
      先左后右，CANN PartitionDim 同序）
    - 无块数上限（对标 torch 递归；CANN tiling 的 MAX_SPLIT_BLOCKS=8 是其自身
      约束，golden 不复刻该上限）
    返回线性有序的 [(lo, hi), ...]。
    """
    limit = 2**31 - 1

    def indexable(size):
        return size <= limit and 1 + (size - 1) * elem_size <= limit

    if indexable(n):
        return [(0, n)]
    blocks = []
    stack = [(0, n)]
    while stack:
        lo, hi = stack.pop()
        size = hi - lo
        if indexable(size):
            blocks.append((lo, hi))
            continue
        mid = lo + size // 2  # 前半 floor，与竞品 split 的 copy_size = shape/2 一致
        stack.append((mid, hi))  # 后半（ceil）后处理
        stack.append((lo, mid))  # 前半先处理
    return blocks


def _scalar_block_offsets(blocks, offset):
    """块级 counter 起点级联，对标竞品递归子调用的 generator fetch 顺序。

    外层调用在递归前先按总 numel 预扣 CO_total（philox_cuda_state fetch），
    子调用 j 以当前累计 offset 启动、再按块内 numel 递增 co_j
    （与 CANN CalcExecutionPoliciesForBlocks 同式）。
    """
    acc = offset
    if len(blocks) > 1:
        acc += _scalar_counter_offset(blocks[-1][1])  # 总 numel（块覆盖全量）
    offsets = []
    for lo, hi in blocks:
        offsets.append(acc)
        acc += _scalar_counter_offset(hi - lo)
    return offsets


def _scalar_stream(seed, offset, n, elem_size=4):
    """标量语义：交织映射 li = thread + call·T·4 + comp·T，返回每元素 u。

    n 超出 32 位索引域时按竞品 TensorIterator 递归切分：每块独立 T_j 与
    counter 起点（kernelOffset 级联），块内交织映射不变。n ≤ 单块域时与
    单公式逐 bit 一致。
    """
    blocks = _split_blocks(n, elem_size)
    if len(blocks) == 1:
        t = _scalar_T(n)
        li = np.arange(n, dtype=np.uint64)
        u_all = _words(
            seed,
            np.uint64((offset + 3) // 4) + li // np.uint64(t * 4),
            li % np.uint64(t),
        )
        comp = ((li // np.uint64(t)) % np.uint64(4)).astype(np.int64)
        return u_all[np.arange(n), comp]
    out = np.empty(n, dtype=np.float32)
    for (lo, hi), koff in zip(blocks, _scalar_block_offsets(blocks, offset)):
        size = hi - lo
        t = _scalar_T(size)
        li = np.arange(size, dtype=np.uint64)
        u_all = _words(
            seed, np.uint64((koff + 3) // 4) + li // np.uint64(t * 4), li % np.uint64(t)
        )
        comp = ((li // np.uint64(t)) % np.uint64(4)).astype(np.int64)
        out[lo:hi] = u_all[np.arange(size), comp]
    return out


def _tensor_stream(seed, offset, n):
    """张量语义：连续映射 g=li//4, j=li%4，返回每元素 u。

    counter 的 call 分量恒 0——完全复刻竞品 functor 每组重新 curand_init 的复位
    行为（grid 封顶 n > ~4.4e12 后同线程多组复用首组随机数；未封顶域 g < T 时
    call 本就为 0，行为不变）。"""
    blocks = min(
        (n + MAX_THREADS_PER_PROCESSOR - 1) // MAX_THREADS_PER_PROCESSOR, GPU_GRID_CAP
    )
    T = max(blocks, 1) * BLOCK_THREAD_TENSOR
    li = np.arange(n, dtype=np.uint64)
    g = li // np.uint64(4)
    u_all = _words(seed, np.uint64((offset + 3) // 4), g % np.uint64(T))  # call 恒 0
    return u_all[np.arange(n), (li % np.uint64(4)).astype(np.int64)]


def _broadcast_strides(output_shape, prob_shape):
    """广播 strides（与 tiling 同式：prob 维为 1 → stride=0，连续维连乘）。"""
    if len(prob_shape) > len(output_shape):
        raise ValueError("prob rank must not exceed output rank")
    strides = [0] * len(output_shape)
    prob_stride = 1
    prob_idx = len(prob_shape) - 1
    for i in range(len(output_shape) - 1, -1, -1):
        if prob_idx < 0:
            continue
        prob_dim = int(prob_shape[prob_idx])
        if prob_dim == 1 and output_shape[i] != 1:
            strides[i] = 0  # 广播维
        elif prob_dim == output_shape[i]:
            strides[i] = prob_stride
            prob_stride *= prob_dim
        else:
            raise ValueError(
                f"prob {list(prob_shape)} not broadcastable to {list(output_shape)}"
            )
        prob_idx -= 1
    return strides


def _prob_offset(li, output_shape, strides):
    """多维索引到 prob 偏移（与 kernel ProbIndexToOffset 同式）。"""
    li = np.asarray(li, dtype=np.int64)
    offset = np.zeros_like(li)
    if not strides:  # 0 维输出：唯一元素偏移恒 0（恒指 prob 首元素）
        return offset
    for i in range(len(output_shape) - 1, 0, -1):
        offset += (li % int(output_shape[i])) * strides[i]
        li = li // int(output_shape[i])
    return offset + li * strides[0]


# ============================================================
# golden 公共实现（标量/张量语义，返回 bool 数组）
# ============================================================

_NP_DT = {
    "float32": "float32",
    "float16": "float16",
    "uint16": "uint16",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
    "uint32": "uint32",
    "uint64": "uint64",
}

# ============================================================
# GPU 远程 golden（配置 BERNOULLI_GOLDEN_HOST 启用，两方对拍）
# ============================================================

_HOST = os.environ.get("BERNOULLI_GOLDEN_HOST", "")
_PORT = int(os.environ.get("BERNOULLI_GOLDEN_PORT", "9999"))
_SM_CHECKED = False
# uint16/32/64：远程服务以 float32 代跑后 numpy cast 到目标位宽（bernoulli 输出
# 仅 0/1、随机值域与输出 dtype 无关），全 dtype 均可两方 bit 级对拍。
# 注意：n 超 32 位索引域后标量语义切分边界依赖输出元素宽度，float32(4B) 代跑
# 与 uint16(2B)/uint64(8B) 的切块不同——超界超大 shape 的标量对拍以本地 CPU
# 复刻（按真实 elem 宽度切分）为准，远程代跑仅覆盖 ≤2GB 输出的常规域。


def _use_remote():
    return bool(_HOST)


def _rpc(request):
    """长度前缀 + pickle 协议（与 bernoulli_remote_golden.py --serve 一致）。"""
    global _SM_CHECKED
    with socket.create_connection((_HOST, _PORT), timeout=120) as sock:
        sock.settimeout(600)

        def send(obj):
            msg = pickle.dumps(obj)
            sock.sendall(struct.pack(">I", len(msg)) + msg)

        def recv():
            raw = b""
            while len(raw) < 4:
                chunk = sock.recv(4 - len(raw))
                if not chunk:
                    return None
                raw += chunk
            (ln,) = struct.unpack(">I", raw)
            data = b""
            while len(data) < ln:
                chunk = sock.recv(ln - len(data))
                if not chunk:
                    return None
                data += chunk
            return pickle.loads(data)

        if not _SM_CHECKED:  # 首次校验对拍 GPU 的 SM 数（标量 T 封顶前提）
            send({"cmd": "hello"})
            info = recv()
            if not info.get("ok") or info.get("sm") != SM_COUNT:
                raise RuntimeError(
                    f"remote GPU SM={info.get('sm')} != {SM_COUNT}, bit 级对齐不成立"
                )
            _SM_CHECKED = True
        send({"cmd": "golden", **request})
        return recv()


def _remote_run(
    mode, output_shape, seed, offset, out_dtype, p=None, prob_flat=None, prob_shape=None
):
    req = {
        "mode": mode,
        "shape": output_shape,
        "seed": seed,
        "offset": offset,
        "dtype": out_dtype,
    }
    if mode == "scalar":
        req["p"] = float(p)
    else:
        req["prob_data"] = prob_flat.astype(np.float32).tobytes()
        req["prob_shape"] = list(prob_shape)
        req["prob_dtype"] = "float32"
    resp = _rpc(req)
    if not resp.get("ok"):
        raise RuntimeError(f"remote golden failed: {resp.get('error')}")
    raw, dt_str = resp["data"], resp["dtype"]
    if dt_str == "bfloat16":  # bf16 以 uint16 位视图传输，按 float32 高 16 位重解释
        return (
            np.frombuffer(bytearray(raw), dtype="<u2").astype(np.uint32)
            << np.uint32(16)
        ).view(np.float32)
    return np.frombuffer(bytearray(raw), dtype=_NP_DT.get(dt_str, "float32"))


def _scalar_of(v):
    return v.item() if hasattr(v, "item") else v


def _to_u64(v):
    return int(np.uint64(np.int64(int(v))))


def _scalar_bool(seed, offset, n, p, elem_size=4):
    return _scalar_stream(seed, offset, n, elem_size) < np.float32(p)


def _tensor_bool(seed, offset, n, output_shape, prob_flat, prob_shape):
    strides = _broadcast_strides(output_shape, prob_shape)
    p_at = prob_flat.astype(np.float32)[
        _prob_offset(np.arange(n, dtype=np.int64), output_shape, strides)
    ]
    return _tensor_stream(seed, offset, n) <= p_at


def _golden_bool(
    mode,
    seed,
    offset,
    n,
    output_shape,
    out_dtype,
    p=None,
    prob_flat=None,
    prob_shape=None,
):
    """统一入口：配置了 GPU 环境取远程真值，否则 CPU 复刻。返回 bool 数组。

    uint 系 dtype 的远程对拍宽度对齐设计（重要，勿回退为统一 float32 代跑）：
      * 竞品 torch 无 uint16/32/64 dtype，须以同字节宽度 dtype 代跑：uint16→float16、
        uint32→float32、uint64→int64（服务端按宽度代理）。超界(n·elemSize > 2^31-1)
        时 TensorIterator 按字节体积切分，宽度不一致会导致 GPU 与 NPU 的分块数/级联
        offset 全部错位，随机流全域不同（p=1.0 等边界值会掩盖成仅数十个临界位差异，
        极易误判为舍入噪声；p 中间值则大规模失败）。
      * 代跑 dtype 的随机流与整型输出完全同构：非 double 输出的 accscalar 恒为
        float（DiscreteDistributionType 仅 double 特化为 double，int64/fp16 均走
        float 路径），curand_uniform4/unroll=4/calc_execution_policy 均与输出
        dtype 无关，仅最后 static_cast 目标不同，0/1 值域经 numpy astype 无损。
        float64 不可代跑（accscalar 为 double：2^-53 缩放、curand_uniform2_double、
        unroll=2，流语义不同）；uint64 由 int64(8B 同宽, float accscalar) 承接。
      * bfloat16(2B) 远程以真实 dtype 直跑（非代跑），不涉及此问题。
    """
    if _use_remote():
        raw = _remote_run(
            mode,
            output_shape,
            seed,
            offset,
            out_dtype,
            p=p,
            prob_flat=prob_flat,
            prob_shape=prob_shape,
        )
        return raw != 0
    if mode == "scalar":
        return _scalar_bool(seed, offset, n, p, _ELEM_SIZE.get(out_dtype, 4))
    return _tensor_bool(seed, offset, n, output_shape, prob_flat, prob_shape)


def _scalar_p(prob_arr):
    """标量 p：fp16 概率先舍入到 fp16 再转 fp32（与 aclnn AllocScalar 路径一致）。"""
    p = prob_arr.reshape(-1)[0]
    return np.float32(np.float16(p)) if prob_arr.dtype == np.float16 else np.float32(p)


# ============================================================
# kernel 层 golden（参数与算子 def 一致，输入 numpy.ndarray）
# ============================================================


def stateless_bernoulli_golden(
    shape, prob, seed, offset, dtype=0, probMode=0, **kwargs
):
    output_shape, prob_arr = _kernel_shape_and_prob(
        shape, prob, kwargs.get("input_ori_shapes")
    )
    n = int(np.prod(output_shape)) if output_shape else 1
    out_dtype = (kwargs.get("output_dtypes") or ["float32"])[0]
    np_dt = np.float32 if out_dtype == "bfloat16" else _NP_DT.get(out_dtype, np.float32)
    if n == 0:
        # Preserve the declared empty shape instead of collapsing to (0,).
        return np.zeros(list(output_shape), dtype=np_dt)

    mode = "scalar" if int(probMode or 0) == 1 else "tensor"
    if mode == "scalar":
        out = _golden_bool(
            mode,
            _to_u64(np.asarray(seed).reshape(-1)[0]),
            _to_u64(np.asarray(offset).reshape(-1)[0]),
            n,
            output_shape,
            out_dtype,
            p=_scalar_p(prob_arr),
        )
    else:
        out = _golden_bool(
            mode,
            _to_u64(np.asarray(seed).reshape(-1)[0]),
            _to_u64(np.asarray(offset).reshape(-1)[0]),
            n,
            output_shape,
            out_dtype,
            prob_flat=prob_arr.reshape(-1),
            prob_shape=prob_arr.shape,
        )
    return out.astype(np_dt).reshape(output_shape)


# ============================================================
# aclnn 层 golden（参数与 GetWorkspaceSize 一致，输入 torch.Tensor；
# 非 inplace 检查显式 out；inplace 注册到独立 wrapper，检查 self 写目标）
# ============================================================


def aclnn_bernoulli_golden(
    self, prob, seed=0, offset=0, out=None, probMode=None, **kwargs
):
    """aclnnBernoulli / aclnnInplaceBernoulli（标量语义，probMode=1 缺省）。"""
    seed_v = _to_u64(_scalar_of(seed))
    offset_v = _to_u64(_scalar_of(offset))
    p = float(_scalar_of(prob))
    dtype_t = out.dtype if out is not None else self.dtype
    shape = list(self.shape)
    n = int(np.prod(shape)) if shape else 1
    if n == 0:
        return torch.empty(shape, dtype=dtype_t)

    _check_writable_tensor(out)

    # probMode=0 显式覆盖为张量语义（单值张量广播）
    if probMode == 0:
        out = _golden_bool(
            "tensor",
            seed_v,
            offset_v,
            n,
            shape,
            str(dtype_t).replace("torch.", ""),
            prob_flat=np.array([p], dtype=np.float32),
            prob_shape=[1],
        )
    else:
        out = _golden_bool(
            "scalar",
            seed_v,
            offset_v,
            n,
            shape,
            str(dtype_t).replace("torch.", ""),
            p=p,
        )
    return torch.from_numpy(out).to(dtype_t).reshape(shape)


def aclnn_bernoulli_tensor_golden(
    self, prob, seed=0, offset=0, out=None, probMode=None, **kwargs
):
    """aclnnBernoulliTensor / aclnnInplaceBernoulliTensor（张量语义，probMode=0 缺省，prob 广播）。"""
    seed_v = _to_u64(_scalar_of(seed))
    offset_v = _to_u64(_scalar_of(offset))
    dtype_t = out.dtype if out is not None else self.dtype
    shape = list(self.shape)
    n = int(np.prod(shape)) if shape else 1
    if n == 0:
        return torch.empty(shape, dtype=dtype_t)

    _check_writable_tensor(out)

    prob_t = prob.detach().cpu() if hasattr(prob, "detach") else prob
    prob_np = (
        prob_t.float().numpy() if prob_t.dtype is torch.bfloat16 else prob_t.numpy()
    )
    prob_flat = prob_np.reshape(-1).astype(np.float32)

    # probMode=1 显式覆盖为标量语义（取 prob 首元素）
    if probMode == 1:
        out = _golden_bool(
            "scalar",
            seed_v,
            offset_v,
            n,
            shape,
            str(dtype_t).replace("torch.", ""),
            p=prob_flat[0],
        )
    else:
        out = _golden_bool(
            "tensor",
            seed_v,
            offset_v,
            n,
            shape,
            str(dtype_t).replace("torch.", ""),
            prob_flat=prob_flat,
            prob_shape=prob_np.shape,
        )
    return torch.from_numpy(out).to(dtype_t).reshape(shape)


def aclnn_inplace_bernoulli_golden(self, prob, seed=0, offset=0, **kwargs):
    return aclnn_bernoulli_golden(self, prob, seed, offset, out=self, **kwargs)


def aclnn_inplace_bernoulli_tensor_golden(self, prob, seed=0, offset=0, **kwargs):
    return aclnn_bernoulli_tensor_golden(self, prob, seed, offset, out=self, **kwargs)


# ============================================================
# e2e 层 golden（torch.Tensor.bernoulli_）
# 签名与 plan.build_args 装配结果对齐：(self, p=..., generator=None)；
# seed/offset 仅从 generator spec 读取；框架负责设备侧物化与逐轮复位。
# p 为 float → 标量语义；p 为 Tensor（广播）→ 张量语义。
# ============================================================


def _e2e_generator_seed_offset(generator):
    """Consume merged PR #578's explicit dict/GeneratorSpec without RNG side effects."""
    if not isinstance(generator, dict):
        raise ValueError(
            "Bernoulli E2E golden requires a generator spec: "
            "attributes={'generator': {'seed': 42, 'offset': 4}}; "
            "use ops-test-kit with PR #578 support, not a materialized Generator."
        )
    unknown = set(generator) - {"seed", "offset"}
    seed = generator.get("seed")
    offset = generator.get("offset", 0)
    if (
        unknown
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not -(2**63) <= seed < 2**64
    ):
        raise ValueError(
            "GENERATOR_SPEC_INVALID: expected seed int in [-2^63, 2^64), keys {seed, offset}"
        )
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or offset % 4
    ):
        raise ValueError(
            "GENERATOR_SPEC_INVALID: offset must be a non-negative int multiple of 4"
        )
    # torch.Generator accepts the full uint64 seed domain and remaps negative seeds.
    # Do not narrow through np.int64: seeds >= 2^63 are valid under PR #578.
    return seed & 0xFFFFFFFFFFFFFFFF, offset


def e2e_bernoulli_golden(self, p=None, generator=None, **kwargs):
    _check_writable_tensor(self)
    if "seed" in kwargs or "offset" in kwargs:
        raise ValueError(
            "Move E2E seed/offset into attributes.generator; top-level seed/offset are no longer supported"
        )
    seed_v, offset_v = _e2e_generator_seed_offset(generator)
    shape = list(self.shape)
    n = int(np.prod(shape)) if shape else 1
    dtype_t = self.dtype
    dt = str(dtype_t).replace("torch.", "")
    if n == 0:
        return torch.empty(shape, dtype=dtype_t)

    if isinstance(p, torch.Tensor):
        prob_t = p.detach().cpu()
        prob_np = (
            prob_t.float().numpy() if prob_t.dtype is torch.bfloat16 else prob_t.numpy()
        )
        prob_flat = prob_np.reshape(-1).astype(np.float32)
        out = _golden_bool(
            "tensor",
            seed_v,
            offset_v,
            n,
            shape,
            dt,
            prob_flat=prob_flat,
            prob_shape=prob_np.shape,
        )
    else:
        pv = float(_scalar_of(p)) if p is not None else 0.5
        out = _golden_bool("scalar", seed_v, offset_v, n, shape, dt, p=pv)
    return torch.from_numpy(out).to(dtype_t).reshape(shape)
