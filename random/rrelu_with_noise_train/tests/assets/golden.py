#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# TTK golden for rrelu_with_noise — 三方参考实现 rrelu_with_noise 的对拍真值。
#
# 真值来源: 三方参考接口 rrelu_with_noise(x, noise, lower, upper, training=True) 直跑
#   返回 y, noise 为传入 buffer 原地填充; training 语义与 CANN kernel 对齐:
#     x <= 0 -> y = x * r, noise = r (x==0 亦走随机斜率)
#     x > 0  -> y = x,      noise = 1 (NaN 比较恒 False, 走正分支)
#
# 随机流对齐(三方直跑真值路径):
#   * 三方 kernel: philox (seed, 全局线程号, offset) + grid-stride + unroll=4
#     交错取数, 与 CANN kernel 的 PhiloxSimtKernelDiscontinuous 流布局一致
#   * rng state 按 seed + ceil(offset/4)*4 注入(三方 state offset 以单个随机数计,
#     CANN kernel 基址为 uint64 域 (offset+3)/4 floor, 负 offset 在加法处翻转,
#     含负 4 倍数在内全 offset 域基址与三方一致)
#   * 三方 grid 封顶 = SM * (2048/256), 仅 SM 数与 CANN tiling 的
#     AIC_CLUSTER_COUNT=78 一致时流布局相同, 其他环境流布局不同(SM 校验失败直接报错)
#
# 值链(独立标杆, 按算子定义的数学本义实现):
#   流布局(counter 映射/Philox)精确复刻——同 seed/offset 产出同随机数是算子的
#   可观测契约; 值链独立计算(f64 域乘加, 不镜像 kernel 的 f32 实现), 单次舍
#   到存储 dtype。kernel 的 f32 近似与真值的残差由 ST 容差兜住; 若 kernel
#   值链有 bug(乘加次序/量化次序/边界错误), golden 不会跟着错, ST 能发现。
#
# 兜底: 无三方直跑环境时退化为本 numpy 独立实现(流精确 + 值链 f64)与随机流;
#   三方语义对齐由直跑真值路径(cross_check)承担; 三方路径绝不复用被测 kernel 任何产物。
#
# 消费路径:
#   1. kernel 模式直挂: ttk kernel --plugin golden.py
#      (走 __spec__["rrelu_with_noise_train"] -> RreluWithNoiseKernelTestSpec)
#   2. 仓库用例集: tests/st/arch35/ttk_kernel_rrelu_with_noise_train_st.csv
#      (走 __golden__["kernel"]["rrelu_with_noise_train"] -> rrelu_with_noise_train_golden;
#       输入按 def 序: x/seed/offset 位置传参, lower/upper 属性按名传参)
# ----------------------------------------------------------------------------
import logging
import struct

import numpy as np

__golden__ = {
    "kernel": {"rrelu_with_noise_train": "rrelu_with_noise_train_golden"},
    "aclnn": {"aclnnRReluWithNoise": "aclnn_rrelu_with_noise_golden"},
}

__spec__ = {
    "rrelu_with_noise_train": "RreluWithNoiseKernelTestSpec",
}

# ---- 与 random_tiling_arch35.h 常量对齐(线程布局封顶, 须与三方对拍环境的 SM 数一致) ----
AIC_CLUSTER_COUNT = 78
SM_COUNT = AIC_CLUSTER_COUNT


def _scalar(value, cast, fallback):
    if value is None:
        return cast(fallback)
    if hasattr(value, "detach"):
        # 标量可能按 scalar_dtypes 物化为张量(bf16 直接 np.asarray 会报
        # "Got unsupported ScalarType BFloat16"), 经 _xpu_np 位视图安全转 numpy
        value = _xpu_np(value)
    return cast(np.asarray(value).reshape(-1)[0])


def _align_offset(offset):
    # CANN kernel 的 ceil(offset/4) 代际语义 -> 三方 state offset(单个随机数粒度)
    return ((int(offset) + 3) // 4) * 4


def _get_bf16():
    try:
        from ml_dtypes import bfloat16 as bf16
    except ImportError:
        raise RuntimeError("bfloat16 golden 需要 ml_dtypes")
    return bf16


def _xpu_np(x):
    """torch tensor（含 device 侧）转 host numpy（bf16 以 ml_dtypes 承载）"""
    if hasattr(x, "detach"):
        import torch

        t = x.detach().cpu()
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().view(_get_bf16())
        return t.numpy()
    return np.asarray(x)


def _np_to_torch(arr):
    """host numpy(含 ml_dtypes bf16) -> torch tensor"""
    import torch

    arr = np.ascontiguousarray(arr)
    if arr.dtype.name == "bfloat16":
        return (
            torch.frombuffer(
                bytearray(arr.view(np.uint16).tobytes()), dtype=torch.int16
            )
            .view(torch.bfloat16)
            .reshape(arr.shape)
        )
    return torch.from_numpy(arr)


def _torch_to_np(t):
    """torch tensor -> host numpy（bf16 经 int16 位视图）"""
    import torch

    tc = t.detach().cpu()
    if tc.dtype == torch.bfloat16:
        return tc.view(torch.int16).numpy().view(_get_bf16())
    return tc.numpy()


def _torch_h20_ready():
    """三方直跑真值仅在 78 SM 环境成立: 三方 grid 封顶 = SM*(2048/256),
    非 78 SM 时流布局与 CANN tiling 的 AIC_CLUSTER_COUNT=78 不一致, 三方输出不可用"""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).multi_processor_count == SM_COUNT


def _rrelu_with_noise_train_torch(x, lower, upper, seed, offset):
    """三方参考实现 rrelu_with_noise(training=True) 直跑（三方 philox 流）

    rng state 按 seed + 对齐 offset 注入;
    返回 (y, noise) 两个 torch tensor。
    """
    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # SM 校验: 三方 grid 封顶 = SM*(2048/256), 仅 78 SM 与 CANN tiling 一致
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    if sm != SM_COUNT:
        raise RuntimeError(
            "third-party SM=%d != %d: 三方实现的 grid 封顶与 CANN tiling 的 "
            "AIC_CLUSTER_COUNT=78 流布局不一致, 随机流无法对齐" % (sm, SM_COUNT)
        )

    torch.cuda.manual_seed(int(seed))
    st = torch.cuda.get_rng_state().clone()
    st[8:16] = torch.tensor(
        list(struct.pack("<Q", _align_offset(offset) & 0xFFFFFFFFFFFFFFFF)),
        dtype=torch.uint8,
    )
    torch.cuda.set_rng_state(st)

    x_t = _np_to_torch(_xpu_np(x)).to(device)
    # 走三方原生入口, noise 为传入 buffer, 由算子原地填充
    # (x<=0 处为随机斜率, x>0 处为 1)
    noise = torch.empty_like(x_t)
    y = torch.ops.aten.rrelu_with_noise(x_t, noise, float(lower), float(upper), True)
    return y, noise


def rrelu_with_noise_train_golden(
    x, seed, offset, lower=0.125, upper=0.333333, **kwargs
):
    """
    Kernel golden for rrelu_with_noise_train.
    All the parameters follow @rrelu_with_noise_train_def.cpp without outputs.
    x/seed/offset 为输入(按 def 顺序); seed/offset 为 INT64 张量(各 1 元素),
    值经 _scalar 提取; lower/upper 为属性(按名传入)。
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    x_arr = _xpu_np(x)
    seed_v = _scalar(seed, int, 0)
    offset_v = _scalar(offset, int, 0)
    lower_v = _scalar(lower, float, 0.125)
    upper_v = _scalar(upper, float, 0.333333)

    dtype = x_arr.dtype
    shape = x_arr.shape
    numel = int(x_arr.size)
    if numel == 0:
        return [np.empty(shape, dtype=dtype), np.empty(shape, dtype=dtype)]

    # 三方直跑(独立真值)
    # 三方直跑真值仅在 78 SM 环境成立; 非 78 SM 环境的三方流布局与 kernel
    # 不一致(SM 校验), 此时落 numpy 复刻(kernel 值链, f32 域); 三方模式下的
    # ThirdPartyImpl 不做兜底, 非 78 SM 直接报错(比对无意义)
    if _torch_h20_ready():
        y_t, noise_t = _rrelu_with_noise_train_torch(
            x_arr, lower_v, upper_v, seed_v, offset_v
        )
        return [_torch_to_np(y_t), _torch_to_np(noise_t)]

    # 兜底: 无三方直跑环境或 SM != 78, numpy 复刻 kernel 值链
    # (f32 域乘加 + f32 边界, 与 kernel 逐步同构; 随机流镜像 kernel 布局)
    logging.info(
        "rrelu_with_noise_train golden: third-party direct run unavailable (no device or SM != 78), fallback to numpy replica"
    )
    return _rrelu_with_noise_train_numpy(x_arr, lower_v, upper_v, seed_v, offset_v)


# ============================================================
# numpy 复刻 kernel 值链（无三方直跑环境的兜底, 随机流与值链均与 kernel
# 对齐: 流布局镜像 kernel, 值链 f32 域乘加 + f32 边界）
# ============================================================

# ---- 与 random_tiling_arch35.h / random_kernel_base.h 常量对齐 ----
SIMT_THREAD_GROUP_SIZE = 256
MAX_THREADS_PER_AIC = 2048
GRID_CLAMP = AIC_CLUSTER_COUNT * (MAX_THREADS_PER_AIC // SIMT_THREAD_GROUP_SIZE)
UNROLL_FACTOR = 4
INDEX_32BIT_LIMIT = 2147483647
# random_unified_tiling_data_arch35.h: constexpr uint32_t MAX_SPLIT_BLOCKS = 8
MAX_SPLIT_BLOCKS = 8
# random_tiling_arch35.h: constexpr uint32_t MAX_PRNG_COUNTER_INCR = 4
MAX_PRNG_COUNTER_INCR = 4

PHILOX_M4X32_A = np.uint64(0xD2511F53)
PHILOX_M4X32_B = np.uint64(0xCD9E8D57)
PHILOX_W32_A = np.uint64(0x9E3779B9)
PHILOX_W32_B = np.uint64(0xBB67AE85)
MASK32 = np.uint64(0xFFFFFFFF)
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
RAND_2POW32_INV = np.float32(2.3283064e-10)  # float32 下恰等于 2^-32
RAND_2POW32_INV_HALF = np.float32(RAND_2POW32_INV / np.float32(2.0))

CHUNK_SIZE = 1 << 22  # 分块生成, 避免超大 shape 下计数器数组占用过多内存


def _philox_counter_base(offset):
    """镜像 PhiloxAlgParsInit 的基址表达式 (offset + VEC_4 - 1) / VEC_4 的真实语义。

    VEC_4 为 uint64_t(random_kernel_base.h:47), int64 offset 在 `offset + VEC_4`
    处即按 usual arithmetic conversions 提升为 uint64(负数在加法处翻转),
    除法为无符号 floor——负 offset 基址按 4 段量化落在 2^62 层
    (-4..-7 -> 2^62-1, -8..-11 -> 2^62-2, 依此类推)。
    """
    off = int(offset) & 0xFFFFFFFFFFFFFFFF
    return np.uint64(((off + UNROLL_FACTOR - 1) & 0xFFFFFFFFFFFFFFFF) // UNROLL_FACTOR)


def _calc_total_threads(numel):
    # 镜像 tiling 侧 CalcExecutionPoliciesForBlocks 的 grid/totalThreads 计算
    grid = (numel + SIMT_THREAD_GROUP_SIZE - 1) // SIMT_THREAD_GROUP_SIZE
    grid = max(grid, 1)
    grid = min(grid, GRID_CLAMP)
    return grid * SIMT_THREAD_GROUP_SIZE


def _philox_10_rounds(c0, c1, c2, c3, k0, k1):
    """向量化 Philox 4x32-10 轮, 与 kernel 侧 PhiloxRandomSimt 逐轮对齐

    c0~c3/k0~k1 均为 uint64(低 32 位有效), 返回 (r0, r1, r2, r3)
    """
    for _ in range(9):
        p0 = PHILOX_M4X32_A * c0
        p1 = PHILOX_M4X32_B * c2
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)) ^ c1 ^ k0,
            p1 & MASK32,
            (p0 >> np.uint64(32)) ^ c3 ^ k1,
            p0 & MASK32,
        )
        k0 = (k0 + PHILOX_W32_A) & MASK32
        k1 = (k1 + PHILOX_W32_B) & MASK32
    p0 = PHILOX_M4X32_A * c0
    p1 = PHILOX_M4X32_B * c2
    return (
        (p1 >> np.uint64(32)) ^ c1 ^ k0,
        p1 & MASK32,
        (p0 >> np.uint64(32)) ^ c3 ^ k1,
        p0 & MASK32,
    )


def _gen_u32(numel, seed, offset):
    """按 kernel 随机流布局生成每元素命中的 philox u32(与数据无关, 每元素一个随机数)

    流布局对标 PhiloxSimtKernelDiscontinuous(unroll=4 交错取数):
      key = [seed_lo, seed_hi]
      counter 基址 = uint64 域 (offset + 3) / 4 的 floor   (PhiloxAlgParsInit,
                    VEC_4 为 uint64_t, 负 offset 在加法处即翻转, 见 _philox_counter_base)
      元素 li -> linearIndex = li % totalThreads
                 groupIdx   = li // totalThreads
                 loopIdx    = groupIdx // 4             (SkipLo 累加, 进位入高 64 位)
                 iStep      = groupIdx % 4              (取 philox 输出第 iStep 列)
                 counter    = 基址 + SkipHi(linearIndex)(高 64 位累加)
    """
    total_threads = _calc_total_threads(numel)
    off64 = _philox_counter_base(offset)
    k0 = np.uint64(seed & 0xFFFFFFFF)
    k1 = np.uint64((seed >> 32) & 0xFFFFFFFF)
    tt64 = np.uint64(total_threads)
    unroll64 = np.uint64(UNROLL_FACTOR)

    u32 = np.empty(numel, dtype=np.uint32)
    for start in range(0, numel, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, numel)
        li = np.arange(start, end, dtype=np.uint64)
        linear_index = li % tt64
        group_idx = li // tt64
        loop_idx = group_idx // unroll64
        i_step = group_idx % unroll64

        low64 = (off64 + loop_idx) & MASK64
        carry = (low64 < loop_idx).astype(np.uint64)
        high64 = linear_index + carry
        r0, r1, r2, r3 = _philox_10_rounds(
            low64 & MASK32,
            (low64 >> np.uint64(32)) & MASK32,
            high64 & MASK32,
            (high64 >> np.uint64(32)) & MASK32,
            k0,
            k1,
        )
        results = np.stack([r0, r1, r2, r3], axis=1)
        u32[start:end] = results[
            np.arange(end - start, dtype=np.int64), i_step.astype(np.int64)
        ]
    return u32


def _gen_noise_chain(u32, dtype, lower_v, upper_v):
    """值链按算子定义的数学本义计算(f64 自然精度), 不镜像 kernel 的 f32 实现。

    golden 是独立标杆: 流布局(counter 映射/Philox)精确复刻算子可观测契约,
    值链独立实现(数学正确的参考值)。kernel 因 aicore 禁 double 走 f32 域
    乘加, 与本链的 f64 残差为 ulp 级, 由 ST 容差兜住——如果 kernel 值链
    有 bug(乘加次序/量化次序/边界处理错误), golden 不会跟着错, ST 能发现。
    """
    # u 口径: u = x * 2^-32 + 2^-33, u 属于 (0, 1] (f32, Philox 输出约定的统一口径)
    u = u32.astype(np.float32) * RAND_2POW32_INV + RAND_2POW32_INV_HALF
    # f64 域乘加(数学本义), 单次舍到存储 dtype
    v = u.astype(np.float64) * (float(upper_v) - float(lower_v)) + float(lower_v)
    return v.astype(dtype)


# ---- 多 splitBlock 镜像(random_tiling_arch35: InitTensorSliceState/CalcSplitBlocks/
#      CalcExecutionPoliciesForBlocks)。超大 tensor(>2^31 元素或字节跨度超限)时
#      kernel 按块切分执行, 每块独立流布局, 流位置逐块累积 ----


def _coalesced_layout(shape):
    """镜像 InitTensorSliceState 的 coalesce: 由 shape 推连续 layout 的元素 strides,
    丢 size-1 维, 相邻维满足 shape[dim]*strides[dim]==strides[dim-1] 则合并。"""
    ndim = len(shape)
    if ndim == 0:
        return [1], [1]
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = shape[d + 1] * strides[d + 1]
    dims = [(shape[d], strides[d]) for d in range(ndim) if shape[d] != 1]
    if not dims:
        return [1], [1]
    merged = [dims[0]]
    for dim, stride in dims[1:]:
        pdim, pstride = merged[-1]
        if dim * stride == pstride:
            merged[-1] = (pdim * dim, stride)
        else:
            merged.append((dim, stride))
    return [d for d, _ in merged], [s for _, s in merged]


def _block_max_offset_bytes(shape, strides, element_size):
    # 镜像 TensorSliceState::GetMaxOffsetBytes(strides 为元素粒度)
    total = 1
    for dim, stride in zip(shape, strides):
        total += (dim - 1) * abs(stride) * element_size
    return total


def _block_32bit_indexable(numel, shape, strides, element_size):
    # 镜像 TensorSliceState::Is32bitIndexable
    if numel > INDEX_32BIT_LIMIT:
        return False
    return _block_max_offset_bytes(shape, strides, element_size) <= INDEX_32BIT_LIMIT


def _dim_to_split(shape, strides, element_size):
    # 镜像 GetDimToSplit: 自高维向低维扫, extent 严格更大才替换(平局取高维)
    max_extent, dim_to_split = -1, -1
    for d in range(len(shape) - 1, -1, -1):
        if shape[d] == 0:
            continue
        extent = (shape[d] - 1) * abs(strides[d]) * element_size
        if extent > max_extent:
            max_extent, dim_to_split = extent, d
    return dim_to_split


def _partition_dim(shape, gm_offset, dim):
    """镜像 PartitionDim: copySize = shape[dim]//2, other 取前半(gm 不变),
    this 取后半(gm 前移 copySize*strides[dim]); strides 全局共享不随切分变化。"""
    copy_size = shape[dim] // 2
    this_size = shape[dim] - copy_size
    other_shape = list(shape)
    other_shape[dim] = copy_size
    this_shape = list(shape)
    this_shape[dim] = this_size
    return (other_shape, gm_offset), (this_shape, gm_offset + copy_size)


def _calc_split_blocks(shape, strides, element_size):
    """镜像 CalcSplitBlocks: 32bit 可索引单块; 否则栈式递归对半切最大 extent 维。
    返回 [(shape, gmOffset, numel), ...](元素粒度 gmOffset; 连续 layout 恒合并为
    1 维, 块即线性连续区间)。"""
    numel = 1
    for d in shape:
        numel *= d
    if _block_32bit_indexable(numel, shape, strides, element_size):
        return [(shape, 0, numel)]
    stack = [(shape, 0)]
    blocks = []
    while stack and len(blocks) < MAX_SPLIT_BLOCKS:
        cur_shape, cur_gm = stack.pop()
        cur_numel = 1
        for d in cur_shape:
            cur_numel *= d
        if _block_32bit_indexable(cur_numel, cur_shape, strides, element_size):
            blocks.append((cur_shape, cur_gm, cur_numel))
            continue
        dim = _dim_to_split(cur_shape, strides, element_size)
        other, cur = _partition_dim(cur_shape, cur_gm, dim)  # other=前半, cur=后半
        if MAX_SPLIT_BLOCKS - len(stack) < 2:
            raise NotImplementedError(
                "rrelu_with_noise_train golden: splitBlock 栈深超过 MAX_SPLIT_BLOCKS(%d), tiling 侧 GRAPH_FAILED"
                % MAX_SPLIT_BLOCKS
            )
        stack.append(cur)  # 后半先入栈(镜像 stack[top++] = cur)
        stack.append(other)  # 前半后入栈, 先弹出 -> 块按 gm 升序出栈
    if stack:
        raise NotImplementedError(
            "rrelu_with_noise_train golden: splitBlock 超过 %d 块, kernel 无法完整覆盖"
            % MAX_SPLIT_BLOCKS
        )
    return blocks


def _calc_block_policies(offset, total_numel, blocks):
    """镜像 CalcExecutionPoliciesForBlocks: 多块时先按总 numel 推进全局流位置,
    再逐块累积 counterOffset。返回 [(numel, gmOffset, kernelOffset), ...]。"""

    def _counter_offset(numel):
        total_threads = _calc_total_threads(numel)
        rounds = (numel + total_threads * UNROLL_FACTOR - 1) // (
            total_threads * UNROLL_FACTOR
        )
        return rounds * MAX_PRNG_COUNTER_INCR

    accumulated = int(offset)
    if len(blocks) > 1:
        accumulated += _counter_offset(total_numel)
    policies = []
    for _, gm, numel in blocks:
        policies.append((numel, gm, accumulated))
        accumulated += _counter_offset(numel)
    return policies


def _rrelu_with_noise_train_numpy(x_arr, lower_v, upper_v, seed_v, offset_v):
    # 与 tiling 侧 CalcSplitBlocks/CalcExecutionPoliciesForBlocks 对齐:
    # 32bit 可索引时单 splitBlock 且 kernelOffset=offset; 超过 2^31 的超大
    # tensor 递归切分为多 splitBlock, 流位置逐块累积(多块时含全张量前缀)
    dtype = x_arr.dtype
    shape = x_arr.shape
    numel = int(x_arr.size)
    cshape, cstrides = _coalesced_layout(shape)
    blocks = _calc_split_blocks(cshape, cstrides, dtype.itemsize)
    policies = _calc_block_policies(offset_v, numel, blocks)

    noise = np.empty(numel, dtype=dtype)
    for bnumel, bgm, koff in policies:
        # 块内 totalThreads 与 _gen_u32 内 _calc_total_threads(numel) 同式
        u32 = _gen_u32(bnumel, seed_v, koff)
        noise[bgm : bgm + bnumel] = _gen_noise_chain(u32, dtype, lower_v, upper_v)
    x_flat = x_arr.reshape(-1)
    xf = x_flat.astype(np.float32)
    # x <= 0 走随机斜率(x==0 亦走随机分支); NaN 比较恒为 False 走正分支, 与 kernel 一致
    neg = xf <= np.float32(0.0)
    factor = noise.astype(np.float32)
    y_neg = (xf * factor).astype(dtype)
    y = np.where(neg, y_neg, x_flat)
    noise_out = np.where(neg, noise, np.ones(numel, dtype=dtype))
    return [y.reshape(shape), noise_out.reshape(shape)]


# ============================================================
# kernel 通路 test spec（TTK --plugin / cross_check 消费）
# ============================================================


class RreluWithNoiseKernelTestSpec:
    """RreluWithNoise kernel 通路 test spec — randomized leaky relu with noise output.

    Formula (README.md training 语义):
        r[i] ~ U(lower, upper)  (philox 流, 由 seed/offset 决定, 与数据无关)
        x[i] <= 0 -> y[i] = x[i] * r[i], noise[i] = r[i]   (x==0 亦走随机斜率)
        x[i] > 0  -> y[i] = x[i],      noise[i] = 1        (NaN 走正分支)

    Reference oracle: 三方参考实现 rrelu_with_noise(training=True) 直跑,
    rng state 按 seed + ceil(offset/4)*4 注入(对齐 CANN kernel 的
    PhiloxAlgParsInit ceil 语义), 仅 78 SM 环境流布局一致。

    Pathway: kernel(本仓库 aclnn_exclude, 无 aclnn/e2e 通路)。

    Golden 走三方直跑(独立真值); 无三方直跑环境时兜底
    numpy 复刻 kernel 值链(f32 域, 与 kernel 逐步同构)。ThirdPartyImpl 同源三方直跑(device_us 仅含
    rrelu_with_noise_train device kernel)。
    """

    golden = staticmethod(rrelu_with_noise_train_golden)

    class ThirdPartyImpl:
        """Third-party reference implementation (cross_check provider).

        每次调用重新注入 rng state(seed + 对齐 offset), 无跨用例状态残留;
        不做 dtype 转换, __call__ 仅计时被测 rrelu_with_noise。
        """

        def __init__(self, **kwargs):
            pass

        def __call__(self, x, seed, offset, lower=0.125, upper=0.333333, **kwargs):
            seed_v = _scalar(seed, int, 0)
            offset_v = _scalar(offset, int, 0)
            y, noise = _rrelu_with_noise_train_torch(
                _xpu_np(x), float(lower), float(upper), seed_v, offset_v
            )
            return [y, noise]

    third_party = {
        "torch": ThirdPartyImpl,
    }

    # cross_check 标准才会让 TTK 取三方输出; 无 XPU provider 的环境裸跑会解析
    # 失败, 需 CLI --compare 覆盖或走 manual_golden_binaries 离线 bin 比对
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


# ============================================================
# aclnn 通路（还原版 aclnnRReluWithNoise: (self, noise, lower, upper,
# training, seed, offset, out), noise 为输入缓冲原地写回, out 为输出）
# ============================================================


def aclnn_rrelu_with_noise_golden(
    x,
    noise,
    lower=0.125,
    upper=0.333333,
    training=True,
    seed=0,
    offset=0,
    out=None,
    **kwargs,
):
    """ACLNN 层 golden（还原版 aclnnRReluWithNoise）。

    参数按还原后 C 签名顺序: (self, noise, lower, upper, training, seed, offset, out);
    out 为输出占位(golden 不消费)。noise 为输入缓冲(含传入数据):
      - training=true: 写回 x<=0 位随机斜率 r、x>0 位 1; noise 可大于 x,
        超出部分保留输入值(镜像 aclnn slice/concat 语义)
      - training=false: 走 LeakyRelu((lower+upper)/2), noise 完全不写(保留输入值)
    返回 [noise, y] —— 与 output_tensor_indexes=(1, 2) 的枚举顺序一致。
    """
    x_arr = _xpu_np(x)
    noise_arr = _xpu_np(noise)
    seed_v = _scalar(seed, int, 0)
    offset_v = _scalar(offset, int, 0)
    lower_v = _scalar(lower, float, 0.125)
    upper_v = _scalar(upper, float, 0.333333)
    dtype = x_arr.dtype
    x_numel = int(x_arr.size)
    n_numel = int(noise_arr.size)

    if x_numel == 0:
        # aclnn 空tensor早退: y 为空, noise 完全不被写(保留输入值)
        return [noise_arr, np.empty_like(x_arr)]

    if not training:
        # eval: y = leaky_relu(x, (lower+upper)/2), noise 不写(保留输入值)
        slope = (lower_v + upper_v) / 2.0
        xf = x_arr.reshape(-1).astype(np.float32)
        y = np.where(
            xf <= 0, (xf * np.float32(slope)).astype(dtype), x_arr.reshape(-1)
        ).reshape(x_arr.shape)
        return [noise_arr, y]

    # training: 950 单 kernel 值链(与 kernel golden 同源)
    y_k, noise_k = rrelu_with_noise_train_golden(
        x_arr, np.int64(seed_v), np.int64(offset_v), lower=lower_v, upper=upper_v
    )
    if n_numel > x_numel:
        # noise 大于 x: 前 x_numel 为斜率, 其余保留输入值(镜像 slice/concat)
        n_flat = noise_arr.reshape(-1)
        noise_out = np.concatenate([noise_k.reshape(-1), n_flat[x_numel:]]).reshape(
            noise_arr.shape
        )
        return [noise_out.astype(dtype), y_k]
    return [noise_k.reshape(noise_arr.shape), y_k]
