# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# ruff: noqa: F821
from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField


@dataclass
class MaskedFillRangeTiling:
    rows: int  # 行视图行数（3D 为 N*C）
    cols: int  # 行视图列数（填充维长度）
    rounds: int  # 填充轮数（start/value 第一维）
    batches: int  # 有效批数 min(B, 行数需求)：行 r 的批 b=r//repeat 仅 b<batches 时填充
    winStride: int  # GM 上 start/end 第二维原值 B：窗口 (i, b) 的 GM flat 索引为 i*winStride+b，
    # 不得用 batches 作步长（1D 恒 batches=1 而 B 可 >1，错位会读成第 0 轮
    # 的第 1/2/... 窗口列）
    repeat: int  # 每 batch 覆盖的 kernel 行数（行 r -> 批 b = r // repeat）
    fillMode: int  # 0=WINDOW 行内 clamp 窗口（1D/2D/3D axis2）；1=ROW_B 3D axis0
    # 整片填充（k=row//repeat=n）；2=ROW_C 3D axis1 整行填充
    # （k=row%repeat=c）


class MaskedFillRangeTilingKey:
    Dtype = TilingKeyField(bits=2, values=[0, 1, 2, 3])  # 0=FP32 1=FP16 2=INT32 3=INT8
    Win = TilingKeyField(bits=1, values=[0, 1])  # 0=前缀掩码域 1=32-bit 向量比较域


# ================================================================
#  Platform & layout constants（dav_3510）
# ================================================================
_REG_BYTES = 256  # 单条向量寄存器字节数；各 dtype lane 数 = _REG_BYTES // 元素字节
_UB_BYTES = 253952  # UB(Vec) 总容量；UB 布局占用见 masked_fill_range 内注释
_META_ELEM_BYTES = 4  # 常驻元数据 start/end 的元素字节数（int32）
_DATA_GROUP_DEPTH = 2  # data tile group 双缓冲深度
_META_GROUP_DEPTH = 1  # 窗口表/取值表 tile group 深度

# dtype 元素字节数（UB 布局与 tile 容量推导用）
_BYTES_32B = 4  # FP32 / INT32
_BYTES_16B = 2  # FP16
_BYTES_8B = 1  # INT8

# 各 dtype 的向量 lane 数（= _REG_BYTES // _BYTES_*B）
_LANES_32B = _REG_BYTES // _BYTES_32B
_LANES_16B = _REG_BYTES // _BYTES_16B
_LANES_8B = _REG_BYTES // _BYTES_8B

# dtype 枚举（TilingKeyField.Dtype values 的镜像；与 op_host
# arch35/masked_fill_range_tiling.cpp 的 DTYPE_* constexpr 一一对应，
# 跨文件契约勿单独改动）
_DTYPE_FP32, _DTYPE_FP16, _DTYPE_INT32, _DTYPE_INT8 = 0, 1, 2, 3


# 数据 tile 宽度（元素数，静态容量）与常驻元数据轮容量
_TILE_D = 16384
_TILE_R = 128


@pl.vector_function
def _fill_tile_cmp(
    data,  # [1, TILE_D] data tile, dtype == Dtype（32-bit lane 域）
    starts,  # [1, TILE_R] int32 tile：窗口 (i, b) 位于 lane i*winStride+b
    ends,  # [1, TILE_R] int32 tile
    values,  # [1, TILE_R] tile，dtype == Dtype：lane i = value[i]
    cols: pl.DT_INT64,
    rounds: pl.DT_INT64,
    base: pl.DT_INT64,
    winStride: pl.DT_INT64,  # GM 窗口步长（start 第二维 B），勿用有效批数 batches
    b: pl.DT_INT64,
):
    # 防御契约（U2 实证终态，2026-11-07）：Win=1 向量比较路径仅支持 32-bit dtype
    # （FP32/INT32），本函数体对 FP16/INT8 为空实现（静默 y=x 直通）。曾按 U2
    # 在此尝试 pl.pto_assert 显式断言，asc_opc release 编译实证拒绝（PRO_PARSER
    # F00005 "Operation 'pl.pto_assert' is not supported inside
    # @pl.vector_function"，见 dev record 9.2/9.3），按预授权回退为双侧注释互引。
    # 安全前提是 tiling 门限：op_host/arch35/masked_fill_range_tiling.cpp 的
    # Win=1 判定（fillMode==0 且 Dtype∈{FP32,INT32} 且窗口表跨度 ≤64）已排除
    # 其余组合；放宽该门限的 dtype 白名单或 fillMode 限制时，必须同步补齐本
    # 函数对 FP16/INT8 的 VF 实现（VF 域无可用断言原语，编译期无兜底）。
    if Dtype in (_DTYPE_FP32, _DTYPE_INT32):  # noqa: F821
        if Dtype == _DTYPE_FP32:  # noqa: F821
            lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        else:
            lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        lane32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        lanes = _LANES_32B
        s_all = vf.load_align(starts, 0)
        e_all = vf.load_align(ends, 0)
        v_all = vf.load_align(values, 0)
        for col in pl.range(0, cols, lanes):
            acc = vf.load_align(data, [0, col])
            gcol = vf.arange(base + col, dtype=pl.DT_INT32)
            for i in pl.range(0, rounds, 1):
                idx_v = vf.full(i, lane32, dtype=pl.DT_UINT32)
                v_b = vf.gather(v_all, idx_v)
                idx = vf.full(i * winStride + b, lane32, dtype=pl.DT_UINT32)
                s_b = vf.gather(s_all, idx)
                e_b = vf.gather(e_all, idx)
                window = vf.ge(gcol, s_b, lane32)
                window = vf.lt(gcol, e_b, window)
                acc = vf.select(v_b, acc, window)
            vf.store_align(data, acc, lane, [0, col])


@pl.vector_function
def _fill_round_chunks(
    data,  # [1, TILE_D] data tile（当前行块）
    s: pl.DT_INT64,
    e: pl.DT_INT64,
    v: pl.DT_FP32,
    v_i: pl.DT_INT64,
    cols: pl.DT_INT64,
    base: pl.DT_INT64,
):
    if Dtype == _DTYPE_FP32:  # noqa: F821
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    elif Dtype == _DTYPE_INT32:  # noqa: F821
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    else:
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    if Dtype in (_DTYPE_FP32, _DTYPE_INT32):  # noqa: F821
        lanes = _LANES_32B
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        lanes = _LANES_16B
    else:
        lanes = _LANES_8B
    if Dtype == _DTYPE_INT32:  # noqa: F821
        vreg = vf.full(v_i, lane, dtype=pl.DT_INT32)
    elif Dtype == _DTYPE_FP32:  # noqa: F821
        vreg = vf.full(v, lane, dtype=pl.DT_FP32)
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        vreg = vf.full(v, lane, dtype=pl.DT_FP16)
    else:
        vreg = vf.full(v, lane, dtype=pl.DT_INT8)
    for col in pl.range(0, cols, lanes):
        active = pl.min(cols - col, lanes)
        acc = vf.load_align(data, [0, col])
        # update_mask 的前缀长度必须限制在当前向量有效 lane 内；窗口完全在
        # 当前向量右侧时，未钳位的 sl 可能大于 lane 数。
        sl = pl.min(pl.max(s - base - col, 0), active)
        el = pl.min(e - base - col, active)
        el = pl.max(el, 0)
        if Dtype == _DTYPE_FP32:  # noqa: F821
            below_start = vf.update_mask(sl, dtype=pl.DT_FP32)
        elif Dtype == _DTYPE_FP16:  # noqa: F821
            below_start = vf.update_mask(sl, dtype=pl.DT_FP16)
        elif Dtype == _DTYPE_INT32:  # noqa: F821
            below_start = vf.update_mask(sl, dtype=pl.DT_INT32)
        else:
            below_start = vf.update_mask(sl, dtype=pl.DT_INT8)
        below_start = vf.not_(below_start, lane)
        if Dtype == _DTYPE_FP32:  # noqa: F821
            below_end = vf.update_mask(el, dtype=pl.DT_FP32)
        elif Dtype == _DTYPE_FP16:  # noqa: F821
            below_end = vf.update_mask(el, dtype=pl.DT_FP16)
        elif Dtype == _DTYPE_INT32:  # noqa: F821
            below_end = vf.update_mask(el, dtype=pl.DT_INT32)
        else:
            below_end = vf.update_mask(el, dtype=pl.DT_INT8)
        window = vf.and_(below_end, below_start, lane)
        acc = vf.select(vreg, acc, window)
        vf.store_align(data, acc, lane, [0, col])


@pl.vector_function
def _fill_one_chunk(
    data,  # [1, TILE_D] data tile
    s: pl.DT_INT64,
    e: pl.DT_INT64,
    v: pl.DT_FP32,
    v_i: pl.DT_INT64,
    cols: pl.DT_INT64,
    col: pl.DT_INT64,
    base: pl.DT_INT64,
):
    if Dtype == _DTYPE_FP32:  # noqa: F821
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    elif Dtype == _DTYPE_INT32:  # noqa: F821
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    else:
        lane = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    if Dtype in (_DTYPE_FP32, _DTYPE_INT32):  # noqa: F821
        lanes = _LANES_32B
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        lanes = _LANES_16B
    else:
        lanes = _LANES_8B
    active = pl.min(cols, lanes)
    acc = vf.load_align(data, [0, col])
    # 窗口完全在当前向量右侧时，未钳位的 sl 可能大于有效 lane 数。
    sl = pl.min(pl.max(s - base, 0), active)
    el = pl.min(e - base, active)
    el = pl.max(el, 0)
    if Dtype == _DTYPE_FP32:  # noqa: F821
        below_start = vf.update_mask(sl, dtype=pl.DT_FP32)
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        below_start = vf.update_mask(sl, dtype=pl.DT_FP16)
    elif Dtype == _DTYPE_INT32:  # noqa: F821
        below_start = vf.update_mask(sl, dtype=pl.DT_INT32)
    else:
        below_start = vf.update_mask(sl, dtype=pl.DT_INT8)
    below_start = vf.not_(below_start, lane)
    if Dtype == _DTYPE_FP32:  # noqa: F821
        below_end = vf.update_mask(el, dtype=pl.DT_FP32)
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        below_end = vf.update_mask(el, dtype=pl.DT_FP16)
    elif Dtype == _DTYPE_INT32:  # noqa: F821
        below_end = vf.update_mask(el, dtype=pl.DT_INT32)
    else:
        below_end = vf.update_mask(el, dtype=pl.DT_INT8)
    window = vf.and_(below_end, below_start, lane)
    if Dtype == _DTYPE_INT32:  # noqa: F821
        vreg = vf.full(v_i, lane, dtype=pl.DT_INT32)
    elif Dtype == _DTYPE_FP32:  # noqa: F821
        vreg = vf.full(v, lane, dtype=pl.DT_FP32)
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        vreg = vf.full(v, lane, dtype=pl.DT_FP16)
    else:
        vreg = vf.full(v, lane, dtype=pl.DT_INT8)
    acc = vf.select(vreg, acc, window)
    vf.store_align(data, acc, lane, [0, col])


@pl.jit(auto_mutex=True, tiling_key=MaskedFillRangeTilingKey)
def masked_fill_range(
    x_ptr: pl.Ptr[pl.DT_UINT8],
    start_ptr: pl.Ptr[pl.DT_INT32],
    end_ptr: pl.Ptr[pl.DT_INT32],
    value_ptr: pl.Ptr[pl.DT_UINT8],
    y_ptr: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: MaskedFillRangeTiling,
):
    if Dtype == _DTYPE_FP32:  # noqa: F821
        dtype = pl.DT_FP32
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        dtype = pl.DT_FP16
    elif Dtype == _DTYPE_INT32:  # noqa: F821
        dtype = pl.DT_INT32
    else:
        dtype = pl.DT_INT8
    if Dtype in (_DTYPE_FP32, _DTYPE_INT32):  # noqa: F821
        elem = _BYTES_32B
    elif Dtype == _DTYPE_FP16:  # noqa: F821
        elem = _BYTES_16B
    else:
        elem = _BYTES_8B
    if Dtype in (_DTYPE_FP32, _DTYPE_INT32):  # noqa: F821
        chunk_lanes = _LANES_32B
    else:
        # FP16/INT8 步长 128：INT8 未吃满 _LANES_8B=256（_fill_round_chunks 为
        # 全宽 256）——沿用既有行为，仅命名化说明
        chunk_lanes = _LANES_16B
    x = pl.make_tensor(x_ptr, [tiling.rows, tiling.cols], dtype=dtype)
    y = pl.make_tensor(y_ptr, [tiling.rows, tiling.cols], dtype=dtype)
    starts_t = pl.make_tensor(
        start_ptr, [tiling.rounds * tiling.winStride], dtype=pl.DT_INT32
    )
    ends_t = pl.make_tensor(
        end_ptr, [tiling.rounds * tiling.winStride], dtype=pl.DT_INT32
    )
    values_t = pl.make_tensor(value_ptr, [tiling.rounds], dtype=dtype)
    data_type = pl.TileType(
        shape=[1, _TILE_D], dtype=dtype, target_memory=pl.MemorySpace.Vec
    )
    meta_type = pl.TileType(
        shape=[1, _TILE_R], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec
    )
    # UB(Vec) 布局（32B 对齐；占用 = data 深度 2 × 64KB + 窗口/取值表 ≈ 129KB，
    # 约 52% of _UB_BYTES=253952B，参照 pdist UB 常量区风格）
    data_bytes = _DATA_GROUP_DEPTH * _TILE_D * elem
    meta_stride = _TILE_R * _META_ELEM_BYTES
    data_group = pl.make_tile_group(
        type=data_type, addrs=0x0, mutex_ids="auto", depth=_DATA_GROUP_DEPTH
    )
    start_group = pl.make_tile_group(
        type=meta_type, addrs=data_bytes, mutex_ids="auto", depth=_META_GROUP_DEPTH
    )
    end_group = pl.make_tile_group(
        type=meta_type,
        addrs=data_bytes + meta_stride,
        mutex_ids="auto",
        depth=_META_GROUP_DEPTH,
    )
    val_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, _TILE_R], dtype=dtype, target_memory=pl.MemorySpace.Vec
        ),
        addrs=data_bytes + 2 * meta_stride,
        mutex_ids="auto",
        depth=_META_GROUP_DEPTH,
    )
    with pl.section_vector():
        if Win:  # noqa: F821 (Win injected by the tiling key)
            # 向量路径的常驻窗口表：gather 索引域 i*winStride+b（b<batches），最大
            # 索引 (rounds-1)*winStride+batches-1，拷贝长度以其为上界——tiling 的
            # Win=1 门限 (rounds-1)*winStride+batches<=64 保证不超 _TILE_R 静态容量。
            # Win=0 不拷表：getval 直读 GM，且大 B 时窗口表会超 tile 容量。
            # 门限互引：Win=1 的完整判定在 op_host/arch35/masked_fill_range_tiling.cpp
            # 的 TilingKey 设置段（fillMode==0 且 Dtype∈{FP32,INT32} 且窗口表跨度
            # ≤64）；_fill_tile_cmp 对 FP16/INT8 为空实现（静默 y=x），该组合仅由
            # tiling 门限排除——放宽门限时须同步补齐 VF 实现（VF 域不支持
            # pl.pto_assert，U2 实证 F00005，无编译期防御兜底）。
            start_tile = start_group.current()
            end_tile = end_group.current()
            val_tile = val_group.current()
            meta_len = (tiling.rounds - 1) * tiling.winStride + tiling.batches
            pl.set_validshape(start_tile, [1, meta_len])
            pl.load(start_tile, starts_t, [0])
            pl.set_validshape(end_tile, [1, meta_len])
            pl.load(end_tile, ends_t, [0])
            pl.set_validshape(val_tile, [1, tiling.rounds])
            pl.load(val_tile, values_t, [0])
        col_blocks = (tiling.cols + _TILE_D - 1) // _TILE_D
        task_count = tiling.rows * col_blocks
        # 一个任务负责一行的一个列块，使 1D 长向量也能跨 block 并行；
        # 行内窗口轮次仍按原顺序执行，保持重叠窗口的覆盖语义。
        for task in pl.range(pl.get_block_idx(), task_count, pl.get_block_num()):
            row = task // col_blocks
            col_block = task - row * col_blocks
            b = row // tiling.repeat
            covered = 0
            if b < tiling.batches:
                covered = 1
            k = b
            if tiling.fillMode == 2:
                k = row - b * tiling.repeat
            for blk in pl.range(
                col_block * _TILE_D, (col_block + 1) * _TILE_D, _TILE_D
            ):
                chunk_cols = pl.min(_TILE_D, tiling.cols - blk)
                data_tile = data_group.next()
                pl.set_validshape(data_tile, [1, chunk_cols])
                pl.load(data_tile, x, [row, blk])
                if Win:  # noqa: F821
                    if covered:
                        _fill_tile_cmp(
                            data_tile,
                            start_tile,
                            end_tile,
                            val_tile,
                            chunk_cols,
                            tiling.rounds,
                            blk,
                            tiling.winStride,
                            b,
                        )
                else:
                    if covered:
                        if tiling.fillMode == 0:
                            for i in pl.range(0, tiling.rounds, 1):
                                v = pl.getval(values_t, i)
                                v = pl.astype(v, pl.DT_FP32)
                                v_i = pl.getval(values_t, i)
                                v_i = pl.astype(v_i, pl.DT_INT64)
                                s = pl.getval(starts_t, i * tiling.winStride + b)
                                e = pl.getval(ends_t, i * tiling.winStride + b)
                                if s < 0:
                                    s = 0
                                if e > tiling.cols:
                                    e = tiling.cols
                                if e > s:
                                    if Dtype == _DTYPE_INT8:  # noqa: F821
                                        _fill_round_chunks(
                                            data_tile, s, e, v, v_i, chunk_cols, blk
                                        )
                                    else:
                                        for col in pl.range(0, chunk_cols, chunk_lanes):
                                            sub = pl.min(chunk_lanes, chunk_cols - col)
                                            _fill_one_chunk(
                                                data_tile,
                                                s,
                                                e,
                                                v,
                                                v_i,
                                                sub,
                                                col,
                                                blk + col,
                                            )
                        else:
                            for i in pl.range(0, tiling.rounds, 1):
                                s = pl.getval(starts_t, i * tiling.winStride + b)
                                e = pl.getval(ends_t, i * tiling.winStride + b)
                                if k >= s and k < e:
                                    v = pl.getval(values_t, i)
                                    v = pl.astype(v, pl.DT_FP32)
                                    v_i = pl.getval(values_t, i)
                                    v_i = pl.astype(v_i, pl.DT_INT64)
                                    if Dtype == _DTYPE_INT8:  # noqa: F821
                                        _fill_round_chunks(
                                            data_tile,
                                            0,
                                            tiling.cols,
                                            v,
                                            v_i,
                                            chunk_cols,
                                            blk,
                                        )
                                    else:
                                        for col in pl.range(0, chunk_cols, chunk_lanes):
                                            sub = pl.min(chunk_lanes, chunk_cols - col)
                                            _fill_one_chunk(
                                                data_tile,
                                                0,
                                                tiling.cols,
                                                v,
                                                v_i,
                                                sub,
                                                col,
                                                blk + col,
                                            )
                pl.set_validshape(data_tile, [1, chunk_cols])
                pl.store(y, data_tile, [row, blk])
