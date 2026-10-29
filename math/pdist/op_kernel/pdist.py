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
# (PyPTO DSL 在 codegen 阶段向 @pl.vector_function/@pl.jit 函数注入编译期常量,
#  如 PBranch/io_dtype/LANES, 静态检查器不可见, 文件级豁免未定义名检查)

"""pdist kernel — pairwise p-norm distances over row pairs (upper-triangular output).

Ported from the Stage-4 verified PyPTO-Pro v2 delivery (custom/pdist/test_pdist.py,
verifier PASS 15/15, 16/16 cases, Stage 5 perf evidence) into the ops-math
AscendC packaging. The v2 m-lane row-major design structurally removes the v1
pair-lane transposed layout, whose transpose lowering (vgather2/vscatter at a
64-element row stride) serialized the UB banks — the fp16 [8192,512] case spent
97% of its 963ms in b16 vscatter; with zero transposes the same case runs at
~20.6ms.

  * m-lane row-major layout: rows are loaded as a [64, M_BLK] tile in GM
    natural row-major order (no transpose; scatter/gather count = 0) and
    consumed pair-by-pair — pair p's 64 consecutive m values occupy one fp32
    VF register via ``vf.load_align(rows, p*M_BLK + c*LANES)``.  The anchor
    row chunks a0/a1 are loaded once per VF call and reused across all 64
    pairs (replaces the v1 per-m BRC broadcast).
  * M-blocking with ping-pong partial-sum state: M blocks of M_BLK=128 columns
    accumulate into a resident acc tile pair ([64, 128] fp32, row p = pair p:
    cols 0-63 = 64-lane partial sum acc, cols 64-127 = the NaN counter for
    PBranch 5), rotated with explicit indices mblk%2 / (mblk+1)%2 and
    ordered by vf.mem_bar(VST_VLD)/(VLD_VST).
  * M-tail lane neutralization: each chunk's tail lanes (beyond vc) are
    zeroed with ``d = vf.select(d, zeros_r, preg_c)`` right after abs_sub
    and BEFORE any p-branch chain (all branches map 0 -> 0 contribution);
    accumulation then uses the FULL predicate so the nested chunk masks
    never clear each other's lanes.  Pair tail = finalize p-loop upper
    bound vp + out32 valid [1, vp].
  * Task tail: per pair one lane-internal ``vf.reduce_sum`` (or
    ``vf.reduce_max`` for p=inf) + inv-pow in the lane-0 scalar domain, then
    a compact 4B write ``vf.store_align(out32, y, preg, p,
    dist=FIRST_ELEMENT)`` — 64 pairs accumulate into a [1, 64] contiguous
    segment written back with one ``pl.store`` (offsets=[0, base]).
  * PBranch folding (TilingKey): PBranch (6 values) compiles a dedicated
    instance per p-class; every variant's inner p loop is branch-free.
  * Multi-core block-domain contiguous split: the B 64-pair task blocks are
    split contiguously per core; each core locates its first block with a
    closed-form binary search over cumsegs(i).
  * dtype instances: fp32 and fp16 specializations of this single @pl.jit
    kernel; the fp16 instance widens rows16/anchor16 with tile-level pl.cast
    CAST_NONE at the KERNEL ENTRY (right after pl.load, before abs_sub —
    earlier than the v1 post-transpose position), computes entirely in
    fp32, and narrows the exit segment with pl.cast CAST_RINT.
  * NaN/Inf semantics: exact chains propagate per pair via IEEE arithmetic;
    p=inf is the Chebyshev comparison max (GPU/CUDA aligned: gt(NaN, acc)
    is false, so NaN dims are ignored and an all-NaN pair keeps 0); general-p
    carries an explicit ne(d,d)-based NaN counter (nan_cnt, acc tile cols
    64-127) that overwrites the output with NaN in finalize; p=0 counts NaN
    dims as mismatched (ne(NaN,0)=1, golden semantics).

Port adaptation (declared deviations vs the Stage-4 source, math identical):
NaN constants in PBranch 5 finalize are materialized with the in-repo idiom
ln(0) = -inf followed by |-inf - (-inf)| = NaN (see dense_lightning_indexer_
softmax_lse_v2.py write_neg_inf_vf for the -inf twin of this idiom), because
the CANN-tree pypto_pro serializes the vf.full(float("nan")) literal as the
libc function name ``nanf``, which bisheng rejects. PBranch 1 (p=+inf) keeps
the remote CUDA-aligned comparison-max chain (gt + select, no NaN counter)
instead of the Stage-4 vf.max + NaN-guard chain: NaN dims are ignored and an
all-NaN pair outputs 0, identical to the remote v1 semantics.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField

# ================================================================
#  Tile dimensions and constants
# ================================================================
TP = 64  # pairs per task block = rows tile row count (row j0+p <-> pair p)
M_BLK = 128  # M block width, shared by fp32/fp16 instances
LANES = 64  # fp32 VL; fp16 data widened to fp32 before VF consumption
CHUNKS = 2  # M_BLK / LANES — register chunks per mblk (compile-time unrolled)
ACC_STR = 128  # acc tile row stride ([64,128] fp32; cols 0-63 acc, 64-127 nan_cnt)
NAN_COL = 64  # in-row nan_cnt half start column (offset p*ACC_STR + 64)

# ================================================================
#  UB(Vec) addresses v2 (32B aligned; fp32 and fp16 instances)
# ================================================================
# fp32 instance (total 0x20600 = 132608B / 253952B = 52.2%)
VA_ROWS_F32 = 0x00000  # rows_group   [64,128] FP32 32768B x2 -> 0x10000
VA_ANCHOR_F32 = 0x10000  # anchor_group [1,128] FP32   512B x2 -> 0x10400
VA_ACC_F32 = (
    0x10400  # acc_group    [64,128] FP32 32768B x2 -> 0x20400 (resident ping-pong)
)
VA_OUT32_F32 = 0x20400  # out32_group  [1,64]  FP32   256B x2 -> 0x20600

# fp16 instance (total 0x28900 = 166144B / 253952B = 65.4%)
VA_ROWS16 = 0x00000  # rows16_group  [64,128] FP16 16384B x2 -> 0x08000 (GM entry)
VA_ROWS32 = 0x08000  # rows32_group  [64,128] FP32 32768B x2 -> 0x18000 (widen dest)
VA_ANCHOR16 = 0x18000  # anchor16_group [1,128] FP16   256B x2 -> 0x18200
VA_ANCHOR32 = 0x18200  # anchor32_group [1,128] FP32   512B x2 -> 0x18600
VA_ACC_F16 = (
    0x18600  # acc_group     [64,128] FP32 32768B x2 -> 0x28600 (resident ping-pong)
)
VA_OUT32_F16 = 0x28600  # out32_group   [1,64]  FP32   256B x2 -> 0x28800
VA_OUT16 = 0x28800  # out16_group   [1,64]  FP16   128B x2 -> 0x28900 (exit narrow)


# ================================================================
#  Tiling data (host tiling fills; kernel consumes)
# ================================================================
@dataclass
class PdistTiling:
    N: int  # INDEX -> int64_t (idx arithmetic safe: N=65535 -> i*(2N-i-1) ~ 8.59e9 > int32)
    M: int
    B: int  # total 64-pair task blocks (host closed form, see below)
    p_val: float  # FP32, used by PBranch 5
    inv_p: float  # FP32, used by PBranch 5


# ================================================================
#  TilingKey
# ================================================================
class PdistTilingKey:
    PBranch = TilingKeyField(bits=3, values=[0, 1, 2, 3, 4, 5])  # p branch table


# ================================================================
#  VF helper functions (register-level implementation, all fp32 domain)
#
# PBranch (a TilingKey constant) and io_dtype (datatype-specialization
# constant) both fold as compile-time constants at the inlined call sites —
# each TilingKey/datatype instance compiles only the instruction chain of its
# surviving branch.
#
# m-lane structure: pair p's 64 consecutive m values occupy one row of the
# rows tile (element offset p*M_BLK + c*LANES, x4B all 32B-aligned NORM
# loads); the anchor chunks a0/a1 are loaded once at VF entry and reused
# across all 64 pairs (replaces the v1 per-m BRC broadcast).
# ================================================================


@pl.vector_function
def pdist_accum_init_vf(
    rows_tile, anchor_tile, acc_out, vp: pl.DT_INT64, vc: pl.DT_INT64, p_val: pl.DT_FP32
):
    """First M block of a task: acc initialized from full(0.0).

    acc_out = acc_group[1] (init always writes slot 1). acc tile [64,128],
    row p = pair p: cols 0-63 = 64-lane partial sum acc (offset p*ACC_STR),
    cols 64-127 = the NaN counter for PBranch 5 (offset p*ACC_STR+NAN_COL).
    The zero init is safe for every PBranch: sums start at 0, and the
    PBranch 1 comparison max is CUDA-aligned with agg = 0 (d=|.|>=0 never
    loses to 0; NaN never wins a gt). fp32/fp16 instances share the same
    structure (fp16 passes the widened rows32/anchor32 FP32 tiles).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # Valid lane count per chunk (M-tail lane zeroing; built once per mblk,
    # reused inside the p loop; vc in [1,128], vc<=64 -> preg_c1 is an empty
    # mask -- legal, all-zeroed)
    preg_c0 = vf.update_mask(pl.min(LANES, vc), dtype=pl.DT_FP32)
    preg_c1 = vf.update_mask(pl.max(vc - LANES, 0), dtype=pl.DT_FP32)
    zeros_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
    if PBranch == 0:
        ones_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
    if PBranch == 5:
        one_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
    # anchor chunks: loaded once at VF entry, reused across all 64 pairs
    a0 = vf.load_align(anchor_tile, 0)
    a1 = vf.load_align(anchor_tile, LANES)
    for p in pl.range(0, vp):
        acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
        if PBranch == 5:
            nan_cnt = vf.full(0.0, preg, dtype=pl.DT_FP32)
        # ---- chunk 0 (M_BLK=128 = 2 chunks, compile-time unrolled) ----
        r = vf.load_align(rows_tile, p * M_BLK)
        d = vf.abs_sub(a0, r, preg)
        d = vf.select(d, zeros_r, preg_c0)  # M-tail lane zeroing (branch(0)=0)
        if PBranch == 0:
            # Hamming: NaN dims count as mismatched (ne(NaN,0)=1, golden semantics)
            ne_m = vf.ne(d, 0.0, preg)
            dp = vf.select(ones_r, zeros_r, ne_m)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 1:
            # Chebyshev comparison max (GPU/CUDA aligned): NaN never wins the
            # comparison (gt(NaN, acc) is false), so NaN dims are ignored and an
            # all-NaN pair keeps acc = 0 — identical to CUDA `if (diff > agg)`.
            gt_m = vf.gt(d, acc, preg)
            acc = vf.select(d, acc, gt_m)
        elif PBranch == 2:
            dp = vf.mul(d, d, preg)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 3:
            acc = vf.add(acc, d, preg)
        elif PBranch == 4:
            dp = vf.sqrt(d, preg)  # d=|.|>=0, in domain
            acc = vf.add(acc, dp, preg)
        else:
            # PBranch 5 (general 0<p<inf): d^p = exp(p*ln d), d=0 clamped to 1 (zero contribution)
            nz = vf.ne(d, 0.0, preg)
            safe = vf.select(d, one_r, nz)
            ld = vf.ln(safe, preg)
            ld = vf.muls(ld, p_val, preg)
            ed = vf.exp(ld, preg)
            dp = vf.select(ed, zeros_r, nz)
            acc = vf.add(acc, dp, preg)
            nan_m = vf.ne(d, d, preg)
            nan_f = vf.select(one_r, zeros_r, nan_m)
            nan_cnt = vf.add(nan_cnt, nan_f, preg)
        # ---- chunk 1 ----
        r = vf.load_align(rows_tile, p * M_BLK + LANES)
        d = vf.abs_sub(a1, r, preg)
        d = vf.select(d, zeros_r, preg_c1)
        if PBranch == 0:
            ne_m = vf.ne(d, 0.0, preg)
            dp = vf.select(ones_r, zeros_r, ne_m)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 1:
            gt_m = vf.gt(d, acc, preg)
            acc = vf.select(d, acc, gt_m)
        elif PBranch == 2:
            dp = vf.mul(d, d, preg)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 3:
            acc = vf.add(acc, d, preg)
        elif PBranch == 4:
            dp = vf.sqrt(d, preg)
            acc = vf.add(acc, dp, preg)
        else:
            nz = vf.ne(d, 0.0, preg)
            safe = vf.select(d, one_r, nz)
            ld = vf.ln(safe, preg)
            ld = vf.muls(ld, p_val, preg)
            ed = vf.exp(ld, preg)
            dp = vf.select(ed, zeros_r, nz)
            acc = vf.add(acc, dp, preg)
            nan_m = vf.ne(d, d, preg)
            nan_f = vf.select(one_r, zeros_r, nan_m)
            nan_cnt = vf.add(nan_cnt, nan_f, preg)
        # full-predicate accumulation (dp tails already zeroed; avoids the add's
        # ZEROING nested chunk masks clearing each other's lanes)
        vf.store_align(acc_out, acc, preg, p * ACC_STR)
        if PBranch == 5:
            vf.store_align(acc_out, nan_cnt, preg, p * ACC_STR + NAN_COL)
    # VST_VLD: this block's store -> next block/finalize load (RAW)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    # VLD_VST: this block's load -> rewrite of the same slot two blocks later (WAR)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)


@pl.vector_function
def pdist_accum_acc_vf(
    rows_tile,
    anchor_tile,
    acc_in,
    acc_out,
    vp: pl.DT_INT64,
    vc: pl.DT_INT64,
    p_val: pl.DT_FP32,
):
    """Subsequent M blocks: read the previous partial sum (acc_in = slot mblk%2)
    and keep accumulating, writing slot (mblk+1)%2."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_c0 = vf.update_mask(pl.min(LANES, vc), dtype=pl.DT_FP32)
    preg_c1 = vf.update_mask(pl.max(vc - LANES, 0), dtype=pl.DT_FP32)
    zeros_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
    if PBranch == 0:
        ones_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
    if PBranch == 5:
        one_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
    a0 = vf.load_align(anchor_tile, 0)
    a1 = vf.load_align(anchor_tile, LANES)
    for p in pl.range(0, vp):
        acc = vf.load_align(acc_in, p * ACC_STR)  # previous M-block partial sum
        if PBranch == 5:
            nan_cnt = vf.load_align(acc_in, p * ACC_STR + NAN_COL)
        # ---- chunk 0 ----
        r = vf.load_align(rows_tile, p * M_BLK)
        d = vf.abs_sub(a0, r, preg)
        d = vf.select(d, zeros_r, preg_c0)
        if PBranch == 0:
            ne_m = vf.ne(d, 0.0, preg)
            dp = vf.select(ones_r, zeros_r, ne_m)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 1:
            # Chebyshev comparison max (GPU/CUDA aligned): NaN never wins the
            # comparison (gt(NaN, acc) is false), so NaN dims are ignored and an
            # all-NaN pair keeps acc = 0 — identical to CUDA `if (diff > agg)`.
            gt_m = vf.gt(d, acc, preg)
            acc = vf.select(d, acc, gt_m)
        elif PBranch == 2:
            dp = vf.mul(d, d, preg)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 3:
            acc = vf.add(acc, d, preg)
        elif PBranch == 4:
            dp = vf.sqrt(d, preg)
            acc = vf.add(acc, dp, preg)
        else:
            nz = vf.ne(d, 0.0, preg)
            safe = vf.select(d, one_r, nz)
            ld = vf.ln(safe, preg)
            ld = vf.muls(ld, p_val, preg)
            ed = vf.exp(ld, preg)
            dp = vf.select(ed, zeros_r, nz)
            acc = vf.add(acc, dp, preg)
            nan_m = vf.ne(d, d, preg)
            nan_f = vf.select(one_r, zeros_r, nan_m)
            nan_cnt = vf.add(nan_cnt, nan_f, preg)
        # ---- chunk 1 ----
        r = vf.load_align(rows_tile, p * M_BLK + LANES)
        d = vf.abs_sub(a1, r, preg)
        d = vf.select(d, zeros_r, preg_c1)
        if PBranch == 0:
            ne_m = vf.ne(d, 0.0, preg)
            dp = vf.select(ones_r, zeros_r, ne_m)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 1:
            gt_m = vf.gt(d, acc, preg)
            acc = vf.select(d, acc, gt_m)
        elif PBranch == 2:
            dp = vf.mul(d, d, preg)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 3:
            acc = vf.add(acc, d, preg)
        elif PBranch == 4:
            dp = vf.sqrt(d, preg)
            acc = vf.add(acc, dp, preg)
        else:
            nz = vf.ne(d, 0.0, preg)
            safe = vf.select(d, one_r, nz)
            ld = vf.ln(safe, preg)
            ld = vf.muls(ld, p_val, preg)
            ed = vf.exp(ld, preg)
            dp = vf.select(ed, zeros_r, nz)
            acc = vf.add(acc, dp, preg)
            nan_m = vf.ne(d, d, preg)
            nan_f = vf.select(one_r, zeros_r, nan_m)
            nan_cnt = vf.add(nan_cnt, nan_f, preg)
        vf.store_align(acc_out, acc, preg, p * ACC_STR)
        if PBranch == 5:
            vf.store_align(acc_out, nan_cnt, preg, p * ACC_STR + NAN_COL)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)


@pl.vector_function
def pdist_finalize_vf(acc_tile, out_tile, vp: pl.DT_INT64, inv_p: pl.DT_FP32):
    """Task tail: in-lane reduction + inv-pow + FIRST_ELEMENT 4B compact write.

    Reads the last written slot (acc_group[n_mblk%2]). acc row p = pair p
    (offset p*ACC_STR), tail lanes are always 0 (init 0 + tail-zeroed
    accumulation) -> reduce_sum counts 0 with no effect / reduce_max (d>=0)
    leaves the maximum unchanged. Each pair's reduction result lands in dst[0]
    (remaining lanes zeroed), then store_align(dist=FIRST_ELEMENT) writes the
    single 4B lane-0 element to out32 element p.

    NaN constants are materialized as |-inf - (-inf)| with -inf = ln(0)
    (in-repo idiom; avoids the float("nan") literal, see module docstring).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for p in pl.range(0, vp):
        acc = vf.load_align(acc_tile, p * ACC_STR)
        if PBranch == 0:
            y = vf.reduce_sum(acc, preg)  # identity: count <= M < 2^24, exact in fp32
        elif PBranch == 1:
            # GPU/CUDA-aligned comparison max already ignored NaN dims in the
            # accumulation (gt never selects NaN), so acc rows hold no NaN:
            # reduce_max is safe and an all-NaN pair reduces to 0.
            y = vf.reduce_max(acc, preg)
        elif PBranch == 2:
            s = vf.reduce_sum(acc, preg)
            y = vf.sqrt(s, preg)
        elif PBranch == 3:
            y = vf.reduce_sum(acc, preg)
        elif PBranch == 4:
            s = vf.reduce_sum(acc, preg)
            y = vf.mul(s, s, preg)
        else:
            # PBranch 5: y = s^(1/p) = exp(ln(s)/p); s=0 -> 0; NaN count -> NaN
            nan_cnt = vf.load_align(acc_tile, p * ACC_STR + NAN_COL)
            s = vf.reduce_sum(acc, preg)
            nz_s = vf.ne(s, 0.0, preg)
            one_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
            zero_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
            s_safe = vf.select(s, one_r, nz_s)  # s=0 -> clamp to 1 (ln domain guard)
            ls = vf.ln(s_safe, preg)
            ls = vf.muls(ls, inv_p, preg)
            es = vf.exp(ls, preg)
            y = vf.select(es, zero_r, nz_s)  # s=0 -> 0 (0^p=0, p>0)
            nc_max = vf.reduce_max(
                nan_cnt, preg
            )  # assignment form (parser requirement)
            bad = vf.ne(nc_max, 0.0, preg)
            ninf_r = vf.ln(zero_r, preg)  # log(0) = -inf
            nan_r = vf.abs_sub(ninf_r, ninf_r, preg)  # |-inf - (-inf)| = NaN
            y = vf.select(nan_r, y, bad)
        # 4B compact write: FIRST_ELEMENT ignores the mask and moves only y's
        # lane 0 (= reduction dst[0]) to out32 element p
        vf.store_align(out_tile, y, preg, p, dist=pl.StoreDist.FIRST_ELEMENT)
    # WAR: this task's acc read -> next task's rewrite of the same slot
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)


# ================================================================
#  Kernel
# ================================================================


@pl.jit(
    auto_mutex=True,
    tiling_key=PdistTilingKey,
    datatype={"x": "io_dtype", "y": "io_dtype"},
)
def pdist(
    x: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: PdistTiling,
):
    """pdist m-lane distance kernel (main @pl.jit kernel, named after the op file).

    x: [N, M] GM (fp32 or fp16, ND row-major); y: [N(N-1)/2] GM (same dtype as x).
    workspace: unused by this kernel, but REQUIRED by the AscendC opc ABI — the runtime
    passes (inputs..., workspace, tiling) in the args area, so the signature must keep
    workspace as the second-to-last parameter (see dense_lightning_indexer_softmax_lse_v2).
    io_dtype is the datatype-specialization compile-time constant: the
    if io_dtype == pl.DT_FP16 branch folds at parse time, so each dtype instance
    compiles only its own address table (fp32 / fp16).
    The rows tile [64,128] is read row-major directly (pl.load GM 2D block);
    the VF consumes it pair-by-pair via load_align(rows, p*M_BLK + c*LANES) —
    no transpose anywhere.
    """
    N = tiling.N
    M = tiling.M
    B = tiling.B
    L = N * (N - 1) // 2
    tensor_x = pl.make_tensor(x, [N, M], dtype=io_dtype)
    tensor_y = pl.make_tensor(
        y, [1, L], dtype=io_dtype
    )  # 1-D output as a [1, L] contiguous view

    with pl.section_vector():
        core_id = pl.get_block_idx()
        # opc MIX delivery (KERNEL_TYPE_MIX_AIC_1_2): blockDim counts tsch blocks, each
        # AIV block carries get_subblock_num() (task ration) vector cores; pl.get_block_idx()
        # is the unique vector-core id (blockIdx * subblockdim + subblockid), while
        # pl.get_block_num() returns the raw AIV block count. Multiply to get the total
        # vector-core count (equals the direct-launch blockDim semantics).
        num_cores = pl.get_block_num() * pl.get_subblock_num()
        # block-domain contiguous split: core c owns [lo, hi)
        lo = core_id * B // num_cores
        hi = (core_id + 1) * B // num_cores

        # closed-form binary search: lo -> (i, jseg), fixed 17 iterations
        # cumsegs(i) = i + F(N-2) - F(N-2-i), F(x) = 32q(q-1) + qr (x+1 = 64q + r)
        qf = (N - 1) // 64
        rf = N - 1 - qf * 64
        F2 = 32 * qf * (qf - 1) + qf * rf
        i = 0
        i_hi = N - 2
        for _b in pl.range(0, 17):
            if i < i_hi:
                mid = (i + i_hi + 1) // 2
                tm = N - 1 - mid
                qm = tm // 64
                fm = 32 * qm * (qm - 1) + qm * (tm - qm * 64)
                if mid + F2 - fm <= lo:
                    i = mid
                else:
                    i_hi = mid - 1
        ti = N - 1 - i
        qi = ti // 64
        fi = 32 * qi * (qi - 1) + qi * (ti - qi * 64)
        jseg = lo - (i + F2 - fi)

        if io_dtype == pl.DT_FP16:
            # ===== fp16 instance =====
            rows16_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TP, M_BLK],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_ROWS16,
                mutex_ids=[0, 1],
            )
            rows32_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TP, M_BLK],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_ROWS32,
                mutex_ids=[2, 3],
            )
            anchor16_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, M_BLK],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_ANCHOR16,
                mutex_ids=[4, 5],
            )
            anchor32_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, M_BLK],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_ANCHOR32,
                mutex_ids=[6, 7],
            )
            acc_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TP, ACC_STR],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=VA_ACC_F16,
                depth=2,
            )  # pure-V resident ping-pong: no mutex
            out32_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, TP],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_OUT32_F16,
                mutex_ids=[8, 9],
            )
            out16_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, TP],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_OUT16,
                mutex_ids=[10, 11],
            )

            for t in pl.range(0, hi - lo):
                j0 = i + 1 + jseg * TP
                vp = pl.min(TP, N - 1 - i - jseg * TP)  # pair tail
                base = (
                    i * (2 * N - i - 1)
                ) // 2 + jseg * TP  # int64 upper-triangular base
                n_mblk = (M + M_BLK - 1) // M_BLK  # ceiling division
                for mblk in pl.range(0, n_mblk):
                    m0 = mblk * M_BLK
                    vc = pl.min(M_BLK, M - m0)  # M tail
                    rows16_slot = rows16_group.next()
                    pl.set_validshape(rows16_slot, [vp, vc])
                    anchor16_slot = anchor16_group.next()
                    pl.set_validshape(anchor16_slot, [1, vc])
                    rows32_slot = rows32_group.next()
                    pl.set_validshape(rows32_slot, [vp, vc])
                    anchor32_slot = anchor32_group.next()
                    pl.set_validshape(anchor32_slot, [1, vc])
                    pl.load(rows16_slot, tensor_x, [j0, m0])
                    pl.load(anchor16_slot, tensor_x, [i, m0])
                    # kernel-entry widen (after load, before abs_sub)
                    pl.cast(rows32_slot, rows16_slot, mode=pl.RoundMode.CAST_NONE)
                    pl.cast(anchor32_slot, anchor16_slot, mode=pl.RoundMode.CAST_NONE)
                    if mblk == 0:
                        pdist_accum_init_vf(
                            rows32_slot,
                            anchor32_slot,
                            acc_group[1],
                            vp,
                            vc,
                            tiling.p_val,
                        )
                    else:
                        pdist_accum_acc_vf(
                            rows32_slot,
                            anchor32_slot,
                            acc_group[mblk % 2],
                            acc_group[(mblk + 1) % 2],
                            vp,
                            vc,
                            tiling.p_val,
                        )
                out32_slot = out32_group.next()
                pl.set_validshape(out32_slot, [1, vp])
                pdist_finalize_vf(acc_group[n_mblk % 2], out32_slot, vp, tiling.inv_p)
                out16_slot = out16_group.next()
                pl.set_validshape(out16_slot, [1, vp])
                pl.cast(
                    out16_slot, out32_slot, mode=pl.RoundMode.CAST_RINT
                )  # exit narrowing
                pl.store(tensor_y, out16_slot, [0, base])  # MTE3 overwrite
                jseg = jseg + 1  # sequential advance (<= 1 row change per step)
                segs_i = (N - 2 - i) // TP + 1
                if jseg == segs_i:
                    i = i + 1
                    jseg = 0
        else:
            # ===== fp32 instance =====
            rows_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TP, M_BLK],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_ROWS_F32,
                mutex_ids=[0, 1],
            )
            anchor_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, M_BLK],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_ANCHOR_F32,
                mutex_ids=[2, 3],
            )
            acc_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TP, ACC_STR],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=VA_ACC_F32,
                depth=2,
            )  # pure-V resident ping-pong: no mutex
            out32_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, TP],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=VA_OUT32_F32,
                mutex_ids=[4, 5],
            )

            for t in pl.range(0, hi - lo):
                j0 = i + 1 + jseg * TP
                vp = pl.min(TP, N - 1 - i - jseg * TP)
                base = (i * (2 * N - i - 1)) // 2 + jseg * TP
                n_mblk = (M + M_BLK - 1) // M_BLK
                for mblk in pl.range(0, n_mblk):
                    m0 = mblk * M_BLK
                    vc = pl.min(M_BLK, M - m0)
                    rows_slot = rows_group.next()
                    pl.set_validshape(rows_slot, [vp, vc])
                    anchor_slot = anchor_group.next()
                    pl.set_validshape(anchor_slot, [1, vc])
                    pl.load(rows_slot, tensor_x, [j0, m0])
                    pl.load(anchor_slot, tensor_x, [i, m0])
                    if mblk == 0:
                        pdist_accum_init_vf(
                            rows_slot, anchor_slot, acc_group[1], vp, vc, tiling.p_val
                        )
                    else:
                        pdist_accum_acc_vf(
                            rows_slot,
                            anchor_slot,
                            acc_group[mblk % 2],
                            acc_group[(mblk + 1) % 2],
                            vp,
                            vc,
                            tiling.p_val,
                        )
                out_slot = out32_group.next()
                pl.set_validshape(out_slot, [1, vp])
                pdist_finalize_vf(acc_group[n_mblk % 2], out_slot, vp, tiling.inv_p)
                pl.store(tensor_y, out_slot, [0, base])
                jseg = jseg + 1
                segs_i = (N - 2 - i) // TP + 1
                if jseg == segs_i:
                    i = i + 1
                    jseg = 0
