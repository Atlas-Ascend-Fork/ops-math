/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hypot_tiling_arch22.cpp
 * \brief hypot arch22 tiling implementation
 */
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "hypot_tiling_arch22.h"

#include <algorithm>
#include <cstdint>
#include <string>

using namespace ge;
using namespace NsHypot;

namespace optiling {

constexpr uint64_t HYPOT_WORK_SPACE_SIZE = 0;
constexpr uint32_t HYPOT_MAX_RANK = 8;
// Available UB budget = ubSize - 10240 (tilingData + alignment padding + Select reserved)
constexpr int64_t HYPOT_UB_EXTRA_SIZE = 10240;
constexpr int64_t HYPOT_ONEDIM_BYTES_PER_ELEM_FP32 = 43;
constexpr int64_t HYPOT_ONEDIM_BYTES_PER_ELEM_FP32_SCALAR = 37;
constexpr int64_t HYPOT_ONEDIM_BYTES_PER_ELEM_FP32_ONE_SCALAR = 27;
constexpr int64_t HYPOT_ONEDIM_BYTES_PER_ELEM_F16 = 35;
constexpr int64_t HYPOT_ONEDIM_BYTES_PER_ELEM_F16_SCALAR = 23;
constexpr int64_t HYPOT_MULTIDIM_BYTES_PER_ELEM_FP32 = 43;
constexpr int64_t HYPOT_MULTIDIM_BYTES_PER_ELEM_F16 = 43;
// STREAM scene: no expansion-engine buffers, so the UB budget buys bigger tiles
constexpr int64_t HYPOT_STREAM_BYTES_PER_ELEM_COL = 21; // full queue + out queue + 3 f32 bufs + masks
constexpr int64_t HYPOT_STREAM_BYTES_PER_ELEM_ROW = 27; // + single-buffered staging queue + b cache
constexpr uint64_t HYPOT_STREAM_MIN_ELEMS = 1048576;    // 1M output elements
// VECIDX scene: single core, whole output in UB, Gather expansion (see kernel ProcessVecIdx)
constexpr uint64_t HYPOT_VECIDX_MAX_ELEMS = 1024;
constexpr uint32_t HYPOT_ALIGN_BLK = 8;   // 32B / sizeof(float)
constexpr uint32_t HYPOT_ALIGN_128B = 32; // OneDim CACHE_LINE alignment in elements
constexpr uint32_t HYPOT_ALIGN_256B = 64; // MultiDimBroadcast REPEAT alignment in elements
constexpr uint32_t HYPOT_ONEDIM_MIN_UB = 256;
constexpr uint32_t HYPOT_SHRINK_MAX_ITER = 100000;
constexpr uint64_t HYPOT_PRELOAD_MAX_ELEMS = 512;
constexpr uint64_t HYPOT_NATIVE_HALF_MIN_ELEMS = 4096;
// Avoid one-core micro launches without collapsing a naturally wide tile space
constexpr uint32_t HYPOT_MIN_USEFUL_CORES = 4;
constexpr uint64_t HYPOT_CORE_SHRINK_ELEMS = 1024;
constexpr uint64_t HYPOT_MIN_ELEMS_PER_CORE = 128;
constexpr uint32_t HYPOT_MIN_TILE_ELEMS = 16;
constexpr uint64_t HYPOT_REAXIS_MIN_ELEMS = 32;
constexpr bool HYPOT_ENABLE_TINY_CROSS3D = false;
constexpr bool HYPOT_ENABLE_TINY_PATTERN = false;

struct FusedShape {
    uint32_t dimNum = 0;
    uint32_t outDims[HYPOT_MAX_RANK] = {0};
    uint32_t x1Dims[HYPOT_MAX_RANK] = {0};
    uint32_t x2Dims[HYPOT_MAX_RANK] = {0};
    uint32_t x1Strides[HYPOT_MAX_RANK] = {0};
    uint32_t x2Strides[HYPOT_MAX_RANK] = {0};
    uint32_t outStrides[HYPOT_MAX_RANK] = {0};
    uint64_t totalLength = 0;
    uint64_t x1Numel = 0;
    uint64_t x2Numel = 0;
};

static uint32_t FloorAlign(uint64_t value, uint32_t align) { return static_cast<uint32_t>(value / align * align); }

static uint32_t CeilDivU64(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }

// pad -> flag bitmap -> merge adjacent same-flag axes -> strides (broadcast axis stride=0)
static ge::graphStatus FuseShapes(gert::TilingContext* tilingContext, const gert::Shape& s1, const gert::Shape& s2,
                                  FusedShape& fs)
{
    size_t rank1 = s1.GetDimNum();
    size_t rank2 = s2.GetDimNum();
    uint32_t rank = static_cast<uint32_t>(rank1 > rank2 ? rank1 : rank2);
    uint32_t pad1[HYPOT_MAX_RANK] = {0};
    uint32_t pad2[HYPOT_MAX_RANK] = {0};
    uint32_t outDims[HYPOT_MAX_RANK] = {0};
    uint8_t flag1[HYPOT_MAX_RANK] = {0};
    uint8_t flag2[HYPOT_MAX_RANK] = {0};
    for (uint32_t i = 0; i < rank; i++) {
        int64_t d1 = (i < rank - rank1) ? 1 : s1.GetDim(i - (rank - rank1));
        int64_t d2 = (i < rank - rank2) ? 1 : s2.GetDim(i - (rank - rank2));
        // guard the int64 -> uint32 narrowing: a dim out of uint32 range would silently truncate
        OP_CHECK_IF(d1 < 0 || d1 > UINT32_MAX || d2 < 0 || d2 > UINT32_MAX,
                    OP_LOGE(tilingContext, "dim out of uint32 range: d1:%ld d2:%ld", d1, d2), return ge::GRAPH_FAILED);
        pad1[i] = static_cast<uint32_t>(d1);
        pad2[i] = static_cast<uint32_t>(d2);
        outDims[i] = pad1[i] > pad2[i] ? pad1[i] : pad2[i];
        flag1[i] = (pad1[i] == 1 && outDims[i] != 1) ? 1 : 0;
        flag2[i] = (pad2[i] == 1 && outDims[i] != 1) ? 1 : 0;
    }

    // merge adjacent axes with identical flag pair
    uint32_t f = 0;
    uint32_t i = 0;
    while (i < rank) {
        uint32_t j = i + 1;
        uint64_t o = outDims[i];
        uint64_t d1 = pad1[i];
        uint64_t d2 = pad2[i];
        while (j < rank && flag1[j] == flag1[i] && flag2[j] == flag2[i]) {
            o *= outDims[j];
            d1 *= pad1[j];
            d2 *= pad2[j];
            j++;
        }
        // guard the uint64 -> uint32 narrowing of the fused products
        OP_CHECK_IF(o > UINT32_MAX || d1 > UINT32_MAX || d2 > UINT32_MAX,
                    OP_LOGE(tilingContext, "fused dim out of uint32 range: o:%lu d1:%lu d2:%lu", o, d1, d2),
                    return ge::GRAPH_FAILED);
        fs.outDims[f] = static_cast<uint32_t>(o);
        fs.x1Dims[f] = static_cast<uint32_t>(d1);
        fs.x2Dims[f] = static_cast<uint32_t>(d2);
        // broadcast axis: original flag of the group (dims collapse to 1 iff flag set)
        fs.x1Strides[f] = (fs.x1Dims[f] == 1 && flag1[i] == 1) ? 0 : 1; // placeholder, fixed below
        fs.x2Strides[f] = (fs.x2Dims[f] == 1 && flag2[i] == 1) ? 0 : 1;
        f++;
        i = j;
    }
    fs.dimNum = f;
    fs.totalLength = 1;
    fs.x1Numel = 1;
    fs.x2Numel = 1;
    for (uint32_t k = 0; k < f; k++) {
        // guard the uint64 accumulation against wrap-around before each multiply (a zero dim keeps
        // the product at zero, so only nonzero multipliers need the overflow check)
        OP_CHECK_IF((fs.outDims[k] != 0 && fs.totalLength > UINT64_MAX / fs.outDims[k]) ||
                        (fs.x1Dims[k] != 0 && fs.x1Numel > UINT64_MAX / fs.x1Dims[k]) ||
                        (fs.x2Dims[k] != 0 && fs.x2Numel > UINT64_MAX / fs.x2Dims[k]),
                    OP_LOGE(tilingContext, "element count overflow: total:%lu x1:%lu x2:%lu", fs.totalLength,
                            fs.x1Numel, fs.x2Numel),
                    return ge::GRAPH_FAILED);
        fs.totalLength *= fs.outDims[k];
        fs.x1Numel *= fs.x1Dims[k];
        fs.x2Numel *= fs.x2Dims[k];
    }
    // strides in elements; broadcast axis stride = 0
    uint64_t st1 = 1;
    uint64_t st2 = 1;
    uint64_t sto = 1;
    for (int64_t k = static_cast<int64_t>(f) - 1; k >= 0; k--) {
        // guard the uint64 -> uint32 narrowing of the strides
        OP_CHECK_IF(st1 > UINT32_MAX || st2 > UINT32_MAX || sto > UINT32_MAX,
                    OP_LOGE(tilingContext, "stride out of uint32 range: st1:%lu st2:%lu sto:%lu", st1, st2, sto),
                    return ge::GRAPH_FAILED);
        fs.outStrides[k] = static_cast<uint32_t>(sto);
        fs.x1Strides[k] = (fs.x1Dims[k] == 1 && fs.outDims[k] != 1) ? 0 : static_cast<uint32_t>(st1);
        fs.x2Strides[k] = (fs.x2Dims[k] == 1 && fs.outDims[k] != 1) ? 0 : static_cast<uint32_t>(st2);
        sto *= fs.outDims[k];
        st1 *= fs.x1Dims[k];
        st2 *= fs.x2Dims[k];
    }
    return ge::GRAPH_SUCCESS;
}

// per-input broadcast expansion mode + [L, B, I] parameters (kernel PrepareInput contract)
static void ComputeInputMode(const FusedShape& fs, const uint32_t strides[], bool isX1, uint32_t ubSplitAxis,
                             uint32_t innerLen, uint32_t& mode, uint32_t& leadInner, uint32_t& brcInner,
                             uint32_t& srcInner)
{
    mode = HYPOT_BRC_NONE;
    leadInner = 1;
    brcInner = 1;
    srcInner = innerLen;
    uint64_t numel = isX1 ? fs.x1Numel : fs.x2Numel;
    if (numel == 1) {
        return; // scalar input, kernel Duplicate path
    }
    const bool fullSplit = strides[ubSplitAxis] != 0;
    // count maximal stride-0 runs among inner axes (ubSplitAxis, dimNum)
    uint32_t runStart = 0;
    uint32_t runEnd = 0;
    uint32_t runCount = 0;
    bool inRun = false;
    uint32_t fullInnerProduct = 1;
    for (uint32_t k = ubSplitAxis + 1; k < fs.dimNum; k++) {
        if (strides[k] == 0) {
            if (!inRun) {
                runStart = k;
                runCount++;
                inRun = true;
            }
            runEnd = k;
        } else {
            fullInnerProduct *= fs.outDims[k];
            inRun = false;
        }
    }
    if (runCount == 0) {
        if (fullSplit) {
            mode = HYPOT_BRC_NONE;
            leadInner = 1;
            brcInner = 1;
            srcInner = innerLen;
        } else {
            // broadcast only on the split axis: repeat a single row per tile
            mode = HYPOT_BRC_ROW;
            leadInner = 1;
            brcInner = 1;
            srcInner = innerLen;
        }
        return;
    }
    uint32_t L = 1;
    for (uint32_t k = ubSplitAxis + 1; k < runStart; k++) {
        L *= fs.outDims[k];
    }
    uint32_t B = 1;
    for (uint32_t k = runStart; k <= runEnd; k++) {
        B *= fs.outDims[k];
    }
    uint32_t I = 1;
    for (uint32_t k = runEnd + 1; k < fs.dimNum; k++) {
        I *= fs.outDims[k];
    }
    if (runCount > 1 || (!fullSplit && L > 1)) {
        // multi-group broadcast, or split-axis broadcast combined with leading full inner axes:
        // kernel [G,B,I] folding does not cover these, use the scalar index-mapping fallback
        mode = HYPOT_BRC_GENERIC;
        leadInner = fullInnerProduct;
        brcInner = 1;
        srcInner = 1;
        return;
    }
    leadInner = L;
    brcInner = B;
    srcInner = I;
    mode = (I == 1) ? HYPOT_BRC_COL : HYPOT_BRC_ROW;
}

static uint32_t BroadcastModeRank(uint32_t mode)
{
    if (mode == HYPOT_BRC_NONE) {
        return 0;
    }
    if (mode == HYPOT_BRC_ROW) {
        return 1;
    }
    if (mode == HYPOT_BRC_COL) {
        return 2;
    }
    return 3;
}

static uint32_t TinyFp32Pattern(const FusedShape& fs, const uint32_t strides[], uint64_t numel)
{
    if (fs.dimNum != 3U) {
        return 4U;
    }
    const uint64_t out0 = fs.outDims[0];
    const uint64_t out1 = fs.outDims[1];
    const uint64_t out2 = fs.outDims[2];
    if (numel == fs.totalLength) {
        return 0U;
    }
    if (numel == out1 && strides[0] == 0U && strides[1] == 1U && strides[2] == 0U) {
        return 1U;
    }
    if (numel == out0 * out2 && strides[0] == out2 && strides[1] == 0U && strides[2] == 1U) {
        return 2U;
    }
    if (numel == out0 * out1 && strides[0] == out1 && strides[1] == 1U && strides[2] == 0U) {
        return 3U;
    }
    return 4U;
}

static bool IsTinyPattern(const FusedShape& fs, bool isFp32)
{
    const uint64_t maxElems = isFp32 ? 4096U : 2048U;
    if (fs.dimNum != 3U || fs.totalLength > maxElems || fs.x1Numel > maxElems || fs.x2Numel > maxElems) {
        return false;
    }
    const uint32_t x1Mode = TinyFp32Pattern(fs, fs.x1Strides, fs.x1Numel);
    const uint32_t x2Mode = TinyFp32Pattern(fs, fs.x2Strides, fs.x2Numel);
    return x1Mode < 4U && x2Mode < 4U && x1Mode + x2Mode > 0U && fs.outDims[2] % HYPOT_ALIGN_BLK == 0U;
}

static bool IsTinyInner3D(const FusedShape& fs, const uint32_t strides[], uint64_t numel)
{
    const uint64_t rows = fs.outDims[0];
    const uint64_t middle = fs.outDims[1];
    const uint64_t inner = fs.outDims[2];
    if (strides[2] != 1U) {
        return false;
    }
    const bool outerInner = strides[0] == inner && strides[1] == 0U && numel == rows * inner;
    const bool middleInner = strides[0] == 0U && strides[1] == inner && numel == middle * inner;
    return outerInner || middleInner;
}

static bool IsTinyBlock3D(const FusedShape& fs, const uint32_t strides[], uint64_t numel)
{
    if (strides[2] != 0U) {
        return false;
    }
    const uint64_t rows = fs.outDims[0];
    const uint64_t middle = fs.outDims[1];
    const bool outerActive = fs.outDims[0] > 1U && strides[0] != 0U;
    const bool middleActive = fs.outDims[1] > 1U && strides[1] != 0U;
    if (outerActive && middleActive) {
        return numel == rows * middle;
    }
    if (outerActive) {
        return numel == rows;
    }
    if (middleActive) {
        return numel == middle;
    }
    return numel == 1U;
}

static bool IsTinyCross3DPattern(const FusedShape& fs)
{
    if (fs.dimNum != 3U || fs.totalLength == 0U || fs.totalLength > 4096U) {
        return false;
    }
    const uint64_t rows = fs.outDims[0] * fs.outDims[1];
    constexpr uint64_t maxAlign = 16U;
    const uint64_t np = (fs.outDims[2] + maxAlign - 1U) / maxAlign * maxAlign;
    if (rows * np > 4096U) {
        return false;
    }
    const bool x1Inner = IsTinyInner3D(fs, fs.x1Strides, fs.x1Numel);
    const bool x2Inner = IsTinyInner3D(fs, fs.x2Strides, fs.x2Numel);
    const bool x1Full = fs.x1Numel == fs.totalLength;
    const bool x2Full = fs.x2Numel == fs.totalLength;
    const bool alignedInner = fs.outDims[2] % HYPOT_ALIGN_BLK == 0U;
    return (x1Inner && IsTinyBlock3D(fs, fs.x2Strides, fs.x2Numel)) ||
           (x2Inner && IsTinyBlock3D(fs, fs.x1Strides, fs.x1Numel)) ||
           (alignedInner && x1Full && IsTinyBlock3D(fs, fs.x2Strides, fs.x2Numel)) ||
           (alignedInner && x2Full && IsTinyBlock3D(fs, fs.x1Strides, fs.x1Numel));
}

// Mirror of kernel TinyCross3DSupported(): [R,1,I] x [1,A,1] on fused 3D, totalLength <= 4096,
// excluding the 192-element dimNum==3 family (it keeps the scene-1 tiny-pattern route).
static bool IsCross3DStructPattern(const FusedShape& fs, bool isFp32)
{
    if (fs.dimNum != 3U || fs.totalLength == 0U || fs.totalLength > 4096U || fs.totalLength == 192U) {
        return false;
    }
    const uint64_t rows = fs.outDims[0] * fs.outDims[1];
    const uint64_t alignElem = isFp32 ? 8U : 16U;
    const uint64_t np = (fs.outDims[2] + alignElem - 1U) / alignElem * alignElem;
    if (rows == 0U || rows * np > 4096U) {
        return false;
    }
    auto outerInner = [&fs](const uint32_t st[], uint64_t numel) {
        return st[2] == 1U && st[0] == fs.outDims[2] && st[1] == 0U && numel == fs.outDims[0] * fs.outDims[2];
    };
    auto middleBlock = [&fs](const uint32_t st[], uint64_t numel) {
        return st[2] == 0U && st[0] == 0U && st[1] != 0U && (numel == fs.outDims[1] || numel == 1U);
    };
    return (outerInner(fs.x1Strides, fs.x1Numel) && middleBlock(fs.x2Strides, fs.x2Numel)) ||
           (outerInner(fs.x2Strides, fs.x2Numel) && middleBlock(fs.x1Strides, fs.x1Numel));
}

// Mirror of kernel rowStaticHighRank (STRUCT): high-rank aligned-ROW inputs avoid the per-element Gather path.
static bool HasHighRankStructRow(const FusedShape& fs, uint32_t x1Mode, uint32_t x1Src, uint32_t x2Mode, uint32_t x2Src)
{
    if (fs.dimNum <= 4U || fs.totalLength <= 256U) {
        return false;
    }
    const bool x1Row = x1Mode == HYPOT_BRC_ROW && x1Src % HYPOT_ALIGN_BLK == 0U;
    const bool x2Row = x2Mode == HYPOT_BRC_ROW && x2Src % HYPOT_ALIGN_BLK == 0U;
    return x1Row || x2Row;
}

static bool HasUnalignedWideColumn(const FusedShape& fs, const uint32_t strides[], uint32_t axis, uint32_t mode,
                                   uint32_t leadInner, uint32_t brcInner)
{
    if (mode != HYPOT_BRC_COL) {
        return false;
    }
    const uint32_t rows = fs.outDims[axis];
    const bool fullSplit = strides[axis] != 0;
    const uint32_t groups = (fullSplit ? rows : 1U) * leadInner;
    const uint32_t width = fullSplit ? brcInner : rows * brcInner;
    return width > 66U && !(groups % HYPOT_ALIGN_BLK == 0 && width % HYPOT_ALIGN_BLK == 0);
}

static bool HasUnalignedWideRow(uint32_t mode, uint32_t srcInner)
{
    return mode == HYPOT_BRC_ROW && srcInner > 96U && srcInner % HYPOT_ALIGN_BLK != 0;
}

static void ComputeCoreSplit(uint64_t fusedProduct, uint32_t coreNum, HypotTilingData& td)
{
    if (fusedProduct == 0) {
        td.usedCoreNum = 1;
        td.blockFormer = 0;
        td.blockTail = 0;
        return;
    }
    uint64_t blockFormer = (fusedProduct + coreNum - 1) / coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    td.blockFormer = static_cast<uint32_t>(blockFormer);
    td.blockTail = static_cast<uint32_t>(blockTail);
    td.usedCoreNum = static_cast<uint32_t>(blockNum);
}

// STREAM pattern: rank-2 output [M,N] with one full contiguous input and the other a COL-single
// ([M,1]) or ROW-periodic ([1,N]) broadcast; returns the per-side HYPOT_BRC_* modes.
static bool MatchStreamPattern(const FusedShape& fs, uint32_t& x1StreamMode, uint32_t& x2StreamMode)
{
    if (fs.dimNum != 2U) {
        return false;
    }
    const uint64_t dimM = fs.outDims[0];
    const uint64_t dimN = fs.outDims[1];
    auto classifySide = [dimM, dimN](const uint32_t strides[], uint64_t numel) -> uint32_t {
        if (numel == dimM * dimN && strides[0] == dimN && strides[1] == 1U) {
            return HYPOT_BRC_NONE;
        }
        if (numel == dimM && numel <= HYPOT_PRELOAD_MAX_ELEMS && strides[0] == 1U && strides[1] == 0U) {
            return HYPOT_BRC_COL;
        }
        if (numel == dimN && strides[0] == 0U && strides[1] == 1U) {
            return HYPOT_BRC_ROW;
        }
        return HYPOT_BRC_GENERIC;
    };
    x1StreamMode = classifySide(fs.x1Strides, fs.x1Numel);
    x2StreamMode = classifySide(fs.x2Strides, fs.x2Numel);
    const bool x1Full = (x1StreamMode == HYPOT_BRC_NONE);
    const bool x2Full = (x2StreamMode == HYPOT_BRC_NONE);
    if (x1Full == x2Full) {
        return false; // single-sided broadcast only
    }
    const uint32_t brcMode = x1Full ? x2StreamMode : x1StreamMode;
    return brcMode == HYPOT_BRC_COL || brcMode == HYPOT_BRC_ROW;
}

// STREAM tiling: split the inner axis into ubFormer tiles and shard M*ubOuter tiles over the AIV cores.
static bool ApplyStreamTiling(const FusedShape& fs, int64_t budget, uint32_t coreNum, uint32_t x1StreamMode,
                              uint32_t x2StreamMode, HypotTilingData& tilingData)
{
    const bool colStream = (x1StreamMode == HYPOT_BRC_COL) || (x2StreamMode == HYPOT_BRC_COL);
    const int64_t bytesPerElem = colStream ? HYPOT_STREAM_BYTES_PER_ELEM_COL : HYPOT_STREAM_BYTES_PER_ELEM_ROW;
    const uint32_t dimM = fs.outDims[0];
    const uint32_t dimN = fs.outDims[1];
    const uint32_t maxElem = FloorAlign(static_cast<uint64_t>(budget) / bytesPerElem, HYPOT_ALIGN_256B);
    const uint32_t ubFormer = std::min(maxElem, dimN);
    if (ubFormer < HYPOT_ONEDIM_MIN_UB) {
        return false; // inner axis too short to stream profitably: keep the general path
    }
    const uint32_t ubOuter = CeilDivU64(dimN, ubFormer);
    const uint64_t fusedProduct = static_cast<uint64_t>(dimM) * ubOuter;

    tilingData.dimNum = fs.dimNum;
    for (uint32_t k = 0; k < fs.dimNum; k++) {
        tilingData.outShape[k] = fs.outDims[k];
        tilingData.x1Strides[k] = fs.x1Strides[k];
        tilingData.x2Strides[k] = fs.x2Strides[k];
        tilingData.outStrides[k] = fs.outStrides[k];
    }
    tilingData.ubSplitAxis = 1U;
    tilingData.dimProduct = dimM;
    tilingData.innerLen = 1U;
    tilingData.rowPitch = HYPOT_ALIGN_BLK;
    tilingData.x1BrcMode = x1StreamMode;
    tilingData.x2BrcMode = x2StreamMode;
    tilingData.x1LeadInner = 1U;
    tilingData.x1BrcInner = 1U;
    tilingData.x1SrcInner = 1U;
    tilingData.x2LeadInner = 1U;
    tilingData.x2BrcInner = 1U;
    tilingData.x2SrcInner = 1U;
    tilingData.ubFormer = ubFormer;
    tilingData.ubOuter = ubOuter;
    tilingData.ubTail = dimN - (ubOuter - 1U) * ubFormer;
    tilingData.x1TileLen = (x1StreamMode == HYPOT_BRC_COL) ? 1U : ubFormer;
    tilingData.x2TileLen = (x2StreamMode == HYPOT_BRC_COL) ? 1U : ubFormer;
    ComputeCoreSplit(fusedProduct, static_cast<uint32_t>(std::min<uint64_t>(fusedProduct, coreNum)), tilingData);
    return true;
}

static ge::graphStatus HypotTilingArch22(gert::TilingContext* tilingContext)
{
    OP_CHECK_IF(tilingContext == nullptr, OP_LOGE("HypotTiling", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Entering HypotTilingArch22");
    auto compileInfo = static_cast<const HypotCompileInfoArch22*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);

    const gert::StorageShape* shape1 = tilingContext->GetInputShape(0);
    const gert::StorageShape* shape2 = tilingContext->GetInputShape(1);
    OP_CHECK_IF(shape1 == nullptr, OP_LOGE_WITH_INVALID_INPUT("Hypot", "self"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape2 == nullptr, OP_LOGE_WITH_INVALID_INPUT("Hypot", "other"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape1->GetStorageShape().GetDimNum() > HYPOT_MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM("Hypot", "self",
                                             std::to_string(shape1->GetStorageShape().GetDimNum()) + "D", "8D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape2->GetStorageShape().GetDimNum() > HYPOT_MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM("Hypot", "other",
                                             std::to_string(shape2->GetStorageShape().GetDimNum()) + "D", "8D"),
                return ge::GRAPH_FAILED);

    HypotTilingData* tilingData = tilingContext->GetTilingData<HypotTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingData);

    // TilingKey carries only the broadcast scene; dtypes reach the kernel via DTYPE_SELF/OTHER/OUT aliases.
    auto inputDesc = tilingContext->GetInputDesc(0);
    auto inputDesc2 = tilingContext->GetInputDesc(1);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    const ge::DataType inputDtype1 = inputDesc->GetDataType();
    const ge::DataType inputDtype2 = inputDesc2->GetDataType();
    const ge::DataType outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(outputDtype != ge::DataType::DT_FLOAT && outputDtype != ge::DataType::DT_FLOAT16 &&
                    outputDtype != ge::DataType::DT_BF16,
                OP_LOGE_WITH_INVALID_INPUT_DTYPE("Hypot", "out", std::to_string(static_cast<int>(outputDtype)),
                                                 "float32, float16, bfloat16"),
                return ge::GRAPH_FAILED);
    // same-dtype launch: inputs and output share one dtype (mixed always promotes to a FP32 output)
    const bool sameDtype = inputDtype1 == inputDtype2 && inputDtype1 == outputDtype;
    const bool sameHalfDtype = sameDtype &&
                               (inputDtype1 == ge::DataType::DT_FLOAT16 || inputDtype1 == ge::DataType::DT_BF16);
    // mixed-dtype launches follow the FP32 per-element budget (conservative)
    const bool isFp32 = !sameHalfDtype;
    const int64_t multiDimBytesPerElem = isFp32 ? HYPOT_MULTIDIM_BYTES_PER_ELEM_FP32 :
                                                  HYPOT_MULTIDIM_BYTES_PER_ELEM_F16;

    FusedShape fs;
    OP_CHECK_IF(
        FuseShapes(tilingContext, shape1->GetStorageShape(), shape2->GetStorageShape(), fs) != ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext, "FuseShapes failed"), return ge::GRAPH_FAILED);
    tilingData->totalLength = fs.totalLength;
    // guard the uint64 -> uint32 narrowing of the input element counts read by the kernel
    OP_CHECK_IF(
        fs.x1Numel > UINT32_MAX || fs.x2Numel > UINT32_MAX,
        OP_LOGE(tilingContext, "input element count out of uint32 range: x1:%lu x2:%lu", fs.x1Numel, fs.x2Numel),
        return ge::GRAPH_FAILED);
    tilingData->x1Numel = static_cast<uint32_t>(fs.x1Numel);
    tilingData->x2Numel = static_cast<uint32_t>(fs.x2Numel);
    tilingData->scalarFlag = 0;
    if (fs.x1Numel == 1) {
        tilingData->scalarFlag |= HYPOT_SCALAR_FLAG_X1;
    }
    if (fs.x2Numel == 1) {
        tilingData->scalarFlag |= HYPOT_SCALAR_FLAG_X2;
    }
    // reserve preload budget; the mode-aware selection below enables it for scalar paths only
    tilingData->x1PreloadLen = fs.x1Numel <= HYPOT_PRELOAD_MAX_ELEMS ? static_cast<uint32_t>(fs.x1Numel) : 0U;
    tilingData->x2PreloadLen = fs.x2Numel <= HYPOT_PRELOAD_MAX_ELEMS ? static_cast<uint32_t>(fs.x2Numel) : 0U;
    tilingData->nativeHalfFlag = (sameDtype && inputDtype1 == ge::DataType::DT_FLOAT16 &&
                                  fs.totalLength >= HYPOT_NATIVE_HALF_MIN_ELEMS) ?
                                     1U :
                                     0U;
    const uint32_t oneScalarFlags = tilingData->scalarFlag & (HYPOT_SCALAR_FLAG_X1 | HYPOT_SCALAR_FLAG_X2);
    const bool oneDimOneScalar = oneScalarFlags == HYPOT_SCALAR_FLAG_X1 || oneScalarFlags == HYPOT_SCALAR_FLAG_X2;
    const int64_t oneDimBytesPerElem = isFp32 ? (oneDimOneScalar ? HYPOT_ONEDIM_BYTES_PER_ELEM_FP32_ONE_SCALAR :
                                                                   (tilingData->scalarFlag != 0U ?
                                                                        HYPOT_ONEDIM_BYTES_PER_ELEM_FP32_SCALAR :
                                                                        HYPOT_ONEDIM_BYTES_PER_ELEM_FP32)) :
                                                (oneDimOneScalar ? HYPOT_ONEDIM_BYTES_PER_ELEM_F16_SCALAR :
                                                                   HYPOT_ONEDIM_BYTES_PER_ELEM_F16);

    // A whole-input scalar broadcast is a 1-D stream of the other input; route it to OneDim to avoid
    // the MultiDimBroadcast index/expansion overhead.
    uint32_t scene = (fs.dimNum <= 1 || fs.x1Numel == 1 || fs.x2Numel == 1) ? HYPOT_SCENE_ONE_DIM :
                                                                              HYPOT_SCENE_MULTI_DIM_BRC;
    const uint32_t coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    const uint32_t minUsefulCores = fs.totalLength >= HYPOT_CORE_SHRINK_ELEMS ? HYPOT_MIN_USEFUL_CORES : 1U;
    const int64_t preloadBytes = static_cast<int64_t>(tilingData->x1PreloadLen + tilingData->x2PreloadLen) *
                                 (isFp32 ? 4 : 2);
    int64_t budget = compileInfo->ubSize - HYPOT_UB_EXTRA_SIZE - preloadBytes;
    OP_CHECK_IF(budget <= 0, OP_LOGE(tilingContext, "ubSize too small: %ld", compileInfo->ubSize),
                return ge::GRAPH_FAILED);

    if (fs.totalLength == 0) {
        // empty tensor is filtered by the aclnn layer; keep a safe no-op configuration
        tilingData->usedCoreNum = 1;
        tilingData->blockFormer = 0;
        tilingData->blockTail = 0;
        tilingData->ubFormer = HYPOT_ONEDIM_MIN_UB;
        tilingData->ubTail = 0;
        tilingData->ubOuter = 0;
        tilingContext->SetBlockDim(1);
        tilingContext->SetTilingKey(GET_TPL_TILING_KEY(HYPOT_SCENE_ONE_DIM));
        size_t* ws = tilingContext->GetWorkspaceSizes(1);
        if (ws != nullptr) {
            ws[0] = HYPOT_WORK_SPACE_SIZE;
        }
        return ge::GRAPH_SUCCESS;
    }

    // STREAM scene: large (>= 1M elems) same-dtype FP16/BF16 rank-2 single-sided broadcast.
    if (scene == HYPOT_SCENE_MULTI_DIM_BRC && sameHalfDtype && fs.totalLength >= HYPOT_STREAM_MIN_ELEMS) {
        uint32_t x1StreamMode = HYPOT_BRC_GENERIC;
        uint32_t x2StreamMode = HYPOT_BRC_GENERIC;
        if (MatchStreamPattern(fs, x1StreamMode, x2StreamMode) &&
            ApplyStreamTiling(fs, budget, coreNum, x1StreamMode, x2StreamMode, *tilingData)) {
            scene = HYPOT_SCENE_MULTI_DIM_STREAM;
        }
    }

    if (scene == HYPOT_SCENE_ONE_DIM) {
        // maxElemNum = budget/bytesPerElem, floor to 128B (32 elements)
        uint32_t maxElem = FloorAlign(static_cast<uint64_t>(budget) / oneDimBytesPerElem, HYPOT_ALIGN_128B);
        uint64_t total = fs.totalLength;
        const uint64_t roundedTotal = (total + HYPOT_ALIGN_128B - 1) / HYPOT_ALIGN_128B * HYPOT_ALIGN_128B;
        const uint64_t desiredTile = std::max<uint64_t>(HYPOT_ONEDIM_MIN_UB, roundedTotal);
        uint32_t ubFormer = desiredTile >= maxElem ? maxElem : static_cast<uint32_t>(desiredTile);
        uint32_t shrinkIter = 0;
        while (true) {
            uint32_t ubOuter = CeilDivU64(total, ubFormer);
            const uint32_t targetCoreNum = std::min(coreNum, std::max(minUsefulCores, ubOuter));
            ComputeCoreSplit(ubOuter, targetCoreNum, *tilingData);
            // shrink ubFormer to feed more cores
            if (tilingData->usedCoreNum >= targetCoreNum || ubFormer <= HYPOT_ONEDIM_MIN_UB ||
                shrinkIter++ > HYPOT_SHRINK_MAX_ITER) {
                tilingData->ubFormer = ubFormer;
                tilingData->ubOuter = ubOuter;
                tilingData->ubTail = static_cast<uint32_t>(total - static_cast<uint64_t>(ubOuter - 1) * ubFormer);
                break;
            }
            ubFormer = (ubFormer - HYPOT_ALIGN_128B < HYPOT_ONEDIM_MIN_UB) ? HYPOT_ONEDIM_MIN_UB :
                                                                             ubFormer - HYPOT_ALIGN_128B;
        }
    } else if (scene != HYPOT_SCENE_MULTI_DIM_STREAM) {
        bool forceSingleCore = false;
        // UB split axis = first axis (from innermost) whose cumulative product
        // exceeds maxElemNum; broadcast axes stay inside/at ubSplitAxis
        uint32_t maxElem = FloorAlign(static_cast<uint64_t>(budget) / multiDimBytesPerElem, HYPOT_ALIGN_256B);
        tilingData->dimNum = fs.dimNum;
        for (uint32_t k = 0; k < fs.dimNum; k++) {
            tilingData->outShape[k] = fs.outDims[k];
            tilingData->x1Strides[k] = fs.x1Strides[k];
            tilingData->x2Strides[k] = fs.x2Strides[k];
            tilingData->outStrides[k] = fs.outStrides[k];
        }
        uint64_t curProduct = 1;
        uint32_t ubSplitAxis = 0;
        bool allFit = true;
        for (int64_t k = static_cast<int64_t>(fs.dimNum) - 1; k >= 0; k--) {
            curProduct *= fs.outDims[k];
            if (curProduct > maxElem) {
                ubSplitAxis = static_cast<uint32_t>(k);
                curProduct /= fs.outDims[k];
                allFit = false;
                break;
            }
        }
        if (allFit) {
            curProduct /= fs.outDims[0];
        }
        uint32_t innerLen = static_cast<uint32_t>(curProduct == 0 ? 1 : curProduct);
        uint32_t rowPitch = (innerLen + HYPOT_ALIGN_BLK - 1) / HYPOT_ALIGN_BLK * HYPOT_ALIGN_BLK;
        // ubFormer is budgeted by innerLen (not the 32B-padded rowPitch); tile UB = CeilAlign(R*innerLen,64).
        uint32_t ubFormerMax = maxElem / innerLen;
        if (ubFormerMax == 0) {
            ubFormerMax = 1;
        }
        uint64_t dimProduct = 1;
        for (uint32_t k = 0; k < ubSplitAxis; k++) {
            dimProduct *= fs.outDims[k];
        }

        uint32_t x1Mode = 0;
        uint32_t x2Mode = 0;
        uint32_t x1Lead = 0;
        uint32_t x2Lead = 0;
        uint32_t x1Brc = 0;
        uint32_t x2Brc = 0;
        uint32_t x1Src = 0;
        uint32_t x2Src = 0;
        ComputeInputMode(fs, fs.x1Strides, true, ubSplitAxis, innerLen, x1Mode, x1Lead, x1Brc, x1Src);
        ComputeInputMode(fs, fs.x2Strides, false, ubSplitAxis, innerLen, x2Mode, x2Lead, x2Brc, x2Src);

        bool forceSplitAxis = false;
        if (fs.dimNum == 3U) {
            const bool x1Full = fs.x1Strides[0] != 0 && fs.x1Strides[1] != 0 && fs.x1Strides[2] != 0;
            const bool x2Full = fs.x2Strides[0] != 0 && fs.x2Strides[1] != 0 && fs.x2Strides[2] != 0;
            const bool x1Cross = fs.x1Strides[0] == 0 && fs.x1Strides[1] != 0 && fs.x1Strides[2] == 0;
            const bool x2Cross = fs.x2Strides[0] == 0 && fs.x2Strides[1] != 0 && fs.x2Strides[2] == 0;
            const bool x1Complementary = fs.x1Strides[0] != 0 && fs.x1Strides[1] == 0 && fs.x1Strides[2] != 0;
            const bool x2Complementary = fs.x2Strides[0] != 0 && fs.x2Strides[1] == 0 && fs.x2Strides[2] != 0;
            const bool middleSplitBroadcast = (x1Cross && x2Full) || (x1Full && x2Cross);
            const bool outerTailCross = ((x1Cross && x2Complementary) || (x1Complementary && x2Cross)) &&
                                        fs.totalLength <= 256U;
            if (outerTailCross) {
                forceSplitAxis = true;
                forceSingleCore = fs.totalLength <= 256U;
            } else if (middleSplitBroadcast) {
                ubSplitAxis = 1U;
                innerLen = fs.outDims[2];
                dimProduct = fs.outDims[0];
                ubFormerMax = std::max<uint32_t>(1U, maxElem / std::max<uint32_t>(1U, innerLen));
                ComputeInputMode(fs, fs.x1Strides, true, ubSplitAxis, innerLen, x1Mode, x1Lead, x1Brc, x1Src);
                ComputeInputMode(fs, fs.x2Strides, false, ubSplitAxis, innerLen, x2Mode, x2Lead, x2Brc, x2Src);
                forceSplitAxis = true;
                forceSingleCore = fs.totalLength <= 256U;
            } else if (fs.outDims[2] >= 128U && fs.totalLength <= 4096U) {
                ubSplitAxis = 2U;
                innerLen = 1U;
                dimProduct = static_cast<uint64_t>(fs.outDims[0]) * fs.outDims[1];
                ubFormerMax = maxElem;
                ComputeInputMode(fs, fs.x1Strides, true, ubSplitAxis, innerLen, x1Mode, x1Lead, x1Brc, x1Src);
                ComputeInputMode(fs, fs.x2Strides, false, ubSplitAxis, innerLen, x2Mode, x2Lead, x2Brc, x2Src);
                forceSplitAxis = true;
            }
        }

        // For small multi-axis broadcasts, prefer a split axis that reduces the expansion to NONE/ROW/COL.
        if (!forceSplitAxis && fs.totalLength >= HYPOT_REAXIS_MIN_ELEMS && fs.totalLength <= 65536U) {
            uint32_t bestAxis = ubSplitAxis;
            uint32_t bestInner = innerLen;
            uint32_t bestRank = std::max(BroadcastModeRank(x1Mode), BroadcastModeRank(x2Mode));
            bool bestBadCol = HasUnalignedWideColumn(fs, fs.x1Strides, ubSplitAxis, x1Mode, x1Lead, x1Brc) ||
                              HasUnalignedWideColumn(fs, fs.x2Strides, ubSplitAxis, x2Mode, x2Lead, x2Brc);
            bestBadCol = bestBadCol || HasUnalignedWideRow(x1Mode, x1Src) || HasUnalignedWideRow(x2Mode, x2Src);
            const uint32_t bestUbFormerMax = std::max<uint32_t>(
                1U, static_cast<uint32_t>(maxElem / std::max<uint32_t>(1U, innerLen)));
            const uint32_t bestUbFormer = std::min<uint32_t>(fs.outDims[ubSplitAxis], bestUbFormerMax);
            const uint64_t bestUbOuter = (fs.outDims[ubSplitAxis] + bestUbFormer - 1) / bestUbFormer;
            uint64_t bestFusedProduct = dimProduct * bestUbOuter;
            const uint32_t coverageTarget = static_cast<uint32_t>(std::max<uint64_t>(
                1U, std::min<uint64_t>(HYPOT_MIN_USEFUL_CORES,
                                       (fs.totalLength + HYPOT_MIN_ELEMS_PER_CORE - 1) / HYPOT_MIN_ELEMS_PER_CORE)));
            uint32_t bestCoverage = static_cast<uint32_t>(std::min<uint64_t>(dimProduct * bestUbOuter, coverageTarget));
            uint32_t bestTileElems = bestUbFormer * innerLen;
            for (uint32_t cand = 0; cand < fs.dimNum; cand++) {
                uint64_t candInner = 1;
                for (uint32_t k = cand + 1; k < fs.dimNum; k++) {
                    candInner *= fs.outDims[k];
                }
                if (candInner == 0 || candInner > maxElem) {
                    continue;
                }
                uint32_t c1Mode = 0;
                uint32_t c2Mode = 0;
                uint32_t c1Lead = 0;
                uint32_t c2Lead = 0;
                uint32_t c1Brc = 0;
                uint32_t c2Brc = 0;
                uint32_t c1Src = 0;
                uint32_t c2Src = 0;
                uint32_t candInner32 = static_cast<uint32_t>(candInner);
                ComputeInputMode(fs, fs.x1Strides, true, cand, candInner32, c1Mode, c1Lead, c1Brc, c1Src);
                ComputeInputMode(fs, fs.x2Strides, false, cand, candInner32, c2Mode, c2Lead, c2Brc, c2Src);
                const uint32_t candRank = std::max(BroadcastModeRank(c1Mode), BroadcastModeRank(c2Mode));
                const bool candBadCol = HasUnalignedWideColumn(fs, fs.x1Strides, cand, c1Mode, c1Lead, c1Brc) ||
                                        HasUnalignedWideColumn(fs, fs.x2Strides, cand, c2Mode, c2Lead, c2Brc) ||
                                        HasUnalignedWideRow(c1Mode, c1Src) || HasUnalignedWideRow(c2Mode, c2Src);
                uint64_t candDimProduct = 1;
                for (uint32_t k = 0; k < cand; k++) {
                    candDimProduct *= fs.outDims[k];
                }
                const uint32_t candUbFormerMax = std::max<uint32_t>(
                    1U, static_cast<uint32_t>(maxElem / std::max<uint32_t>(1U, candInner32)));
                const uint32_t candUbFormer = std::min<uint32_t>(fs.outDims[cand], candUbFormerMax);
                const uint64_t candUbOuter = (fs.outDims[cand] + candUbFormer - 1) / candUbFormer;
                const uint64_t candFusedProduct = candDimProduct * candUbOuter;
                const uint32_t candCoverage = static_cast<uint32_t>(
                    std::min<uint64_t>(candDimProduct * candUbOuter, coverageTarget));
                const uint32_t candTileElems = candUbFormer * candInner32;
                // Keep a materially larger tile over dozens of 8-element tiles once the minimum core count is fed.
                const bool candEligible = candTileElems >= HYPOT_MIN_TILE_ELEMS;
                const bool bestEligible = bestTileElems >= HYPOT_MIN_TILE_ELEMS;
                bool better;
                if (candEligible != bestEligible) {
                    better = candEligible;
                } else if (candRank != bestRank) {
                    const bool rankGainCutsTile = candRank + 1U == bestRank && bestRank <= HYPOT_BRC_COL &&
                                                  fs.totalLength > HYPOT_CORE_SHRINK_ELEMS &&
                                                  2U * candTileElems < bestTileElems;
                    const bool rankTuneEnabled = isFp32 && fs.dimNum == 3U && fs.totalLength <= 1024U;
                    const bool rankTradeCutsTileCount = rankTuneEnabled && candRank == bestRank + 1U &&
                                                        2U * candFusedProduct <= bestFusedProduct &&
                                                        candTileElems >= bestTileElems;
                    const bool rankGainWorsensTileCount = rankTuneEnabled && candFusedProduct > bestFusedProduct;
                    better = (candRank < bestRank && !rankGainCutsTile && !rankGainWorsensTileCount) ||
                             rankTradeCutsTileCount;
                } else if (candCoverage != bestCoverage) {
                    better = candCoverage > bestCoverage;
                } else if (candBadCol != bestBadCol) {
                    better = candBadCol < bestBadCol;
                } else {
                    better = candTileElems > bestTileElems;
                }
                if (better) {
                    bestAxis = cand;
                    bestInner = candInner32;
                    bestRank = candRank;
                    bestBadCol = candBadCol;
                    bestCoverage = candCoverage;
                    bestTileElems = candTileElems;
                    bestFusedProduct = candFusedProduct;
                }
            }
            if (bestRank < HYPOT_BRC_GENERIC) {
                ubSplitAxis = bestAxis;
                innerLen = bestInner;
                rowPitch = (innerLen + HYPOT_ALIGN_BLK - 1) / HYPOT_ALIGN_BLK * HYPOT_ALIGN_BLK;
                ubFormerMax = maxElem / innerLen;
                if (ubFormerMax == 0) {
                    ubFormerMax = 1;
                }
                dimProduct = 1;
                for (uint32_t k = 0; k < ubSplitAxis; k++) {
                    dimProduct *= fs.outDims[k];
                }
                ComputeInputMode(fs, fs.x1Strides, true, ubSplitAxis, innerLen, x1Mode, x1Lead, x1Brc, x1Src);
                ComputeInputMode(fs, fs.x2Strides, false, ubSplitAxis, innerLen, x2Mode, x2Lead, x2Brc, x2Src);
            }
        }

        const bool genericBroadcast = x1Mode == HYPOT_BRC_GENERIC || x2Mode == HYPOT_BRC_GENERIC;
        const bool colBroadcast = x1Mode == HYPOT_BRC_COL || x2Mode == HYPOT_BRC_COL;
        const bool heavyBroadcast = genericBroadcast || colBroadcast;

        uint32_t ubFormer = ubFormerMax;
        uint32_t shrinkIter = 0;
        const bool colNeedsRows = fs.dimNum <= 3U && colBroadcast && !genericBroadcast &&
                                  ((x1Mode == HYPOT_BRC_COL &&
                                    (fs.x1Strides[ubSplitAxis] != 0 || x1Lead * x1Src > 1U)) ||
                                   (x2Mode == HYPOT_BRC_COL &&
                                    (fs.x2Strides[ubSplitAxis] != 0 || x2Lead * x2Src > 1U)));
        if (colNeedsRows) {
            // Column expansion is efficient when the split extent is 8-aligned and the tile has a few thousand
            // elements.
            uint32_t desiredRows = static_cast<uint32_t>((4096U + innerLen - 1) / innerLen);
            desiredRows = std::max<uint32_t>(8U, desiredRows);
            desiredRows = std::min(desiredRows, ubFormerMax);
            if (desiredRows >= HYPOT_ALIGN_BLK) {
                desiredRows = FloorAlign(desiredRows, HYPOT_ALIGN_BLK);
            }
            ubFormer = std::max<uint32_t>(1U, std::min(desiredRows, fs.outDims[ubSplitAxis]));
            const uint32_t ubOuter = CeilDivU64(fs.outDims[ubSplitAxis], ubFormer);
            const uint64_t fusedProduct = dimProduct * ubOuter;
            const uint32_t naturalCoreNum = static_cast<uint32_t>(std::min<uint64_t>(fusedProduct, coreNum));
            ComputeCoreSplit(fusedProduct, std::max(minUsefulCores, naturalCoreNum), *tilingData);
            tilingData->ubFormer = ubFormer;
            tilingData->ubOuter = ubOuter;
            tilingData->ubTail = fs.outDims[ubSplitAxis] - (ubOuter - 1) * ubFormer;
        } else {
            while (true) {
                uint32_t ubOuter = CeilDivU64(fs.outDims[ubSplitAxis], ubFormer);
                uint64_t fusedProduct = dimProduct * ubOuter;
                const uint32_t naturalCoreNum = static_cast<uint32_t>(std::min<uint64_t>(fusedProduct, coreNum));
                uint64_t effectiveMinElemsPerCore = fs.totalLength <= 256U ?
                                                        ((x1Mode <= HYPOT_BRC_ROW && x2Mode <= HYPOT_BRC_ROW) ? 48U :
                                                                                                                64U) :
                                                        HYPOT_MIN_ELEMS_PER_CORE;
                const uint32_t workLimitedCoreNum = fs.totalLength <= 65536U ?
                                                        static_cast<uint32_t>(std::max<uint64_t>(
                                                            1U, (fs.totalLength + effectiveMinElemsPerCore - 1) /
                                                                    effectiveMinElemsPerCore)) :
                                                        coreNum;
                const uint32_t targetCoreNum = heavyBroadcast ? std::min(coreNum, workLimitedCoreNum) :
                                                                std::max(minUsefulCores,
                                                                         std::min(naturalCoreNum, workLimitedCoreNum));
                ComputeCoreSplit(fusedProduct, std::min(coreNum, targetCoreNum), *tilingData);
                if (tilingData->usedCoreNum >= std::min(coreNum, targetCoreNum) || ubFormer <= 1 ||
                    shrinkIter++ > HYPOT_SHRINK_MAX_ITER) {
                    tilingData->ubFormer = ubFormer;
                    tilingData->ubOuter = ubOuter;
                    tilingData->ubTail = fs.outDims[ubSplitAxis] - (ubOuter - 1) * ubFormer;
                    break;
                }
                ubFormer--;
            }
        }

        if ((HYPOT_ENABLE_TINY_CROSS3D && tilingData->scalarFlag == 0U && IsTinyCross3DPattern(fs)) ||
            (HYPOT_ENABLE_TINY_PATTERN && isFp32 && tilingData->scalarFlag == 0U && IsTinyPattern(fs, true))) {
            forceSingleCore = true;
        }

        tilingData->ubSplitAxis = ubSplitAxis;
        tilingData->ubFormer = std::min(tilingData->ubFormer, fs.outDims[ubSplitAxis]);
        tilingData->dimProduct = static_cast<uint32_t>(dimProduct);
        tilingData->innerLen = innerLen;
        tilingData->rowPitch = rowPitch;

        tilingData->x1BrcMode = x1Mode;
        tilingData->x1LeadInner = x1Lead;
        tilingData->x1BrcInner = x1Brc;
        tilingData->x1SrcInner = x1Src;
        tilingData->x2BrcMode = x2Mode;
        tilingData->x2LeadInner = x2Lead;
        tilingData->x2BrcInner = x2Brc;
        tilingData->x2SrcInner = x2Src;
        uint64_t x1TileSrc = static_cast<uint64_t>(tilingData->x1LeadInner) * tilingData->x1SrcInner;
        uint64_t x2TileSrc = static_cast<uint64_t>(tilingData->x2LeadInner) * tilingData->x2SrcInner;
        bool x1Full = fs.x1Strides[ubSplitAxis] != 0;
        bool x2Full = fs.x2Strides[ubSplitAxis] != 0;
        tilingData->x1TileLen = static_cast<uint32_t>((x1Full ? tilingData->ubFormer : 1) * x1TileSrc);
        tilingData->x2TileLen = static_cast<uint32_t>((x2Full ? tilingData->ubFormer : 1) * x2TileSrc);

        const bool x1ScalarSource = fs.x1Numel > 1 && fs.x1Numel <= HYPOT_PRELOAD_MAX_ELEMS &&
                                    fs.x1Strides[ubSplitAxis] == 0 && tilingData->x1LeadInner == 1 &&
                                    tilingData->x1SrcInner == 1;
        const bool x2ScalarSource = fs.x2Numel > 1 && fs.x2Numel <= HYPOT_PRELOAD_MAX_ELEMS &&
                                    fs.x2Strides[ubSplitAxis] == 0 && tilingData->x2LeadInner == 1 &&
                                    tilingData->x2SrcInner == 1;
        // the fused-COL preload is a same-dtype FP32 kernel feature (kernel kPreloadColSupported)
        const bool fp32FusedColPreload = isFp32 && sameDtype && (fs.dimNum == 4U || fs.totalLength <= 256U);
        const bool x1PreloadSource = x1ScalarSource || (fp32FusedColPreload && x1Mode == HYPOT_BRC_COL);
        const bool x2PreloadSource = x2ScalarSource || (fp32FusedColPreload && x2Mode == HYPOT_BRC_COL);
        tilingData->x1PreloadLen = x1PreloadSource ? static_cast<uint32_t>(fs.x1Numel) : 0U;
        tilingData->x2PreloadLen = x2PreloadSource ? static_cast<uint32_t>(fs.x2Numel) : 0U;
        if (forceSingleCore) {
            const uint64_t fusedProduct = static_cast<uint64_t>(tilingData->dimProduct) * tilingData->ubOuter;
            ComputeCoreSplit(fusedProduct, 1U, *tilingData);
        }

        // STRUCT scene: structured small-shape paths isolated in their own TilingKey (same-dtype only).
        if (sameDtype && tilingData->scalarFlag == 0U &&
            (IsCross3DStructPattern(fs, isFp32) || HasHighRankStructRow(fs, x1Mode, x1Src, x2Mode, x2Src))) {
            scene = HYPOT_SCENE_MULTI_DIM_STRUCT;
        }

        // VECIDX scene: remaining small (<= 1024 elems) multi-dim broadcasts go to the single-core
        // whole-output Gather kernel; STRUCT keeps priority for its already-tuned patterns.
        if (scene == HYPOT_SCENE_MULTI_DIM_BRC && tilingData->scalarFlag == 0U && fs.dimNum >= 2U &&
            fs.totalLength <= HYPOT_VECIDX_MAX_ELEMS) {
            tilingData->usedCoreNum = 1U;
            tilingData->blockFormer = 1U;
            tilingData->blockTail = 1U;
            tilingData->ubFormer = static_cast<uint32_t>(fs.totalLength);
            tilingData->ubTail = static_cast<uint32_t>(fs.totalLength);
            tilingData->ubOuter = 1U;
            tilingData->x1PreloadLen = 0U;
            tilingData->x2PreloadLen = 0U;
            scene = HYPOT_SCENE_MULTI_DIM_VECIDX;
        }
    }

    if (scene == HYPOT_SCENE_ONE_DIM) {
        tilingData->x1PreloadLen = 0U;
        tilingData->x2PreloadLen = 0U;
    }

    tilingContext->SetBlockDim(tilingData->usedCoreNum);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(scene);
    tilingContext->SetTilingKey(tilingKey);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = HYPOT_WORK_SPACE_SIZE;
    }
    OP_LOGD(tilingContext,
            "scene:%u core:%u ubFormer:%u ubTail:%u ubOuter:%u dim:%u axis:%u inner:%u modes:%u/%u pre:%u/%u", scene,
            tilingData->usedCoreNum, tilingData->ubFormer, tilingData->ubTail, tilingData->ubOuter, tilingData->dimNum,
            tilingData->ubSplitAxis, tilingData->innerLen, tilingData->x1BrcMode, tilingData->x2BrcMode,
            tilingData->x1PreloadLen, tilingData->x2PreloadLen);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4HypotArch22(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<HypotCompileInfoArch22>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context, "GetHardwareInfo Failed, totalCoreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
                        compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Hypot).Tiling(HypotTilingArch22).TilingParse<HypotCompileInfoArch22>(TilingPrepare4HypotArch22);
} // namespace optiling
