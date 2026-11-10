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
 * \file hypot_tiling_data.h
 * \brief hypot tiling template declaration and tiling data struct (shared by host tiling and kernel)
 */

#ifndef HYPOT_TILING_DATA_H
#define HYPOT_TILING_DATA_H

#include "ascendc/host_api/tiling/template_argument.h"

#include <cstdint>

namespace NsHypot {

// TilingKey encodes only the broadcast scene; dtypes come from the per-binary
// DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT aliases.

// broadcast scene ids (the only TilingKey dimension)
#define HYPOT_SCENE_ONE_DIM 0U
#define HYPOT_SCENE_MULTI_DIM_BRC 1U
// MultiDimBroadcast plus structured small-shape extensions, isolated so the default BRC binary stays unchanged.
#define HYPOT_SCENE_MULTI_DIM_STRUCT 2U
// Large (>= 1M elems) same-dtype FP16/BF16 2D single-sided broadcast: the expansion engine buffers
// are not allocated, so the tile is larger. FP16/BF16 only.
#define HYPOT_SCENE_MULTI_DIM_STREAM 3U
// Small (<= 1024 elems) multi-dim broadcast: single core, whole output in UB, one Gather per input.
#define HYPOT_SCENE_MULTI_DIM_VECIDX 4U

// per-input broadcast expansion mode (MultiDimBroadcast scene)
#define HYPOT_BRC_NONE 0U    // no broadcast axis in tile, plain copy
#define HYPOT_BRC_ROW 1U     // single broadcast group, trailing full product I > 1: [G,B,I] row expansion
#define HYPOT_BRC_COL 2U     // single broadcast group, I == 1 (innermost broadcast): splat expansion
#define HYPOT_BRC_GENERIC 3U // multi-group broadcast: scalar index-mapping fallback

// scalar input bitmap bits of HypotTilingData.scalarFlag: bit0 = x1 is scalar, bit1 = x2 is scalar
constexpr uint32_t HYPOT_SCALAR_FLAG_X1 = 0x1U;
constexpr uint32_t HYPOT_SCALAR_FLAG_X2 = 0x2U;

ASCENDC_TPL_ARGS_DECL(Hypot,
                      ASCENDC_TPL_UINT_DECL(BROADCAST_SCENE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, HYPOT_SCENE_ONE_DIM,
                                            HYPOT_SCENE_MULTI_DIM_BRC, HYPOT_SCENE_MULTI_DIM_STRUCT,
                                            HYPOT_SCENE_MULTI_DIM_STREAM, HYPOT_SCENE_MULTI_DIM_VECIDX), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BROADCAST_SCENE, ASCENDC_TPL_UI_RANGE, 1, HYPOT_SCENE_ONE_DIM,
                                              HYPOT_SCENE_ONE_DIM), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BROADCAST_SCENE, ASCENDC_TPL_UI_RANGE, 1, HYPOT_SCENE_MULTI_DIM_BRC,
                                              HYPOT_SCENE_MULTI_DIM_BRC), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BROADCAST_SCENE, ASCENDC_TPL_UI_RANGE, 1, HYPOT_SCENE_MULTI_DIM_STRUCT,
                                              HYPOT_SCENE_MULTI_DIM_STRUCT), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BROADCAST_SCENE, ASCENDC_TPL_UI_RANGE, 1, HYPOT_SCENE_MULTI_DIM_STREAM,
                                              HYPOT_SCENE_MULTI_DIM_STREAM), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(BROADCAST_SCENE, ASCENDC_TPL_UI_RANGE, 1, HYPOT_SCENE_MULTI_DIM_VECIDX,
                                              HYPOT_SCENE_MULTI_DIM_VECIDX), ), );

struct HypotTilingData {
    // ---- common: multi-core split (unit: tile) and UB split ----
    uint64_t totalLength = 0;    // broadcast output total element num
    uint32_t usedCoreNum = 0;    // actual used AIV core num (blockDim)
    uint32_t blockFormer = 0;    // tiles per non-last core (unit: tile)
    uint32_t blockTail = 0;      // tiles of last core (unit: tile)
    uint32_t ubFormer = 0;       // OneDim: elements per tile; MultiDim: rows per tile on ubSplitAxis
    uint32_t ubTail = 0;         // OneDim: tail tile elements; MultiDim: tail tile rows on ubSplitAxis
    uint32_t ubOuter = 0;        // tile count on ubSplitAxis = ceil(dims[ubSplitAxis]/ubFormer)
    uint32_t tileNum = 0;        // reserved (kernel derives per-core tile num from blockFormer/blockTail)
    uint32_t scalarFlag = 0;     // scalar input bitmap: bit0=x1, bit1=x2 (fused input numel == 1)
    uint32_t x1PreloadLen = 0;   // small x1 source staged in UB once at Init (0 = disabled)
    uint32_t x2PreloadLen = 0;   // small x2 source staged in UB once at Init (0 = disabled)
    uint32_t nativeHalfFlag = 0; // FP16 native vector chain is allowed for this launch
    uint32_t x1Numel = 0;        // compact input element count (mixed small-kernel path)
    uint32_t x2Numel = 0;

    // ---- MultiDimBroadcast only (0 / ignored in OneDim) ----
    uint32_t dimNum = 0;          // fused dim num (<= 8)
    uint32_t outShape[8] = {0};   // fused output dims
    uint32_t x1Strides[8] = {0};  // x1 strides in elements (broadcast axis stride = 0)
    uint32_t x2Strides[8] = {0};  // x2 strides in elements (broadcast axis stride = 0)
    uint32_t outStrides[8] = {0}; // output strides in elements
    uint32_t ubSplitAxis = 0;     // UB split axis
    uint32_t dimProduct = 0;      // product of dims[0..ubSplitAxis-1]
    uint32_t x1TileLen = 0;       // x1 source elements of a full tile (unbroadcast)
    uint32_t x2TileLen = 0;       // x2 source elements of a full tile (unbroadcast)
    uint32_t innerLen = 0;        // C = product of dims[ubSplitAxis+1..dimNum-1]
    uint32_t rowPitch = 0;        // ceil_align(innerLen, 8), padded row pitch in elements (32B)
    uint32_t x1BrcMode = 0;       // HYPOT_BRC_*
    uint32_t x2BrcMode = 0;       // HYPOT_BRC_*
    uint32_t x1LeadInner = 0;     // L: product of full inner axes before the broadcast run (after ubSplitAxis)
    uint32_t x2LeadInner = 0;
    uint32_t x1BrcInner = 0; // B_inner: broadcast extent inside the inner region
    uint32_t x2BrcInner = 0;
    uint32_t x1SrcInner = 0; // I: trailing full product after the broadcast run
    uint32_t x2SrcInner = 0;
};

} // namespace NsHypot
#endif // HYPOT_TILING_DATA_H
