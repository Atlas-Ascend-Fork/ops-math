/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_tiling_with_nchwconv_arch31.h"

#include <algorithm>
#include <limits>

namespace optiling {
namespace {
constexpr uint64_t TRANS_ELEMS = 16;
constexpr uint64_t BUFFER_COUNT = 4; // Input/output queues, each double buffered.
constexpr uint64_t KIRIN_MAX_COPY_BLOCK_COUNT = std::numeric_limits<uint16_t>::max();
constexpr uint64_t KIRIN_MAX_REPEAT = 255;

struct CoreSplit {
    uint64_t alignFactor = 0;
    uint64_t factor = 0;
    uint64_t count = 0;
    uint64_t alignTail = 0;
    uint64_t tail = 0;
};

struct UbSplit {
    uint64_t mainAlign = 0;
    uint64_t main = 0;
    uint64_t mainCount = 0;
    uint64_t mainTailAlign = 0;
    uint64_t mainTail = 0;
    uint64_t tailAlign = 0;
    uint64_t tail = 0;
    uint64_t tailCount = 0;
    uint64_t tailTailAlign = 0;
    uint64_t tailTail = 0;
};

struct TilingState {
    uint64_t availableUb = 0;
    uint64_t usedCores = 0;
    uint64_t mainLoops = 0;
    uint64_t tailLoops = 0;
    uint64_t r = 0;
    uint64_t c = 0;
    uint64_t rBlocks = 0;
    uint64_t cBlocks = 0;
    uint64_t rAligned = 0;
    uint64_t cAligned = 0;
    bool rSplit = false;
    bool rcSplit = false;
    CoreSplit rCore;
    CoreSplit cCore;
    UbSplit rUb;
    UbSplit cUb;
};

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return value / divisor + (value % divisor != 0); }

void SplitCore(uint64_t alignedBlocks, uint64_t coreCount, CoreSplit& split, uint64_t& usedCores)
{
    split.alignFactor = CeilDiv(alignedBlocks, coreCount);
    split.factor = split.alignFactor * TRANS_ELEMS;
    split.count = CeilDiv(alignedBlocks, split.alignFactor);
    split.alignTail = alignedBlocks - (split.count - 1) * split.alignFactor;
    split.tail = split.alignTail * TRANS_ELEMS;
    usedCores = split.count;
}

void SetFullCoreSplit(uint64_t blocks, uint64_t aligned, CoreSplit& split)
{
    split = {blocks, aligned, 1, blocks, aligned};
}

void SetFullUbSplit(uint64_t blocks, uint64_t aligned, UbSplit& split)
{
    split = {blocks, aligned, 1, blocks, aligned, blocks, aligned, 1, blocks, aligned};
}

void SetCoreFactorFullUb(const CoreSplit& core, UbSplit& split)
{
    split = {core.alignFactor, core.factor, 1, core.alignFactor, core.factor,
             core.alignTail,   core.tail,   1, core.alignTail,   core.tail};
}

bool SplitUbFactor(uint64_t availableUb, uint64_t elementBytes, uint64_t otherAligned, const CoreSplit& core,
                   UbSplit& split)
{
    const uint64_t alignFactor = availableUb / elementBytes / otherAligned / TRANS_ELEMS;
    if (alignFactor == 0) {
        return false;
    }
    split.mainAlign = alignFactor;
    split.main = alignFactor * TRANS_ELEMS;
    split.mainCount = CeilDiv(core.factor, split.main);
    split.mainTailAlign = core.alignFactor - (split.mainCount - 1) * split.mainAlign;
    split.mainTail = split.mainTailAlign * TRANS_ELEMS;
    split.tailAlign = std::min(split.mainAlign, core.alignTail);
    split.tail = split.tailAlign * TRANS_ELEMS;
    split.tailCount = CeilDiv(core.tail, split.tail);
    split.tailTailAlign = core.alignTail - (split.tailCount - 1) * split.tailAlign;
    split.tailTail = split.tailTailAlign * TRANS_ELEMS;
    return true;
}

bool IsCanonicalB16Transpose(const KirinTransposeShapeInfo& shape, const KirinReducedShapeInfo& reduced)
{
    if (shape.elementBytes != sizeof(uint16_t) || reduced.rank != 2 || reduced.outShape[0] == 0 ||
        reduced.outShape[1] <= 5) {
        return false;
    }
    const uint64_t c = reduced.outShape[0];
    const uint64_t r = reduced.outShape[1];
    if (c > std::numeric_limits<uint64_t>::max() / r || shape.totalElements != r * c) {
        return false;
    }
    return reduced.srcStride[0] == 1 && reduced.srcStride[1] == c && reduced.outStride[0] == r &&
           reduced.outStride[1] == 1;
}

bool ValidateKirinLimits(const TilingState& state)
{
    const uint64_t maxCopyBytes = std::numeric_limits<uint32_t>::max();
    const uint64_t maxDimension = std::max(state.r, state.c);
    if (state.availableUb > std::numeric_limits<uint32_t>::max() || maxDimension > maxCopyBytes / sizeof(uint16_t) ||
        state.c > KIRIN_MAX_COPY_BLOCK_COUNT) {
        return false;
    }

    const uint64_t maxRFactor = std::max({state.rUb.main, state.rUb.mainTail, state.rUb.tail, state.rUb.tailTail});
    const uint64_t maxCFactor = std::max({state.cUb.main, state.cUb.mainTail, state.cUb.tail, state.cUb.tailTail});
    if (maxRFactor == 0 || maxCFactor == 0 || maxRFactor > KIRIN_MAX_COPY_BLOCK_COUNT ||
        maxCFactor > KIRIN_MAX_COPY_BLOCK_COUNT) {
        return false;
    }
    if (!state.rSplit && state.r > KIRIN_MAX_COPY_BLOCK_COUNT) {
        return false;
    }

    const uint64_t maxRepeat = std::max(maxRFactor, maxCFactor) / TRANS_ELEMS;
    return maxRepeat <= KIRIN_MAX_REPEAT;
}

void FillCore(KirinTransposeVConvCoreSplitTilingData& output, const CoreSplit& input)
{
    output.set_alignBlockFactor(input.alignFactor);
    output.set_blockFactor(input.factor);
    output.set_blockCount(input.count);
    output.set_alignBlockTailFactor(input.alignTail);
    output.set_blockTailFactor(input.tail);
}

void FillUb(KirinTransposeVConvUbSplitTilingData& output, const UbSplit& input)
{
    output.set_mainCoreUbAlignFactor(input.mainAlign);
    output.set_mainCoreUbFactor(input.main);
    output.set_mainCoreUbCount(input.mainCount);
    output.set_mainCoreTailUbAlignFactor(input.mainTailAlign);
    output.set_mainCoreTailUbFactor(input.mainTail);
    output.set_tailCoreUbAlignFactor(input.tailAlign);
    output.set_tailCoreUbFactor(input.tail);
    output.set_tailCoreUbCount(input.tailCount);
    output.set_tailCoreTailUbAlignFactor(input.tailTailAlign);
    output.set_tailCoreTailUbFactor(input.tailTail);
}

void FillTiling(KirinTransposeVConvOpTilingData& output, const TilingState& state)
{
    output.set_availableUbSize(state.availableUb);
    output.set_usedCoreNum(state.usedCores);
    output.set_mainCoreLoopCount(state.mainLoops);
    output.set_tailCoreLoopCount(state.tailLoops);
    output.set_rLen(state.r);
    output.set_cLen(state.c);
    output.set_rAlignBlock(state.rBlocks);
    output.set_cAlignBlock(state.cBlocks);
    output.set_rAlignBlockElem(state.rAligned);
    output.set_cAlignBlockElem(state.cAligned);
    output.set_isRSplit(state.rSplit);
    output.set_isRCSplit(state.rcSplit);
    FillCore(output.rSplitPara, state.rCore);
    FillCore(output.cSplitPara, state.cCore);
    FillUb(output.rUbSplitPara, state.rUb);
    FillUb(output.cUbSplitPara, state.cUb);
}
} // namespace

bool BuildKirinTransposeVConvTiling(const TransposeCompilerInfo& compilerInfo, const KirinTransposeShapeInfo& shapeInfo,
                                    const KirinReducedShapeInfo& reducedShape,
                                    KirinTransposeVConvTilingData& tilingData, uint32_t& blockDim)
{
    if (!IsCanonicalB16Transpose(shapeInfo, reducedShape) || compilerInfo.coreNum <= 0 || compilerInfo.ubSize <= 0) {
        return false;
    }

    TilingState state;
    state.availableUb = static_cast<uint64_t>(compilerInfo.ubSize) / BUFFER_COUNT;
    state.r = reducedShape.outShape[1];
    state.c = reducedShape.outShape[0];
    const uint64_t maxCopyDimension = std::numeric_limits<uint32_t>::max() / shapeInfo.elementBytes;
    if (state.availableUb > std::numeric_limits<uint32_t>::max() || state.r > maxCopyDimension ||
        state.c > maxCopyDimension) {
        return false;
    }
    state.rBlocks = CeilDiv(state.r, TRANS_ELEMS);
    state.cBlocks = CeilDiv(state.c, TRANS_ELEMS);
    state.rAligned = state.rBlocks * TRANS_ELEMS;
    state.cAligned = state.cBlocks * TRANS_ELEMS;
    const uint64_t coreCount = static_cast<uint64_t>(compilerInfo.coreNum);
    const uint64_t elementBytes = shapeInfo.elementBytes;

    if (TRANS_ELEMS * state.cAligned * elementBytes <= state.availableUb) {
        SplitCore(state.rBlocks, coreCount, state.rCore, state.usedCores);
        state.rSplit = true;
        SetFullCoreSplit(state.cBlocks, state.cAligned, state.cCore);
        SetFullUbSplit(state.cBlocks, state.cAligned, state.cUb);
        if (state.rCore.factor * state.cAligned * elementBytes <= state.availableUb) {
            SetCoreFactorFullUb(state.rCore, state.rUb);
            state.mainLoops = 1;
            state.tailLoops = 1;
        } else {
            if (!SplitUbFactor(state.availableUb, elementBytes, state.cAligned, state.rCore, state.rUb)) {
                return false;
            }
            state.mainLoops = state.rUb.mainCount;
            state.tailLoops = state.rUb.tailCount;
        }
    } else if (TRANS_ELEMS * state.rAligned * elementBytes <= state.availableUb) {
        SplitCore(state.cBlocks, coreCount, state.cCore, state.usedCores);
        SetFullCoreSplit(state.rBlocks, state.rAligned, state.rCore);
        SetFullUbSplit(state.rBlocks, state.rAligned, state.rUb);
        if (state.cCore.factor * state.rAligned * elementBytes <= state.availableUb) {
            SetCoreFactorFullUb(state.cCore, state.cUb);
            state.mainLoops = 1;
            state.tailLoops = 1;
        } else {
            if (!SplitUbFactor(state.availableUb, elementBytes, state.rAligned, state.cCore, state.cUb)) {
                return false;
            }
            state.mainLoops = state.cUb.mainCount;
            state.tailLoops = state.cUb.tailCount;
        }
    } else {
        SplitCore(state.rBlocks, coreCount, state.rCore, state.usedCores);
        state.rSplit = true;
        state.rcSplit = true;
        if (state.rCore.factor * TRANS_ELEMS * elementBytes > state.availableUb) {
            return false;
        }
        SetCoreFactorFullUb(state.rCore, state.rUb);
        SetFullCoreSplit(state.cBlocks, state.cAligned, state.cCore);
        const uint64_t cAlignFactor = state.availableUb / elementBytes / state.rCore.factor / TRANS_ELEMS;
        if (cAlignFactor == 0) {
            return false;
        }
        state.cUb.mainAlign = cAlignFactor;
        state.cUb.main = cAlignFactor * TRANS_ELEMS;
        state.cUb.mainCount = CeilDiv(state.cAligned, state.cUb.main);
        state.cUb.mainTailAlign = state.cBlocks - (state.cUb.mainCount - 1) * state.cUb.mainAlign;
        state.cUb.mainTail = state.cUb.mainTailAlign * TRANS_ELEMS;
        state.cUb.tailAlign = state.cUb.mainAlign;
        state.cUb.tail = state.cUb.main;
        state.cUb.tailCount = state.cUb.mainCount;
        state.cUb.tailTailAlign = state.cUb.mainTailAlign;
        state.cUb.tailTail = state.cUb.mainTail;
        state.mainLoops = state.cUb.mainCount;
        state.tailLoops = state.cUb.tailCount;
    }

    if (!ValidateKirinLimits(state) || state.usedCores > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    FillTiling(tilingData.transposeVConvTiling, state);
    blockDim = static_cast<uint32_t>(state.usedCores);
    return true;
}
} // namespace optiling
