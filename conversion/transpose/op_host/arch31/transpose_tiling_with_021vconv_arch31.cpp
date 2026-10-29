/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_tiling_with_021vconv_arch31.h"

#include <algorithm>
#include <limits>

namespace optiling {
namespace {
constexpr uint64_t TRANS_ELEMS = 16;
constexpr uint64_t MIN_HW_DIM = 8;
constexpr uint64_t MIN_HW_PRODUCT = 448;
constexpr uint64_t MIN_TOTAL_BYTES = 70000;
constexpr uint64_t BUFFER_COUNT = 2;
constexpr uint64_t INPUT_OUTPUT_COUNT = 2;
constexpr uint64_t MAX_REPEAT_TIMES = 255;
constexpr uint64_t MAX_UINT16 = std::numeric_limits<uint16_t>::max();
constexpr uint64_t UB_RESERVED_BYTES = 256;
constexpr uint64_t MAX_TWO_DIMENSIONAL_DIM = 4096;
constexpr int64_t MIN_TWO_DIMENSIONAL_UB_SIZE = 128 * 1024;

struct UbInfo {
    uint64_t alignFactor = 0;
    uint64_t factor = 0;
    uint64_t count = 0;
    uint64_t tailAlignFactor = 0;
    uint64_t tailFactor = 0;
};

struct BuildInfo {
    uint64_t availableUb = 0;
    uint64_t usedCores = 0;
    uint64_t n = 0;
    uint64_t h = 0;
    uint64_t w = 0;
    uint64_t hAlign = 0;
    uint64_t wAlign = 0;
    uint64_t nPerCore = 0;
    uint64_t nTailCore = 0;
    uint64_t loopCount = 0;
    uint64_t blockElems = 0;
    uint64_t hPerCore = 0;
    uint64_t hTailCore = 0;
    uint32_t elementBytes = 0;
    bool useRConv = false;
    bool useHSplit = false;
    bool useTwoDimensionalSplit = false;
    UbInfo r;
    UbInfo c;
};

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedProduct3(uint64_t first, uint64_t second, uint64_t third, uint64_t& result)
{
    return CheckedMultiply(first, second, result) && CheckedMultiply(result, third, result);
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result)
{
    const uint64_t units = CeilDiv(value, alignment);
    return CheckedMultiply(units, alignment, result);
}

bool Match021(const KirinTransposeShapeInfo& shapeInfo, const KirinReducedShapeInfo& reducedShape, uint64_t& n,
              uint64_t& h, uint64_t& w)
{
    if (reducedShape.rank != 3 ||
        (shapeInfo.elementBytes != 1 && shapeInfo.elementBytes != 2 && shapeInfo.elementBytes != 4)) {
        return false;
    }
    n = reducedShape.outShape[0];
    w = reducedShape.outShape[1];
    h = reducedShape.outShape[2];
    uint64_t hw = 0;
    if (n == 0 || h <= MIN_HW_DIM || w <= MIN_HW_DIM || !CheckedMultiply(h, w, hw) || hw < MIN_HW_PRODUCT) {
        return false;
    }
    if (reducedShape.outStride[2] != 1 || reducedShape.outStride[1] != h || reducedShape.outStride[0] != hw ||
        reducedShape.srcStride[2] != w || reducedShape.srcStride[1] != 1 || reducedShape.srcStride[0] != hw) {
        return false;
    }

    uint64_t hAlign = 0;
    uint64_t wAlign = 0;
    uint64_t alignedArea = 0;
    uint64_t doubledArea = 0;
    uint64_t totalElements = 0;
    uint64_t totalBytes = 0;
    return AlignUp(h, TRANS_ELEMS, hAlign) && AlignUp(w, TRANS_ELEMS, wAlign) &&
           CheckedMultiply(hAlign, wAlign, alignedArea) && CheckedMultiply(hw, 2, doubledArea) &&
           doubledArea > alignedArea && CheckedMultiply(n, hw, totalElements) &&
           totalElements == shapeInfo.totalElements &&
           CheckedMultiply(totalElements, shapeInfo.elementBytes, totalBytes) && totalBytes >= MIN_TOTAL_BYTES;
}

bool MatchTwoDimensionalVConv(const KirinTransposeShapeInfo& shapeInfo, const KirinReducedShapeInfo& reducedShape,
                              uint64_t& n, uint64_t& h, uint64_t& w)
{
    const bool supportedDtype = shapeInfo.inputDtype == ge::DT_FLOAT16 || shapeInfo.inputDtype == ge::DT_FLOAT ||
                                shapeInfo.inputDtype == ge::DT_INT32 || shapeInfo.inputDtype == ge::DT_INT8 ||
                                shapeInfo.inputDtype == ge::DT_UINT8;
    if (!supportedDtype || reducedShape.rank != 2 || reducedShape.outShape[0] > MAX_TWO_DIMENSIONAL_DIM ||
        reducedShape.outShape[1] > MAX_TWO_DIMENSIONAL_DIM) {
        return false;
    }
    w = reducedShape.outShape[0];
    h = reducedShape.outShape[1];
    n = 1;
    uint64_t totalElements = 0;
    return reducedShape.srcStride[0] == 1 && reducedShape.srcStride[1] == w && reducedShape.outStride[0] == h &&
           reducedShape.outStride[1] == 1 && CheckedMultiply(h, w, totalElements) &&
           shapeInfo.totalElements == totalElements;
}

void SplitN(uint64_t coreCount, BuildInfo& info)
{
    info.nPerCore = CeilDiv(info.n, coreCount);
    info.usedCores = CeilDiv(info.n, info.nPerCore);
    info.nTailCore = info.n - (info.usedCores - 1) * info.nPerCore;
}

bool SplitH(uint64_t coreCount, uint32_t elementBytes, BuildInfo& info)
{
    const uint64_t minHPerCore = elementBytes == 4 ? TRANS_ELEMS : 2 * TRANS_ELEMS;
    const uint64_t effectiveCores = std::min(coreCount, CeilDiv(info.hAlign, minHPerCore));
    if (effectiveCores == 0 || !AlignUp(CeilDiv(info.hAlign, effectiveCores), TRANS_ELEMS, info.hPerCore)) {
        return false;
    }
    if (elementBytes == 1 && !AlignUp(info.hPerCore, 2 * TRANS_ELEMS, info.hPerCore)) {
        return false;
    }
    info.usedCores = CeilDiv(info.hAlign, info.hPerCore);
    info.hTailCore = info.hAlign - (info.usedCores - 1) * info.hPerCore;
    info.nPerCore = info.n;
    info.nTailCore = info.n;
    return true;
}

bool FitsUb(uint64_t rows, uint64_t columns, uint32_t elementBytes, uint64_t availableUb)
{
    uint64_t bytes = 0;
    return CheckedProduct3(rows, columns, elementBytes, bytes) && bytes <= availableUb;
}

bool BuildHSplit(uint32_t elementBytes, BuildInfo& info)
{
    const uint64_t wBlocks = CeilDiv(info.w, info.blockElems);
    uint64_t wBlockElems = 0;
    if (!CheckedMultiply(wBlocks, info.blockElems, wBlockElems)) {
        return false;
    }
    uint64_t wPerUbBlocks = wBlocks;
    if (!FitsUb(info.hPerCore, wBlockElems, elementBytes, info.availableUb)) {
        uint64_t rowBlockBytes = 0;
        if (!CheckedProduct3(info.hPerCore, info.blockElems, elementBytes, rowBlockBytes) || rowBlockBytes == 0) {
            return false;
        }
        wPerUbBlocks = info.availableUb / rowBlockBytes;
    }
    if (elementBytes != 1) {
        wPerUbBlocks = std::min(wPerUbBlocks, MAX_REPEAT_TIMES);
    }
    if (wPerUbBlocks == 0) {
        return false;
    }
    info.c.alignFactor = wPerUbBlocks;
    info.c.factor = wPerUbBlocks * info.blockElems;
    info.c.count = CeilDiv(wBlocks, wPerUbBlocks);
    info.c.tailAlignFactor = wBlocks - (info.c.count - 1) * wPerUbBlocks;
    info.c.tailFactor = info.c.tailAlignFactor * info.blockElems;
    info.r.alignFactor = CeilDiv(info.hPerCore, TRANS_ELEMS);
    info.r.factor = info.hPerCore;
    info.r.count = 1;
    info.r.tailAlignFactor = CeilDiv(info.hTailCore, TRANS_ELEMS);
    info.r.tailFactor = info.hTailCore;
    info.loopCount = info.c.count;
    return true;
}

bool BuildTwoDimensionalSplit(BuildInfo& info)
{
    const uint64_t hBlocks = CeilDiv(info.h, TRANS_ELEMS);
    const uint64_t wBlocks = CeilDiv(info.w, info.blockElems);
    uint64_t bestRGroups = 0;
    uint64_t bestCBlocks = 0;
    uint64_t bestLoopCount = std::numeric_limits<uint64_t>::max();
    uint64_t bestTileArea = 0;

    const uint64_t maxRGroups = std::min(hBlocks, MAX_REPEAT_TIMES);
    for (uint64_t rGroups = 1; rGroups <= maxRGroups; ++rGroups) {
        uint64_t rowBytes = 0;
        if (!CheckedProduct3(rGroups * TRANS_ELEMS, info.blockElems, info.elementBytes, rowBytes) || rowBytes == 0) {
            return false;
        }
        const uint64_t cBlocks = std::min({wBlocks, MAX_REPEAT_TIMES, info.availableUb / rowBytes});
        if (cBlocks == 0) {
            continue;
        }
        const uint64_t loopCount = CeilDiv(hBlocks, rGroups) * CeilDiv(wBlocks, cBlocks);
        const uint64_t tileArea = rGroups * cBlocks;
        if (loopCount < bestLoopCount || (loopCount == bestLoopCount && tileArea > bestTileArea)) {
            bestRGroups = rGroups;
            bestCBlocks = cBlocks;
            bestLoopCount = loopCount;
            bestTileArea = tileArea;
        }
    }
    if (bestRGroups == 0 || bestCBlocks == 0) {
        return false;
    }

    info.r.alignFactor = bestRGroups;
    info.r.factor = bestRGroups * TRANS_ELEMS;
    info.r.count = CeilDiv(hBlocks, bestRGroups);
    info.r.tailAlignFactor = hBlocks - (info.r.count - 1) * bestRGroups;
    info.r.tailFactor = info.r.tailAlignFactor * TRANS_ELEMS;
    info.c.alignFactor = bestCBlocks;
    info.c.factor = bestCBlocks * info.blockElems;
    info.c.count = CeilDiv(wBlocks, bestCBlocks);
    info.c.tailAlignFactor = wBlocks - (info.c.count - 1) * bestCBlocks;
    info.c.tailFactor = info.c.tailAlignFactor * info.blockElems;
    info.loopCount = info.c.count;
    return FitsUb(info.r.factor, info.c.factor, info.elementBytes, info.availableUb);
}

bool BuildRConv(uint32_t elementBytes, BuildInfo& info)
{
    if (!FitsUb(TRANS_ELEMS, info.wAlign, elementBytes, info.availableUb)) {
        return false;
    }
    const uint64_t hBlocks = CeilDiv(info.h, TRANS_ELEMS);
    uint64_t hPerUbBlocks = info.availableUb / elementBytes / info.wAlign / TRANS_ELEMS;
    hPerUbBlocks = std::min(hPerUbBlocks, elementBytes == 1 ? 2 * MAX_REPEAT_TIMES : MAX_REPEAT_TIMES);
    if (elementBytes == 1 && hPerUbBlocks % 2 != 0) {
        --hPerUbBlocks;
    }
    if (hPerUbBlocks == 0) {
        return false;
    }
    info.r.alignFactor = std::min(hBlocks, hPerUbBlocks);
    info.r.factor = info.r.alignFactor * TRANS_ELEMS;
    info.r.count = CeilDiv(hBlocks, info.r.alignFactor);
    info.r.tailAlignFactor = hBlocks - (info.r.count - 1) * info.r.alignFactor;
    if (elementBytes == 1 && info.r.tailAlignFactor % 2 != 0) {
        ++info.r.tailAlignFactor;
    }
    info.r.tailFactor = info.r.tailAlignFactor * TRANS_ELEMS;
    info.c.alignFactor = CeilDiv(info.w, info.blockElems);
    info.c.factor = info.wAlign;
    info.c.count = 1;
    info.c.tailAlignFactor = info.c.alignFactor;
    info.c.tailFactor = info.c.factor;
    info.loopCount = info.r.count;
    if (elementBytes == 4 && info.r.factor > TRANS_ELEMS && info.c.alignFactor > MAX_UINT16 / TRANS_ELEMS) {
        return false;
    }
    return true;
}

bool BuildCConv(uint32_t elementBytes, BuildInfo& info)
{
    if (!FitsUb(TRANS_ELEMS, info.hAlign, elementBytes, info.availableUb)) {
        return false;
    }
    const uint64_t wBlocks = CeilDiv(info.w, TRANS_ELEMS);
    uint64_t wPerUbBlocks = info.availableUb / elementBytes / info.hAlign / TRANS_ELEMS;
    if (elementBytes != 1) {
        const uint64_t maxBlocks = MAX_REPEAT_TIMES * info.blockElems / TRANS_ELEMS;
        wPerUbBlocks = std::min(wPerUbBlocks, maxBlocks);
    }
    if (elementBytes == 1 && wPerUbBlocks % 2 != 0) {
        --wPerUbBlocks;
    }
    if (wPerUbBlocks == 0) {
        return false;
    }
    info.c.alignFactor = std::min(wBlocks, wPerUbBlocks);
    info.c.factor = info.c.alignFactor * TRANS_ELEMS;
    info.c.count = CeilDiv(wBlocks, info.c.alignFactor);
    info.c.tailAlignFactor = wBlocks - (info.c.count - 1) * info.c.alignFactor;
    if (elementBytes == 1 && info.c.tailAlignFactor % 2 != 0) {
        ++info.c.tailAlignFactor;
    }
    info.c.tailFactor = info.c.tailAlignFactor * TRANS_ELEMS;
    info.r.alignFactor = CeilDiv(info.h, TRANS_ELEMS);
    info.r.factor = info.hAlign;
    info.r.count = 1;
    info.r.tailAlignFactor = info.r.alignFactor;
    info.r.tailFactor = info.r.factor;
    info.loopCount = info.c.count;
    if (elementBytes == 1 && info.r.alignFactor > 2 * MAX_REPEAT_TIMES) {
        return false;
    }
    return true;
}

void FillSplit(KirinTranspose021VConvUbSplitPara& output, const UbInfo& input)
{
    output.set_ubAlignFactor(input.alignFactor);
    output.set_ubFactor(input.factor);
    output.set_ubCount(input.count);
    output.set_ubTailAlignFactor(input.tailAlignFactor);
    output.set_ubTailFactor(input.tailFactor);
}

void FillTiling(KirinTranspose021VConvOpTilingData& output, const BuildInfo& info)
{
    output.set_availableUbSize(info.availableUb);
    output.set_usedCoreNum(info.usedCores);
    output.set_nLen(info.n);
    output.set_hLen(info.h);
    output.set_wLen(info.w);
    output.set_hAlignBlockElem(info.hAlign);
    output.set_wAlignBlockElem(info.wAlign);
    output.set_nPerCore(info.nPerCore);
    output.set_nTailCore(info.nTailCore);
    output.set_ubLoopCount(info.loopCount);
    output.set_elementBytes(static_cast<uint8_t>(info.elementBytes));
    output.set_useRConv(info.useRConv);
    output.set_useHSplit(info.useHSplit);
    output.set_useTwoDimensionalSplit(info.useTwoDimensionalSplit);
    output.set_hPerCore(info.hPerCore);
    output.set_hTailCore(info.hTailCore);
    FillSplit(output.rUbSplitPara, info.r);
    FillSplit(output.cUbSplitPara, info.c);
}
} // namespace

bool BuildKirinTranspose021VConvTiling(const TransposeCompilerInfo& compilerInfo,
                                       const KirinTransposeShapeInfo& shapeInfo,
                                       const KirinReducedShapeInfo& reducedShape,
                                       KirinTranspose021VConvTilingData& tilingData, uint32_t& blockDim)
{
    BuildInfo info;
    if (compilerInfo.coreNum <= 0 || compilerInfo.ubSize <= 0) {
        return false;
    }
    const bool supportsTwoDimensionalSplit = compilerInfo.coreNum == 1 &&
                                             compilerInfo.ubSize >= MIN_TWO_DIMENSIONAL_UB_SIZE;
    info.useTwoDimensionalSplit = supportsTwoDimensionalSplit &&
                                  MatchTwoDimensionalVConv(shapeInfo, reducedShape, info.n, info.h, info.w);
    if (!info.useTwoDimensionalSplit && !Match021(shapeInfo, reducedShape, info.n, info.h, info.w)) {
        return false;
    }
    const uint64_t ubSize = static_cast<uint64_t>(compilerInfo.ubSize);
    if (info.useTwoDimensionalSplit && ubSize <= UB_RESERVED_BYTES) {
        return false;
    }
    const uint64_t usableUb = info.useTwoDimensionalSplit ? ubSize - UB_RESERVED_BYTES : ubSize;
    const uint64_t bufferCount = info.useTwoDimensionalSplit ? 1 : BUFFER_COUNT;
    info.availableUb = usableUb / INPUT_OUTPUT_COUNT / bufferCount;
    info.availableUb = info.availableUb / TRANSPOSE_BLOCK_BYTES * TRANSPOSE_BLOCK_BYTES;
    info.elementBytes = shapeInfo.elementBytes;
    info.blockElems = TRANSPOSE_BLOCK_BYTES / shapeInfo.elementBytes;
    info.useRConv = info.useTwoDimensionalSplit || info.h >= info.w;
    info.useHSplit = info.useTwoDimensionalSplit ||
                     (shapeInfo.elementBytes != 1 && info.n < 5 && !info.useRConv && info.n <= info.h / TRANS_ELEMS);
    const uint64_t hAlignment = shapeInfo.elementBytes == 1 ? info.blockElems : TRANS_ELEMS;
    const uint64_t wAlignment = info.useRConv ? info.blockElems :
                                                (shapeInfo.elementBytes == 1 ? info.blockElems : TRANS_ELEMS);
    if (info.availableUb < TRANSPOSE_BLOCK_BYTES || !AlignUp(info.h, hAlignment, info.hAlign) ||
        !AlignUp(info.w, wAlignment, info.wAlign)) {
        return false;
    }

    const uint64_t cores = static_cast<uint64_t>(compilerInfo.coreNum);
    if (info.useTwoDimensionalSplit) {
        info.usedCores = 1;
        info.nPerCore = 1;
        info.nTailCore = 1;
        info.hPerCore = info.h;
        info.hTailCore = info.h;
    } else if (info.useHSplit) {
        if (!SplitH(cores, shapeInfo.elementBytes, info) ||
            info.usedCores < std::max<uint64_t>(2, std::min(info.n, cores))) {
            info.useHSplit = false;
            SplitN(cores, info);
        }
    } else {
        SplitN(cores, info);
    }

    const bool feasible = info.useTwoDimensionalSplit ? BuildTwoDimensionalSplit(info) :
                          info.useHSplit              ? BuildHSplit(shapeInfo.elementBytes, info) :
                          info.useRConv               ? BuildRConv(shapeInfo.elementBytes, info) :
                                                        BuildCConv(shapeInfo.elementBytes, info);
    if (!feasible || info.usedCores == 0 || info.usedCores > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    uint64_t hBytes = 0;
    uint64_t wBytes = 0;
    if (info.availableUb > std::numeric_limits<uint32_t>::max() ||
        !CheckedMultiply(info.h, shapeInfo.elementBytes, hBytes) ||
        !CheckedMultiply(info.w, shapeInfo.elementBytes, wBytes) || hBytes > std::numeric_limits<uint32_t>::max() ||
        wBytes > std::numeric_limits<uint32_t>::max() || info.r.factor > std::numeric_limits<uint16_t>::max() ||
        info.c.factor > std::numeric_limits<uint16_t>::max() ||
        info.r.alignFactor > std::numeric_limits<uint16_t>::max() ||
        info.c.alignFactor > std::numeric_limits<uint16_t>::max() ||
        info.r.tailFactor > std::numeric_limits<uint16_t>::max() ||
        info.c.tailFactor > std::numeric_limits<uint16_t>::max() ||
        info.r.tailAlignFactor > std::numeric_limits<uint16_t>::max() ||
        info.c.tailAlignFactor > std::numeric_limits<uint16_t>::max() ||
        info.r.count > std::numeric_limits<uint32_t>::max() || info.c.count > std::numeric_limits<uint32_t>::max() ||
        info.loopCount > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    FillTiling(tilingData.transpose021VConvOpTiling, info);
    blockDim = static_cast<uint32_t>(info.usedCores);
    return true;
}
} // namespace optiling
