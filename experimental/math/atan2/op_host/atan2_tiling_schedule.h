/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATAN2_TILING_SCHEDULE_H
#define ATAN2_TILING_SCHEDULE_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include "../op_kernel/atan2_tiling_data.h"

namespace optiling {
namespace atan2 {

constexpr uint64_t DMA_BYTES = 32;
constexpr uint64_t COMPARE_ELEMENTS = 256 / sizeof(float);
constexpr uint64_t SELECT_BYTES = 8192;
constexpr uint64_t ATAN_FLOAT_BUFFERS = 8;

struct CoreSlice {
    int64_t elements;
    int64_t tiles;
    int64_t lastTile;
};

inline CoreSlice DescribeCore(uint64_t blocks, uint64_t elementsPerBlock, uint64_t tile)
{
    const uint64_t elements = blocks * elementsPerBlock;
    return {static_cast<int64_t>(elements), static_cast<int64_t>(elements / tile + (elements % tile != 0)),
            static_cast<int64_t>((elements - 1) % tile + 1)};
}

// Pure scheduling arithmetic, shared with the host and device regression
// probes. The buffer layout is three double-buffered queues, five float
// temporaries, eight float-sized Atan buffers, and Select's fixed scratch.
inline bool BuildSchedule(uint64_t elements, uint32_t elementBytes, uint64_t ubBytes, uint32_t availableCores,
                          Atan2TilingData& data, uint32_t& blocks)
{
    if (elements == 0 || availableCores == 0 || (elementBytes != 2 && elementBytes != 4)) {
        return false;
    }
    const uint64_t bytesPerElement = 3 * 2 * elementBytes + (5 + ATAN_FLOAT_BUFFERS) * sizeof(float);
    if (ubBytes < SELECT_BYTES + COMPARE_ELEMENTS * bytesPerElement) {
        return false;
    }
    const uint64_t tile = (ubBytes - SELECT_BYTES) / bytesPerElement / COMPARE_ELEMENTS * COMPARE_ELEMENTS;
    if (tile > std::numeric_limits<uint32_t>::max() / (ATAN_FLOAT_BUFFERS * sizeof(float))) {
        return false;
    }
    const uint64_t elementsPerBlock = DMA_BYTES / elementBytes;
    const uint64_t dmaBlocks = elements / elementsPerBlock + (elements % elementsPerBlock != 0);
    // Kernel offsets and lengths are uint32_t, including padded DMA elements.
    if (dmaBlocks > std::numeric_limits<uint32_t>::max() / elementsPerBlock) {
        return false;
    }
    const uint32_t activeCores = static_cast<uint32_t>(std::min<uint64_t>(availableCores, dmaBlocks));
    const uint64_t regularBlocks = dmaBlocks / activeCores;
    const CoreSlice regular = DescribeCore(regularBlocks, elementsPerBlock, tile);
    const CoreSlice extra = DescribeCore(regularBlocks + 1, elementsPerBlock, tile);
    data.smallCoreDataNum = regular.elements;
    data.bigCoreDataNum = extra.elements;
    data.finalBigTileNum = extra.tiles;
    data.finalSmallTileNum = regular.tiles;
    data.tileDataNum = static_cast<int64_t>(tile);
    data.smallTailDataNum = regular.lastTile;
    data.bigTailDataNum = extra.lastTile;
    data.tailBlockNum = static_cast<int64_t>(dmaBlocks % activeCores);
    data.atanTmpSize = static_cast<uint32_t>(ATAN_FLOAT_BUFFERS * tile * sizeof(float));
    blocks = activeCores;
    return true;
}

} // namespace atan2
} // namespace optiling
#endif
