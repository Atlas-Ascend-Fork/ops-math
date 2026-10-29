/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_BASE_ARCH31_H
#define TRANSPOSE_BASE_ARCH31_H

#include "kernel_operator.h"

namespace Transpose {
using namespace AscendC;

constexpr uint32_t TRANSPOSE_BLOCK_BYTES = 32;
constexpr uint32_t TRANSPOSE_MAX_AXIS_NUM = 8;

__aicore__ inline bool GetCoreRange(const KirinTransposeOpTilingData* tiling, uint32_t blockIdx, uint64_t& start,
                                    uint64_t& count)
{
    const uint64_t totalElements = tiling->totalElements;
    const uint64_t elementsPerBlock = TRANSPOSE_BLOCK_BYTES / tiling->elementBytes;
    const uint64_t fullBlockCount = totalElements / elementsPerBlock;
    const uint64_t tailElements = totalElements % elementsPerBlock;
    const uint64_t blocksPerCore = tiling->blocksPerCore;
    const uint64_t extraBlockCoreCount = tiling->extraBlockCoreCount;
    const uint32_t usedCoreCount = tiling->usedCoreCount;

    if (blockIdx >= usedCoreCount) {
        return false;
    }

    if (blockIdx < extraBlockCoreCount) {
        start = blockIdx * (blocksPerCore + 1) * elementsPerBlock;
        count = (blocksPerCore + 1) * elementsPerBlock;
    } else {
        start = (extraBlockCoreCount * (blocksPerCore + 1) + (blockIdx - extraBlockCoreCount) * blocksPerCore) *
                elementsPerBlock;
        count = blocksPerCore * elementsPerBlock;
    }

    if (blockIdx == usedCoreCount - 1) {
        count += tailElements;
    }

    return true;
}
} // namespace Transpose

#endif // TRANSPOSE_BASE_ARCH31_H
