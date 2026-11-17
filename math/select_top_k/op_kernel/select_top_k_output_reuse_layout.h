/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_OUTPUT_REUSE_LAYOUT_H_
#define SELECT_TOP_K_OUTPUT_REUSE_LAYOUT_H_
#include <cstdint>
#include "select_top_k_tiling_data.h"
#if defined(__CCE_AICORE__)
#define SELECT_TOP_K_LAYOUT_FUNCTION __aicore__
#else
#define SELECT_TOP_K_LAYOUT_FUNCTION
#endif
namespace SelectTopK {
// Host 容量计算与 Device 地址偏移共用此布局，统一计入 DMA 尾部填充。
struct OutputReuseLayout {
    uint32_t rows;
    uint32_t pitch;
    uint32_t capacity;
    uint32_t inputCapacity;
    uint32_t bytes;
    uint32_t inputValuesOffset = 0;
    uint32_t outputValuesOffset = 0;
    uint32_t inputIndicesOffset = 0;
    uint32_t outputIndices32Offset = 0;
    uint32_t offsetsOffset = 0;
    uint32_t outputIndicesOffset = 0;

    // 计算分阶段复用 UB 的输出重排布局，隔离存活的值、索引和偏移表。
    SELECT_TOP_K_LAYOUT_FUNCTION constexpr OutputReuseLayout(uint32_t inner, uint32_t block, uint64_t k,
                                                             uint32_t valueBytes, uint32_t indexBytes,
                                                             bool enabled = true)
        : rows(inner),
          pitch(block + (inner > PAD_MIN_INNER && k > LARGE_K_THRESHOLD ?
                             (enabled && valueBytes == SelectTopKBase::VALUE32_BYTES ? REUSED_FP32_PADDING :
                                                                                       OUTPUT_ROW_PADDING) :
                             0)),
          capacity(rows * block),
          inputCapacity(rows * pitch),
          bytes(inputCapacity * (valueBytes + SelectTopKBase::INDEX32_BYTES) +
                capacity * (valueBytes + SelectTopKBase::INDEX_AND_OFFSET_BYTES + indexBytes))
    {
        outputValuesOffset = inputCapacity * valueBytes;
        inputIndicesOffset = (inputCapacity + capacity) * valueBytes;
        outputIndices32Offset = inputIndicesOffset + inputCapacity * SelectTopKBase::INDEX32_BYTES;
        offsetsOffset = outputIndices32Offset + capacity * SelectTopKBase::INDEX32_BYTES;
        outputIndicesOffset = offsetsOffset + capacity * SelectTopKBase::INDEX32_BYTES;
        if (enabled && k > LARGE_K_THRESHOLD && indexBytes == SelectTopKBase::INDEX64_BYTES) {
            // 两路 Gather 完成后才扩宽索引；输出值、INT32 索引和偏移表不与输入/扩宽临时区重叠。
            const uint32_t inputBytes = inputCapacity * (valueBytes + SelectTopKBase::INDEX32_BYTES);
            const uint32_t wideBytes = capacity * indexBytes;
            const uint32_t scratch = inputBytes > wideBytes ? inputBytes : wideBytes;
            inputIndicesOffset = inputCapacity * valueBytes;
            outputValuesOffset = scratch;
            outputIndices32Offset = scratch + capacity * valueBytes;
            offsetsOffset = outputIndices32Offset + capacity * SelectTopKBase::INDEX32_BYTES;
            outputIndicesOffset = 0;
            bytes = offsetsOffset + capacity * SelectTopKBase::INDEX32_BYTES;
        }
    }
};
} // namespace SelectTopK
#undef SELECT_TOP_K_LAYOUT_FUNCTION
#endif
