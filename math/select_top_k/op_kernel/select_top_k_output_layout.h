/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_OUTPUT_LAYOUT_H_
#define SELECT_TOP_K_OUTPUT_LAYOUT_H_
#include <cstdint>
#include "select_top_k_tiling_data.h"
#if defined(__CCE_AICORE__)
#define SELECT_TOP_K_LAYOUT_FUNCTION __aicore__
#else
#define SELECT_TOP_K_LAYOUT_FUNCTION
#endif
namespace SelectTopK {
// Host 容量计算与 Device 地址偏移共用此布局，统一计入 DMA 尾部填充。
struct OutputLayout {
    uint32_t rows;
    uint32_t pitch;
    uint32_t capacity;
    uint32_t inputCapacity;
    uint32_t bytes;

    // 计算常规输出重排的源行步长、Gather 容量及 UB 总量。
    SELECT_TOP_K_LAYOUT_FUNCTION constexpr OutputLayout(uint32_t inner, uint32_t block, uint64_t k, uint32_t valueBytes,
                                                        uint32_t indexBytes)
        : rows(inner),
          pitch(block + (inner > PAD_MIN_INNER && k > LARGE_K_THRESHOLD ? OUTPUT_ROW_PADDING : 0)),
          capacity(rows * block),
          inputCapacity(rows * pitch),
          bytes(inputCapacity * (valueBytes + SelectTopKBase::INDEX32_BYTES) +
                capacity * (valueBytes + SelectTopKBase::INDEX_AND_OFFSET_BYTES + indexBytes))
    {}
};
} // namespace SelectTopK
#undef SELECT_TOP_K_LAYOUT_FUNCTION
#endif
