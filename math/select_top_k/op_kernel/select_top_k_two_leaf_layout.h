/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_TWO_LEAF_LAYOUT_H_
#define SELECT_TOP_K_TWO_LEAF_LAYOUT_H_
#include "select_top_k_tiling_data.h"

namespace SelectTopK {
// 两路归并仍按Sort32粒度对齐解码缓冲，并遵守单路最多4095项的硬件限制。
constexpr uint32_t TWO_LEAF_WINDOW = SelectTopKBase::MERGE_MAX_ALIGNED_ITEMS / SelectTopKBase::SORT32_ELEMENTS *
                                     SelectTopKBase::SORT32_ELEMENTS;

// 两路输入与输出转换分时复用前部；归并结果独立存活，不能与解码缓冲重叠。
struct TwoLeafMergeLayout {
    uint64_t merged;
    uint64_t bytes;

    // 排序阶段仍使用基础叶子布局；归并阶段只为两路结果保留空间。
#if defined(__CCE_AICORE__)
    __aicore__
#endif
        constexpr TwoLeafMergeLayout(uint32_t tile, uint32_t window)
        : merged(uint64_t(window) * SelectTopKBase::COMPACT_SORT_BYTES),
          bytes(merged + uint64_t(window) * SelectTopKBase::TWO_MERGE_RUNS * SelectTopKBase::PAIR_BYTES)
    {
        const uint64_t sortBytes = uint64_t(tile) * SelectTopKBase::COMPACT_SORT_BYTES;
        if (sortBytes > bytes)
            bytes = sortBytes;
    }
};
} // namespace SelectTopK
#endif
