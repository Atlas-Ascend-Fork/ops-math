/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_SORT_SCHEDULE_H_
#define SELECT_TOP_K_SORT_SCHEDULE_H_
#include <cstdint>
#include "select_top_k_strategy.h"

namespace SelectTopK {
// 分别表示整行排序、核内归并、跨核归并和核内加跨核的分层归并。
enum class SortPattern : uint32_t { ROWS = 0, LOCAL_MERGE = 1, CROSS_MERGE = 2, LOCAL_CROSS_MERGE = 3 };
constexpr uint32_t SHARD_ALIGNMENT = 2048;     // 跨核分段和核内叶子的对齐粒度。
constexpr uint32_t MAX_LEAF_SORT_ITEMS = 8192; // 两次 Sort32 可覆盖的最大叶子容量。
constexpr uint32_t ROW_SORT_ARRAY_BYTES = 24;  // key/index 各 4B，排序双缓冲各 8B。
constexpr uint32_t ROW_INPUT_BUFFER_COUNT = 2; // 整行批处理交替预取两份输入。
constexpr uint32_t SORT_SCHEDULE_BYTES = 48;   // 基础规划私有 ABI，不改变公开接口。

// 公共 base 字段保存资源；本结构保存排序调度的分层关系。
struct SortSchedule {
    uint64_t rowsPerCore = 0;    // 无跨核归并时，每核负责的连续行数。
    uint64_t shardElements = 0;  // 每核负责的连续轴片段，最后一个片段可不足。
    uint64_t leavesPerShard = 0; // 轴片段内的叶子槽位数，各核相同。
    uint64_t leafStride = 0;     // 工作区叶子槽位的 pair 数，含对齐空间。
    SortPattern pattern = SortPattern::ROWS;
    uint32_t rowWidth = 0;                        // 整行路径在 UB 中补齐到 Sort32 的行宽。
    MergeStorage storage = MergeStorage::WINDOWS; // 归并驻留方式，与分核模式独立。
};
static_assert(sizeof(SortSchedule) == SORT_SCHEDULE_BYTES, "排序调度结构的 ABI 已改变");
} // namespace SelectTopK
#endif
