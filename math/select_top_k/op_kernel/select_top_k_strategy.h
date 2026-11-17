/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_STRATEGY_H_
#define SELECT_TOP_K_STRATEGY_H_
#include <cstdint>

namespace SelectTopK {
// 完整入口的算法族；Host 按算法族规划，在最终编码阶段统一生成连续的 tiling key。
enum class KernelTemplate : uint32_t {
    ROW_SORT,
    MERGE_SORT,
    RANGE_MERGE,
    TOP1,
    SHORT_ROWS,
    SELECT,
    WIDE_SELECT,
    LEAF_SELECT,
    COPY_ALL,
    TINY_ROW,
    PREFIX_ROWS,
    COUNT
};

// 过滤扫描的任务粒度：整行由一核维护候选，或一行拆成多个核的页段。
enum class SelectPartition : uint32_t { ROWS, SHARDS };

// 候选归并读取完整列表、约半 K 前缀、或约 K/3 的稠密前缀；不足时保留完整回退。
enum class CandidateMerge : uint32_t { FULL_LISTS, HALF_PREFIX, DENSE_PREFIX };

// 归并的驻留布局，独立于核内/跨核调度；两叶布局使用两路较宽的流式窗口。
enum class MergeStorage : uint32_t { WINDOWS, ALL_CANDIDATES, GROUPED_CANDIDATES, TWO_LEAF_WINDOWS };
} // namespace SelectTopK
#endif
