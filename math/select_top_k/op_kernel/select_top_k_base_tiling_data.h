/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_TILING_DATA_H
#define SELECT_TOP_K_BASE_TILING_DATA_H
#include <cstdint>
#include "select_top_k_base_config.h"
#include "select_top_k_strategy.h"

// Host/Device 共用 ABI；长度默认以元素或 pair 计，Bytes 后缀明确表示字节。
// 本版为 128 字节，必须同时更新 Host、Device 和独立测试描述器，不能混装旧 tiling。
struct SelectTopKBaseTilingData {
    uint64_t rows = 0;
    uint64_t axis = 0;
    uint64_t k = 0;
    uint64_t tiles = 0;
    uint64_t rowWorkspacePairs = 0;
    uint64_t workspaceBytes = 0;
    uint32_t route = SelectTopKBase::ROUTE_EMPTY; // 路径族，供通用核分发及诊断。
    uint32_t cores = 1;
    uint32_t tile = 0;
    uint32_t mergeWindow = 0; // 通用归并每路窗口的 pair 数。
    uint32_t outputTile = 0;  // 最终输出搬运的分段元素数。
    uint32_t largest = 1;
    uint32_t sorted = 1;
    uint32_t dtypeBytes = 0;
    uint32_t ubBytes = 0;
    uint32_t pageStep = 0; // 按行过滤时的互质页遍历步长。
    SelectTopK::KernelTemplate kernel = SelectTopK::KernelTemplate::RANGE_MERGE; // 完整入口算法族。
    uint32_t shardCount = 0;        // 每行分片数；不再与页步长或批量行数复用。
    uint32_t batchRows = 0;         // 短轴 Sort32 的每批行数。
    uint32_t pendingLimit = 0;      // 多页过滤触发候选压缩的槽位数。
    uint32_t seedItems = 0;         // 宽块过滤建立初始阈值的元素数。
    uint32_t candidateSortTile = 0; // 宽块筛选后，每批候选排序的元素数。
    uint32_t finalCapacity = 0;     // 单块分片收尾时，每份 UB 归并缓冲的 pair 容量。
    SelectTopK::SelectPartition partition = SelectTopK::SelectPartition::ROWS;          // 仅过滤选择使用。
    SelectTopK::CandidateMerge candidateMerge = SelectTopK::CandidateMerge::FULL_LISTS; // 过滤/叶子的候选归并策略。
};
static_assert(sizeof(SelectTopKBaseTilingData) == SelectTopKBase::TILING_DATA_BYTES,
              "SelectTopK base tiling ABI changed");
#endif
