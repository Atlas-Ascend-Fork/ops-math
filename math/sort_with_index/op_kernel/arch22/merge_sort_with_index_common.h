/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_MERGE_COMMON_ARCH22_H
#define SORT_WITH_INDEX_MERGE_COMMON_ARCH22_H

#include "sort_with_index_common_arch22.h"

namespace SortWithIndexA2 {
// 将连续的 Sort32 有序段按四路归并，兼容最后一段不足整段的情况。
__aicore__ inline void MergeStage(LocalTensor<float> dst, LocalTensor<float> src, uint32_t total, uint32_t runLength,
                                  uint32_t runCount)
{
    uint32_t fullGroups = runCount / MERGE_LIST_COUNT;
    uint32_t tailRuns = runCount % MERGE_LIST_COUNT;
    if (tailRuns == 0 && total != runCount * runLength) {
        --fullGroups;
        tailRuns = MERGE_LIST_COUNT;
    }
    uint32_t mergedCounts[MERGE_LIST_COUNT];
    if (fullGroups > 0) {
        uint16_t counts[MERGE_LIST_COUNT] = {uint16_t(runLength), uint16_t(runLength), uint16_t(runLength),
                                             uint16_t(runLength)};
        uint32_t stride = GetSortOffset<float>(runLength);
        MrgSortSrcList<float> lists(src, src[stride], src[stride * 2], src[stride * 3]);
        MrgSort<float, false>(dst, lists, counts, mergedCounts, (1U << MERGE_LIST_COUNT) - 1, fullGroups);
    }
    uint32_t tailStart = fullGroups * MERGE_LIST_COUNT * runLength;
    if (tailStart == total) {
        return;
    }
    if (tailRuns == 1) {
        DataCopy(dst[GetSortOffset<float>(tailStart)], src[GetSortOffset<float>(tailStart)],
                 GetSortOffset<float>(total - tailStart));
        return;
    }
    uint16_t counts[MERGE_LIST_COUNT] = {0, 0, 0, 0};
    uint32_t offsets[MERGE_LIST_COUNT] = {tailStart, tailStart, tailStart, tailStart};
    for (uint32_t list = 0; list < tailRuns; ++list) {
        offsets[list] = tailStart + list * runLength;
        uint32_t remaining = total - offsets[list];
        counts[list] = remaining < runLength ? remaining : runLength;
    }
    MrgSortSrcList<float> lists(src[GetSortOffset<float>(offsets[0])], src[GetSortOffset<float>(offsets[1])],
                                src[GetSortOffset<float>(offsets[2])], src[GetSortOffset<float>(offsets[3])]);
    MrgSort<float, false>(dst[GetSortOffset<float>(tailStart)], lists, counts, mergedCounts, (1U << tailRuns) - 1, 1);
}

} // namespace SortWithIndexA2
#endif // SORT_WITH_INDEX_MERGE_COMMON_ARCH22_H
