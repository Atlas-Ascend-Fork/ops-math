/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sort_with_index.cpp
 * \brief SortWithIndex 在 Atlas A2/A3 上的 kernel 入口。
 */

#include "arch22/radix_sort_with_index_multi_block_arch22.h"
#include "arch22/merge_sort_with_index_arch22.h"
#include "arch22/merge_sort_with_index_stream_native.h"
#include "arch22/merge_sort_with_index_single_run.h"
#include "arch22/merge_sort_with_index_native.h"

extern "C" __global__ __aicore__ void sort_with_index(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 跨核基数排序和归并使用硬件 SyncAll，启动时需启用 FFTS。
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(data, tiling);
    // CANN 9.2 扫描 TILING_KEY_IS 的字面量识别分支，须与 sort_with_index_constants.h 中的模式值保持一致。
    if constexpr ((AscendC::IsSameType<DTYPE_X, half>::value || AscendC::IsSameType<DTYPE_X, bfloat16_t>::value) &&
                  AscendC::IsSameType<DTYPE_INDEX, int32_t>::value) {
        if (TILING_KEY_IS(4)) { // UB 内短行归并
            SortWithIndexA2::MergeSortWithIndex<DTYPE_X> op;
            op.Init(x, index, y, sortedIndex, workspace, data);
            op.Process();
        } else if (TILING_KEY_IS(6)) { // UB 内整行排序
            SortWithIndexA2::MergeSortWithIndexSingleRun<DTYPE_X> op;
            op.Init(x, index, y, sortedIndex, workspace, data);
            op.Process();
        } else if (TILING_KEY_IS(9)) { // 四段流式归并，直接输出值和索引
            SortWithIndexA2::MergeSortWithIndexStreamNative<DTYPE_X> op;
            op.Init(x, index, y, sortedIndex, workspace, data);
            op.Process();
        } else if (TILING_KEY_IS(8)) { // UB 内分段排序，GM 中归并
            SortWithIndexA2::NativeMergeSortWithIndex<DTYPE_X, true> op;
            op.Init(x, index, y, sortedIndex, workspace, data);
            op.Process();
        } else if (TILING_KEY_IS(2)) { // 基数排序
            SortWithIndexA2::RadixSortWithIndexMultiBlock<DTYPE_X> op;
            op.Init(x, index, y, sortedIndex, workspace, data);
            op.Process();
        }
    }
}
