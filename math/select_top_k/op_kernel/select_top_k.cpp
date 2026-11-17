/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "select_top_k_entry.h"

// 11 个 key 与 11 个完整入口模板一一对应；容量、分片及归并策略从 tiling 数据读取。
extern "C" __global__ __aicore__ void select_top_k(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR values,
                                                   GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(SELECT_TOP_K_TILING_TINY_ROW, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(SELECT_TOP_K_TILING_PREFIX_ROWS, KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SelectTopKTilingData);
    if (TILING_KEY_IS(SELECT_TOP_K_TILING_TINY_ROW)) {
        GET_TILING_DATA_PTR_WITH_STRUCT(SelectTopKTilingData, source, tiling);
        SelectTopK::RunTinyRow<DTYPE_X>(x, values, indices, source);
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(SelectTopKTilingData, data, tiling);
    if (TILING_KEY_IS(SELECT_TOP_K_TILING_ROW_SORT)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::ROW_SORT>(x, begin, end, values,
                                                                                            indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_MERGE_SORT)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::MERGE_SORT>(x, begin, end, values,
                                                                                              indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_RANGE_MERGE)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::RANGE_MERGE>(
            x, begin, end, values, indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_TOP1)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::TOP1>(x, begin, end, values, indices,
                                                                                        workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_SHORT_ROWS)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::SHORT_ROWS>(x, begin, end, values,
                                                                                              indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_SELECT)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::SELECT>(x, begin, end, values,
                                                                                          indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_WIDE_SELECT)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::WIDE_SELECT>(
            x, begin, end, values, indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_LEAF_SELECT)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::LEAF_SELECT>(
            x, begin, end, values, indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_COPY_ALL)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::COPY_ALL>(x, begin, end, values,
                                                                                            indices, workspace, data);
    } else if (TILING_KEY_IS(SELECT_TOP_K_TILING_PREFIX_ROWS)) {
        SelectTopK::RunKernel<DTYPE_X, DTYPE_INDICES, SelectTopK::KernelTemplate::PREFIX_ROWS>(
            x, begin, end, values, indices, workspace, data);
    }
}
