/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_apt.cpp
 * \brief StridedSliceAssignV2
 */

#include "kernel_operator.h"
#include "arch35/strided_slice_assign_v2_tiling_data.h"
#include "arch35/strided_slice_assign_v2_tilingkey.h"
#include "arch35/strided_slice_assign_v2_contig.h"
#include "arch35/strided_slice_assign_v2_hybrid.h"
#include "arch35/strided_slice_assign_v2_scatter.h"

using namespace AscendC;

template <uint32_t TemplateNum>
__global__ __aicore__ void strided_slice_assign_v2(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin, GM_ADDR end,
                                                   GM_ADDR strides, GM_ADDR axes, GM_ADDR var_out, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;

    if constexpr (TemplateNum == SSA_TPL_SCATTER) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(StridedSliceAssignV2ScatterTilingData, tilingData, tiling);
        StridedSliceAssignV2ScatterNs::StridedSliceAssignV2Scatter<DTYPE_VAR> op(&pipe);
        op.Init(var, input_value, var_out, &tilingData);
        op.Process();
    } else if constexpr (TemplateNum == SSA_TPL_CONTIG) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(StridedSliceAssignV2ContigTilingData, tilingData, tiling);
        StridedSliceAssignV2ContigNs::StridedSliceAssignV2Contig<DTYPE_VAR> op(&pipe);
        op.Init(var, input_value, var_out, &tilingData);
        op.Process();
    } else if constexpr (TemplateNum == SSA_TPL_HYBRID) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(StridedSliceAssignV2HybridTilingData, tilingData, tiling);
        StridedSliceAssignV2HybridNs::StridedSliceAssignV2Hybrid<DTYPE_VAR> op(&pipe);
        op.Init(var, input_value, var_out, &tilingData);
        op.Process();
    } else if constexpr (TemplateNum == SSA_TPL_EMPTY) {
        GET_TILING_DATA_WITH_STRUCT(StridedSliceAssignV2EmptyTilingData, tilingData, tiling);
    }
}
