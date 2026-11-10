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
 * \file hypot.cpp
 * \brief hypot kernel entry (TilingKey template dispatch)
 */

#include "hypot_struct.h"
#include "hypot_tiling_data.h"

using namespace NsHypot;

// Kernel function. The TilingKey carries only the broadcast scene; the concrete tensor dtypes are
// stamped per OpDef dtype combination by the opc framework and read through DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT.
template <int BROADCAST_SCENE>
__global__ __aicore__ void hypot(GM_ADDR self, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(HypotTilingData);
    GET_TILING_DATA_WITH_STRUCT(HypotTilingData, tilingData, tiling);

    HypotKernelImpl<DTYPE_SELF, DTYPE_OTHER, DTYPE_OUT, BROADCAST_SCENE>(self, other, out, &tilingData);
}
