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
 * \file histogram_v2_tiling_data.h
 * \brief Pure-struct tiling data shared by host tiling and arch35 kernels.
 *        Field names/order/types are identical to the legacy macro-defined
 *        HistogramV2SimtTilingData, so the binary layout is unchanged.
 */
#ifndef HISTOGRAM_V2_TILING_DATA_H
#define HISTOGRAM_V2_TILING_DATA_H

#include <cstdint>

struct HistogramV2SimtTilingData {
    int64_t bins;
    int64_t ubNumCanUse;
    int64_t ubLoopNum;
    int64_t needXCoreNum;
    int64_t formerLength;
    int64_t tailLength;
    int64_t clearYCoreNum;
    int64_t clearYFactor;
    int64_t clearYTail;
};

#endif // HISTOGRAM_V2_TILING_DATA_H
