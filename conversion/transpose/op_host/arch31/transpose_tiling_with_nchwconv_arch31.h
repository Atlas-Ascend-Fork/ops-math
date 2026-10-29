/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_TILING_WITH_NCHWCONV_ARCH31_H
#define TRANSPOSE_TILING_WITH_NCHWCONV_ARCH31_H

#include <cstdint>

#include "transpose_tiling_arch31.h"
#include "transpose_tiling_base_arch31.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KirinTransposeVConvCoreSplitTilingData)
TILING_DATA_FIELD_DEF(uint64_t, alignBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockCount);
TILING_DATA_FIELD_DEF(uint64_t, alignBlockTailFactor);
TILING_DATA_FIELD_DEF(uint64_t, blockTailFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTransposeVConvCoreSplitTilingDataOp, KirinTransposeVConvCoreSplitTilingData)

BEGIN_TILING_DATA_DEF(KirinTransposeVConvUbSplitTilingData)
TILING_DATA_FIELD_DEF(uint64_t, mainCoreUbAlignFactor);
TILING_DATA_FIELD_DEF(uint64_t, mainCoreUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, mainCoreUbCount);
TILING_DATA_FIELD_DEF(uint64_t, mainCoreTailUbAlignFactor);
TILING_DATA_FIELD_DEF(uint64_t, mainCoreTailUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreUbAlignFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreUbFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreUbCount);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailUbAlignFactor);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailUbFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTransposeVConvUbSplitTilingDataOp, KirinTransposeVConvUbSplitTilingData)

BEGIN_TILING_DATA_DEF(KirinTransposeVConvOpTilingData)
TILING_DATA_FIELD_DEF(uint64_t, availableUbSize);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, mainCoreLoopCount);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreLoopCount);
TILING_DATA_FIELD_DEF(uint64_t, rLen);
TILING_DATA_FIELD_DEF(uint64_t, cLen);
TILING_DATA_FIELD_DEF(uint64_t, rAlignBlock);
TILING_DATA_FIELD_DEF(uint64_t, cAlignBlock);
TILING_DATA_FIELD_DEF(uint64_t, rAlignBlockElem);
TILING_DATA_FIELD_DEF(uint64_t, cAlignBlockElem);
TILING_DATA_FIELD_DEF(bool, isRSplit);
TILING_DATA_FIELD_DEF(bool, isRCSplit);
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeVConvCoreSplitTilingData, rSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeVConvCoreSplitTilingData, cSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeVConvUbSplitTilingData, rUbSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeVConvUbSplitTilingData, cUbSplitPara);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTransposeVConvOpTilingDataOp, KirinTransposeVConvOpTilingData)

BEGIN_TILING_DATA_DEF(KirinTransposeVConvTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeVConvOpTilingData, transposeVConvTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose_10007, KirinTransposeVConvTilingData);

bool BuildKirinTransposeVConvTiling(const TransposeCompilerInfo& compilerInfo, const KirinTransposeShapeInfo& shapeInfo,
                                    const KirinReducedShapeInfo& reducedShape,
                                    KirinTransposeVConvTilingData& tilingData, uint32_t& blockDim);
} // namespace optiling

#endif // TRANSPOSE_TILING_WITH_NCHWCONV_ARCH31_H
