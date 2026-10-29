/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_TILING_WITH_021VCONV_ARCH31_H
#define TRANSPOSE_TILING_WITH_021VCONV_ARCH31_H

#include <cstdint>

#include "register/tilingdata_base.h"
#include "transpose_tiling_arch31.h"
#include "transpose_tiling_base_arch31.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KirinTranspose021VConvUbSplitPara)
TILING_DATA_FIELD_DEF(uint64_t, ubAlignFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubCount);
TILING_DATA_FIELD_DEF(uint64_t, ubTailAlignFactor);
TILING_DATA_FIELD_DEF(uint64_t, ubTailFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTranspose021VConvUbSplitParaOp, KirinTranspose021VConvUbSplitPara);

BEGIN_TILING_DATA_DEF(KirinTranspose021VConvOpTilingData)
TILING_DATA_FIELD_DEF(uint64_t, availableUbSize);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, nLen);
TILING_DATA_FIELD_DEF(uint64_t, hLen);
TILING_DATA_FIELD_DEF(uint64_t, wLen);
TILING_DATA_FIELD_DEF(uint64_t, hAlignBlockElem);
TILING_DATA_FIELD_DEF(uint64_t, wAlignBlockElem);
TILING_DATA_FIELD_DEF(uint64_t, nPerCore);
TILING_DATA_FIELD_DEF(uint64_t, nTailCore);
TILING_DATA_FIELD_DEF(uint64_t, ubLoopCount);
TILING_DATA_FIELD_DEF(uint8_t, elementBytes);
TILING_DATA_FIELD_DEF(bool, useRConv);
TILING_DATA_FIELD_DEF(bool, useHSplit);
TILING_DATA_FIELD_DEF(bool, useTwoDimensionalSplit);
TILING_DATA_FIELD_DEF(uint64_t, hPerCore);
TILING_DATA_FIELD_DEF(uint64_t, hTailCore);
TILING_DATA_FIELD_DEF_STRUCT(KirinTranspose021VConvUbSplitPara, rUbSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(KirinTranspose021VConvUbSplitPara, cUbSplitPara);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTranspose021VConvOpTilingDataOp, KirinTranspose021VConvOpTilingData);

BEGIN_TILING_DATA_DEF(KirinTranspose021VConvTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KirinTranspose021VConvOpTilingData, transpose021VConvOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose_10008, KirinTranspose021VConvTilingData);

bool BuildKirinTranspose021VConvTiling(const TransposeCompilerInfo& compilerInfo,
                                       const KirinTransposeShapeInfo& shapeInfo,
                                       const KirinReducedShapeInfo& reducedShape,
                                       KirinTranspose021VConvTilingData& tilingData, uint32_t& blockDim);
} // namespace optiling

#endif // TRANSPOSE_TILING_WITH_021VCONV_ARCH31_H
