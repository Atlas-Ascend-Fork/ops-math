/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_TILING_WITH_STRIDED_DATACOPY_ARCH31_H
#define TRANSPOSE_TILING_WITH_STRIDED_DATACOPY_ARCH31_H

#include <cstdint>

#include "transpose_tiling_arch31.h"
#include "transpose_tiling_base_arch31.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KirinTransposeStridedDataCopyOpTilingData)
TILING_DATA_FIELD_DEF(uint64_t, mLen);        // M dimension size (output rows per n)
TILING_DATA_FIELD_DEF(uint64_t, nLen);        // N dimension size (transpose columns)
TILING_DATA_FIELD_DEF(uint64_t, wLen);        // W dimension size (lowest dim element count)
TILING_DATA_FIELD_DEF(uint64_t, bLen);        // B dimension size (batch count)
TILING_DATA_FIELD_DEF(uint64_t, wBytes);      // W dimension byte count (= W * elementBytes)
TILING_DATA_FIELD_DEF(uint64_t, mTile);       // M rows moved per DataCopyPad invocation
TILING_DATA_FIELD_DEF(uint8_t, elementBytes); // bytes per element
TILING_DATA_FIELD_DEF(uint64_t, batchStride); // batch stride = M*N*W (element units, same for src and dst)
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTransposeStridedDataCopyOpTilingDataOp, KirinTransposeStridedDataCopyOpTilingData)

BEGIN_TILING_DATA_DEF(KirinTransposeStridedDataCopyTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeStridedDataCopyOpTilingData, transposeStridedDataCopyOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose_10010, KirinTransposeStridedDataCopyTilingData)

bool BuildKirinStridedDataCopyTransposeTiling(const TransposeCompilerInfo& compilerInfo,
                                              const KirinTransposeShapeInfo& shapeInfo,
                                              const KirinReducedShapeInfo& reducedShape,
                                              KirinTransposeStridedDataCopyTilingData& tilingData, uint32_t& blockDim);

} // namespace optiling

#endif // TRANSPOSE_TILING_WITH_STRIDED_DATACOPY_ARCH31_H
