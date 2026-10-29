/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH31_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH31_H

#include <cstdint>

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t TRANSPOSE_MAX_AXIS_NUM = 8;
constexpr uint32_t TRANSPOSE_BLOCK_BYTES = 32;
constexpr uint64_t TRANSPOSE_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t INPUT_IDX_X = 0;
constexpr uint64_t INPUT_IDX_PERM = 1;
constexpr uint64_t OUTPUT_IDX_Y = 0;

enum class KirinTransposeKey : uint64_t {
    TENSOR_MOVE = 10000,
    VCONV_TRANSPOSE = 10007,
    VCONV_021_TRANSPOSE = 10008,
    GENERIC_DATACOPY = 10009,
    STRIDED_DATACOPY_TRANSPOSE = 10010,
};

BEGIN_TILING_DATA_DEF(KirinTransposeOpTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalElements);
TILING_DATA_FIELD_DEF(uint64_t, blocksPerCore);
TILING_DATA_FIELD_DEF(uint64_t, extraBlockCoreCount);
TILING_DATA_FIELD_DEF(uint64_t, tailElements);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreCount);
TILING_DATA_FIELD_DEF(uint32_t, ubTileElements);
TILING_DATA_FIELD_DEF(uint8_t, reducedRank);
TILING_DATA_FIELD_DEF(uint8_t, elementBytes);
TILING_DATA_FIELD_DEF(uint16_t, reserved);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, srcStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KirinTransposeOpTilingDataOp, KirinTransposeOpTilingData)

// Wrapper for base op_type "Transpose"
BEGIN_TILING_DATA_DEF(KirinTransposeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeOpTilingData, transposeOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose, KirinTransposeTilingData);

// Separate wrapper classes for each tiling key (same layout, different class names)
// to avoid redefinition in OPC-generated _tiling_data.h
BEGIN_TILING_DATA_DEF(KirinTransposeTensorMoveTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeOpTilingData, transposeOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose_10000, KirinTransposeTensorMoveTilingData);

BEGIN_TILING_DATA_DEF(KirinTransposeGenericDataCopyTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KirinTransposeOpTilingData, transposeOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose_10009, KirinTransposeGenericDataCopyTilingData);

struct TransposeCompilerInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context);
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_ARCH31_H
