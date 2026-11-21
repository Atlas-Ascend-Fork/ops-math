/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file space_to_depth_tiling.h
 * \brief tiling for SpaceToDepth
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SPACE_TO_DEPTH_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SPACE_TO_DEPTH_H

#include "conversion/transpose/op_host/arch35/transpose_tiling_base.h"
#include "conversion/transpose/op_host/arch35/transpose_tiling_arch35.h"
#include "register/tilingdata_base.h"

namespace optiling {

/* 展开后的维度数量（成员数组长度） */
static constexpr int64_t EXPANDED_DIM_NUM = 6;

/**
 * @brief SpaceToDepth 算子参数
 */
struct ParamInfo {
    gert::Shape xShape;
    ge::DataType xDtype;
    int64_t blockSize;
    const char* modePtr;
    const char* dataFormatPtr;
};

ge::graphStatus SpaceToDepthTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo);

class SpaceToDepthTiling {
public:
    explicit SpaceToDepthTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus ParametersVerifying();
    void ProcessShapeInfo(ShapeInfo& shapeInfo);

private:
    ge::graphStatus CheckShapeAndFormat(ge::Format& xFormat);
    ge::graphStatus CheckAttrsAndBlockSize(ge::Format xFormat);

private:
    gert::TilingContext* tilingContext_ = nullptr;
    ParamInfo paramInfo_{};
    int64_t nchwPerm_[EXPANDED_DIM_NUM] = {0, 1, 3, 5, 2, 4};
    int64_t nhwcPerm_[EXPANDED_DIM_NUM] = {0, 1, 3, 2, 4, 5};
};
} // namespace optiling
#endif
