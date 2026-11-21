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
 * \file depth_to_space_tiling_arch35.h
 * \brief tiling for DepthToSpace
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DEPTH_TO_SPACE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DEPTH_TO_SPACE_H

#include "conversion/transpose/op_host/arch35/transpose_tiling_arch35.h"

namespace optiling {

namespace DepthToSpace {

/* 展开后的维度数量（成员数组长度） */
static constexpr int64_t EXPANDED_DIM_NUM = 6;

/**
 * @brief DepthToSpace 算子参数
 */
struct ParamInfo {
    gert::Shape xShape;
    ge::DataType xDtype;
    int64_t blockSize;
    const char* modePtr;
    const char* dataFormatPtr;
};

ge::graphStatus DepthToSpaceTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo);

class DepthToSpaceTiling {
public:
    explicit DepthToSpaceTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus ParametersVerifying();
    void ProcessShapeInfo(ShapeInfo& shapeInfo);

private:
    ge::graphStatus CheckFormatAndShape(int64_t xDimNum, int64_t yDimNum, ge::Format xFormat, ge::Format yFormat);
    ge::graphStatus CheckAttrValues(ge::Format xFormat);
    void SetPermAndOutShape(ShapeInfo& shapeInfo, const int64_t* perm);
    void SetInShapeNhwcDcr(ShapeInfo& shapeInfo);
    void SetInShapeNchwDcr(ShapeInfo& shapeInfo);
    void SetInShapeNhwcCrd(ShapeInfo& shapeInfo);
    void SetInShapeNchwCrd(ShapeInfo& shapeInfo);
    void BuildInShapeByFormatMode(ShapeInfo& shapeInfo);
    gert::TilingContext* tilingContext_ = nullptr;
    ParamInfo paramInfo_{};
    int64_t nhwcDcrPerm_[EXPANDED_DIM_NUM] = {0, 1, 3, 2, 4, 5};
    int64_t nchwDcrPerm_[EXPANDED_DIM_NUM] = {0, 3, 4, 1, 5, 2};
    int64_t crdPerm_[EXPANDED_DIM_NUM] = {0, 1, 4, 2, 5, 3};
};
} // namespace DepthToSpace
} // namespace optiling
#endif
