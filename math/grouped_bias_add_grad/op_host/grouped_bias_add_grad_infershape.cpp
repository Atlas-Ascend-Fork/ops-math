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
 * \file grouped_bias_add_grad_infer.cpp
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t INDEX_GRAD_Y = 0;
static constexpr size_t INDEX_GROUP_IDX = 1;

static constexpr size_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr size_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

static constexpr size_t GROUP_INDEX_SHAPE = 1;
static constexpr size_t GRAD_BIAS_SHAPE_SIZE = 2;

static constexpr size_t OTHER_SHAPE = -1;

static constexpr int64_t MAX_GROUP_NUM = 2048;

static ge::graphStatus GroupedBiasAddGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do GroupedBiasAddGradInferShape.");

    const gert::Shape* gradYShape = context->GetInputShape(INDEX_GRAD_Y);
    OP_CHECK_IF(gradYShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "grad_y"),
                return ge::GRAPH_FAILED);

    auto gradYDimNum = gradYShape->GetDimNum();
    auto groupNum = gradYShape->GetDim(0);
    int64_t hNum = 0;

    if (gradYDimNum == 1 && groupNum == ge::UNKNOWN_DIM_NUM) {
        gert::Shape* gradBiasShape = context->GetOutputShape(0);
        gradBiasShape->SetDimNum(GRAD_BIAS_SHAPE_SIZE);
        gradBiasShape->SetDim(0, OTHER_SHAPE);
        gradBiasShape->SetDim(1, OTHER_SHAPE);
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape* groupIdxShape = context->GetOptionalInputShape(INDEX_GROUP_IDX);
    if (groupIdxShape == nullptr) {
        OP_CHECK_IF(gradYDimNum != GRAD_Y_SHAPE_NO_GROUP_IDX,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "grad_y",
                                                             std::to_string(gradYDimNum).c_str(),
                                                             "grad_y must be 3D when group_idx is null."),
                    return ge::GRAPH_FAILED);
        hNum = gradYShape->GetDim(GRAD_Y_SHAPE_NO_GROUP_IDX - 1);
    } else {
        auto groupIdxDimNum = groupIdxShape->GetDimNum();
        OP_CHECK_IF(
            gradYDimNum != GRAD_Y_SHAPE_WITH_GROUP_IDX || groupIdxDimNum != GROUP_INDEX_SHAPE,
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "grad_y,group_idx",
                ("grad_y:" + std::to_string(gradYDimNum) + ",group_idx:" + std::to_string(groupIdxDimNum)).c_str(),
                "grad_y must be 2D and group_idx must be 1D when group_idx is not null."),
            return ge::GRAPH_FAILED);

        auto groupIdxShapeSize = groupIdxShape->GetDim(0);

        OP_CHECK_IF(
            groupIdxShapeSize != -1 && groupIdxShapeSize > MAX_GROUP_NUM,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "group_idx", std::to_string(groupIdxShapeSize).c_str(),
                ("group_idx shape size must be less than or equal to " + std::to_string(MAX_GROUP_NUM) + ".").c_str()),
            return ge::GRAPH_FAILED);

        hNum = gradYShape->GetDim(GRAD_Y_SHAPE_WITH_GROUP_IDX - 1);
        groupNum = groupIdxShapeSize;
    }

    gert::Shape* gradBiasShape = context->GetOutputShape(0);
    gradBiasShape->SetDimNum(GRAD_BIAS_SHAPE_SIZE);
    gradBiasShape->SetDim(0, groupNum);
    gradBiasShape->SetDim(1, hNum);

    OP_LOGD(context->GetNodeName(), "End to do GroupedBiasAddGradInferShape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedBiasAddGrad).InferShape(GroupedBiasAddGradInferShape);
} // namespace ops
