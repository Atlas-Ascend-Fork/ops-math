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
 * \file stateless_bernoulli_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"
#include "random/random_common/op_host/random_infershape_base.h"

using namespace ge;
namespace ops {
static constexpr size_t STATELESS_BERNOULLI_SHAPE = 0;
static constexpr size_t STATELESS_BERNOULLI_PROB = 1;
static constexpr size_t STATELESS_BERNOULLI_SEED = 2;
static constexpr size_t STATELESS_BERNOULLI_OFFSET = 3;
static constexpr size_t STATELESS_BERNOULLI_Y = 0;

static graphStatus InferShapeStatelessBernoulli(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const std::unordered_map<std::string, size_t>& inputMap = {{"shape", STATELESS_BERNOULLI_SHAPE},
                                                               {"prob", STATELESS_BERNOULLI_PROB},
                                                               {"seed", STATELESS_BERNOULLI_SEED},
                                                               {"offset", STATELESS_BERNOULLI_OFFSET}};
    const std::unordered_map<std::string, size_t>& outputMap = {{"y", STATELESS_BERNOULLI_Y}};
    const auto* shape = context->GetInputShape(STATELESS_BERNOULLI_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    // A logical scalar is the GEIR placeholder for an empty dimension list.
    // Actual rank-one lists, including empty lists, retain the common path.
    if (shape->GetDimNum() == 0) {
        const auto* shapeDesc = context->GetInputDesc(STATELESS_BERNOULLI_SHAPE);
        OP_CHECK_NULL_WITH_CONTEXT(context, shapeDesc);
        const auto shapeDtype = shapeDesc->GetDataType();
        OP_CHECK_IF(shapeDtype != ge::DT_INT32 && shapeDtype != ge::DT_INT64,
                    OP_LOGE(context->GetNodeName(), "shape input must be int32 or int64."), return ge::GRAPH_FAILED);
        for (size_t index = STATELESS_BERNOULLI_SHAPE; index <= STATELESS_BERNOULLI_OFFSET; ++index) {
            const auto* inputShape = context->GetInputShape(index);
            OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
        }
        auto* outputShape = context->GetOutputShape(STATELESS_BERNOULLI_Y);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        outputShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }
    int32_t mode = ops::randomCommon::MODE_DEPENDENCY;
    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}
IMPL_OP_INFERSHAPE(StatelessBernoulli).InputsDataDependency({0}).InferShape(InferShapeStatelessBernoulli);

} // namespace ops
