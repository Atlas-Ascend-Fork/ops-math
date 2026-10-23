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
 * \file strided_slice_grad_infershape.cpp
 * \brief
 */

#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/const_util.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
// define the input idx
static constexpr size_t IN_SHAPE_IDX = 0;
static constexpr size_t IN_BEGIN_IDX = 1;
static constexpr size_t IN_END_IDX = 2;
static constexpr size_t IN_STRIDES_IDX = 3;
static constexpr size_t IN_DY_IDX = 4;
// define the output idx
static constexpr size_t OUT_OUTPUT_IDX = 0;
// define the attr idx
static constexpr size_t ATTR_BEGIN_IDX = 0;
static constexpr size_t ATTR_END_IDX = 1;
static constexpr size_t ATTR_ELLIPSIS_IDX = 2;
static constexpr size_t ATTR_NEW_AXIS_IDX = 3;
static constexpr size_t ATTR_SHRINK_AXIS_IDX = 4;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1L;

static const char* StrideInputName(size_t inputIdx)
{
    if (inputIdx == IN_END_IDX) {
        return "end";
    }
    if (inputIdx == IN_STRIDES_IDX) {
        return "strides";
    }
    return "begin";
}

static const char* MaskAttrName(size_t attrIdx)
{
    if (attrIdx == ATTR_END_IDX) {
        return "end_mask";
    }
    if (attrIdx == ATTR_ELLIPSIS_IDX) {
        return "ellipsis_mask";
    }
    if (attrIdx == ATTR_NEW_AXIS_IDX) {
        return "new_axis_mask";
    }
    if (attrIdx == ATTR_SHRINK_AXIS_IDX) {
        return "shrink_axis_mask";
    }
    return "begin_mask";
}

static ge::graphStatus IsMasksAllZero(gert::InferShapeContext* context, bool& is_mask_all_zero)
{
    is_mask_all_zero = true;
    static const std::vector<size_t> attr_list = {ATTR_BEGIN_IDX, ATTR_END_IDX, ATTR_ELLIPSIS_IDX, ATTR_NEW_AXIS_IDX,
                                                  ATTR_SHRINK_AXIS_IDX};
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "begin_mask", "null",
                                                      "required attribute is missing"),
                return ge::GRAPH_FAILED);
    for (const auto& param : attr_list) {
        const int64_t* mask_value = attrs->GetAttrPointer<int64_t>(param);
        OP_CHECK_IF(mask_value == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), MaskAttrName(param), "null",
                                                          "required attribute is missing"),
                    return ge::GRAPH_FAILED);
        if (*mask_value != 0) {
            is_mask_all_zero = false;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMaxStridedLen(gert::InferShapeContext* context, int64_t& begin_len)
{
    begin_len = -1;
    static const std::vector<size_t> input_list = {IN_BEGIN_IDX, IN_END_IDX, IN_STRIDES_IDX};
    for (const auto& param : input_list) {
        const gert::Shape* param_shape = context->GetInputShape(param);
        OP_CHECK_IF(param_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), StrideInputName(param)),
                    return ge::GRAPH_FAILED);
        const int64_t param_shape_value = param_shape->IsScalar() ? 1 : param_shape->GetDim(0);
        begin_len = std::max(param_shape_value, begin_len);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4StridedSliceGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin do infershape for StridedSliceGrad");
    gert::Shape* output_shape = context->GetOutputShape(OUT_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // get the input const value for shape, save to output_shape
    OP_CHECK_IF(Ops::Base::GetConstIntToShape(context, IN_SHAPE_IDX, *output_shape),
                OP_LOGD(context->GetNodeName(), "do infershape of StridedSliceGrad succ, output = %s",
                        Ops::Base::ToString(*output_shape).c_str()),
                return ge::GRAPH_SUCCESS);

    // dynamic infershape scenario
    const gert::Shape* in_dy_shape = context->GetInputShape(IN_DY_IDX);
    OP_CHECK_IF(in_dy_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "dy"),
                return ge::GRAPH_FAILED);
    const gert::Shape* in_shape_shape = context->GetInputShape(IN_SHAPE_IDX);
    OP_CHECK_IF(in_shape_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(in_shape_shape->GetDimNum() > 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "shape", std::to_string(in_shape_shape->GetDimNum()).c_str(),
                    ("rank must be 1, but shape is " + Ops::Base::ToString(*in_shape_shape)).c_str()),
                return ge::GRAPH_FAILED);
    const int64_t in_shape_shape_value = in_shape_shape->IsScalar() ? 1 : in_shape_shape->GetDim(0);

    OP_CHECK_IF(in_shape_shape_value == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "shape",
                                                          std::to_string(in_shape_shape_value).c_str(),
                                                          "shape cannot be an empty tensor"),
                return ge::GRAPH_FAILED);

    // shape is unknown, out_shape is -2
    if (in_shape_shape_value < 0) {
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }

    // shape_dy is _UNKNOWN_RANK
    if (Ops::Base::IsUnknownRank(*in_dy_shape)) {
        // when in_shape_shape is not -1 or -2, will set all [[-1] * shape[0]] shape
        Ops::Base::SetUnknownShape(in_shape_shape_value, *output_shape);
        return ge::GRAPH_SUCCESS;
    }

    // special branch: when shape is not const and rank of begin < rank of dy
    int64_t begin_len = -1;
    bool no_mask = true;
    OP_CHECK_IF(GetMaxStridedLen(context, begin_len) == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "get max strided len failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsMasksAllZero(context, no_mask) == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "check masks zero failed!"), return ge::GRAPH_FAILED);

    if (no_mask && begin_len > 0 && begin_len < in_shape_shape_value) {
        OP_LOGD(context->GetNodeName(), "Enter the special branch when shape is not const.");
        *output_shape = *in_dy_shape;
        for (int64_t dim = 0; dim < begin_len; dim++) {
            output_shape->SetDim(dim, UNKNOWN_DIM_VALUE_);
        }
        OP_LOGD(context->GetNodeName(), "special branch: output shape is %s",
                Ops::Base::ToString(*in_shape_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    // Set outputShape
    // The SSG's out_range needs to get value of "shape". In compilation, infer_shape will not get
    // value of "shape" while "shape" is variable, set range as (0,-1).
    Ops::Base::SetUnknownShape(in_shape_shape_value, *output_shape);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceGrad)
    .InferShape(InferShape4StridedSliceGrad)
    .InputsDataDependency({IN_SHAPE_IDX, IN_BEGIN_IDX, IN_END_IDX, IN_STRIDES_IDX});
} // namespace ops
