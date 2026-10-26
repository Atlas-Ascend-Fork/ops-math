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
 * \file reshape_infershape.cpp
 * \brief InferShape and InferDataType implementation for Reshape operator.
 */
#include <limits>
#include <vector>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
// both operands must be non-negative
bool MulOverflow(const int64_t lhs, const int64_t rhs)
{
    return (lhs != 0) && (rhs != 0) && (lhs > (std::numeric_limits<int64_t>::max() / rhs));
}

template <typename T>
ge::graphStatus InferShapeForEmptyTensorX(const gert::Shape& x_shape, const T* reshape_data, const size_t reshape_size,
                                          gert::Shape* y_shape)
{
    int64_t neg_index = -1;
    int64_t num_of_neg_1 = 0;
    bool target_has_zero = false;
    int64_t shape_des_size = 1;
    y_shape->SetDimNum(reshape_size);
    for (size_t i = 0U; i < reshape_size; ++i) {
        y_shape->SetDim(i, reshape_data[i]);
        if (reshape_data[i] == -1) {
            neg_index = static_cast<int64_t>(i);
            ++num_of_neg_1;
            continue;
        }
        if (reshape_data[i] == 0) {
            target_has_zero = true;
        } else if (reshape_data[i] < 0) {
            OP_LOGE("Reshape", "shape dim[%zu] must be -1 or non-negative, but got %ld!", i,
                    static_cast<int64_t>(reshape_data[i]));
            return ge::GRAPH_FAILED;
        } else {
            if (MulOverflow(shape_des_size, static_cast<int64_t>(reshape_data[i]))) {
                OP_LOGE("Reshape", "The product of the target dims overflows int64 at dim[%zu]!", i);
                return ge::GRAPH_FAILED;
            }
            shape_des_size = shape_des_size * reshape_data[i];
        }
    }
    // target contains 0 and no -1: keep the target dims as-is
    if ((num_of_neg_1 == 0) && target_has_zero) {
        return ge::GRAPH_SUCCESS;
    }
    // target contains one -1
    if (num_of_neg_1 == 1) {
        // target has no 0: the -1 dim is the only dim that can make the output empty
        if (!target_has_zero) {
            y_shape->SetDim(static_cast<size_t>(neg_index), 0);
            return ge::GRAPH_SUCCESS;
        }
        // target contains 0: infer from the non-zero products, the remainder is discarded as RT1.0 does
        int64_t x_shape_size = 1;
        for (size_t i = 0U; i < x_shape.GetDimNum(); ++i) {
            if (x_shape.GetDim(i) < 0) {
                OP_LOGE("Reshape", "x dim[%zu] is %ld, cannot infer the -1 dim of shape input for empty tensor x!", i,
                        x_shape.GetDim(i));
                return ge::GRAPH_FAILED;
            }
            if (x_shape.GetDim(i) != 0) {
                if (MulOverflow(x_shape_size, x_shape.GetDim(i))) {
                    OP_LOGE("Reshape", "The product of the x dims overflows int64 at dim[%zu]!", i);
                    return ge::GRAPH_FAILED;
                }
                x_shape_size = x_shape_size * x_shape.GetDim(i);
            }
        }
        const auto real_dim = (x_shape_size == shape_des_size) ? 0 : (x_shape_size / shape_des_size);
        y_shape->SetDim(static_cast<size_t>(neg_index), real_dim);
        return ge::GRAPH_SUCCESS;
    }
    // more than one -1 in target
    if (num_of_neg_1 > 1) {
        OP_LOGE("Reshape", "Only one dim of shape input can be -1, while the count is %ld!", num_of_neg_1);
        return ge::GRAPH_FAILED;
    }
    // no -1 and no 0 in target
    OP_LOGE("Reshape", "Target shape has no 0 or -1 dim, cannot infer shape for empty tensor x!");
    return ge::GRAPH_FAILED;
}

// case 5 impl, see the case 5 comment in InferShapeForReshape for the 0/-1 semantics
template <typename T>
ge::graphStatus ReshapeInferShapeImpl(const gert::Shape& x_shape, const T* reshape_dims, const size_t reshape_size,
                                      const bool literal_zero, gert::Shape* y_shape)
{
    constexpr T kUnknownDim = -1;
    constexpr size_t kNoInferIndex = std::numeric_limits<size_t>::max();
    y_shape->SetDimNum(reshape_size);
    const auto x_shape_size = x_shape.GetShapeSize();
    const bool x_is_dynamic = x_shape_size < 0;
    int64_t output_shape_size = 1;
    size_t infer_index = kNoInferIndex;
    bool has_literal_zero = false;
    size_t dynamic_infer_idx = kNoInferIndex;
    int64_t dynamic_infer_product = 1;
    bool copied_unknown_dim[gert::Shape::kMaxDimNum] = {false};
    for (size_t i = 0U; i < reshape_size; ++i) {
        if (reshape_dims[i] == 0) {
            if (literal_zero) {
                y_shape->SetDim(i, 0);
                output_shape_size = 0;
                has_literal_zero = true;
                continue;
            }
            if ((i >= x_shape.GetDimNum()) || (x_shape.GetDim(i) < -1)) {
                OP_LOGE("Reshape", "shape dim[%zu] is 0, cannot copy from the dim of x!", i);
                return ge::GRAPH_FAILED;
            }
            const auto copied_dim = x_shape.GetDim(i);
            y_shape->SetDim(i, copied_dim);
            if (copied_dim == ge::UNKNOWN_DIM) {
                // the copied unknown dim counts as 1 in the size math (RT1.0)
                copied_unknown_dim[i] = true;
            } else if (x_is_dynamic) {
                if (MulOverflow(dynamic_infer_product, copied_dim)) {
                    OP_LOGE("Reshape", "The output shape size overflows int64 at dim[%zu]!", i);
                    return ge::GRAPH_FAILED;
                }
                dynamic_infer_product *= copied_dim;
            } else {
                if (MulOverflow(output_shape_size, copied_dim)) {
                    OP_LOGE("Reshape", "The output shape size overflows int64 at dim[%zu]!", i);
                    return ge::GRAPH_FAILED;
                }
                output_shape_size *= copied_dim;
            }
        } else if (reshape_dims[i] == kUnknownDim) {
            if (x_is_dynamic) {
                if (dynamic_infer_idx != kNoInferIndex) {
                    OP_LOGE("Reshape",
                            "Only one dim of shape input can be -1, while both shape[%zu] and shape[%zu] "
                            "are -1!",
                            dynamic_infer_idx, i);
                    return ge::GRAPH_FAILED;
                }
                y_shape->SetDim(i, ge::UNKNOWN_DIM);
                dynamic_infer_idx = i;
            } else {
                if (infer_index != kNoInferIndex) {
                    OP_LOGE("Reshape",
                            "Only one dim of shape input can be -1, while both shape[%zu] and shape[%zu] "
                            "are -1!",
                            infer_index, i);
                    return ge::GRAPH_FAILED;
                }
                y_shape->SetDim(i, 1);
                infer_index = i;
            }
        } else if (reshape_dims[i] < 0) {
            OP_LOGE("Reshape", "shape dim[%zu] must be -1 or non-negative, but got %ld!", i,
                    static_cast<int64_t>(reshape_dims[i]));
            return ge::GRAPH_FAILED;
        } else {
            y_shape->SetDim(i, reshape_dims[i]);
            if (x_is_dynamic) {
                if (MulOverflow(dynamic_infer_product, static_cast<int64_t>(reshape_dims[i]))) {
                    OP_LOGE("Reshape", "The output shape size overflows int64 at dim[%zu]!", i);
                    return ge::GRAPH_FAILED;
                }
                dynamic_infer_product *= reshape_dims[i];
            } else {
                if (MulOverflow(output_shape_size, static_cast<int64_t>(reshape_dims[i]))) {
                    OP_LOGE("Reshape", "The output shape size overflows int64 at dim[%zu]!", i);
                    return ge::GRAPH_FAILED;
                }
                output_shape_size *= reshape_dims[i];
            }
        }
    }
    // dynamic x skips the size check: a literal 0 dim is rejected, and the -1 dim is inferred
    // from the effective x size when the only unknown dims of x are the copied ones (RT1.0)
    if (x_is_dynamic) {
        if (has_literal_zero) {
            OP_LOGE("Reshape", "The output shape contains a literal 0 dim, cannot match the unknown size of x!");
            return ge::GRAPH_FAILED;
        }
        int64_t effective_x_size = 1;
        bool x_effectively_static = true;
        for (size_t i = 0U; i < x_shape.GetDimNum(); ++i) {
            const auto x_dim = x_shape.GetDim(i);
            if ((x_dim < 0) && !copied_unknown_dim[i]) {
                x_effectively_static = false;
                break;
            }
            const auto dim_value = copied_unknown_dim[i] ? 1 : x_dim;
            if (MulOverflow(effective_x_size, dim_value)) {
                OP_LOGE("Reshape", "The effective x shape size overflows int64 at dim[%zu]!", i);
                return ge::GRAPH_FAILED;
            }
            effective_x_size *= dim_value;
        }
        if (!x_effectively_static || (dynamic_infer_idx == kNoInferIndex)) {
            return ge::GRAPH_SUCCESS;
        }
        if ((dynamic_infer_product <= 0) || ((effective_x_size % dynamic_infer_product) != 0)) {
            OP_LOGE("Reshape", "The effective x shape size %ld cannot be divided by the known output shape size %ld!",
                    effective_x_size, dynamic_infer_product);
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDim(dynamic_infer_idx, effective_x_size / dynamic_infer_product);
        return ge::GRAPH_SUCCESS;
    }
    if (infer_index == kNoInferIndex) {
        if (output_shape_size != x_shape_size) {
            OP_LOGE("Reshape", "The output shape size %ld is not equal to x shape size %ld!", output_shape_size,
                    x_shape_size);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    // x is not an empty tensor here
    if (output_shape_size == 0) {
        OP_LOGE("Reshape", "The output shape contains a 0 dim, cannot infer the -1 dim for the non-empty x!");
        return ge::GRAPH_FAILED;
    }
    if ((x_shape_size % output_shape_size) == 0) {
        y_shape->SetDim(infer_index, x_shape_size / output_shape_size);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE("Reshape", "x shape size %ld cannot be divided by the known output shape size %ld!", x_shape_size,
            output_shape_size);
    return ge::GRAPH_FAILED;
}
} // namespace

static ge::graphStatus InferShapeForReshape(gert::InferShapeContext* context)
{
    const auto x_shape = context->GetInputShape(0U);
    const auto shape_tensor = context->GetInputTensor(1U);
    auto y_shape = context->GetOutputShape(0U);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    // allowzero is a private attr set by name by the ONNX plugin, materialized after the IR
    // segment (node value first, the default 1 as fallback), so it always sits at the last index
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t allowzero = 1;
    if (attrs->GetAttrNum() > 0U) {
        const auto attr_value = attrs->GetInt(attrs->GetAttrNum() - 1U);
        if (attr_value != nullptr) {
            allowzero = *attr_value;
        }
    }

    // case 1: shape data is not got on compile stage, infer output rank from the shape of shape
    // input, output dims are all unknown
    if (shape_tensor->GetTensorData().GetAddr() == nullptr) {
        const auto shape_input_shape = context->GetInputShape(1U);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape_input_shape);
        OP_LOGD(context, "Shape data is not got on compile stage. Infer output rank from the shape of shape input.");
        if (shape_input_shape->GetDimNum() > 1U) {
            OP_LOGE(context, "Shape input rank is %zu, should be not larger than 1!", shape_input_shape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        const auto y_rank = (shape_input_shape->GetDimNum() == 0U) ? 0 : shape_input_shape->GetDim(0U);
        if (y_rank < 0) {
            y_shape->SetDimNum(1U);
            y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
            return ge::GRAPH_SUCCESS;
        }
        if (y_rank > static_cast<int64_t>(gert::Shape::kMaxDimNum)) {
            OP_LOGE(context, "Element num of shape input is %ld, larger than kMaxDimNum which is %zu!", y_rank,
                    gert::Shape::kMaxDimNum);
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDimNum(static_cast<size_t>(y_rank));
        for (size_t i = 0U; i < static_cast<size_t>(y_rank); ++i) {
            y_shape->SetDim(i, ge::UNKNOWN_DIM);
        }
        return ge::GRAPH_SUCCESS;
    }

    const auto reshape_size = shape_tensor->GetShapeSize();
    // case 2: element num of shape input is invalid
    if (reshape_size < 0) {
        OP_LOGE(context, "Element num of shape input is %ld, cannot be less than 0!", reshape_size);
        return ge::GRAPH_FAILED;
    }
    // case 3: shape input contains no element, output is scalar; x must be a scalar or dynamic
    if (reshape_size == 0) {
        const auto x_size = x_shape->GetShapeSize();
        if ((x_size != 1) && (x_size >= 0)) {
            OP_LOGE(context, "Shape input contains no element, x shape size is %ld, must be 1 to reshape to scalar!",
                    x_size);
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(context, "Shape input contains no element. Set output shape as scalar.");
        y_shape->SetDimNum(0U);
        return ge::GRAPH_SUCCESS;
    }
    if (static_cast<uint64_t>(reshape_size) > gert::Shape::kMaxDimNum) {
        OP_LOGE(context, "Element num of shape input is %ld, larger than kMaxDimNum which is %zu!", reshape_size,
                gert::Shape::kMaxDimNum);
        return ge::GRAPH_FAILED;
    }
    const auto target_size = static_cast<size_t>(reshape_size);

    // the data is about to be read, other dtypes would be misread as int64 or read out of range
    const auto shape_dtype = shape_tensor->GetDataType();
    if ((shape_dtype != ge::DT_INT32) && (shape_dtype != ge::DT_INT64)) {
        OP_LOGE(context, "Data type of shape input is %d, must be DT_INT32 or DT_INT64!",
                static_cast<int32_t>(shape_dtype));
        return ge::GRAPH_FAILED;
    }

    // case 4: x is an empty tensor (shape size is 0)
    if (x_shape->GetShapeSize() == 0) {
        if (shape_dtype == ge::DT_INT32) {
            return InferShapeForEmptyTensorX(*x_shape, shape_tensor->GetData<int32_t>(), target_size, y_shape);
        }
        return InferShapeForEmptyTensorX(*x_shape, shape_tensor->GetData<int64_t>(), target_size, y_shape);
    }
    // case 5: normal or dynamic x, -1 as unknown dim; 0 copies the x dim when allowzero is 0,
    // otherwise it is a literal dim value
    const bool literal_zero = (allowzero != 0);
    if (shape_dtype == ge::DT_INT32) {
        return ReshapeInferShapeImpl(*x_shape, shape_tensor->GetData<int32_t>(), target_size, literal_zero, y_shape);
    }
    return ReshapeInferShapeImpl(*x_shape, shape_tensor->GetData<int64_t>(), target_size, literal_zero, y_shape);
}

static ge::graphStatus InferDataTypeForReshape(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(Reshape)
    .InferShape(InferShapeForReshape)
    .InferDataType(InferDataTypeForReshape)
    .InputsDataDependency({1U})
    .PrivateAttr("allowzero", int64_t{1});
} // namespace ops
