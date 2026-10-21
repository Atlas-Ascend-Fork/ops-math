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
 * \file bitcast_infershape.cpp
 * \brief infershape func of Bitcast
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"

using namespace ge;
namespace ops {
namespace {
constexpr int32_t kUnknownSizeByDtype = -1;
constexpr int32_t kKeepdimAtrrIndex = 1;
const std::unordered_set<int32_t> kUnsupportDtypes = {
    static_cast<int32_t>(ge::DT_STRING), static_cast<int32_t>(ge::DT_STRING_REF),
    static_cast<int32_t>(ge::DT_UNDEFINED), static_cast<int32_t>(ge::DT_MAX)};

ge::graphStatus InferShapeRangeImplForBitcast(gert::Range<gert::Shape>* y_shape_range,
                                              const gert::Range<gert::Shape>* x_shape_range, const ge::DataType x_dtype,
                                              const ge::DataType target_dtype)
{
    auto x_dtype_size = ge::GetSizeByDataType(x_dtype);
    auto target_dtype_size = ge::GetSizeByDataType(target_dtype);
    const auto min_shape = x_shape_range->GetMin();
    const auto max_shape = x_shape_range->GetMax();
    if (x_dtype_size == target_dtype_size) {
        *(y_shape_range->GetMin()) = *min_shape;
        *(y_shape_range->GetMax()) = *max_shape;
        return ge::GRAPH_SUCCESS;
    }

    bool is_use_bit = (x_dtype_size > kDataTypeSizeBitOffset) || (target_dtype_size > kDataTypeSizeBitOffset);
    if (is_use_bit) {
        x_dtype_size = x_dtype_size > kDataTypeSizeBitOffset ? (x_dtype_size - kDataTypeSizeBitOffset) :
                                                               x_dtype_size * kBitNumOfOneByte;
        target_dtype_size = target_dtype_size > kDataTypeSizeBitOffset ? (target_dtype_size - kDataTypeSizeBitOffset) :
                                                                         target_dtype_size * kBitNumOfOneByte;
    }

    if (x_dtype_size > target_dtype_size) {
        *(y_shape_range->GetMin()) = *min_shape;
        *(y_shape_range->GetMax()) = *max_shape;
        const int32_t kTimesOfDtypeSize = x_dtype_size / target_dtype_size;
        y_shape_range->GetMin()->AppendDim(kTimesOfDtypeSize);
        y_shape_range->GetMax()->AppendDim(kTimesOfDtypeSize);
    } else {
        *(y_shape_range->GetMin()) = *min_shape;
        *(y_shape_range->GetMax()) = *max_shape;
        y_shape_range->GetMin()->SetDimNum(y_shape_range->GetMin()->GetDimNum() - 1);
        y_shape_range->GetMax()->SetDimNum(y_shape_range->GetMax()->GetDimNum() - 1);
    }
    return ge::GRAPH_SUCCESS;
}

bool IsKeepDim(const gert::RuntimeAttrs* attrs)
{
    const auto* keep_dim_attr = attrs->GetBool(kKeepdimAtrrIndex);
    return (keep_dim_attr == nullptr) ? false : *keep_dim_attr;
}

ge::graphStatus CalcAndUpdateShapeByDtypeHigh2Low(gert::InferShapeContext* context, const int32_t x_dtype_size,
                                                  const int32_t target_dtype_size, const bool is_use_bit)
{
    const auto x_shape = context->GetInputShape(0);
    auto y_shape = context->GetOutputShape(0);

    const auto x_desc = context->GetInputDesc(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto x_dtype = x_desc->GetDataType();
    const auto target_dtype = *(attrs->GetAttrPointer<ge::DataType>(0));

    if (x_dtype_size % target_dtype_size != 0) {
        OP_LOGE(context, "x dtype[%s, size[%d]] was not divisible by attr dtype[%s, size[%d]], unit[%s].",
                ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), x_dtype_size,
                ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString(), target_dtype_size,
                is_use_bit ? "bit" : "byte");
        return GRAPH_FAILED;
    }
    *y_shape = *x_shape;
    auto multiple = x_dtype_size / target_dtype_size;
    if (IsKeepDim(attrs)) {
        if (x_shape->IsScalar()) {
            OP_LOGE(context, "x is scalar, x dtype[%s], attr dtype[%s], x can not be scalar, when keepdim is true.",
                    ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(),
                    ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString());
            return ge::GRAPH_FAILED;
        }
        const auto rank = x_shape->GetDimNum();
        if (x_shape->GetDim(rank - 1) != ge::UNKNOWN_DIM) {
            y_shape->SetDim(rank - 1, y_shape->GetDim(rank - 1) * multiple);
        }
    } else {
        y_shape->AppendDim(multiple);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcAndUpdateShapeByDtypeLow2High(gert::InferShapeContext* context, const int32_t x_dtype_size,
                                                  const int32_t target_dtype_size, const bool is_use_bit)
{
    const auto x_shape = context->GetInputShape(0);
    auto y_shape = context->GetOutputShape(0);

    const auto x_desc = context->GetInputDesc(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto x_dtype = x_desc->GetDataType();
    const auto target_dtype = *(attrs->GetAttrPointer<ge::DataType>(0));
    if (x_shape->IsScalar()) {
        OP_LOGE(context, "x is scalar, x dtype[%s], attr dtype[%s], can not squeeze any more.",
                ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString());
        return ge::GRAPH_FAILED;
    }
    if (target_dtype_size % x_dtype_size != 0) {
        OP_LOGE(context, "attr dtype[%s, size[%d]] was not divisible by x dtype[%s, size[%d]], unit[%s]",
                ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString(), target_dtype_size,
                ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), x_dtype_size, is_use_bit ? "bit" : "byte");
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;
    const auto rank = x_shape->GetDimNum();
    if (IsKeepDim(attrs)) {
        auto multiple = target_dtype_size / x_dtype_size;
        if ((x_shape->GetDim(rank - 1) != ge::UNKNOWN_DIM) && (x_shape->GetDim(rank - 1) % multiple != 0)) {
            OP_LOGE(context,
                    "last dim[%ld] of x shape is not a multiple of (attr dtype[%s, size[%d]] / x dtype[%s, size[%d]], "
                    "unit[%s]).",
                    x_shape->GetDim(x_shape->GetDimNum() - 1),
                    ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString(), target_dtype_size,
                    ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), x_dtype_size,
                    is_use_bit ? "bit" : "byte");
            return ge::GRAPH_FAILED;
        }
        if (x_shape->GetDim(rank - 1) != ge::UNKNOWN_DIM) {
            y_shape->SetDim(rank - 1, x_shape->GetDim(rank - 1) / multiple);
        }
    } else {
        if ((x_shape->GetDim(rank - 1) != ge::UNKNOWN_DIM) &&
            (x_shape->GetDim(rank - 1) != (target_dtype_size / x_dtype_size))) {
            OP_LOGE(context,
                    "last dim[%ld] of x shape is not equal to (attr dtype[%s, size[%d]] / x dtype[%s, size[%d]], "
                    "unit[%s]).",
                    x_shape->GetDim(x_shape->GetDimNum() - 1),
                    ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString(), target_dtype_size,
                    ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), x_dtype_size,
                    is_use_bit ? "bit" : "byte");
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDimNum(y_shape->GetDimNum() - 1);
    }
    return ge::GRAPH_SUCCESS;
}

/**
keepdim = false:
  1)sizeof(xDtype)>sizeof(yDtype):在shape尾部扩1维，维度值=sizeof(xDtype)/sizeof(yDtype)，需满足：
    a)sizeof(xDtype)/%sizeof(yDtype) == 0
    b)标量扩充为1维向量
    c)处理空tensor：扩维后依然是空tensor
  2)sizeof(xDtype)<sizeof(yDtype):将shape尾部降1维，此时需满足：
    a)sizeof(xDtype)/%sizeof(yDtype) == 0
    b)xShape的最后一维度 == sizeof(xDtype)/sizeof(yDtype)
    c)x不能为标量
    d)不处理空tensor,b)条可拦截
  3)sizeof(xDtype)==sizeof(yDtype):维度不变
keepdim = true:
  1)sizeof(xDtype)>sizeof(yDtype):尾轴乘倍数，倍数=sizeof(xDtype)/sizeof(yDtype),需满足：
    a)sizeof(xDtype)/%sizeof(yDtype) == 0
    b)x不能为标量
    c)处理空tensor:空tensor的shape不变
  2)sizeof(xDtype)<sizeof(yDtype):尾轴除倍数，此时需满足：
    a)sizeof(xDtype)/%sizeof(yDtype) == 0
    b)xShape的最后一维度 == sizeof(xDtype)/sizeof(yDtype)的倍数
    c)x不能为标量
    d)处理空tensor:空tensor的shape不变
  3)sizeof(xDtype)==sizeof(yDtype):维度不变
*/

ge::graphStatus InferShapeImplForBitcast(gert::InferShapeContext* context, gert::Shape* y_shape,
                                         const gert::Shape* x_shape, const ge::DataType x_dtype,
                                         const ge::DataType target_dtype)
{
    auto x_dtype_size = ge::GetSizeByDataType(x_dtype);
    auto target_dtype_size = ge::GetSizeByDataType(target_dtype);
    if (x_dtype_size == kUnknownSizeByDtype || target_dtype_size == kUnknownSizeByDtype) {
        OP_LOGE(context, "x dtype[%s, size[%d]] or attr dtype[%s, size[%d]] contains size -1, not supported.",
                ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), x_dtype_size,
                ge::TypeUtils::DataTypeToAscendString(target_dtype).GetString(), target_dtype_size);
        return ge::GRAPH_FAILED;
    }
    if (x_dtype_size == target_dtype_size) {
        *y_shape = *x_shape;
        return ge::GRAPH_SUCCESS;
    }
    bool is_use_bit = (x_dtype_size > kDataTypeSizeBitOffset) || (target_dtype_size > kDataTypeSizeBitOffset);
    if (is_use_bit) {
        x_dtype_size = x_dtype_size > kDataTypeSizeBitOffset ? (x_dtype_size - kDataTypeSizeBitOffset) :
                                                               x_dtype_size * kBitNumOfOneByte;
        target_dtype_size = target_dtype_size > kDataTypeSizeBitOffset ? (target_dtype_size - kDataTypeSizeBitOffset) :
                                                                         target_dtype_size * kBitNumOfOneByte;
    }
    if (x_dtype_size > target_dtype_size) {
        return CalcAndUpdateShapeByDtypeHigh2Low(context, x_dtype_size, target_dtype_size, is_use_bit);
    }
    return CalcAndUpdateShapeByDtypeLow2High(context, x_dtype_size, target_dtype_size, is_use_bit);
}
} // namespace

static ge::graphStatus InferDataTypeForBitCast(gert::InferDataTypeContext* context)
{
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto target_dtype = attrs->GetAttrPointer<ge::DataType>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_dtype);
    OP_LOGI(context, "attr dtype = [%d, %s].", static_cast<int32_t>(*target_dtype),
            ge::TypeUtils::DataTypeToAscendString(*target_dtype).GetString());
    return context->SetOutputDataType(0, *target_dtype);
}

static ge::graphStatus InferShapeRangeForBitcast(gert::InferShapeRangeContext* context)
{
    const auto x_shape_range = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape_range);
    const auto min_shape = x_shape_range->GetMin();
    const auto max_shape = x_shape_range->GetMax();
    OP_CHECK_NULL_WITH_CONTEXT(context, min_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, max_shape);

    if (min_shape->GetDimNum() != max_shape->GetDimNum()) {
        OP_LOGE(context, "shape range min size[%zu] is not equal to max size[%zu]", min_shape->GetDimNum(),
                max_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (min_shape->GetDimNum() == 0 && max_shape->GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    auto y_shape_range = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape_range->GetMax());

    const auto x_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_desc);
    const auto x_dtype = x_desc->GetDataType();

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto target_dtype = attrs->GetAttrPointer<ge::DataType>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_dtype);
    OP_LOGI(context, "x dtype = [%d, %s], attr dtype = [%d, %s].", static_cast<int32_t>(x_dtype),
            ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), static_cast<int32_t>(*target_dtype),
            ge::TypeUtils::DataTypeToAscendString(*target_dtype).GetString());

    return InferShapeRangeImplForBitcast(y_shape_range, x_shape_range, x_dtype, *target_dtype);
}

static ge::graphStatus InferShapeForBitcast(gert::InferShapeContext* context)
{
    const auto x_shape = context->GetInputShape(0);
    auto y_shape = context->GetOutputShape(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto x_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_desc);
    const auto x_dtype = x_desc->GetDataType();

    const auto target_dtype = attrs->GetAttrPointer<ge::DataType>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_dtype);
    OP_LOGI(context, "x dtype = [%d, %s], attr dtype = [%d, %s].", static_cast<int32_t>(x_dtype),
            ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(), static_cast<int32_t>(*target_dtype),
            ge::TypeUtils::DataTypeToAscendString(*target_dtype).GetString());

    if (kUnsupportDtypes.count(static_cast<int32_t>(x_dtype)) > 0U ||
        kUnsupportDtypes.count(static_cast<int32_t>(*target_dtype)) > 0U) {
        OP_LOGE(context, "x dtype[%s] or attr dtype[%s] has not been supported in current version of BitCast.",
                ge::TypeUtils::DataTypeToAscendString(x_dtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(*target_dtype).GetString());
        return ge::GRAPH_FAILED;
    }
    return InferShapeImplForBitcast(context, y_shape, x_shape, x_dtype, *target_dtype);
}

IMPL_OP_INFERSHAPE(Bitcast)
    .InferShape(InferShapeForBitcast)
    .InferShapeRange(InferShapeRangeForBitcast)
    .InferDataType(InferDataTypeForBitCast);
} // namespace ops
