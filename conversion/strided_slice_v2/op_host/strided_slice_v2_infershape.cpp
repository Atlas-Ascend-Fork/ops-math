/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file strided_slice_v2.cc
 * \brief
 */
#include <algorithm>
#include <cstdint>
#include <vector>
#include "register/op_impl_registry.h"
#include "../../strided_slice/op_host/strided_slice_util.h"
#include "log/log.h"
#include "util/const_util.h"
#include "util/shape_util.h"
#include "op_api/op_util.h"

namespace ops {
using namespace ge;

static const size_t IDX_X = 0;
static const size_t IDX_BEGIN = 1;
static const size_t IDX_END = 2;
static const size_t IDX_AXES = 3;
static const size_t IDX_STRIDES = 4;
static const size_t IDX_Y = 0;

static constexpr int64_t MAX_INT = static_cast<int64_t>((static_cast<uint32_t>(-1)) >> 1);
static constexpr int64_t MIN_INT = ~MAX_INT;
static constexpr int64_t MAX_INT64 = static_cast<int64_t>((static_cast<uint64_t>(-1)) >> 1);
static constexpr int64_t UNKNOWN_DIM_VALUE = -1;

struct StridedSliceV2Inputs {
    int64_t begin_len = -1;
    bool strides_present = false;
    bool strides_scalar = false;
    bool begin_valid = true;
    bool end_valid = true;
    bool stride_valid = true;
    bool axes_valid = true;
    std::vector<int64_t> begin_list;
    std::vector<int64_t> end_list;
    std::vector<int64_t> stride_list;
    std::vector<int64_t> axes_list;
};

static bool GetConstListValue(const gert::Tensor* tensor, std::vector<int64_t>& valueList)
{
    if (tensor == nullptr || !IsConstTensor(tensor)) {
        return false;
    }
    const int64_t valueNum = static_cast<int64_t>(tensor->GetShapeSize());
    valueList.clear();
    if (valueNum <= 0) {
        return true;
    }
    valueList.resize(static_cast<size_t>(valueNum));
    const ge::DataType dtype = tensor->GetDataType();
    if (dtype == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        if (data == nullptr) {
            valueList.clear();
            return false;
        }
        for (int64_t i = 0; i < valueNum; ++i) {
            valueList[static_cast<size_t>(i)] = static_cast<int64_t>(data[i]);
        }
    } else if (dtype == ge::DT_INT64) {
        const int64_t* data = tensor->GetData<int64_t>();
        if (data == nullptr) {
            valueList.clear();
            return false;
        }
        for (int64_t i = 0; i < valueNum; ++i) {
            valueList[static_cast<size_t>(i)] = data[i];
        }
    } else {
        valueList.clear();
        return false;
    }
    return true;
}

static std::vector<int64_t> ShapeDimsToVector(const gert::Shape& shape)
{
    std::vector<int64_t> dims;
    const size_t dimNum = shape.GetDimNum();
    dims.reserve(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

static void MakeUpShapeRange(const gert::Shape& shape, std::vector<std::pair<int64_t, int64_t>>& range)
{
    if (Ops::Base::IsUnknownRank(shape) || !range.empty()) {
        return;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        range.push_back(dim == UNKNOWN_DIM_VALUE ? std::pair<int64_t, int64_t>(0, -1) :
                                                   std::pair<int64_t, int64_t>(dim, dim));
    }
}

static void SetAllUnknownDim(int64_t rank, gert::Shape* shape)
{
    shape->SetDimNum(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        shape->SetDim(static_cast<size_t>(i), UNKNOWN_DIM_VALUE);
    }
}

static ge::graphStatus InferShapeAndRangeCore(const gert::Shape& shape_x,
                                              const std::vector<std::pair<int64_t, int64_t>>& input_ranges,
                                              const StridedSliceV2Inputs& info, gert::Shape* out_shape,
                                              std::vector<std::pair<int64_t, int64_t>>* output_ranges)
{
    const int64_t rank_num = static_cast<int64_t>(shape_x.GetDimNum());

    std::vector<int64_t> stride_list = info.stride_list;
    bool stride_valid = info.stride_valid;

    // If strides is absent or scalar, the default stride is 1.
    if (info.begin_len > 0 && !stride_valid && (!info.strides_present || info.strides_scalar)) {
        stride_valid = true;
        stride_list.assign(static_cast<size_t>(info.begin_len), 1);
    }

    // begin/end/strides are not all usable, set output shape to all -1 of the input rank.
    if (info.begin_len == -1 || !stride_valid || (!info.end_valid && info.begin_len == 0) ||
        (!info.begin_valid && info.begin_len == 0)) {
        SetAllUnknownDim(rank_num, out_shape);
        if (output_ranges != nullptr) {
            output_ranges->assign(static_cast<size_t>(rank_num), std::pair<int64_t, int64_t>(0, -1));
        }
        return GRAPH_SUCCESS;
    }

    std::vector<int64_t> begin_list = info.begin_list;
    std::vector<int64_t> end_list = info.end_list;

    // If begin is invalid, set begin with begin_len count of 0.
    if (!info.begin_valid) {
        begin_list.assign(static_cast<size_t>(info.begin_len), 0);
    }

    // If end is invalid, set end with begin_len count of the input shape dims.
    if (!info.end_valid) {
        std::vector<int64_t> shape_dims = ShapeDimsToVector(shape_x);
        if (info.begin_len < rank_num) {
            end_list.assign(shape_dims.begin(), shape_dims.begin() + static_cast<size_t>(info.begin_len));
        } else {
            end_list = shape_dims;
            if (!shape_dims.empty()) {
                for (int64_t i = rank_num; i < info.begin_len; ++i) {
                    end_list.push_back(shape_dims.back());
                }
            }
        }
    }

    // If stride is invalid, set stride with begin_len count of 1.
    if (!stride_valid) {
        stride_list.assign(static_cast<size_t>(info.begin_len), 1);
    }

    // Expand the sparse begin/end/stride into full rank according to axes, and derive begin_mask/end_mask from axes.
    uint64_t axes_mask = 0;
    uint64_t ends_mask = 0;
    if (info.axes_valid) {
        auto clamp = [](int64_t x) { return x > MAX_INT ? MAX_INT : (x < MIN_INT ? MIN_INT : x); };
        std::vector<int64_t> processed_begin(static_cast<size_t>(rank_num), 0);
        std::vector<int64_t> processed_end = ShapeDimsToVector(shape_x);
        std::vector<int64_t> processed_stride(static_cast<size_t>(rank_num), 1);
        std::vector<int64_t> axes_list = info.axes_list;
        for (size_t i = 0;
             i < axes_list.size() && i < end_list.size() && i < begin_list.size() && i < stride_list.size(); ++i) {
            int64_t axes_index = axes_list[i];
            if (axes_index < 0) {
                axes_index = axes_index + rank_num;
            }
            if (axes_index >= rank_num) {
                axes_index = rank_num - 1;
            }
            if (axes_index < 0) {
                axes_index = 0;
            }
            axes_mask = (static_cast<uint64_t>(1) << static_cast<uint64_t>(axes_index)) | axes_mask;
            processed_end[static_cast<size_t>(axes_index)] = clamp(end_list[i]);
            processed_begin[static_cast<size_t>(axes_index)] = clamp(begin_list[i]);
            processed_stride[static_cast<size_t>(axes_index)] = stride_list[i];
        }
        axes_mask = ~axes_mask;
        ends_mask = axes_mask;
        for (size_t i = 0; i < processed_end.size(); ++i) {
            if (processed_end[i] == MAX_INT64 || processed_end[i] == MAX_INT) {
                ends_mask = (static_cast<uint64_t>(1) << static_cast<uint64_t>(i)) | ends_mask;
            }
        }
        begin_list.assign(processed_begin.begin(), processed_begin.end());
        end_list.assign(processed_end.begin(), processed_end.end());
        stride_list.assign(processed_stride.begin(), processed_stride.end());
    }

    if (rank_num == 0) {
        OP_LOGE(OP_NAME, "input x's dimnum must be in [1, 8].");
        return ge::GRAPH_FAILED;
    }

    if (end_list.size() != begin_list.size()) {
        OP_LOGE(OP_NAME, "end shape, begin shape length mismatch!");
        return ge::GRAPH_FAILED;
    }

    StridedSliceParams input_params;
    input_params.input_shape = shape_x;
    input_params.ranges = input_ranges;
    input_params.begin_mask = axes_mask;
    input_params.end_mask = ends_mask;
    input_params.ellipsis_mask = 0;
    input_params.new_axis_mask = 0;
    input_params.shrink_axis_mask = 0;
    input_params.begin_valid = info.begin_valid;
    input_params.end_valid = info.end_valid;
    input_params.stride_valid = stride_valid;
    for (const int64_t v : begin_list) {
        input_params.begin.AppendDim(v);
    }
    for (const int64_t v : end_list) {
        input_params.end.AppendDim(v);
    }
    for (const int64_t v : stride_list) {
        input_params.strides.AppendDim(v);
    }

    if (output_ranges != nullptr) {
        if (!InferShapeAndRange(input_params, out_shape, *output_ranges)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!InferShape(input_params, out_shape)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static int64_t GetDim0(const gert::Shape* shape)
{
    if (shape == nullptr || shape->GetDimNum() == 0) {
        return 0;
    }
    return shape->GetDim(0);
}

static ge::graphStatus InferShape4StridedSliceV2(gert::InferShapeContext* context)
{
    const auto shape_x = context->GetInputShape(IDX_X);
    OP_CHECK_IF(shape_x == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    auto shape_y = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_y);

    if (Ops::Base::IsUnknownRank(*shape_x)) {
        Ops::Base::SetUnknownRank(*shape_y);
        OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s.", Ops::Base::ToString(*shape_y).c_str());
        return GRAPH_SUCCESS;
    }

    const auto shape_begin = context->GetInputShape(IDX_BEGIN);
    OP_CHECK_IF(shape_begin == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "begin"),
                return ge::GRAPH_FAILED);
    const auto shape_end = context->GetInputShape(IDX_END);
    OP_CHECK_IF(shape_end == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "end"),
                return ge::GRAPH_FAILED);
    const auto tensor_begin = context->GetInputTensor(IDX_BEGIN);
    OP_CHECK_IF(tensor_begin == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "begin"),
                return ge::GRAPH_FAILED);
    const auto tensor_end = context->GetInputTensor(IDX_END);
    OP_CHECK_IF(tensor_end == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "end"),
                return ge::GRAPH_FAILED);

    const gert::Tensor* tensor_axes = context->GetOptionalInputTensor(IDX_AXES);
    const gert::Tensor* tensor_strides = context->GetOptionalInputTensor(IDX_STRIDES);
    const gert::Shape* shape_axes = context->GetOptionalInputShape(IDX_AXES);
    const gert::Shape* shape_strides = context->GetOptionalInputShape(IDX_STRIDES);

    int64_t begin_len = -1;
    begin_len = std::max(GetDim0(shape_begin), begin_len);
    begin_len = std::max(GetDim0(shape_end), begin_len);
    if (shape_axes != nullptr) {
        begin_len = std::max(GetDim0(shape_axes), begin_len);
    }
    if (shape_strides != nullptr) {
        begin_len = std::max(GetDim0(shape_strides), begin_len);
    }

    StridedSliceV2Inputs info;
    info.begin_len = begin_len;
    info.strides_present = (tensor_strides != nullptr);
    info.strides_scalar = (tensor_strides != nullptr && tensor_strides->GetOriginShape().GetDimNum() == 0);

    bool all_const = true;
    all_const = GetConstListValue(tensor_begin, info.begin_list) && all_const;
    all_const = GetConstListValue(tensor_end, info.end_list) && all_const;
    all_const = GetConstListValue(tensor_strides, info.stride_list) && all_const;
    all_const = GetConstListValue(tensor_axes, info.axes_list) && all_const;
    if (!all_const) {
        info.begin_valid = !info.begin_list.empty();
        info.end_valid = !info.end_list.empty();
        info.stride_valid = !info.stride_list.empty();
        info.axes_valid = !info.axes_list.empty();
    }

    std::vector<std::pair<int64_t, int64_t>> input_ranges;
    MakeUpShapeRange(*shape_x, input_ranges);

    const ge::graphStatus ret = InferShapeAndRangeCore(*shape_x, input_ranges, info, shape_y, nullptr);
    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s.", Ops::Base::ToString(*shape_y).c_str());
    return ret;
}

IMPL_OP_INFERSHAPE(StridedSliceV2)
    .InferShape(InferShape4StridedSliceV2)
    .InputsDataDependency({IDX_BEGIN, IDX_END, IDX_AXES, IDX_STRIDES});
} // namespace ops
