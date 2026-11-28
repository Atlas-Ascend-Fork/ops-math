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
 * \file strided_slice_v2_v3_infershape_common.h
 * \brief StridedSliceV2 与 StridedSliceV3 共用的 InferShape 实现（两算子共有逻辑收敛于此）。
 *
 * 两个算子的 InferShape 流程一致：读取 const 输入 -> 缺省回填 -> axes 展开 ->
 * 复用 strided_slice_util.h 的 InferShape/InferShapeAndRange。版本间语义差异通过
 * StridedSliceInferTraits 由各算子 cpp 注入：
 *  - V2: begin/end 为 int32 语义，需 clamp 到 [MIN_INT32, MAX_INT32]；end 的
 *        int32/int64 双哨兵都按"取到末尾"处理；空 const 列表视为有值；
 *        begin_len == 0 且 begin/end 非法时判为信息不足。
 *  - V3: begin/end 为 int64 语义，无 clamp；仅 int64 哨兵；空 const 列表视为非法。
 */

#ifndef STRIDED_SLICE_V2_V3_INFERSHAPE_COMMON_H_
#define STRIDED_SLICE_V2_V3_INFERSHAPE_COMMON_H_

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

static constexpr int64_t SS_V2_V3_MAX_INT64 = static_cast<int64_t>((static_cast<uint64_t>(-1)) >> 1);
static constexpr int64_t SS_V2_V3_UNKNOWN_DIM = -1;

struct StridedSliceInferInputs {
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

// StridedSliceV2 / StridedSliceV3 的版本语义差异全部集中在此结构，由各算子 cpp 提供实例。
struct StridedSliceInferTraits {
    // begin_len == 0 且 begin/end 非法时也判为信息不足（V2 特有，V3 为 false）。
    bool insufficient_at_zero_len;
    // 全部输入为 const 且列表为空时，是否仍视为"有值"（V2 true / V3 false）。
    bool empty_const_list_valid;
    // begin/end 数值裁剪（V2: clamp 到 int32 范围；V3: 原样返回）。
    int64_t (*clip_begin_end)(int64_t);
    // end 等于这些值时按"取到末尾"处理，参与 ends_mask 置位（V2 双哨兵 / V3 单哨兵）。
    int64_t end_sentinels[2];
    size_t end_sentinel_count;
};

// 读取 const tensor 的整型列表（int32/int64）。列表为空时返回 true 但不填充内容（V2 语义）。
inline bool GetConstListValue(const gert::Tensor* tensor, std::vector<int64_t>& valueList)
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

inline std::vector<int64_t> ShapeDimsToVector(const gert::Shape& shape)
{
    std::vector<int64_t> dims;
    const size_t dimNum = shape.GetDimNum();
    dims.reserve(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

inline void MakeUpShapeRange(const gert::Shape& shape, std::vector<std::pair<int64_t, int64_t>>& range)
{
    if (Ops::Base::IsUnknownRank(shape) || !range.empty()) {
        return;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        range.push_back(dim == SS_V2_V3_UNKNOWN_DIM ? std::pair<int64_t, int64_t>(0, -1) :
                                                      std::pair<int64_t, int64_t>(dim, dim));
    }
}

inline int64_t GetStridedSliceDim0(const gert::Shape* shape)
{
    if (shape == nullptr || shape->GetDimNum() == 0) {
        return 0;
    }
    return shape->GetDim(0);
}

inline void SetAllUnknownOutput(int64_t rank, gert::Shape* out_shape,
                                std::vector<std::pair<int64_t, int64_t>>* output_ranges)
{
    out_shape->SetDimNum(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        out_shape->SetDim(static_cast<size_t>(i), SS_V2_V3_UNKNOWN_DIM);
    }
    if (output_ranges != nullptr) {
        output_ranges->assign(static_cast<size_t>(rank), std::pair<int64_t, int64_t>(0, -1));
    }
}

inline void TryFillDefaultStrides(const StridedSliceInferInputs& info, bool& stride_valid,
                                  std::vector<int64_t>& stride_list)
{
    // If strides is absent or scalar, the default stride is 1.
    if (info.begin_len > 0 && !stride_valid && (!info.strides_present || info.strides_scalar)) {
        stride_valid = true;
        stride_list.assign(static_cast<size_t>(info.begin_len), 1);
    }
}

inline bool IsInfoInsufficient(const StridedSliceInferInputs& info, bool stride_valid,
                               const StridedSliceInferTraits& traits)
{
    if (info.begin_len == -1 || !stride_valid) {
        return true;
    }
    if (traits.insufficient_at_zero_len && info.begin_len == 0 && (!info.end_valid || !info.begin_valid)) {
        return true;
    }
    return false;
}

inline void BackfillInvalidLists(const StridedSliceInferInputs& info, const gert::Shape& shape_x,
                                 std::vector<int64_t>& begin_list, std::vector<int64_t>& end_list)
{
    // If begin is invalid, set begin with begin_len count of 0.
    if (!info.begin_valid) {
        begin_list.assign(static_cast<size_t>(info.begin_len), 0);
    }

    // If end is invalid, set end with begin_len count of the input shape dims.
    if (!info.end_valid) {
        const std::vector<int64_t> shape_dims = ShapeDimsToVector(shape_x);
        if (info.begin_len < static_cast<int64_t>(shape_dims.size())) {
            end_list.assign(shape_dims.begin(), shape_dims.begin() + static_cast<size_t>(info.begin_len));
        } else {
            end_list = shape_dims;
            if (!shape_dims.empty()) {
                for (int64_t i = static_cast<int64_t>(shape_dims.size()); i < info.begin_len; ++i) {
                    end_list.push_back(shape_dims.back());
                }
            }
        }
    }
}

inline int64_t NormalizeAxisIndex(int64_t axes_index, int64_t rank_num)
{
    if (axes_index < 0) {
        axes_index += rank_num;
    }
    if (axes_index >= rank_num) {
        axes_index = rank_num - 1;
    }
    if (axes_index < 0) {
        axes_index = 0;
    }
    return axes_index;
}

inline bool IsEndSentinel(int64_t end_value, const StridedSliceInferTraits& traits)
{
    for (size_t i = 0; i < traits.end_sentinel_count; ++i) {
        if (end_value == traits.end_sentinels[i]) {
            return true;
        }
    }
    return false;
}

// Expand the sparse begin/end/stride into full rank according to axes, and derive begin_mask/end_mask from axes.
inline void ExpandSparseByAxes(const StridedSliceInferInputs& info, const gert::Shape& shape_x,
                               const StridedSliceInferTraits& traits, std::vector<int64_t>& begin_list,
                               std::vector<int64_t>& end_list, std::vector<int64_t>& stride_list, uint64_t& axes_mask,
                               uint64_t& ends_mask)
{
    if (!info.axes_valid) {
        return;
    }
    const int64_t rank_num = static_cast<int64_t>(shape_x.GetDimNum());
    if (rank_num == 0) {
        // 0 维 x：无轴可展开，直接跳过（rank_num==0 由调用方 InferShapeAndRangeCore 报错兜底），
        // 避免 NormalizeAxisIndex 把任意下标归到 0 后对空 processed_* 容器 operator[] 写入。
        return;
    }
    std::vector<int64_t> processed_begin(static_cast<size_t>(rank_num), 0);
    std::vector<int64_t> processed_end = ShapeDimsToVector(shape_x);
    std::vector<int64_t> processed_stride(static_cast<size_t>(rank_num), 1);
    const size_t len = std::min({info.axes_list.size(), end_list.size(), begin_list.size(), stride_list.size()});
    for (size_t i = 0; i < len; ++i) {
        const int64_t axes_index = NormalizeAxisIndex(info.axes_list[i], rank_num);
        axes_mask = (static_cast<uint64_t>(1) << static_cast<uint64_t>(axes_index)) | axes_mask;
        processed_end[static_cast<size_t>(axes_index)] = traits.clip_begin_end(end_list[i]);
        processed_begin[static_cast<size_t>(axes_index)] = traits.clip_begin_end(begin_list[i]);
        processed_stride[static_cast<size_t>(axes_index)] = stride_list[i];
    }
    axes_mask = ~axes_mask;
    ends_mask = axes_mask;
    for (size_t i = 0; i < processed_end.size(); ++i) {
        if (IsEndSentinel(processed_end[i], traits)) {
            ends_mask = (static_cast<uint64_t>(1) << static_cast<uint64_t>(i)) | ends_mask;
        }
    }
    begin_list.assign(processed_begin.begin(), processed_begin.end());
    end_list.assign(processed_end.begin(), processed_end.end());
    stride_list.assign(processed_stride.begin(), processed_stride.end());
}

inline ge::graphStatus RunSharedInferShape(const char* op_name, StridedSliceParams& input_params,
                                           gert::Shape* out_shape,
                                           std::vector<std::pair<int64_t, int64_t>>* output_ranges)
{
    if (output_ranges != nullptr) {
        if (!InferShapeAndRange(input_params, out_shape, *output_ranges)) {
            OP_LOGE(op_name, "invoke InferShapeAndRange failed.");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    if (!InferShape(input_params, out_shape)) {
        OP_LOGE(op_name, "invoke InferShape failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferShapeAndRangeCore(const char* op_name, const gert::Shape& shape_x,
                                              const std::vector<std::pair<int64_t, int64_t>>& input_ranges,
                                              const StridedSliceInferInputs& info,
                                              const StridedSliceInferTraits& traits, gert::Shape* out_shape,
                                              std::vector<std::pair<int64_t, int64_t>>* output_ranges)
{
    const int64_t rank_num = static_cast<int64_t>(shape_x.GetDimNum());

    std::vector<int64_t> stride_list = info.stride_list;
    bool stride_valid = info.stride_valid;
    TryFillDefaultStrides(info, stride_valid, stride_list);

    // begin/end/strides are not all usable, set output shape to all -1 of the input rank.
    if (IsInfoInsufficient(info, stride_valid, traits)) {
        SetAllUnknownOutput(rank_num, out_shape, output_ranges);
        return GRAPH_SUCCESS;
    }

    std::vector<int64_t> begin_list = info.begin_list;
    std::vector<int64_t> end_list = info.end_list;
    BackfillInvalidLists(info, shape_x, begin_list, end_list);

    uint64_t axes_mask = 0;
    uint64_t ends_mask = 0;
    ExpandSparseByAxes(info, shape_x, traits, begin_list, end_list, stride_list, axes_mask, ends_mask);

    if (rank_num == 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(op_name, "x", "input x's dimnum must be in [1, 8]");
        return ge::GRAPH_FAILED;
    }

    if (end_list.size() != begin_list.size()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(op_name, "end, begin", "end shape and begin shape length must match");
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

    return RunSharedInferShape(op_name, input_params, out_shape, output_ranges);
}

// StridedSliceV2/V3 共用的 InferShape 入口：x/begin/end 必选输入校验、unknown-rank 出口、
// 输入信息收集与核心推导。算子名与版本差异由调用方传入。
inline ge::graphStatus InferShape4StridedSliceCommon(gert::InferShapeContext* context, const char* op_name,
                                                     const StridedSliceInferTraits& traits)
{
    constexpr size_t IDX_X = 0;
    constexpr size_t IDX_Y = 0;
    constexpr size_t IDX_BEGIN = 1;
    constexpr size_t IDX_END = 2;
    constexpr size_t IDX_AXES = 3;
    constexpr size_t IDX_STRIDES = 4;

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

    StridedSliceInferInputs info;
    info.begin_len = -1;
    info.begin_len = std::max(GetStridedSliceDim0(shape_begin), info.begin_len);
    info.begin_len = std::max(GetStridedSliceDim0(shape_end), info.begin_len);
    if (shape_axes != nullptr) {
        info.begin_len = std::max(GetStridedSliceDim0(shape_axes), info.begin_len);
    }
    if (shape_strides != nullptr) {
        info.begin_len = std::max(GetStridedSliceDim0(shape_strides), info.begin_len);
    }

    info.strides_present = (tensor_strides != nullptr);
    info.strides_scalar = (tensor_strides != nullptr && tensor_strides->GetOriginShape().GetDimNum() == 0);

    bool all_const = true;
    all_const = GetConstListValue(tensor_begin, info.begin_list) && all_const;
    all_const = GetConstListValue(tensor_end, info.end_list) && all_const;
    all_const = GetConstListValue(tensor_strides, info.stride_list) && all_const;
    all_const = GetConstListValue(tensor_axes, info.axes_list) && all_const;
    if (!all_const || !traits.empty_const_list_valid) {
        // 非全 const 时按列表非空降级 valid；V3 语义下空 const 列表同样降级。
        info.begin_valid = !info.begin_list.empty();
        info.end_valid = !info.end_list.empty();
        info.stride_valid = !info.stride_list.empty();
        info.axes_valid = !info.axes_list.empty();
    }

    std::vector<std::pair<int64_t, int64_t>> input_ranges;
    MakeUpShapeRange(*shape_x, input_ranges);

    const ge::graphStatus ret = InferShapeAndRangeCore(op_name, *shape_x, input_ranges, info, traits, shape_y, nullptr);
    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s.", Ops::Base::ToString(*shape_y).c_str());
    return ret;
}
} // namespace ops
#endif // STRIDED_SLICE_V2_V3_INFERSHAPE_COMMON_H_
