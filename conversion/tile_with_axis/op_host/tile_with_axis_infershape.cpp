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
 * \file tile_with_axis_infershape.cpp
 * \brief TileWithAxis 算子形状推导实现
 *
 * 设计依据: DESIGN.md v1.6
 * 输出 y 的 shape: y.shape[axis] = x.shape[axis] * tiles, 其余维度与 x 相同
 * rank=0 时 y.shape = [tiles]（人工裁决：支持标量输入）
 * 动态 shape: axis 维为未知维标记（-1/-2）时原样传播，不参与乘法
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "op_common/log/log.h"
#include "op_host/util/shape_util.h"

#include <cstdint>

using namespace ge;

namespace ops {

namespace {
constexpr int64_t ATTR_AXIS_IDX = 0;
constexpr int64_t ATTR_TILES_IDX = 1;
// 仓库约定（对齐 circular_pad_infershape.cpp）：-1 为未知维标记；
// -2 为未知 rank 标记，known-rank shape 中出现时同样原样传播（对齐 canndev legacy 基线）。
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_RANK_DIM = -2;

// axis 维输出值：未知维标记原样传播，其余乘 tiles 并做 int64 溢出检查
ge::graphStatus ScaleAxisDim(int64_t xDim, int64_t tiles, int64_t& yDim, const char* node_name)
{
    if (xDim == UNKNOWN_DIM || xDim == UNKNOWN_RANK_DIM) {
        yDim = xDim;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(xDim > 0 && xDim > INT64_MAX / tiles,
                OP_LOGE(node_name, "axis dim %ld * tiles %ld overflows int64", xDim, tiles), return ge::GRAPH_FAILED);
    yDim = xDim * tiles;
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShape4TileWithAxis(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // 获取属性 axis 和 tiles
    auto attrs = context->GetAttrs();
    const int64_t* axis_ptr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    const int64_t* tiles_ptr = attrs->GetAttrPointer<int64_t>(ATTR_TILES_IDX);
    // tiles 为必选属性（proto REQUIRED_ATTR / def REQUIRED / README）：缺失时拒绝，不静默回退
    OP_CHECK_IF(tiles_ptr == nullptr, OP_LOGE(context, "tiles is required but missing"), return ge::GRAPH_FAILED);
    int64_t tiles = *tiles_ptr;
    // axis 默认 1（与 TBE/AICPU 缺省一致）；属性缺失时兜底 1
    int64_t axis = (axis_ptr != nullptr) ? *axis_ptr : 1;

    OP_CHECK_IF(tiles <= 0, OP_LOGE(context, "tiles must be > 0, got %ld", tiles), return ge::GRAPH_FAILED);

    // 获取输入 rank
    int64_t rank = input_shape->GetDimNum();

    // unknown rank 处理: input rank 未知时 output 也设未知
    if (Ops::Base::IsUnknownRank(*input_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
        OP_LOGI(context->GetNodeName(), "TileWithAxis output shape: %s.", Ops::Base::ToString(*output_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    // 空 tensor 检查
    if (rank == 0) {
        // 标量输入: 输出 shape = [tiles]
        output_shape->SetDimNum(1);
        output_shape->SetDim(0, tiles);
        OP_LOGI(context->GetNodeName(), "TileWithAxis output shape: %s.", Ops::Base::ToString(*output_shape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    // 归一化 axis 到非负
    if (axis < 0) {
        axis = axis + rank;
    }

    OP_CHECK_IF(axis < 0 || axis >= rank, OP_LOGE(context, "axis is out of range"), return ge::GRAPH_FAILED);

    // 复制 input shape 到 output shape（axis 维: 未知维传播 + 溢出检查）
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; i++) {
        if (i == axis) {
            int64_t yDim = 0;
            OP_CHECK_IF(ScaleAxisDim(input_shape->GetDim(i), tiles, yDim, context->GetNodeName()) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "scale axis dim failed"), return ge::GRAPH_FAILED);
            output_shape->SetDim(i, yDim);
        } else {
            output_shape->SetDim(i, input_shape->GetDim(i));
        }
    }

    OP_LOGI(context->GetNodeName(), "TileWithAxis output shape: %s.", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

// 动态 shape 的 range 推导：输出 range = 输入 range，axis 维乘 tiles（未知维标记原样传播）
static ge::graphStatus InferShapeRange4TileWithAxis(gert::InferShapeRangeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis_ptr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    const int64_t* tiles_ptr = attrs->GetAttrPointer<int64_t>(ATTR_TILES_IDX);
    OP_CHECK_IF(tiles_ptr == nullptr || *tiles_ptr <= 0, OP_LOGE(context, "tiles is required and must be > 0"),
                return ge::GRAPH_FAILED);
    const int64_t tiles = *tiles_ptr;
    int64_t axis = (axis_ptr != nullptr) ? *axis_ptr : 1;

    const auto* xRange = context->GetInputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRange);
    auto* xRangeMin = xRange->GetMin();
    auto* xRangeMax = xRange->GetMax();
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMin);
    OP_CHECK_NULL_WITH_CONTEXT(context, xRangeMax);

    auto* yRange = context->GetOutputShapeRange(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRange);
    auto* yRangeMin = yRange->GetMin();
    auto* yRangeMax = yRange->GetMax();
    OP_CHECK_NULL_WITH_CONTEXT(context, yRangeMin);
    OP_CHECK_NULL_WITH_CONTEXT(context, yRangeMax);

    for (auto* yBound : {yRangeMin, yRangeMax}) {
        const auto* xBound = (yBound == yRangeMin) ? xRangeMin : xRangeMax;
        *yBound = *xBound;
        int64_t boundRank = static_cast<int64_t>(xBound->GetDimNum());
        if (boundRank == 0) {
            // 标量: 输出 range = [tiles, tiles]
            yBound->SetDimNum(1);
            yBound->SetDim(0, tiles);
            continue;
        }
        int64_t boundAxis = axis;
        if (boundAxis < 0) {
            boundAxis += boundRank;
        }
        OP_CHECK_IF(boundAxis < 0 || boundAxis >= boundRank, OP_LOGE(context, "axis is out of range"),
                    return ge::GRAPH_FAILED);
        int64_t yDim = 0;
        OP_CHECK_IF(ScaleAxisDim(xBound->GetDim(boundAxis), tiles, yDim, context->GetNodeName()) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "scale axis range dim failed"), return ge::GRAPH_FAILED);
        yBound->SetDim(boundAxis, yDim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4TileWithAxis(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TileWithAxis)
    .InferShape(InferShape4TileWithAxis)
    .InferDataType(InferDataType4TileWithAxis)
    .InferShapeRange(InferShapeRange4TileWithAxis);

} // namespace ops
