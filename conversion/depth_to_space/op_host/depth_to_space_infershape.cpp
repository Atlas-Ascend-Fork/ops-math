/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file depth_to_space_infershape.cpp
 * \brief
 */
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
namespace {
constexpr size_t SUPPORTED_DIM_NUM = 4;
constexpr int64_t MIN_BLOCK_SIZE = 2;
constexpr size_t ATTR_INDEX_BLOCK_SIZE = 0;
constexpr size_t ATTR_INDEX_MODE = 1;
constexpr size_t ATTR_INDEX_DATA_FORMAT = 2;
} // namespace

static ge::graphStatus InferShape4DepthToSpace(gert::InferShapeContext* context)
{
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    // -2 透传：输入维数未知，输出维数未知
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS;
    }

    // verify 并入：输入 rank >= 4
    OP_CHECK_IF(
        x_shape->GetDimNum() < SUPPORTED_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "The dim num of input x must be >= 4, but got %zu.", x_shape->GetDimNum()),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // verify 并入：block_size >= 2
    auto block_size_ptr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, block_size_ptr);
    const int64_t block_size = *block_size_ptr;
    OP_CHECK_IF(block_size < MIN_BLOCK_SIZE,
                OP_LOGE(context->GetNodeName(), "block_size must be >= 2, but got %ld.", block_size),
                return ge::GRAPH_FAILED);

    // verify 并入：mode 仅支持 DCR / CRD
    auto mode = attrs->GetAttrPointer<char>(ATTR_INDEX_MODE);
    OP_CHECK_IF(
        mode == nullptr || (std::string(mode) != "DCR" && std::string(mode) != "CRD"),
        OP_LOGE(context->GetNodeName(), "mode must be DCR or CRD, but got %s.", (mode == nullptr ? "nullptr" : mode)),
        return ge::GRAPH_FAILED);

    // verify 并入：data_format 仅支持 NCHW / NHWC / NC1HWC0
    auto data_format = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    OP_CHECK_IF(data_format == nullptr || (std::string(data_format) != "NCHW" && std::string(data_format) != "NHWC" &&
                                           std::string(data_format) != "NC1HWC0"),
                OP_LOGE(context->GetNodeName(), "data_format must be NCHW, NHWC or NC1HWC0, but got %s.",
                        (data_format == nullptr ? "nullptr" : data_format)),
                return ge::GRAPH_FAILED);

    const bool is_nchw = (std::string(data_format) == "NCHW");
    const int64_t block_area = block_size * block_size;

    // verify 并入：C 维可被 block_size * block_size 整除（仅 NCHW/NHWC 校验，-1 维跳过）
    if (std::string(data_format) != "NC1HWC0") {
        const size_t c_dim_idx = is_nchw ? 1U : 3U;
        const int64_t c_dim = x_shape->GetDim(c_dim_idx);
        OP_CHECK_IF(c_dim != -1 && (c_dim % block_area) != 0,
                    OP_LOGE(context->GetNodeName(),
                            "Depth size must be divisible by block_size * block_size, but got depth[%ld], "
                            "block_size[%ld], data_format[%s].",
                            c_dim, block_size, data_format),
                    return ge::GRAPH_FAILED);
    }

    // shape 计算：-1 维透传，其余维照算；输出固定 4 维（对齐 RT1.0 行为）
    y_shape->SetDimNum(SUPPORTED_DIM_NUM);
    y_shape->SetDim(0, x_shape->GetDim(0));
    if (is_nchw) {
        // NCHW: out = (N, C / b^2, H * b, W * b)
        y_shape->SetDim(1, x_shape->GetDim(1) == -1 ? -1 : x_shape->GetDim(1) / block_area);
        y_shape->SetDim(2, x_shape->GetDim(2) == -1 ? -1 : x_shape->GetDim(2) * block_size);
        y_shape->SetDim(3, x_shape->GetDim(3) == -1 ? -1 : x_shape->GetDim(3) * block_size);
    } else {
        // NHWC 及其他: out = (N, H * b, W * b, C / b^2)
        y_shape->SetDim(1, x_shape->GetDim(1) == -1 ? -1 : x_shape->GetDim(1) * block_size);
        y_shape->SetDim(2, x_shape->GetDim(2) == -1 ? -1 : x_shape->GetDim(2) * block_size);
        y_shape->SetDim(3, x_shape->GetDim(3) == -1 ? -1 : x_shape->GetDim(3) / block_area);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DepthToSpace).InferShape(InferShape4DepthToSpace);
} // namespace ops
