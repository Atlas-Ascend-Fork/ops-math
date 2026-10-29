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
 * \file feeds_repeat_infershape.cpp
 * \brief
 */
#include <cstdint>
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t IDX_0 = 0;
static constexpr size_t IDX_1 = 1;
static constexpr size_t MAX_FEEDS_RANK = 8; // FORMAT_ND 秩上限
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

inline bool IsUnknownRankForFeeds(const gert::Shape* checkShape)
{
    return checkShape->GetDimNum() == 1 && checkShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

inline ge::graphStatus SetUnknownRankForFeeds(gert::Shape* outShape)
{
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4FeedsRepeat(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4FeedsRepeat");
    const gert::Shape* feeds_shape = context->GetInputShape(IDX_0);
    if (feeds_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "feeds");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* repeat_times_shape = context->GetInputShape(IDX_1);
    if (repeat_times_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "feeds_repeat_times");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* y_shape = context->GetOutputShape(IDX_0);
    if (y_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    // 必选属性判空必须先于未知 Rank 早退：[-2] 传播路径同样不得跳过 REQUIRED 属性校验
    const int64_t* output_feeds_size = attrs->GetAttrPointer<int64_t>(0);
    if (output_feeds_size == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "output_feeds_size");
        return ge::GRAPH_FAILED;
    }
    // 取值域校验与 tiling 保持同一口径：负值在第 0 维写入前即拒绝，两入口一致
    if (*output_feeds_size < 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "output_feeds_size",
            ("the attribute output_feeds_size must be non-negative, but got " + std::to_string(*output_feeds_size))
                .c_str());
        return ge::GRAPH_FAILED;
    }

    if (IsUnknownRankForFeeds(feeds_shape)) { // feeds 输入为未知 Rank [-2] 时，输出传播为未知 Rank
        OP_LOGD(context, "Feeds is unknown rank, set output shape to unknown rank");
        return SetUnknownRankForFeeds(y_shape);
    }

    const size_t feeds_rank = feeds_shape->GetDimNum();
    OP_CHECK_IF(feeds_rank < 1 || feeds_rank > MAX_FEEDS_RANK,
                OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds rank must be in [1, %zu], but got %zu",
                        MAX_FEEDS_RANK, feeds_rank),
                return ge::GRAPH_FAILED);

    const int64_t feeds_dim0 = feeds_shape->GetDim(0);
    if (feeds_dim0 >= 0) { // feeds 第 0 维为动态维(-1)时无法做数值校验
        OP_CHECK_IF(
            feeds_dim0 < 1,
            OP_LOGE(context->GetNodeName(),
                    "[FeedsRepeat] feeds dim0 must be >= 1 (empty tensor is not supported), but got %ld", feeds_dim0),
            return ge::GRAPH_FAILED);
        // 元素总数是否已知须按逐维判定：GetShapeSize() 对含偶数个 -1 维的 shape 会返回正乘积
        // （如 [-1,-1] 得 1），不能作为"已知"依据；存在任一负维即视为构图期未知，校验顺延至执行阶段
        bool times_count_known = true;
        for (size_t i = 0; i < repeat_times_shape->GetDimNum(); ++i) {
            if (repeat_times_shape->GetDim(i) < 0) {
                times_count_known = false;
                break;
            }
        }
        if (times_count_known) {
            const int64_t times_numel = repeat_times_shape->GetShapeSize();
            OP_CHECK_IF(times_numel >= 0 && feeds_dim0 != times_numel,
                        OP_LOGE(context->GetNodeName(),
                                "[FeedsRepeat] element count of feeds_repeat_times (%ld) must equal feeds dim0 (%ld)",
                                times_numel, feeds_dim0),
                        return ge::GRAPH_FAILED);
        }
        int64_t elem_row = 1;
        bool row_shape_dynamic = false;
        for (size_t i = 1; i < feeds_rank; ++i) {
            const int64_t dim = feeds_shape->GetDim(i);
            if (dim < 0) {
                row_shape_dynamic = true;
                break;
            }
            // 逐维累乘前做溢出保护，避免有符号乘法回绕后绕过后续 elem_row 合法性判断
            OP_CHECK_IF(dim > 0 && elem_row > INT64_MAX / dim,
                        OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds element count per row is too large"),
                        return ge::GRAPH_FAILED);
            elem_row *= dim;
        }
        OP_CHECK_IF(!row_shape_dynamic && elem_row < 1,
                    OP_LOGE(context->GetNodeName(),
                            "[FeedsRepeat] feeds element count per row must be >= 1, but got %ld", elem_row),
                    return ge::GRAPH_FAILED);
    }

    *y_shape = *feeds_shape;
    y_shape->SetDim(0, *output_feeds_size);
    OP_LOGD(context, "End to do InferShape4FeedsRepeat");
    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s", Ops::Base::ToString(*y_shape).c_str());
    return GRAPH_SUCCESS;
}

// 不再注册 InferDataType：InferDataType 仅图场景使用，交付在 op_graph（feeds_repeat_graph_infer.cpp），
// 输出 dtype 恒等于 feeds dtype，由框架按 proto（feeds_repeat_proto.h）声明的 dtype 组合推导，
// 避免 IMPL_OP 与 IMPL_OP_INFERSHAPE 双注册因共享库加载顺序产生不确定的生效者。
IMPL_OP_INFERSHAPE(FeedsRepeat).InferShape(InferShape4FeedsRepeat);
} // namespace ops
