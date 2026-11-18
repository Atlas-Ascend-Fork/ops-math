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
 * \file strided_slice_v3_infershape.cpp
 * \brief StridedSliceV3 InferShape：共有逻辑见 strided_slice_v2_v3_infershape_common.h。
 */

#include "../../strided_slice_v2/op_host/strided_slice_v2_v3_infershape_common.h"

namespace ops {
namespace {
constexpr size_t IDX_BEGIN = 1;
constexpr size_t IDX_END = 2;
constexpr size_t IDX_AXES = 3;
constexpr size_t IDX_STRIDES = 4;
constexpr int64_t MAX_INT64 = static_cast<int64_t>((static_cast<uint64_t>(-1)) >> 1);
const char* const kOpNameStridedSliceV3 = "StridedSliceV3";

// V3：begin/end 为 int64 语义，无需裁剪。
int64_t ClipInt64Identity(int64_t value) { return value; }

// V3 版本语义：begin_len==0 不额外判信息不足；空 const 列表视为非法；仅 int64 end 哨兵。
const StridedSliceInferTraits kStridedSliceV3Traits = {
    false, // insufficient_at_zero_len
    false, // empty_const_list_valid
    ClipInt64Identity,
    {MAX_INT64, MAX_INT64}, // 仅第一个哨兵生效
    1UL,
};
} // namespace

static ge::graphStatus InferShape4StridedSliceV3(gert::InferShapeContext* context)
{
    return InferShape4StridedSliceCommon(context, kOpNameStridedSliceV3, kStridedSliceV3Traits);
}

IMPL_OP_INFERSHAPE(StridedSliceV3)
    .InferShape(InferShape4StridedSliceV3)
    .InputsDataDependency({IDX_BEGIN, IDX_END, IDX_AXES, IDX_STRIDES});
} // namespace ops
