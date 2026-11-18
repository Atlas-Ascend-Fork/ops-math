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
 * \file strided_slice_v2_infershape.cpp
 * \brief StridedSliceV2 InferShape：共有逻辑见 strided_slice_v2_v3_infershape_common.h。
 */

#include "strided_slice_v2_v3_infershape_common.h"

namespace ops {
namespace {
constexpr size_t IDX_BEGIN = 1;
constexpr size_t IDX_END = 2;
constexpr size_t IDX_AXES = 3;
constexpr size_t IDX_STRIDES = 4;
constexpr int64_t MAX_INT = static_cast<int64_t>((static_cast<uint32_t>(-1)) >> 1);
constexpr int64_t MIN_INT = ~MAX_INT;
constexpr int64_t MAX_INT64 = static_cast<int64_t>((static_cast<uint64_t>(-1)) >> 1);
const char* const kOpNameStridedSliceV2 = "StridedSliceV2";

// V2：begin/end 为 int32 语义，超出 int32 表示范围的值裁剪回边界。
int64_t ClipInt32Range(int64_t value) { return value > MAX_INT ? MAX_INT : (value < MIN_INT ? MIN_INT : value); }

// V2 版本语义：begin_len==0 且 begin/end 非法也判信息不足；空 const 列表视为有值；
// end 的 int32/int64 双哨兵都按"取到末尾"处理。
const StridedSliceInferTraits kStridedSliceV2Traits = {
    true, // insufficient_at_zero_len
    true, // empty_const_list_valid
    ClipInt32Range,
    {MAX_INT64, MAX_INT}, // 双哨兵
    2UL,
};
} // namespace

static ge::graphStatus InferShape4StridedSliceV2(gert::InferShapeContext* context)
{
    return InferShape4StridedSliceCommon(context, kOpNameStridedSliceV2, kStridedSliceV2Traits);
}

IMPL_OP_INFERSHAPE(StridedSliceV2)
    .InferShape(InferShape4StridedSliceV2)
    .InputsDataDependency({IDX_BEGIN, IDX_END, IDX_AXES, IDX_STRIDES});
} // namespace ops
