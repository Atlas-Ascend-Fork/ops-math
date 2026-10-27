/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file minimum_grad_graph_infer.cpp
 * \brief MinimumGrad 算子图模式 InferDataType 推导实现
 */

#include "register/op_impl_registry.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/types.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
namespace {

constexpr size_t INPUT_NUM = 3U;

// 4-dtype 白名单成员判定（与 _def.cpp / proto.h 声明一致；ge::DataType 取值稀疏，
// 无法用区间检查，只能做成员包含判断）。
inline bool IsSupportedDtype(const ge::DataType dt)
{
    constexpr ge::DataType kWhitelist[] = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32};
    return std::find(std::begin(kWhitelist), std::end(kWhitelist), dt) != std::end(kWhitelist);
}

const char* const DTYPE_WHITELIST_TEXT = "float16/float32/bfloat16/int32";

} // namespace

/**
 * InferDataTypeForMinimumGrad: GE dtype 推导回调。
 *
 * 校验链：空指针防御 -> grads(0) dtype 白名单门禁 -> x1/x2 与 grads dtype
 * 一致性（无类型提升）-> 写 y1/y2 输出 dtype（均传播 grads dtype）。
 */
ge::graphStatus InferDataTypeForMinimumGrad(gert::InferDataTypeContext* context)
{
    // 空指针防御（null_input）—— 宏覆盖 context 自身。
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "InferDataType: enter");

    // 实例 0（grads）dtype：白名单门禁 + 传播源。
    const ge::DataType dtype0 = context->GetInputDataType(0);
    if (!IsSupportedDtype(dtype0)) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, "grads", std::to_string(static_cast<int>(dtype0)),
                                         DTYPE_WHITELIST_TEXT);
        return ge::GRAPH_FAILED; // dtype_not_supported
    }
    // 跨输入一致性：x1/x2 必须与 grads 同 dtype（无提升），不一致在写输出前拒绝。
    for (size_t i = 1U; i < INPUT_NUM; ++i) {
        const ge::DataType dt = context->GetInputDataType(i);
        if (dt != dtype0) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(nodeName, i == 1U ? "x1" : "x2", std::to_string(static_cast<int>(dt)),
                                             "identical to grads (no promotion)");
            return ge::GRAPH_FAILED; // dtype_not_supported
        }
    }

    // 全部校验通过后再写 y1/y2 输出 dtype；检查 SetOutputDataType 返回值
    // （写失败不得上报成功）。
    if (context->SetOutputDataType(0, dtype0) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, dtype0) != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "InferDataType: SetOutputDataType(0/1, %d) failed!", static_cast<int>(dtype0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MinimumGrad).InferDataType(InferDataTypeForMinimumGrad);

} // namespace ops
