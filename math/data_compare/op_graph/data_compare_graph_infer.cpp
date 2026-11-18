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
 * \file data_compare_graph_infer.cpp
 * \brief DataCompare graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
namespace ops {
static constexpr int64_t IDX_OUT_0 = 0;

ge::graphStatus InferDataTypeDataCompare(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeDataCompare");
    // 输出 num 为比较结果统计量，proto 定义恒为 DT_FLOAT（OUTPUT(num, TensorType({DT_FLOAT}))），
    // 与输入 dtype 无关，故此处无条件设置 DT_FLOAT。
    context->SetOutputDataType(IDX_OUT_0, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeDataCompare");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(DataCompare).InferDataType(InferDataTypeDataCompare);
} // namespace ops
