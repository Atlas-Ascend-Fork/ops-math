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
 * \file masked_fill_range_graph_infer.cpp
 * \brief MaskedFillRange 算子 GE graph 侧 dtype 推导
 *
 * y 与 x 同形同型，InferDataType 为 same-as-input 推导，直接复用本仓公共函数
 * InferDataTypeOutputSameAsInput（common/inc/op_graph/graph_infer_comm_funcs.h）。
 * 按 conversion 仓 op_host/op_graph 分工（matrix_diag/masked_fill 等）：
 * shape 推导在 op_host/masked_fill_range_infershape.cpp 的 IMPL_OP_INFERSHAPE
 * 注册，本文件仅注册 InferDataType，避免双注册。
 */
#include "register/op_impl_registry.h"
#include "op_graph/graph_infer_comm_funcs.h"

using namespace ge;

namespace ops {
IMPL_OP(MaskedFillRange).InferDataType(InferDataTypeOutputSameAsInput);
} // namespace ops
