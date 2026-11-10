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
 * \file diag_graph_infer.cpp
 * \brief Diag 的 op_graph 推导注册：输出 dtype 与输入一致。
 *
 * Diag 迁移到 op_host 侧 InferShape/InferShapeRange 后（IMPL_OP_INFERSHAPE），
 * GE 2.0 推导链在 InferShapeAndType 阶段要求算子注册 InferDataType 回调，
 * 缺失会在图编译期报 E19999（Cannot find custom infer_datatype func）。
 * 参照 Roll/BiasAddGrad 的 graph_infer 注册同款推导。
 */
#include "register/op_impl_registry.h"
#include "op_graph/graph_infer_comm_funcs.h"

using namespace ge;
namespace ops {
IMPL_OP(Diag).InferDataType(InferDataTypeOutputSameAsInput);
} // namespace ops
