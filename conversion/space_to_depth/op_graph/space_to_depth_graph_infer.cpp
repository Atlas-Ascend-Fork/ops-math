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
 * \file space_to_depth_graph_infer.cpp
 * \brief SpaceToDepth operator graph infer resource (InferDataType only).
 *        InferShape is implemented in op_host/space_to_depth_infershape.cpp
 */

#include "register/op_impl_registry.h"
#include "op_graph/graph_infer_comm_funcs.h"

using namespace ge;
namespace ops {

IMPL_OP(SpaceToDepth).InferDataType(InferDataTypeOutputSameAsInput);

} // namespace ops
