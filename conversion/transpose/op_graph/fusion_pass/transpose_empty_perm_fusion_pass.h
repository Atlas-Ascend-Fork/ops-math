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
 * \file transpose_empty_perm_fusion_pass.h
 * \brief Transpose empty perm fusion pass (fill the ONNX default reverse perm into the perm Const input)
 *
 * ONNX Transpose allows an absent `perm` attribute, whose default semantics is the reverse of the
 * input ranks, i.e. perm = [n-1, n-2, ..., 0]. The rank is unknown while parsing the ONNX node, so
 * the onnx plugin keeps an empty perm Const; this pass runs after InferShape (rank known) and
 * materializes the reverse perm into that Const.
 */
#ifndef OPS_MATH_CONVERSION_TRANSPOSE_OP_GRAPH_FUSION_PASS_TRANSPOSE_EMPTY_PERM_FUSION_PASS_H_
#define OPS_MATH_CONVERSION_TRANSPOSE_OP_GRAPH_FUSION_PASS_TRANSPOSE_EMPTY_PERM_FUSION_PASS_H_

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {
using namespace ge;
using namespace ge::fusion;

/**
 * @brief Fill the ONNX default reverse perm for Transpose nodes whose perm Const is empty.
 *
 * Graph route (FusionBasePass): traverse the whole graph, locate every Transpose whose perm input
 * comes from an empty Const, and rewrite that Const's value to the reverse permutation derived from
 * the (post-InferShape) rank of the data input. No topology change is performed.
 */
class __attribute__((visibility("default"))) TransposeEmptyPermFusionPass : public FusionBasePass {
public:
    Status Run(GraphPtr& graph, CustomPassContext& passContext) override;
};

} // namespace ops

#endif // OPS_MATH_CONVERSION_TRANSPOSE_OP_GRAPH_FUSION_PASS_TRANSPOSE_EMPTY_PERM_FUSION_PASS_H_
