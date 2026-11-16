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
 * \file concat_to_concatd_fusion_pass.h
 * \brief Concat/ConcatV2/ConcatV2D --> ConcatD fusion pass (regbase platforms only)
 */
#ifndef OPS_MATH_CONVERSION_CONCAT_D_OP_GRAPH_FUSION_PASS_CONCAT_TO_CONCATD_FUSION_PASS_H_
#define OPS_MATH_CONVERSION_CONCAT_D_OP_GRAPH_FUSION_PASS_CONCAT_TO_CONCATD_FUSION_PASS_H_

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {
using namespace ge;
using namespace ge::fusion;

class __attribute__((visibility("default"))) ConcatToConcatDFusionPass : public FusionBasePass {
public:
    Status Run(GraphPtr& graph, CustomPassContext& passContext) override;

protected:
    // Type filter for rewrite targets. Virtual so the early-stage twin can narrow its scope
    // (see ConcatToConcatDPrepareFusionPass); the main pass keeps the full concat family.
    virtual bool IsTargetType(const std::string& opType) const;
};

// Early-stage twin of ConcatToConcatDFusionPass, registered at kBeforeInferShape (the new-framework
// counterpart of the legacy BUILT_IN_PREPARE_GRAPH_PASS timing: before shape inference and before
// the main fusion round, so downstream infershape/const-folding already sees ConcatD).
// The new-framework PassRegistry deduplicates registrations by name (= class name), and the main
// registration must keep the name "ConcatToConcatDFusionPass" to match the built-in pass for the
// kCompatibleInherited takeover, so the early slot needs this separately named subclass that
// reuses the same Run implementation.
// Scope narrowing: the early slot rewrites ONLY the classic Concat (axis as input 0) — the type
// whose axis must be const-to-attr'd before infershape; ConcatV2/ConcatV2D are left for the
// main-round takeover pass, keeping the early intervention minimal.
class __attribute__((visibility("default"))) ConcatToConcatDPrepareFusionPass : public ConcatToConcatDFusionPass {
protected:
    bool IsTargetType(const std::string& opType) const override;
};

} // namespace ops

#endif // OPS_MATH_CONVERSION_CONCAT_D_OP_GRAPH_FUSION_PASS_CONCAT_TO_CONCATD_FUSION_PASS_H_
