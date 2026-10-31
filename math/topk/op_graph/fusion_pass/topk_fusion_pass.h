/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_TOPK_FUSION_PASS_TOPK_FUSION_PASS_H_
#define OPS_MATH_TOPK_FUSION_PASS_TOPK_FUSION_PASS_H_

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
using namespace ge;
using namespace fusion;

/**
 * @brief TopK fusion pass (TopK/TopKV2 --> TopKV2D/TopKV2, with TransposeD when dim is not last)
 * @details Rebuilt from the legacy fe::TopKFusionPass in canndev.
 *
 *  Transformation rules (aligned with the legacy pass on non-Ascend310 platforms):
 *  1. k is a const int32 input:
 *       if dim == -1 : TopK --> TopKV2D (x, k, assist_seq), dim set to -1.
 *          (legacy pass converts to TopKD with k as attribute; TopKV2D is the
 *           dynamic-k equivalent which keeps k as an input and also takes the
 *           same assist_seq input, functionally equivalent on new platforms)
 *       if dim != -1 : TransposeD(perm) --> TopKV2D(dim=-1) --> TransposeD(perm) x2
 *  2. k is a non-const input:
 *       if dim == -1 : TopK --> TopKV2 (dynamic k)
 *       if dim != -1 : TransposeD(perm) --> TopKV2(dim=-1) --> TransposeD(perm) x2
 *  3. x dtype must be float16/float/bfloat16, sorted must be true, otherwise no fusion.
 */
class __attribute__((visibility("default"))) TopKFusionPass : public PatternFusionPass {
protected:
    std::vector<PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<MatchResult>& matchResult) override;

    GraphUniqPtr Replacement(const std::unique_ptr<MatchResult>& matchResult) override;
};

} // namespace ops

#endif // OPS_MATH_TOPK_FUSION_PASS_TOPK_FUSION_PASS_H_
