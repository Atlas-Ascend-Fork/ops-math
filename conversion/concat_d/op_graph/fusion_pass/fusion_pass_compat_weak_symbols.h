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
 * \file fusion_pass_compat_weak_symbols.h
 * \brief Weak-symbol declarations for GE APIs added after 8.5.0, for fusion-pass
 *        runtime compatibility (see the fusion-pass-compat guideline).
 *
 * DO NOT include the original "ge/fusion/graph_rewriter.h" together with this
 * header: both define class SubgraphRewriter and would collide (ODR).
 *
 * - The two plain Replace overloads exist since GE 8.5.0: declared normally
 *   (strong reference, present in every runtime).
 * - The CustomPassContext overloads exist since GE 9.1.0 only: declared weak so
 *   that a pass .so linked with -z now still loads on older GE runtimes (weak
 *   references resolve to null instead of failing dlopen). Callers must guard
 *   the call with a runtime version check (aclsysGetVersionNum("ge_compiler"))
 *   and fall back to a plain overload.
 */

#ifndef OPS_MATH_CONVERSION_CONCAT_D_OP_GRAPH_FUSION_PASS_FUSION_PASS_COMPAT_WEAK_SYMBOLS_H_
#define OPS_MATH_CONVERSION_CONCAT_D_OP_GRAPH_FUSION_PASS_FUSION_PASS_COMPAT_WEAK_SYMBOLS_H_

#include "graph/graph.h"
#include "ge/fusion/subgraph_boundary.h"
#include "register/register_custom_pass.h"

namespace ge {
namespace fusion {

class SubgraphRewriter {
public:
    // Since 8.5.0: plain replacement, no built-in fusibility check / reporting.
    static Status Replace(const SubgraphBoundary& subgraph, const Graph& replacement);
    static Status Replace(const SubgraphBoundary& subgraph, Graph&& replacement);
    // Since 9.1.0: performs the built-in fusibility check and fusion-result
    // reporting using the pass context. Weak reference on purpose.
    __attribute__((weak)) static Status Replace(const SubgraphBoundary& subgraph, Graph&& replacement,
                                                CustomPassContext& ctx);
    __attribute__((weak)) static Status Replace(const SubgraphBoundary& subgraph, const Graph& replacement,
                                                CustomPassContext& ctx);
};

} // namespace fusion
} // namespace ge

#endif // OPS_MATH_CONVERSION_CONCAT_D_OP_GRAPH_FUSION_PASS_FUSION_PASS_COMPAT_WEAK_SYMBOLS_H_
