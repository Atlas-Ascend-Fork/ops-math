/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_SELECT_PLAN_H_
#define SELECT_TOP_K_SELECT_PLAN_H_
#include "select_top_k_sort_plan.h"

namespace SelectTopK {
// 仅复制语义字段，为每种选择方案独立计算资源，避免残留基础归并工作区。
inline SelectTopKBaseTilingData SelectionInput(const SelectTopKBaseTilingData& source)
{
    SelectTopKBaseTilingData result{};
    result.rows = source.rows;
    result.axis = source.axis;
    result.k = source.k;
    result.dtypeBytes = source.dtypeBytes;
    result.largest = source.largest;
    result.sorted = source.sorted;
    return result;
}

// 在长轴小 K 上尝试扫描、过滤、压缩；资源不适用时保留已建立的 基础计划。
inline void TryFilterSelectPlan(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.axis < SelectTopKBase::Tuning::FILTER_MIN_AXIS ||
        plan.k < SelectTopKBase::Tuning::FILTER_MIN_K || plan.k > SelectTopKBase::Tuning::FILTER_MAX_K)
        return;
    const PlanResources resources{ub - UB_RESERVE, cores, SelectTopKBase::Tuning::FILTER_MIN_AXIS,
                                  SelectTopKBase::Tuning::MAX_LEAF_BLOCK, true};
    // 各候选由资源规划器直接给出模板/分片/候选策略，成功后整体提交。
    using Planner = PlanAttempt (*)(SelectTopKBaseTilingData&, const PlanResources&);
    const Planner planners[] = {TryWideFilter, TryRowFilter, TryShardedFilter};
    for (auto planner : planners) {
        auto candidate = SelectionInput(plan);
        if (planner(candidate, resources) == PlanAttempt::SUCCESS) {
            plan = candidate;
            schedule = {};
            return;
        }
    }
}
} // namespace SelectTopK
#endif
