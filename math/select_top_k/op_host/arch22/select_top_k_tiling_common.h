#ifndef SELECT_TOP_K_TILING_COMMON_H_
#define SELECT_TOP_K_TILING_COMMON_H_
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "select_top_k_plan.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include <cstring>

namespace optiling {
namespace select_top_k_detail {
// 读取可选边界张量的描述，校验广播长度并设置区间标记。
inline bool ReadBounds(gert::TilingContext* context, size_t input, uint64_t rows, uint32_t& flags)
{
    auto shape = context->GetOptionalInputShape(input);
    if (shape == nullptr) {
        return true;
    }
    auto desc = context->GetOptionalInputDesc(input);
    if (desc == nullptr || desc->GetDataType() != ge::DT_INT32) {
        return false;
    }
    const int64_t count = shape->GetStorageShape().GetShapeSize();
    if (count != 1 && (count < 0 || static_cast<uint64_t>(count) != rows)) {
        return false;
    }
    flags |= input == SelectTopK::INPUT_BEGIN ? SelectTopK::HAS_BEGIN : SelectTopK::HAS_END;
    if (count == 1) {
        flags |= input == SelectTopK::INPUT_BEGIN ? SelectTopK::BROADCAST_BEGIN : SelectTopK::BROADCAST_END;
    }
    return true;
}
} // namespace select_top_k_detail

// 读取框架输入与平台资源，生成计划并提交 tiling、核数和工作区。
inline ge::graphStatus TileSelectTopKPlan(gert::TilingContext* context)
{
    auto input = context->GetInputShape(0);
    auto desc = context->GetInputDesc(0);
    auto indexDesc = context->GetOutputDesc(1);
    auto attrs = context->GetAttrs();
    auto platformInfo = context->GetPlatformInfo();
    if (!input || !desc || !indexDesc || !attrs || !platformInfo) {
        return ge::GRAPH_FAILED;
    }
    const auto& shape = input->GetStorageShape();
    const auto* k = attrs->GetInt(0);
    const auto* largest = attrs->GetBool(1);
    const auto* order = attrs->GetInt(SelectTopK::ATTR_ORDER);
    const auto* indicesType = attrs->GetInt(SelectTopK::ATTR_INDICES_TYPE);
    const auto* offset = attrs->GetInt(SelectTopK::ATTR_INDICES_OFFSET);
    const auto* mode = attrs->GetInt(SelectTopK::ATTR_OUTPUT_MODE);
    const auto* valueFill = attrs->GetFloat(SelectTopK::ATTR_VALUE_FILL);
    const auto* indexFill = attrs->GetInt(SelectTopK::ATTR_INDEX_FILL);
    const auto dtype = desc->GetDataType();
    if (shape.GetDimNum() != SelectTopK::INPUT_RANK || shape.GetDim(0) < 0 || shape.GetDim(1) < 0 ||
        shape.GetDim(1) > INT32_MAX || shape.GetDim(SelectTopK::INNER_LAYOUT_DIM) < 1 ||
        shape.GetDim(SelectTopK::INNER_LAYOUT_DIM) > SelectTopK::MAX_LAYOUT_INNER || !k || !largest || !order ||
        !indicesType || !offset || !mode || !valueFill || !indexFill || *k < 0 || *k > INT32_MAX ||
        (*order != SelectTopK::ORDER_VALUE && *order != SelectTopK::ORDER_NONE) || *mode < 0 ||
        *mode > SelectTopK::OUTPUT_BOTH || (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
        (*indicesType != ge::DT_INT32 && *indicesType != ge::DT_INT64) || indexDesc->GetDataType() != *indicesType) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t inner = shape.GetDim(SelectTopK::INNER_LAYOUT_DIM);
    const uint64_t outer = shape.GetDim(0), axis = shape.GetDim(1);
    if (outer > static_cast<uint64_t>(INT64_MAX) / inner) {
        return ge::GRAPH_FAILED;
    }
    // 三维形状保留原轴后跨度；内核内部仍按 rows 行计算，不增加 GM 搬运。
    const uint64_t rows = outer * inner;
    const int64_t maxIndex = axis == 0 ? 0 : static_cast<int64_t>(axis - 1);
    const int64_t upper = *indicesType == ge::DT_INT32 ? INT32_MAX : INT64_MAX;
    const int64_t lower = *indicesType == ge::DT_INT32 ? INT32_MIN : INT64_MIN;
    if (*offset < lower || *offset > upper - maxIndex || *indexFill < lower || *indexFill > upper) {
        return ge::GRAPH_FAILED;
    }
    SelectTopKTilingData data;
    data.outputK = *k;
    data.indicesOffset = *offset;
    data.indexFill = *indexFill;
    data.valueFill = *valueFill;
    if (!select_top_k_detail::ReadBounds(context, SelectTopK::INPUT_BEGIN, rows, data.flags) ||
        !select_top_k_detail::ReadBounds(context, SelectTopK::INPUT_END, rows, data.flags)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t bytes = dtype == ge::DT_FLOAT ? SelectTopKBase::VALUE32_BYTES : SelectTopKBase::VALUE16_BYTES;
    if (!SelectTopK::BuildPlan(
            rows, axis, *k, bytes, *largest, *order == SelectTopK::ORDER_VALUE,
            *indicesType == ge::DT_INT64 ? SelectTopKBase::INDEX64_BYTES : SelectTopKBase::INDEX32_BYTES, *offset,
            *mode, data.flags, *valueFill, *indexFill, ub, platform.GetCoreNumAiv(), data, inner, inner)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t key = SelectTopK::TilingKey(data.base.kernel);
    if (key == UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    auto raw = context->GetRawTilingData();
    auto workspace = context->GetWorkspaceSizes(1);
    const uint64_t system = platform.GetLibApiWorkSpaceSize();
    if (!raw || !workspace || raw->GetCapacity() < sizeof(data) || data.workspaceBytes > UINT64_MAX - system) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = data.workspaceBytes + system;
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(data.base.cores);
    context->SetTilingKey(key);
    context->SetScheduleMode(1);
    OP_LOGI(context->GetNodeName(),
            "SelectTopK A2: rows=%lu N=%lu k=%lu coreK=%lu key=%u flags=%u offset=%ld inputInner=%u inputBlock=%u "
            "outputInner=%u outputBlock=%u tile=%u shards=%u cores=%u "
            "sortPattern=%u rowBlock=%lu rowBatch=%u shardItems=%lu localLeaves=%lu leafStride=%lu mergeWindow=%u "
            "template=%u partition=%u candidateMerge=%u mergeStorage=%u",
            rows, axis, data.outputK, data.base.k, key, data.flags, data.indicesOffset, data.inputInner,
            data.inputBlock, data.outputInner, data.outputBlock, data.base.tile, data.base.shardCount, data.base.cores,
            static_cast<uint32_t>(data.schedule.pattern), data.schedule.rowsPerCore, data.base.batchRows,
            data.schedule.shardElements, data.schedule.leavesPerShard, data.schedule.leafStride, data.base.mergeWindow,
            static_cast<uint32_t>(data.base.kernel), static_cast<uint32_t>(data.base.partition),
            static_cast<uint32_t>(data.base.candidateMerge), static_cast<uint32_t>(data.schedule.storage));
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
#endif
