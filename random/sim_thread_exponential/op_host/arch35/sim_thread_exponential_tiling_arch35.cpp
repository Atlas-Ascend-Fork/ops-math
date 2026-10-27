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
 * \file sim_thread_exponential_tiling_arch35.cpp
 * \brief Tiling implementation for sim_thread_exponential operator
 */
#include "sim_thread_exponential_tiling_arch35.h"
#include "log/log.h"
#include <string>
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {
static constexpr uint16_t OUTPUT_IDX_SELF = 0;
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t OFFSET_MULTIPLE = 4;
static constexpr uint64_t TILING_KEY_FP16 = 1;
static constexpr uint64_t TILING_KEY_BF16 = 2;
static constexpr uint64_t TILING_KEY_FP32 = 3;
static constexpr uint32_t NUM_FOUR = 4;

// 空指针检查仅写日志不注册错误码，GetNodeName为空时回退到字面算子名（参照官方整改基线）
static const char* OpNameOf(gert::TilingContext* ctx)
{
    const char* name = ctx->GetNodeName();
    return name == nullptr ? "SimThreadExponential" : name;
}

OpTilingConfig SimThreadExponentialTilingSimt::BuildOpConfig()
{
    OpTilingConfig config;

    config.outputCheckRules = {
        {OUTPUT_IDX_SELF, {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}, -1, {}, nullptr}},
    };

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto attrs = ctx->GetAttrs();
        if (attrs == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "attrs is nullptr");
            return ge::GRAPH_FAILED;
        }
        const auto* countAttr = attrs->GetAttrPointer<int64_t>(0);
        if (countAttr == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "countAttr is nullptr");
            return ge::GRAPH_FAILED;
        }
        size = *countAttr;

        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_SELF);
        if (outputShape == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "outputShape is nullptr");
            return ge::GRAPH_FAILED;
        }
        auto outTensor = outputShape->GetStorageShape();
        int64_t outputShapeSize = outTensor.GetShapeSize();
        if (size != outputShapeSize) {
            std::string valueStr = std::to_string(size) + " and " + std::to_string(outputShapeSize);
            std::string reasonMsg = "count must be equal to output shape size";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "attr count", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    };

    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        auto attrs = ctx->GetAttrs();
        if (attrs == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "attrs is nullptr");
            return ge::GRAPH_FAILED;
        }
        const auto* seedAttr = attrs->GetAttrPointer<int64_t>(2);
        if (seedAttr == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "seedAttr is nullptr");
            return ge::GRAPH_FAILED;
        }
        seed = *seedAttr;
        const auto* offsetAttr = attrs->GetAttrPointer<int64_t>(3);
        if (offsetAttr == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "offsetAttr is nullptr");
            return ge::GRAPH_FAILED;
        }
        offset = *offsetAttr;
        if (offset % OFFSET_MULTIPLE != 0) {
            std::string valueStr = std::to_string(offset);
            std::string reasonMsg = "offset must be a multiple of 4";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "input offset", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    config.attrCheckRules = {
        {1,
         [](gert::TilingContext* ctx) -> bool {
             auto attrs = ctx->GetAttrs();
             if (attrs == nullptr)
                 return false;
             const auto* lambdAttr = attrs->GetAttrPointer<float>(1);
             if (lambdAttr == nullptr)
                 return false;
             if (*lambdAttr <= 0.0f) {
                 std::string valueStr = std::to_string(*lambdAttr);
                 std::string reasonMsg = "lambd must be greater than 0";
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "attr lambd", valueStr.c_str(),
                                                       reasonMsg.c_str());
                 return false;
             }
             return true;
         }},
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = false;
    config.unrollFactor = NUM_FOUR;
    config.enableSplitBlocks = true;
    return config;
}

ge::graphStatus SimThreadExponentialTilingSimt::UniqueProcess()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context_), "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }

    const auto* lambdAttr = attrs->GetAttrPointer<float>(1);
    if (lambdAttr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context_), "lambdAttr is nullptr");
        return ge::GRAPH_FAILED;
    }
    simtTilingData_.prob = *lambdAttr;

    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_SELF);
    if (outputDesc == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context_), "outputDesc is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto outputDtype = outputDesc->GetDataType();
    if (outputDtype == ge::DT_FLOAT16) {
        tilingKey_ = TILING_KEY_FP16;
    } else if (outputDtype == ge::DT_BF16) {
        tilingKey_ = TILING_KEY_BF16;
    } else {
        tilingKey_ = TILING_KEY_FP32;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace optiling {

static ge::graphStatus Tiling4SimThreadExponentialArch35(gert::TilingContext* context)
{
    SimThreadExponentialTilingSimt tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4SimThreadExponentialArch35(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "SimThreadExponential");
}

IMPL_OP_OPTILING(SimThreadExponential)
    .Tiling(Tiling4SimThreadExponentialArch35)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4SimThreadExponentialArch35)
    .TilingInputsDataDependency({});
} // namespace optiling
