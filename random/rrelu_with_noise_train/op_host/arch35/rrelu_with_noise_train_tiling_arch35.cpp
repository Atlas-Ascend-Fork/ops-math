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
 * \file rrelu_with_noise_train_tiling_arch35.cpp
 * \brief Tiling implementation for rrelu_with_noise_train operator
 */
#include "rrelu_with_noise_train_tiling_arch35.h"
#include "log/log.h"
#include <string>
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_X = 0;
static constexpr uint16_t INPUT_IDX_SEED = 1;
static constexpr uint16_t INPUT_IDX_OFFSET = 2;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t OUTPUT_IDX_NOISE = 1;
static constexpr int64_t DCACHE_SIZE = 32 * 1024;
static constexpr uint32_t NUM_FOUR = 4;
static constexpr uint16_t ATTR_IDX_LOWER = 0;
static constexpr uint16_t ATTR_IDX_UPPER = 1;

static const char* OpNameOf(gert::TilingContext* ctx)
{
    const char* name = ctx->GetNodeName();
    return name == nullptr ? "RreluWithNoiseTrain" : name;
}

OpTilingConfig RreluWithNoiseTrainTilingSimt::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        {INPUT_IDX_X, {{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}},
    };

    config.outputCheckRules = {
        {OUTPUT_IDX_Y, {{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}, -1, {}, nullptr}},
        {OUTPUT_IDX_NOISE, {{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}, -1, {}, nullptr}},
    };

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_Y);
        if (outputShape == nullptr) {
            OP_LOGE_WITHOUT_REPORT(OpNameOf(ctx), "outputShape is nullptr");
            return ge::GRAPH_FAILED;
        }
        size = outputShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    config.getSeedAndOffset = [](gert::TilingContext* /*ctx*/, int64_t& seed, int64_t& offset) {
        seed = 0;
        offset = 0;
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = false;
    config.unrollFactor = NUM_FOUR;
    config.enableSplitBlocks = true;
    return config;
}

ge::graphStatus RreluWithNoiseTrainTilingSimt::UniqueProcess()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context_), "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }

    // lower/upper 填入统一 TilingData 既有扩展槽 fromFp32/toFp32
    const auto* lowerAttr = attrs->GetAttrPointer<float>(ATTR_IDX_LOWER);
    if (lowerAttr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context_), "lowerAttr is nullptr");
        return ge::GRAPH_FAILED;
    }
    const auto* upperAttr = attrs->GetAttrPointer<float>(ATTR_IDX_UPPER);
    if (upperAttr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context_), "upperAttr is nullptr");
        return ge::GRAPH_FAILED;
    }
    simtTilingData_.fromFp32 = *lowerAttr;
    simtTilingData_.toFp32 = *upperAttr;

    // y/noise 同形防御（图模式直调 L0 时不经过 L2 校验）
    auto xShape = context_->GetInputShape(INPUT_IDX_X);
    auto yShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    auto noiseShape = context_->GetOutputShape(OUTPUT_IDX_NOISE);
    if (xShape == nullptr || yShape == nullptr || noiseShape == nullptr) {
        OP_LOGE(context_, "input/output shape is nullptr: x=%d y=%d noise=%d", static_cast<int>(xShape != nullptr),
                static_cast<int>(yShape != nullptr), static_cast<int>(noiseShape != nullptr));
        return ge::GRAPH_FAILED;
    }
    if (yShape->GetStorageShape() != xShape->GetStorageShape() ||
        noiseShape->GetStorageShape() != xShape->GetStorageShape()) {
        std::string shapesStr = Ops::Base::ToString(xShape->GetStorageShape()) + ", " +
                                Ops::Base::ToString(yShape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(noiseShape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OpNameOf(context_), "x, y and noise", shapesStr.c_str(),
                                               "The shapes of input x, output y and output noise must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace optiling {

static ge::graphStatus Tiling4RreluWithNoiseTrainArch35(gert::TilingContext* context)
{
    RreluWithNoiseTrainTilingSimt tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4RreluWithNoiseTrainArch35(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "RreluWithNoiseTrain");
}

IMPL_OP_OPTILING(RreluWithNoiseTrain)
    .Tiling(Tiling4RreluWithNoiseTrainArch35)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4RreluWithNoiseTrainArch35)
    .TilingInputsDataDependency({});
} // namespace optiling
