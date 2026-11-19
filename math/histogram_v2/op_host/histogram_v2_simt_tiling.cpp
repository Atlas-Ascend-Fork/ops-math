/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_simt_tiling.cpp
 * \brief
 */
#include "op_common/op_host/util/op_const_def.h"
#include "histogram_v2_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"

namespace optiling {

constexpr size_t DEFAULT_WORKSPACE_SIZE = 0;
static const std::unordered_map<ge::DataType, uint32_t> INPUT_DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1},   {ge::DataType::DT_INT32, 2}, {ge::DataType::DT_INT8, 3},
    {ge::DataType::DT_UINT8, 4},   {ge::DataType::DT_INT16, 5}, {ge::DataType::DT_INT64, 6},
    {ge::DataType::DT_FLOAT16, 7}, {ge::DataType::DT_BF16, 8}};
constexpr int64_t INPUT_NUM = 3;
constexpr int64_t INPUT_IDX_X = 0;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int64_t BINS_IDX = 0;
constexpr int64_t SIZE_OF_INT32 = 4;
constexpr int64_t SIZE_OF_FLOAT32 = 4;
constexpr int64_t GM_ATOMIC_ADD_FACTOR = 100;
constexpr int64_t DEFAULT_BINS = 100;
// kernel 侧 bin 下标为 int32，bins 超过 INT32_MAX 时下标无法精确表示
constexpr int64_t MAX_BINS = 2147483647;
constexpr uint64_t SIMT_DCACHE_SIZE = 32 * 1024;

class HistogramV2SimtTiling : public HistogramV2BaseClass {
public:
    explicit HistogramV2SimtTiling(gert::TilingContext* context) : HistogramV2BaseClass(context) {};
    ~HistogramV2SimtTiling() override = default;
    void Reset(gert::TilingContext* context) override { HistogramV2BaseClass::Reset(context); }

protected:
    bool IsCapable() override
    {
        if (!(Ops::Base::IsRegbaseSocVersion(context_) || npuArch == Ops::Base::DAV_9201 ||
              npuArch == Ops::Base::DAV_9202)) {
            return false;
        }
        return true;
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    ge::graphStatus TilingDataForCore();
    void TilingDataPrint() const;
    void SelectSchId();

    uint64_t schId_ = TPL_SCH_ID_UB_FULL;
    int64_t inputDtypeVal_ = 0;
    int64_t coreNum_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t totalLength_ = 0; // the length of input
    int64_t tailNum_ = 0;
    bool isFp32Output_ = false;
    int64_t isDeterministic_ = 0;

private:
    // kernel needed.
    int64_t bins_ = 0;
    int64_t ubNumCanUse_ = 0;
    int64_t ubLoopNum_ = 0;

    int64_t needXCoreNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;

    int64_t clearYCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYTail_ = 0;
};

ge::graphStatus HistogramV2SimtTiling::DoOpTiling()
{
    aicoreParams_.ubSize = aicoreParams_.ubSize - SIMT_DCACHE_SIZE;
    Init();
    return SetKernelTiling();
}

ge::graphStatus HistogramV2SimtTiling::PostTiling()
{
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimtTiling::GetShapeAttrsInfo()
{
    auto inputShape = context_->GetInputShape(INPUT_IDX_X);
    if (inputShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "inputShape");
        return ge::GRAPH_FAILED;
    }
    totalLength_ = inputShape->GetStorageShape().GetShapeSize();

    auto compileInfo = reinterpret_cast<const HistogramV2CompileInfo*>(context_->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "compileInfo");
        return ge::GRAPH_FAILED;
    }
    coreNum_ = compileInfo->totalCoreNum;
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_, "coreNum must be > 0, but got %ld.", coreNum_),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    const int64_t* binsPtr = attrs->GetAttrPointer<int64_t>(BINS_IDX);
    bins_ = (binsPtr == nullptr) ? DEFAULT_BINS : *binsPtr;
    OP_CHECK_IF(
        bins_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "bins", std::to_string(bins_).c_str(), "a positive integer"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(bins_ > MAX_BINS,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "bins", std::to_string(bins_).c_str(),
                                          "a positive integer no greater than 2147483647"),
                return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX);
    if (outputShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "outputShape");
        return ge::GRAPH_FAILED;
    }
    auto outputDataLength = outputShape->GetStorageShape().GetShapeSize();
    if (outputDataLength != bins_) {
        std::string sizeStr = std::to_string(outputDataLength) + " and " + std::to_string(bins_);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "y and bins", sizeStr.c_str(),
                                                   "The shape size of y should be the same as bins");
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context_->GetInputDesc(INPUT_IDX_X);
    if (inputDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "inputDesc");
        return ge::GRAPH_FAILED;
    }
    auto dType = inputDesc->GetDataType();

    for (int64_t i = 1; i < INPUT_NUM; i++) {
        auto inputMinMaxDesc = context_->GetInputDesc(i);
        if (inputMinMaxDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "inputMinMaxDesc");
            return ge::GRAPH_FAILED;
        }
        auto minMaxDtype = inputMinMaxDesc->GetDataType();
        if (minMaxDtype != dType) {
            std::string dtypesStr = Ops::Base::ToString(dType) + " and " + Ops::Base::ToString(minMaxDtype);
            std::string paramNames = std::string("x and ") + (i == 1 ? "min" : "max");
            std::string reason = std::string("The dtypes of ") + paramNames + " must be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramNames.c_str(), dtypesStr.c_str(),
                                                   reason.c_str());
            return ge::GRAPH_FAILED;
        }
        auto minMaxShape = context_->GetInputShape(i);
        if (minMaxShape == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "minMaxShape");
            return ge::GRAPH_FAILED;
        }
        auto minMaxLength = minMaxShape->GetStorageShape().GetShapeSize();
        if (minMaxLength != 1) {
            std::string paramName = (i == 1) ? "min" : "max";
            OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), paramName.c_str(),
                                          std::to_string(minMaxLength).c_str(), "1");
            return ge::GRAPH_FAILED;
        }
    }

    auto iter = INPUT_DATA_TYPE_TO_INT.find(dType);
    OP_CHECK_IF((iter == INPUT_DATA_TYPE_TO_INT.end()),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(dType).c_str(),
                                          "float, int32, int8, uint8, int16, int64, float16 or bfloat16"),
                return ge::GRAPH_FAILED);
    inputDtypeVal_ = iter->second;

    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX);
    if (outputDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "outputDesc");
        return ge::GRAPH_FAILED;
    }
    auto outputDType = outputDesc->GetDataType();

    // Determine fp32 output flag directly from output dtype
    isFp32Output_ = (outputDType == ge::DT_FLOAT);

    // Validate fp32 output: only allowed for fp16/fp32/bf16 input on ascend950
    if (isFp32Output_) {
        if (inputDtypeVal_ != 1 && inputDtypeVal_ != 7 && inputDtypeVal_ != 8) { // 1=fp32, 7=fp16, 8=bf16
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(outputDType).c_str(), "int32");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (outputDType != ge::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(outputDType).c_str(),
                                      "int32, float");
            return ge::GRAPH_FAILED;
        }
    }

    // Parse deterministic flag from context
    isDeterministic_ = (context_->GetDeterministic() == 1) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

uint64_t HistogramV2SimtTiling::GetTilingKey() const { return context_->GetTilingKey(); }

ge::graphStatus HistogramV2SimtTiling::Init()
{
    OP_LOGD(context_, "Tiling initing.");
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "currentWorkSpace");
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = DEFAULT_WORKSPACE_SIZE;
    // For deterministic fp32 output path
    int64_t ubElemSize = (isFp32Output_ && isDeterministic_ == 1) ? SIZE_OF_FLOAT32 : SIZE_OF_INT32;
    ubNumCanUse_ = aicoreParams_.ubSize / ubElemSize;
    ubLoopNum_ = Ops::Base::CeilDiv(bins_, ubNumCanUse_);
    SelectSchId();
    TilingDataForCore();
    OP_LOGD(context_, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

void HistogramV2SimtTiling::SelectSchId()
{
    if (bins_ < ubNumCanUse_) {
        schId_ = TPL_SCH_ID_UB_FULL;
        context_->SetLocalMemorySize(aicoreParams_.ubSize);
    } else if (totalLength_ > bins_ / GM_ATOMIC_ADD_FACTOR) {
        schId_ = TPL_SCH_ID_UB_WINDOW;
        context_->SetLocalMemorySize(aicoreParams_.ubSize);
    } else {
        schId_ = TPL_SCH_ID_GM_ATOMIC;
    }
    // 确定性语义只作用于 fp32 输出（int32 输出的整数原子加本身即确定）
    if (isFp32Output_ && isDeterministic_ == 1) {
        schId_ += TPL_SCH_ID_UB_FULL_DET; // det variants are the base modes + 3
    }
}

ge::graphStatus HistogramV2SimtTiling::TilingDataForCore()
{
    OP_LOGD(context_, "TilingDataForCore start.");
    formerLength_ = Ops::Base::CeilDiv(totalLength_, coreNum_);
    needXCoreNum_ = Ops::Base::CeilDiv(totalLength_, formerLength_);
    tailLength_ = totalLength_ - (needXCoreNum_ - 1) * formerLength_;

    clearYFactor_ = Ops::Base::CeilDiv(bins_, coreNum_);
    OP_CHECK_IF(clearYFactor_ == 0, OP_LOGE(context_, "clearYFactor must not be 0."), return ge::GRAPH_FAILED);
    clearYCoreNum_ = Ops::Base::CeilDiv(bins_, clearYFactor_);
    clearYTail_ = bins_ - (clearYCoreNum_ - 1) * clearYFactor_;
    needCoreNum_ = std::max(needXCoreNum_, clearYCoreNum_);
    OP_LOGD(context_, "TilingDataForCore end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimtTiling::SetKernelTiling()
{
    context_->SetBlockDim(needCoreNum_);

    auto tilingData = context_->GetTilingData<HistogramV2SimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->bins = bins_;
    tilingData->ubNumCanUse = ubNumCanUse_;
    tilingData->ubLoopNum = ubLoopNum_;

    tilingData->needXCoreNum = needXCoreNum_;
    tilingData->formerLength = formerLength_;
    tilingData->tailLength = tailLength_;

    tilingData->clearYCoreNum = clearYCoreNum_;
    tilingData->clearYFactor = clearYFactor_;
    tilingData->clearYTail = clearYTail_;

    context_->SetTilingKey(GET_TPL_TILING_KEY(schId_));
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void HistogramV2SimtTiling::TilingDataPrint() const
{
    OP_LOGD(context_, "tilingKey_: %lu.", context_->GetTilingKey());
    OP_LOGD(context_, "schId_: %lu.", schId_);
    OP_LOGD(context_, "needCoreNum_: %ld.", needCoreNum_);
    OP_LOGD(context_, "totalLength_: %ld.", totalLength_);
    OP_LOGD(context_, "bins_: %ld.", bins_);
    OP_LOGD(context_, "ubNumCanUse_: %ld.", ubNumCanUse_);
    OP_LOGD(context_, "ubLoopNum_: %ld.", ubLoopNum_);
    OP_LOGD(context_, "needXCoreNum_: %ld.", needXCoreNum_);
    OP_LOGD(context_, "formerLength_: %ld.", formerLength_);
    OP_LOGD(context_, "tailLength_: %ld.", tailLength_);
    OP_LOGD(context_, "clearYCoreNum_: %ld.", clearYCoreNum_);
    OP_LOGD(context_, "clearYFactor_: %ld.", clearYFactor_);
    OP_LOGD(context_, "clearYTail_: %ld.", clearYTail_);
    OP_LOGD(context_, "isFp32Output_: %d.", isFp32Output_);
    OP_LOGD(context_, "isDeterministic_: %ld.", isDeterministic_);
}

REGISTER_OPS_TILING_TEMPLATE(HistogramV2, HistogramV2SimtTiling, 30000);
} // namespace optiling
