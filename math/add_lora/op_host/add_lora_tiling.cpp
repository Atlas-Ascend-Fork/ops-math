/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_lora_tiling.cpp
 * \brief
 */

#include "util/shape_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "add_lora_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
// E9 remediation: replace generic internal error codes with registered codes

namespace optiling {
void AddLoraTiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

ge::graphStatus AddLoraTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"AddLora", "Tiling framework object platformInfo is unavailable; check "
                                                         "context construction and platform metadata."})),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"AddLora",
                                              (std::string("Platform core count must be greater than zero; actual=") +
                                               std::to_string(coreNum) + "; check platform resource metadata.")
                                                  .c_str()})),
                return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm <= 0,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"AddLora",
                                              (std::string("Platform UB bytes must be greater than zero; actual=") +
                                               std::to_string(ubSizePlatForm) + "; check platform resource metadata.")
                                                  .c_str()})),
                return ge::GRAPH_FAILED);
    coreNum_ = ascendcPlatform.GetCoreNumAic();
    isAscend310P_ = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::GetShapeAttrsInfo()
{
    yParaInfo_.desc = context_->GetInputDesc(Y_INPUT_INDEX);
    yParaInfo_.shape = context_->GetInputShape(Y_INPUT_INDEX);
    xParaInfo_.desc = context_->GetInputDesc(X_INPUT_INDEX);
    xParaInfo_.shape = context_->GetInputShape(X_INPUT_INDEX);
    weightBParaInfo_.desc = context_->GetInputDesc(WEIGHTB_INPUT_INDEX);
    weightBParaInfo_.shape = context_->GetInputShape(WEIGHTB_INPUT_INDEX);

    indicesParaInfo_.desc = context_->GetInputDesc(INDICES_INPUT_INDEX);
    indicesParaInfo_.shape = context_->GetInputShape(INDICES_INPUT_INDEX);
    weightAParaInfo_.desc = context_->GetOptionalInputDesc(WEIGHTA_INPUT_INDEX);
    weightAParaInfo_.tensor = context_->GetOptionalInputTensor(WEIGHTA_INPUT_INDEX);
    yOutParaInfo_.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    yOutParaInfo_.shape = context_->GetOutputShape(OUTPUT_INDEX);

    attrs_ = context_->GetAttrs();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckInputDataType()
{
    const gert::CompileTimeTensorDesc *yDesc = yParaInfo_.desc;
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "y"), return ge::GRAPH_FAILED);
    ge::DataType yDataType = yDesc->GetDataType();
    OP_CHECK_IF(
        yDataType != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE("AddLora", "y", ge::TypeUtils::DataTypeToSerialString(yDataType).c_str(), "FLOAT16"),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc *xDesc = xParaInfo_.desc;
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "x"), return ge::GRAPH_FAILED);
    ge::DataType xDataType = xDesc->GetDataType();
    OP_CHECK_IF(
        xDataType != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE("AddLora", "x", ge::TypeUtils::DataTypeToSerialString(xDataType).c_str(), "FLOAT16"),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc *weightBDesc = weightBParaInfo_.desc;
    OP_CHECK_IF(weightBDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "weightB"), return ge::GRAPH_FAILED);
    ge::DataType weightBDataType = weightBDesc->GetDataType();
    OP_CHECK_IF(weightBDataType != ge::DT_FLOAT16,
                OP_LOGE_FOR_INVALID_DTYPE("AddLora", "weightB",
                                          ge::TypeUtils::DataTypeToSerialString(weightBDataType).c_str(), "FLOAT16"),
                return ge::GRAPH_FAILED);

    const gert::Tensor *weightATensor = weightAParaInfo_.tensor;
    const gert::CompileTimeTensorDesc *weightADesc = weightAParaInfo_.desc;
    if (weightATensor != nullptr && weightADesc != nullptr) {
        ge::DataType weightADataType = weightADesc->GetDataType();
        OP_CHECK_IF(weightADataType != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE("AddLora", "weightA",
                                              ge::TypeUtils::DataTypeToSerialString(weightADataType).c_str(), "FLOAT16"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool AddLoraTiling::IsCapable()
{
    return true;
}

ge::graphStatus AddLoraTiling::CheckInputTensorShape()
{
    // Check base input tensors
    const gert::StorageShape *yShape = yParaInfo_.shape;
    OP_CHECK_IF(yShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "y"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM("AddLora", "y",
                                             std::to_string(yShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    
    const gert::StorageShape *xShape = xParaInfo_.shape;
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "x"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM("AddLora", "x",
                                             std::to_string(xShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(yShape->GetStorageShape().GetDim(DIM_INDEX_ZERO) != xShape->GetStorageShape().GetDim(DIM_INDEX_ZERO),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("AddLora", "y,x",
                                                       (Ops::Base::ToString(yShape->GetStorageShape()) + "," +
                                                        Ops::Base::ToString(xShape->GetStorageShape()))
                                                           .c_str(),
                                                       "Dimension 0 of y and x must match."),
                return ge::GRAPH_FAILED);

    batchSize_ = static_cast<uint32_t>(yShape->GetStorageShape().GetDim(DIM_INDEX_ZERO));
    H3_ = static_cast<uint32_t>(yShape->GetStorageShape().GetDim(DIM_INDEX_ONE));
    H1_ = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(DIM_INDEX_ONE));

    const gert::StorageShape *weightBShape = weightBParaInfo_.shape;
    OP_CHECK_IF(weightBShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "weightB"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightBShape->GetStorageShape().GetDimNum() != 4,
                OP_LOGE_FOR_INVALID_SHAPEDIM("AddLora", "weightB",
                                             std::to_string(weightBShape->GetStorageShape().GetDimNum()).c_str(), "4"),
                return ge::GRAPH_FAILED);

    wBatch_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_ZERO));
    L_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_ONE));
    H2_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_TWO));
    R_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_THREE));

    OP_CHECK_IF(H2_ > H3_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    "AddLora", "weightB,y",
                    (Ops::Base::ToString(weightBShape->GetStorageShape()) + "," +
                     Ops::Base::ToString(yShape->GetStorageShape()))
                        .c_str(),
                    "Dimension 2 of weightB must be less than or equal to dimension 1 of y."),
                return ge::GRAPH_FAILED);

    const gert::StorageShape *indicesShape = indicesParaInfo_.shape;
    OP_CHECK_IF(indicesShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("AddLora", "indices"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM("AddLora", "indices",
                                             std::to_string(indicesShape->GetStorageShape().GetDimNum()).c_str(), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        indicesShape->GetStorageShape().GetDim(DIM_INDEX_ZERO) != yShape->GetStorageShape().GetDim(DIM_INDEX_ZERO),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("AddLora", "indices,y",
                                               (Ops::Base::ToString(indicesShape->GetStorageShape()) + "," +
                                                Ops::Base::ToString(yShape->GetStorageShape()))
                                                   .c_str(),
                                               "Dimension 0 of indices must equal dimension 0 of y and x."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckOptionalInputTensorShape()
{
    const gert::StorageShape *xShape = xParaInfo_.shape;
    const gert::StorageShape *weightBShape = weightBParaInfo_.shape;
    const gert::Tensor *weightATensor = weightAParaInfo_.tensor;
    if (weightATensor != nullptr && weightBShape != nullptr) {
        OP_CHECK_IF(weightATensor->GetStorageShape().GetDim(DIM_INDEX_ZERO) !=
                            weightBShape->GetStorageShape().GetDim(DIM_INDEX_ZERO) ||
                        weightATensor->GetStorageShape().GetDim(DIM_INDEX_ONE) !=
                            weightBShape->GetStorageShape().GetDim(DIM_INDEX_ONE),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("AddLora", "weightA,weightB",
                                                           (Ops::Base::ToString(weightATensor->GetStorageShape()) + "," +
                                                            Ops::Base::ToString(weightBShape->GetStorageShape()))
                                                               .c_str(),
                                                           "Dimensions 0 and 1 of weightA and weightB must match."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(weightATensor->GetStorageShape().GetDim(DIM_INDEX_TWO) !=
                        weightBShape->GetStorageShape().GetDim(DIM_INDEX_THREE),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("AddLora", "weightA,weightB",
                                                           (Ops::Base::ToString(weightATensor->GetStorageShape()) + "," +
                                                            Ops::Base::ToString(weightBShape->GetStorageShape()))
                                                               .c_str(),
                                                           "Dimension 2 of weightA must equal dimension 3 of weightB (R)."),
                    return ge::GRAPH_FAILED);
        addLoraFlag_ = 1;
    }

    if (weightATensor != nullptr) {
        OP_CHECK_IF(
            weightATensor->GetStorageShape().GetDim(DIM_INDEX_THREE) != xShape->GetStorageShape().GetDim(DIM_INDEX_ONE),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("AddLora", "weightA,x",
                                                   (Ops::Base::ToString(weightATensor->GetStorageShape()) + "," +
                                                    Ops::Base::ToString(xShape->GetStorageShape()))
                                                       .c_str(),
                                                   "Dimension 3 of weightA must equal dimension 1 of x."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckInputAttrLimit()
{
    // check base input attrs
    OP_CHECK_IF(attrs_ == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"AddLora",
                                              "GetAttrs returned null; check the TilingContext attribute container."})),
                return ge::GRAPH_FAILED);

    const int64_t *layerIdxPtr = attrs_->GetAttrPointer<int64_t>(LAYER_IDX_ATTR_INDEX);
    OP_CHECK_IF(
        layerIdxPtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"AddLora", "Registered optional attribute layer_idx is unavailable; check "
                                                 "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    layerIdx_ = static_cast<uint32_t>(*layerIdxPtr);
    OP_CHECK_IF(
        layerIdx_ >= L_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "AddLora", "layer_idx", std::to_string(*layerIdxPtr).c_str(),
            (std::string("layer_idx must be less than weightB dimension 1, which is ") + std::to_string(L_) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);

    const float *scalePtr = attrs_->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    OP_CHECK_IF(
        scalePtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"AddLora", "Registered optional attribute scale is unavailable; check context "
                                                 "construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    scale_ = *scalePtr;

    return CheckYSliceAttrs();
}

ge::graphStatus AddLoraTiling::CheckYSliceAttrs()
{
    const int64_t *yOffsetPtr = attrs_->GetAttrPointer<int64_t>(Y_OFFSET_ATTR_INDEX);
    OP_CHECK_IF(
        yOffsetPtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"AddLora", "Registered optional attribute y_offset is unavailable; check context "
                                                 "construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    int64_t yOffsetValue = *yOffsetPtr;
    OP_CHECK_IF(yOffsetValue < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("AddLora", "y_offset", std::to_string(yOffsetValue).c_str(),
                                                      "y_offset must be non-negative."),
                return ge::GRAPH_FAILED);
    yOffset_ = static_cast<uint32_t>(yOffsetValue);
    OP_CHECK_IF(
        yOffset_ >= H3_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "AddLora", "y_offset", std::to_string(yOffsetValue).c_str(),
            (std::string("y_offset must be less than y dimension 1, which is ") + std::to_string(H3_) + ".").c_str()),
        return ge::GRAPH_FAILED);

    const int64_t *ySliceSizePtr = attrs_->GetAttrPointer<int64_t>(Y_SLICE_SIZE_ATTR_INDEX);
    OP_CHECK_IF(
        ySliceSizePtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"AddLora", "Registered optional attribute y_slice_size is unavailable; check "
                                                 "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    int64_t ySliceSizeValue = *ySliceSizePtr;
    if (ySliceSizeValue == Y_SLICE_SIZE_DEFAULT) {
        ySliceSize_ = H2_;
    } else {
        OP_CHECK_IF(
            ySliceSizeValue < 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("AddLora", "y_slice_size", std::to_string(ySliceSizeValue).c_str(),
                                                  (std::string("y_slice_size must be non-negative or the default value ") +
                                                   std::to_string(Y_SLICE_SIZE_DEFAULT) + ".")
                                                      .c_str()),
            return ge::GRAPH_FAILED);
        ySliceSize_ = static_cast<uint32_t>(ySliceSizeValue);
    }
    OP_CHECK_IF(
        ySliceSize_ > H2_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "AddLora", "y_slice_size", std::to_string(ySliceSizeValue).c_str(),
            (std::string("y_slice_size must not exceed weightB dimension 2, which is ") + std::to_string(H2_) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(yOffset_ + H2_ > H3_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    "AddLora", "y_offset,weightB,y",
                    (std::string("y_offset=") + std::to_string(yOffset_) + ",weightB.dim2=" + std::to_string(H2_) +
                     ",y.dim1=" + std::to_string(H3_))
                        .c_str(),
                    "y_offset plus weightB dimension 2 must not exceed y dimension 1."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckInputLimits()
{
    OP_CHECK_IF(
        addLoraFlag_ && R_ > MAX_RANK_SIZE,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "weightB", std::to_string(R_).c_str(),
                                                  "When weightA is provided, dimension 3 (R) must not exceed 128."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(R_ % CUBESIZE != 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "weightB", std::to_string(R_).c_str(),
                                                          "Dimension 3 (R) must be divisible by 16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(wBatch_ > MAX_WEIGHT_NUM,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "weightB", std::to_string(wBatch_).c_str(),
                                                          "Dimension 0 (weight batch count) must not exceed 32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(L_ > MAX_LAYER_NUM,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "weightB", std::to_string(L_).c_str(),
                                                          "Dimension 1 (layer count) must not exceed 32."),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(H1_ > MAX_HIDDEN_SIZE,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "x", std::to_string(H1_).c_str(),
                                                          "Dimension 1 (H1) must not exceed 131072."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H1_ % CUBESIZE != 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "x", std::to_string(H1_).c_str(),
                                                          "Dimension 1 (H1) must be divisible by 16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(H2_ > MAX_HIDDEN_SIZE,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "weightB", std::to_string(H2_).c_str(),
                                                          "Dimension 2 (H2) must not exceed 131072."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H2_ % CUBESIZE != 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "weightB", std::to_string(H2_).c_str(),
                                                          "Dimension 2 (H2) must be divisible by 16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(H3_ > MAX_HIDDEN_SIZE,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "y", std::to_string(H3_).c_str(),
                                                          "Dimension 1 (H3) must not exceed 131072."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H3_ % CUBESIZE != 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("AddLora", "y", std::to_string(H3_).c_str(),
                                                          "Dimension 1 (H3) must be divisible by 16."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::DoOpTiling()
{
    if ((CheckInputDataType() != ge::GRAPH_SUCCESS) ||
        (CheckInputTensorShape() != ge::GRAPH_SUCCESS) ||
        (CheckOptionalInputTensorShape() != ge::GRAPH_SUCCESS) ||
        (CheckInputAttrLimit() != ge::GRAPH_SUCCESS) ||
        (CheckInputLimits() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    
    uint32_t maxCore = static_cast<uint32_t>(coreNum_) * VECTORDOUBLE;
    if (isAscend310P_) {
        maxCore = static_cast<uint32_t>(coreNum_);
    }

    uint32_t usedCoreNum = static_cast<uint32_t>(coreNum_);
    uint32_t taskNumPerCore = batchSize_ / maxCore;
    uint32_t H2PerCore = Ops::Base::CeilDiv(H2_, CUBESIZE) / maxCore * CUBESIZE;
    tilingData_.set_layer(L_);
    tilingData_.set_batch(batchSize_);
    tilingData_.set_H2(H2_);
    tilingData_.set_H1(H1_);
    tilingData_.set_wBatch(wBatch_);
    tilingData_.set_usedCoreNum(usedCoreNum);
    tilingData_.set_R(R_);
    tilingData_.set_layer_idx(layerIdx_);
    tilingData_.set_scale(scale_);
    tilingData_.set_y_offset(yOffset_);
    tilingData_.set_y_slice_size(ySliceSize_);
    tilingData_.set_taskNumPerCore(taskNumPerCore);
    tilingData_.set_H2PerCore(H2PerCore);
    tilingData_.set_addLoraFlag(addLoraFlag_);
    tilingData_.set_y_column(H3_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::DoLibApiTiling()
{
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t AddLoraTiling::GetTilingKey() const
{
    uint64_t tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_DEFAULT_SCENE);
    if (batchSize_ <= static_cast<uint32_t>(coreNum_)) {
        tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_SPARSE_SCENE);
    }

    uint32_t socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_910);
    if (isAscend310P_) {
        socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_310);

        if (!addLoraFlag_) {
            tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_BGMV_SCENE);
        }
    }

    tilingKey += static_cast<uint64_t>(socVersionFlag) * SOC_TILINGKEY_OFFSET;
    return tilingKey;
}

ge::graphStatus AddLoraTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"AddLora", "Tiling framework object workspaces is unavailable; check "
                                                         "context construction and platform metadata."})),
                return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = WORKSIZE;
    /* usrWorkspaceSize = workspace for other gm */
    size_t usrWorkspaceSize =
        tilingData_.get_batch() * tilingData_.get_H1() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * tilingData_.get_R() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * tilingData_.get_H2() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * sizeof(uint32_t) * 3 + 1024 * 1024;
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "final tiling data size: %zu", tilingData_.GetDataSize());
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        "AddLora", (std::string("Serialized tiling data bytes must be aligned to 8; actual=") +
                                    std::to_string(tilingData_.GetDataSize()) + "; check the tiling data definition.")
                                       .c_str()})),
                return ge::GRAPH_FAILED);
    
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    auto numBlocks = tilingData_.get_usedCoreNum();
    context_->SetBlockDim(numBlocks);
    context_->SetTilingKey(GetTilingKey());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void AddLoraTiling::PrintTilingData()
{
    std::stringstream ss;
    ss << " layer: " << tilingData_.get_layer() << " batch: " << tilingData_.get_batch()
       << " H2: " << tilingData_.get_H2() << " H1: " << tilingData_.get_H1() << " wBatch: " << tilingData_.get_wBatch()
       << " usedCoreNum: " << tilingData_.get_usedCoreNum() << " R: " << tilingData_.get_R()
       << " layer_idx: " << tilingData_.get_layer_idx() << " scale: " << tilingData_.get_scale()
       << " y_offset: " << tilingData_.get_y_offset() << " y_slice_size: " << tilingData_.get_y_slice_size()
       << " taskNumPerCore: " << tilingData_.get_taskNumPerCore() << " H2PerCore: " << tilingData_.get_H2PerCore()
       << " addLoraFlag: " << tilingData_.get_addLoraFlag() << " y_column: " << tilingData_.get_y_column();
    OP_LOGD(context_->GetNodeName(), "api tiling: %s", ss.str().c_str());
}

static ge::graphStatus AddLoraTilingFunc(gert::TilingContext* context)
{
    AddLoraTiling tiling(context);
    context->SetScheduleMode(BATCH_MODE);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForAddLora([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AddLora, AddLoraTiling, 0);

IMPL_OP_OPTILING(AddLora).Tiling(AddLoraTilingFunc).TilingParse<Tiling4AddLoraCompileInfo>(TilingParseForAddLora);
} // namespace optiling
