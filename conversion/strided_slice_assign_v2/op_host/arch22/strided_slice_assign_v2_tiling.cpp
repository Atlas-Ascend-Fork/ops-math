/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_tiling.cpp
 * \brief
 */

#include <string>
#include <vector>
#include "util/math_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "strided_slice_assign_v2_tiling.h"

namespace optiling {
static const size_t INDEX_BEGIN = 2;
static const size_t INDEX_END = 3;
static const size_t INDEX_STRIDES = 4;
static const size_t INDEX_AXES = 5;

constexpr int32_t INPUT_VAR_INDEX = 0;
constexpr int32_t INPUT_INPUTVAL_INDEX = 1;

static const char* ConstInputParamName(size_t idxInput)
{
    if (idxInput == INDEX_END) {
        return "end";
    }
    if (idxInput == INDEX_STRIDES) {
        return "strides";
    }
    if (idxInput == INDEX_AXES) {
        return "axes";
    }
    return "begin";
}

static void ReportTilingFailure(const char* opName, const char* reason)
{
    const std::string opNameStr(opName);
    const std::string reasonStr(reason);
    const std::string msg = "Failed to execute operator " + opNameStr + ". Reason: " + reasonStr + ".";
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opNameStr.c_str(), reasonStr.c_str()};
    OP_LOGE_WITHOUT_REPORT(opNameStr.c_str(), "%s", msg.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

template <typename T>
static ge::graphStatus CopyData2Array(gert::TilingContext* context, const gert::Tensor* listTensor, int64_t listSize,
                                      int64_t dataList[], const char* paramName)
{
    const T* listDataPtr = listTensor->GetData<T>();
    OP_CHECK_IF(listDataPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), paramName),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < listSize; i++) {
        dataList[i] = listDataPtr[i];
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus CopyData2Array(gert::TilingContext* context, const gert::Tensor* listTensor, int64_t listSize,
                                      std::vector<int64_t>& dataList, const char* paramName)
{
    const T* listDataPtr = listTensor->GetData<T>();
    OP_CHECK_IF(listDataPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), paramName),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < listSize; i++) {
        dataList.emplace_back(listDataPtr[i]);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus GetConstInputData(gert::TilingContext* context, const size_t idxInput, T& dataList)
{
    const char* paramName = ConstInputParamName(idxInput);
    auto listTensor = context->GetInputTensor(idxInput);
    OP_CHECK_IF(listTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), paramName),
                return ge::GRAPH_FAILED);
    auto listDataType = listTensor->GetDataType();
    int64_t listSize = listTensor->GetShapeSize();
    if (listDataType == ge::DT_INT32) {
        return CopyData2Array<int32_t>(context, listTensor, listSize, dataList, paramName);
    }
    if (listDataType == ge::DT_INT64) {
        return CopyData2Array<int64_t>(context, listTensor, listSize, dataList, paramName);
    }

    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), paramName, Ops::Base::ToString(listDataType).c_str(),
                                          "only int32 and int64 are supported");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus AdjustWithAxes(gert::TilingContext* context, size_t dimNum, int64_t varDim[], int64_t begin[],
                                      int64_t end[], int64_t strides[])
{
    std::vector<int64_t> axes;
    OP_CHECK_IF(GetConstInputData(context, INDEX_AXES, axes) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Get const input axes data failed."),
                return ge::GRAPH_FAILED);
    // adjust begin/end/strides
    std::vector<int64_t> beginAdjust(dimNum, 0);
    std::vector<int64_t> endAdjust(dimNum, 0);
    std::vector<int64_t> stridesAdjust(dimNum, 1);
    std::vector<bool> axisUsed(dimNum, false);
    for (size_t i = 0; i < dimNum; i++) {
        endAdjust[i] = varDim[i];
    }
    const int64_t dimNumI64 = static_cast<int64_t>(dimNum);
    for (size_t i = 0; i < axes.size(); i++) {
        int64_t axis = axes[i];
        if (axis < 0) {
            axis += dimNumI64;
        }
        OP_CHECK_IF(axis < 0 || axis >= dimNumI64,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "axes", std::to_string(axes[i]).c_str(),
                        ("axis is out of range [-" + std::to_string(dimNumI64) + ", " + std::to_string(dimNumI64) + ")")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(axisUsed[static_cast<size_t>(axis)],
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes",
                                                          std::to_string(axes[i]).c_str(), "axis value is duplicated"),
                    return ge::GRAPH_FAILED);
        axisUsed[static_cast<size_t>(axis)] = true;
        beginAdjust[static_cast<size_t>(axis)] = begin[i];
        endAdjust[static_cast<size_t>(axis)] = end[i];
        stridesAdjust[static_cast<size_t>(axis)] = strides[i];
    }
    for (size_t i = 0; i < dimNum; i++) {
        begin[i] = beginAdjust[i];
        end[i] = endAdjust[i];
        strides[i] = stridesAdjust[i];
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AdjustBegin(size_t dimNum, int64_t varDim[], int64_t begin[])
{
    for (size_t i = 0; i < dimNum; i++) {
        if (begin[i] < 0) {
            begin[i] = varDim[i] + begin[i];
        }
        if (begin[i] < 0) {
            begin[i] = 0;
        }
        if (begin[i] > varDim[i]) {
            begin[i] = varDim[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AdjustEnd(size_t dimNum, int64_t varDim[], int64_t end[])
{
    for (size_t i = 0; i < dimNum; i++) {
        if (end[i] < 0) {
            end[i] = varDim[i] + end[i];
        }
        if (end[i] < 0) {
            end[i] = 0;
        }
        if (end[i] > varDim[i]) {
            end[i] = varDim[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(const gert::TilingContext* context, size_t dimNum, const int64_t inputValueDim[],
                                       const int64_t begin[], const int64_t end[], const int64_t strides[])
{
    for (size_t i = 0; i < dimNum; i++) {
        int64_t calcValue = Ops::Base::CeilDiv(end[i] - begin[i], strides[i]);
        calcValue = calcValue < 0 ? 0 : calcValue;
        OP_CHECK_IF(
            calcValue != inputValueDim[i],
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "input_value,begin,end,strides",
                (std::to_string(inputValueDim[i]) + "," + std::to_string(calcValue)).c_str(),
                ("input_value dim " + std::to_string(i) + " must equal the slice length from begin, end and strides")
                    .c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StridedSliceAssignV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), " Tiling4StridedSliceAssignV2 is running.");
    StridedSliceAssignV2TilingData tiling;
    auto compileInfo = reinterpret_cast<const StridedSliceAssignV2CompileInfo*>(context->GetCompileInfo());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = (compileInfo == nullptr) ? ascendcPlatform.GetCoreNumAiv() : compileInfo->totalCoreNum;

    const gert::StorageShape* varShape = context->GetInputShape(INPUT_VAR_INDEX);
    const gert::StorageShape* inputValueShape = context->GetInputShape(INPUT_INPUTVAL_INDEX);
    OP_CHECK_IF(varShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputValueShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "input_value"),
                return ge::GRAPH_FAILED);

    size_t dimNum = varShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        dimNum != inputValueShape->GetStorageShape().GetDimNum(),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "var,input_value",
            (std::to_string(dimNum) + "," + std::to_string(inputValueShape->GetStorageShape().GetDimNum())).c_str(),
            "var rank must equal input_value rank"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNum <= 0,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "var", std::to_string(dimNum).c_str(),
                                                         "rank must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNum > 8,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "var", std::to_string(dimNum).c_str(),
                                                         "rank must not be greater than 8"),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < dimNum; i++) {
        OP_CHECK_IF(varShape->GetStorageShape().GetDim(i) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "var", std::to_string(varShape->GetStorageShape().GetDim(i)).c_str(),
                        ("dim " + std::to_string(i) + " must be greater than 0").c_str()),
                    return ge::GRAPH_FAILED);
    }

    int64_t varDim[MAX_DIM_NUM] = {0};
    int64_t inputValueDim[MAX_DIM_NUM] = {0};

    for (size_t i = 0; i < dimNum; i++) {
        varDim[i] = varShape->GetStorageShape().GetDim(i);
        inputValueDim[i] = inputValueShape->GetStorageShape().GetDim(i);
    }

    auto axesTensor = context->GetOptionalInputTensor(5);
    bool isHasAxes = (nullptr == axesTensor) ? 0 : 1;

    int64_t begin[MAX_DIM_NUM] = {0};
    int64_t end[MAX_DIM_NUM] = {0};
    int64_t strides[MAX_DIM_NUM] = {0};
    OP_CHECK_IF(GetConstInputData(context, INDEX_BEGIN, begin) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Get const input begin data failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetConstInputData(context, INDEX_END, end) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Get const input end data failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetConstInputData(context, INDEX_STRIDES, strides) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Get const input strides data failed."),
                return ge::GRAPH_FAILED);
    if (isHasAxes) {
        OP_CHECK_IF(AdjustWithAxes(context, dimNum, varDim, begin, end, strides) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Adjust begin, end and strides failed."),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(AdjustBegin(dimNum, varDim, begin) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Adjust begin data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AdjustEnd(dimNum, varDim, end) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Adjust end data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShape(context, dimNum, inputValueDim, begin, end, strides) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Check input_value shape failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strides[dimNum - 1] != 1,
        OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "strides", std::to_string(strides[dimNum - 1]).c_str(), "1"),
        return ge::GRAPH_FAILED);

    int64_t varCumShape[MAX_DIM_NUM] = {0};
    int64_t inputCumShape[MAX_DIM_NUM] = {0};

    varCumShape[dimNum - 1U] = varDim[dimNum - 1U];
    inputCumShape[dimNum - 1U] = inputValueDim[dimNum - 1U];

    for (int32_t i = static_cast<int32_t>(dimNum) - 2; i > 0; i--) {
        varCumShape[i] = varDim[i] * varCumShape[i + 1];
        inputCumShape[i] = inputValueDim[i] * inputCumShape[i + 1];
    }
    uint32_t useCoreNum = totalCoreNum;

    context->SetBlockDim(useCoreNum);

    uint32_t tilingKey = 1;
    context->SetTilingKey(tilingKey);

    tiling.set_dimNum(static_cast<int64_t>(dimNum));
    tiling.set_varDim(varDim);
    tiling.set_inputValueDim(inputValueDim);
    tiling.set_begin(begin);
    tiling.set_strides(strides);
    tiling.set_varCumShape(varCumShape);
    tiling.set_inputCumShape(inputCumShape);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = 16777216U; // 16777216U = 16U * 1024U * 1024U;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4StridedSliceAssignV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2 running.");
    auto compileInfo = context->GetCompiledInfo<StridedSliceAssignV2CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingFailure(context->GetNodeName(),
                                    "TilingPrepare4StridedSliceAssignV2 failed because compile info is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                ReportTilingFailure(context->GetNodeName(),
                                    "TilingPrepare4StridedSliceAssignV2 failed because platform info is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // no vector core enabled
    if (compileInfo->totalCoreNum <= 0) {
        std::string reason = "TilingPrepare4StridedSliceAssignV2 failed to get AIV core number, actual=" +
                             std::to_string(compileInfo->totalCoreNum);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    if (compileInfo->ubSize <= 0) {
        std::string reason = "TilingPrepare4StridedSliceAssignV2 failed to get UB size, actual=" +
                             std::to_string(compileInfo->ubSize);
        ReportTilingFailure(context->GetNodeName(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StridedSliceAssignV2)
    .Tiling(Tiling4StridedSliceAssignV2)
    .TilingParse<StridedSliceAssignV2CompileInfo>(TilingPrepare4StridedSliceAssignV2)
    .TilingInputsDataDependency({2, 3, 4, 5});
} // namespace optiling
