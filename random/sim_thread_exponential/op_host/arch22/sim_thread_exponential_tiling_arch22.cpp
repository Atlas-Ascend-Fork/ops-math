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
 * \file sim_thread_exponential_tiling_arch22.cpp
 * \brief
 */

#include "sim_thread_exponential_tiling_arch22.h"
#include <string>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_ascendc.h"

namespace {
constexpr uint32_t BLOCKSIZE = 256;
constexpr uint32_t BATCHNUMPERHANDLE = 6;
constexpr uint32_t UNROLLFACTOR = 4;
constexpr uint32_t INPUT_SELF_IDX = 0;
constexpr uint32_t OUTPUT_SELF_IDX = 0;
constexpr uint32_t FP32_TYPESIZE = 4;
constexpr uint64_t ATTR_0 = 0;
constexpr uint64_t ATTR_1 = 1;
constexpr uint64_t ATTR_2 = 2;
constexpr uint64_t ATTR_3 = 3;
constexpr int64_t THREAD_PER_PROCESSOR = 2048;
constexpr int64_t STREAM_PROCESSOR_COUNT = 78;
constexpr float SIM_THREAD_EXPONENTIAL_LOW = 0.0;
constexpr float SIM_THREAD_EXPONENTIAL_HIGH = 1.0;
constexpr uint32_t TILING_K_TWO = 2;
constexpr uint32_t TILING_K_THREE = 3;
constexpr uint32_t SHIFT_LEFT_32 = 32;
inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// 空指针检查仅写日志不注册错误码，GetNodeName为空时回退到字面算子名（参照官方整改基线）
const char* OpNameOf(gert::TilingContext* tilingContext)
{
    const char* name = tilingContext->GetNodeName();
    return name == nullptr ? "SimThreadExponential" : name;
}
} // namespace

namespace optiling {
using namespace std;
void SimThreadExponentialTiling::PrintInfo()
{
    auto nodeName = context;
    OP_LOGD(nodeName, "----------------Start to print SimThreadExponentialTiling tiling data.----------------");
    OP_LOGD(nodeName, "key0 = %u.", tiling.get_key0());
    OP_LOGD(nodeName, "key1 = %u.", tiling.get_key1());
    OP_LOGD(nodeName, "offset_t_low = %u.", tiling.get_offset_t_low());
    OP_LOGD(nodeName, "offset_t_high = %u.", tiling.get_offset_t_high());
    OP_LOGD(nodeName, "useCoreNum = %u.", tiling.get_useCoreNum());
    OP_LOGD(nodeName, "batchNumPerCore = %u.", tiling.get_batchNumPerCore());
    OP_LOGD(nodeName, "batchNumTailCore = %u.", tiling.get_batchNumTailCore());
    OP_LOGD(nodeName, "batchNumTotal = %u.", tiling.get_batchNumTotal());
    OP_LOGD(nodeName, "numel = %ld.", tiling.get_numel());
    OP_LOGD(nodeName, "stepBlock = %u.", tiling.get_stepBlock());
    OP_LOGD(nodeName, "roundedSizeBlock = %u.", tiling.get_roundedSizeBlock());
    OP_LOGD(nodeName, "range = %f.", tiling.get_range());
    OP_LOGD(nodeName, "handleNumLoop = %u.", tiling.get_handleNumLoop());
    OP_LOGD(nodeName, "handleNumTail = %u.", tiling.get_handleNumTail());
    OP_LOGD(nodeName, "state = %lu.", tiling.get_state());
    OP_LOGD(nodeName, "start = %f.", tiling.get_start());
    OP_LOGD(nodeName, "end = %f.", tiling.get_end());
    OP_LOGD(nodeName, "lambda = %f.", tiling.get_lambda());
    OP_LOGD(nodeName, "----------------print SimThreadExponentialTiling tiling data end.----------------");
}

void SimThreadExponentialTiling::SetTilingData()
{
    context->SetBlockDim(useCoreNum);
    tiling.set_key0(key0);
    tiling.set_key1(key1);
    tiling.set_offset_t_low(offset_t_low);
    tiling.set_offset_t_high(offset_t_high);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_batchNumPerCore(batchNumPerCore);
    tiling.set_batchNumTailCore(batchNumTailCore);
    tiling.set_batchNumTotal(batchNumTotal);
    tiling.set_numel(numel);
    tiling.set_stepBlock(stepBlock);
    tiling.set_roundedSizeBlock(roundedSizeBlock);
    tiling.set_range(range);
    tiling.set_handleNumLoop(handleNumLoop);
    tiling.set_handleNumTail(handleNumTail);
    tiling.set_state(state);
    tiling.set_start(start);
    tiling.set_end(end);
    tiling.set_lambda(lambda);
}

ge::graphStatus SimThreadExponentialTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    totalCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = static_cast<int64_t>(ubSizePlatForm);

    return ge::GRAPH_SUCCESS;
}

bool SimThreadExponentialTiling::GetDataTypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dataSizeType = 1;
            break;
        case ge::DT_BF16:
            dataSizeType = TILING_K_TWO;
            break;
        case ge::DT_FLOAT:
            dataSizeType = TILING_K_THREE;
            break;
        default:
            return false;
    }

    return true;
}

ge::graphStatus SimThreadExponentialTiling::GetInputTensorInfo()
{
    auto nodeName = context->GetNodeName();

    OP_CHECK_IF(count < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attr count", std::to_string(count).c_str(),
                                                      "count must be greater than or equal to 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(threadPerProcessor <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "threadPerProcessor",
                                                      std::to_string(threadPerProcessor).c_str(),
                                                      "threadPerProcessor must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(streamProcessorCount <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "streamProcessorCount",
                                                      std::to_string(streamProcessorCount).c_str(),
                                                      "streamProcessorCount must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(start > end, // 如何获取start、end？
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "start, end",
                                                      (std::to_string(start) + " and " + std::to_string(end)).c_str(),
                                                      "start must be less than or equal to end"),
                return ge::GRAPH_FAILED);

    // 获取第一个输入gradOut的信息：空指针为框架契约防御检查，仅写日志透传，不再隐式注册EZ9999
    auto selfShapePtr = context->GetInputShape(INPUT_SELF_IDX);
    if (selfShapePtr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "selfShapePtr is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto selfShape = selfShapePtr->GetStorageShape();

    for (size_t i = 0; i < selfShape.GetDimNum(); ++i) {
        usrWorkspaceSize *= selfShape[i];
    }
    OP_CHECK_IF(
        usrWorkspaceSize != count,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "attr count", (std::to_string(count) + " and " + std::to_string(usrWorkspaceSize)).c_str(),
            "count must be equal to the product of the elements in "
            "selfShape"),
        return ge::GRAPH_FAILED);
    usrWorkspaceSize = Ceil(usrWorkspaceSize, BATCHNUMPERHANDLE * BLOCKSIZE);
    usrWorkspaceSize *= FP32_TYPESIZE;

    auto selfDesc = context->GetInputDesc(INPUT_SELF_IDX);
    if (selfDesc == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "selfDesc is nullptr");
        return ge::GRAPH_FAILED;
    }
    selfDType = selfDesc->GetDataType();
    GetDataTypeKey(selfDType);
    OP_CHECK_IF(
        GetDataTypeKey(selfDType) == false,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "self", std::to_string(static_cast<int>(selfDType)).c_str(),
                                              "dtype must be one of [float32, float16, bfloat16]"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SimThreadExponentialTiling::Tiling4Block()
{
    auto nodeName = context->GetNodeName();

    uint32_t default_block_num = threadPerProcessor / BLOCKSIZE * streamProcessorCount;
    batchNumTotal = Min<uint32_t>((count + BLOCKSIZE - 1) / BLOCKSIZE, default_block_num);
    stepNum = BLOCKSIZE * batchNumTotal * UNROLLFACTOR;
    stepBlock = batchNumTotal * UNROLLFACTOR;
    roundedSizeNum = ((numel - 1) / stepNum + 1) * stepNum;
    roundedSizeBlock = roundedSizeNum / BLOCKSIZE;
    range = end - start;

    key0 = static_cast<uint32_t>(seed);
    key1 = static_cast<uint32_t>(seed >> SHIFT_LEFT_32);
    state = 0;
    state += offset & static_cast<uint64_t>(ATTR_3);
    uint64_t offset_t = offset / UNROLLFACTOR;
    if (state > ATTR_3) {
        offset_t += 1;
        state -= UNROLLFACTOR;
    }
    offset_t_low = static_cast<uint32_t>(offset_t);
    offset_t_high = static_cast<uint32_t>(offset_t >> SHIFT_LEFT_32);

    // 分核计算
    useCoreNum = static_cast<int64_t>(
        Ops::Base::CeilDiv(batchNumTotal, Ops::Base::CeilDiv(batchNumTotal, totalCoreNum)));
    OP_CHECK_IF(useCoreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "useCoreNum", std::to_string(useCoreNum).c_str(),
                                                      "useCoreNum must not be equal to 0"),
                return ge::GRAPH_FAILED);
    // useCoreNum = static_cast<int64_t>(CeilDiv(batchNumTotal, CeilDiv(batchNumTotal, totalCoreNum)));
    batchNumPerCore = (batchNumTotal + useCoreNum - 1) / useCoreNum;
    batchNumTailCore = batchNumTotal - (useCoreNum - 1) * batchNumPerCore;
    // 分块计算
    handleNumLoop = batchNumPerCore / BATCHNUMPERHANDLE;
    handleNumTail = batchNumPerCore - handleNumLoop * BATCHNUMPERHANDLE;

    OP_CHECK_IF(
        batchNumPerCore <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "batchNumPerCore", std::to_string(batchNumPerCore).c_str(),
                                              "batchNumPerCore must be greater than 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SimThreadExponentialTiling::SetAttrParams()
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const int64_t* countPtr = attrs->GetAttrPointer<int64_t>(ATTR_0);
    if (countPtr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "countPtr is nullptr");
        return ge::GRAPH_FAILED;
    }
    count = static_cast<int64_t>(*countPtr);
    start = SIM_THREAD_EXPONENTIAL_LOW;
    end = SIM_THREAD_EXPONENTIAL_HIGH; // sim_expinential功能与sim_random_uniform类似，start固定为0.0，end固定为1.0
    const float* lambdaPtr = attrs->GetAttrPointer<float>(ATTR_1);
    if (lambdaPtr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "lambdaPtr is nullptr");
        return ge::GRAPH_FAILED;
    }
    lambda = static_cast<float>(*lambdaPtr);
    OP_CHECK_IF(
        lambda == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attr lambd", std::to_string(lambda).c_str(),
                                              "lambd is the denominator and cannot be zero"),
        return ge::GRAPH_FAILED);
    const int64_t* seedPtr = attrs->GetAttrPointer<int64_t>(ATTR_2);
    if (seedPtr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "seedPtr is nullptr");
        return ge::GRAPH_FAILED;
    }
    seed = static_cast<uint64_t>(*seedPtr);
    const int64_t* offsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_3);
    if (offsetPtr == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "offsetPtr is nullptr");
        return ge::GRAPH_FAILED;
    }
    offset = static_cast<uint64_t>(*offsetPtr);
    threadPerProcessor = THREAD_PER_PROCESSOR;
    streamProcessorCount = STREAM_PROCESSOR_COUNT;
    numel = count;

    return ge::GRAPH_SUCCESS;
}

void SimThreadExponentialTiling::SetTilingKey() { tilingKey_ = dataSizeType; }

uint64_t SimThreadExponentialTiling::GetTilingKey() { return tilingKey_; }

ge::graphStatus SimThreadExponentialTiling::DoTiling()
{
    auto nodeName = context->GetNodeName();

    // 下层函数失败时已完成正式错误码首报，此处仅补PLOG日志并原样透传，避免重复首报
    OP_CHECK_IF(SetAttrParams() != ge::GRAPH_SUCCESS, OP_LOGW(nodeName, "SetAttrParams failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGW(nodeName, "GetInputTensorInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGW(nodeName, "GetPlatformInfo failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Tiling4Block() != ge::GRAPH_SUCCESS, OP_LOGW(nodeName, "Tiling4Block failed."),
                return ge::GRAPH_FAILED);
    SetTilingData();

    SetTilingKey();
    context->SetTilingKey(GetTilingKey());

    PrintInfo();
    // 固定写法
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace
        [0] = usrWorkspaceSize +
              ascendcPlatform
                  .GetLibApiWorkSpaceSize(); // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
                                             // 该workspace作为中转空间用来放T2类型数据

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SimThreadExponential(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling4SimThreadExponential started.");

    SimThreadExponentialTiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4SimThreadExponential(gert::TilingParseContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepare4SimThreadExponential finished.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SimThreadExponential)
    .Tiling(Tiling4SimThreadExponential)
    .TilingParse<Tiling4SimThreadExponentialCompileInfo>(TilingPrepare4SimThreadExponential);
} // namespace optiling
