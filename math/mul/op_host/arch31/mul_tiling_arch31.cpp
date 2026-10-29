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
 * \file mul_tiling_arch31.cpp
 * \brief mul tiling source file for arch31
 */

#include <functional>
#include <graph/utils/type_utils.h>
#include <unordered_map>

#include "mul_tiling_arch31.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "math/mul/op_kernel/arch31/mul_dag_arch31.h"
#include "math/mul/op_kernel/arch31/mul_struct.h"

namespace optiling {
using namespace AscendC;
using namespace ge;
using namespace MulDag;
using namespace Ops::Base;

constexpr static uint64_t MUL_COMMON_TILING_PRIORITY = 0;
constexpr static int64_t MUL_BROADCAST_EXTRA_BUFFER_NUM = 1;
constexpr static std::size_t HASH_PRIME = 31;
constexpr static std::size_t HASH_INIT = 17;
constexpr static std::size_t HASH_SHIFT_3 = 3;
constexpr static std::size_t HASH_SHIFT_5 = 5;

template <typename OpDag>
ge::graphStatus DoTiling(gert::TilingContext* context, uint64_t& tilingKey)
{
    BroadcastBaseTiling<OpDag> brcBaseTiling(context,
                                             static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_UB_BROADCAST));
    OP_CHECK_IF((brcBaseTiling.DoTiling(0, MUL_BROADCAST_EXTRA_BUFFER_NUM) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "Broadcast template do base tiling failed."), return ge::GRAPH_FAILED);

    const uint64_t schMode = brcBaseTiling.GetSchMode();
    const bool isSupportedSchMode = schMode == 101 || schMode == 102 || schMode == 103 || schMode == 104 ||
                                    schMode == 109 || schMode == 201 || schMode == 202;
    OP_CHECK_IF(
        !isSupportedSchMode,
        OP_LOGE(context->GetNodeName(), "Unsupported arch31 schMode: %llu.", static_cast<unsigned long long>(schMode)),
        return ge::GRAPH_FAILED);
    tilingKey = GET_TPL_TILING_KEY(schMode);
    OP_LOGI("MulTiling", "schMode is %llu, tilingKey is %llu.", static_cast<unsigned long long>(schMode),
            static_cast<unsigned long long>(tilingKey));
    return ge::GRAPH_SUCCESS;
}

namespace {
struct DtypeCombination {
    ge::DataType input0;
    ge::DataType input1;
    ge::DataType output;

    bool operator==(const DtypeCombination& other) const
    {
        return input0 == other.input0 && input1 == other.input1 && output == other.output;
    }
};

struct DtypeCombinationHash {
    std::size_t operator()(const DtypeCombination& comb) const
    {
        const std::size_t prime = HASH_PRIME;
        std::size_t hash = HASH_INIT;
        hash = hash * prime + std::hash<ge::DataType>()(comb.input0);
        hash = hash * prime + (std::hash<ge::DataType>()(comb.input1) << HASH_SHIFT_3);
        hash = hash * prime + (std::hash<ge::DataType>()(comb.output) << HASH_SHIFT_5);
        return hash;
    }
};

using TilingFunc = std::function<ge::graphStatus(MulTiling*)>;

const std::unordered_map<DtypeCombination, TilingFunc, DtypeCombinationHash> DTYPE_MAP = {
    {{ge::DT_INT8, ge::DT_INT8, ge::DT_INT8},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter int8 branch.");
         return DoTiling<MulInt8Op::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter uint8 branch.");
         return DoTiling<MulUint8Op::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter fp16 branch.");
         return DoTiling<MulXfp16Op<half>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
     [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter no cast branch.");
         return DoTiling<MulOp<float>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }},
    {{ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}, [](MulTiling* tiling) {
         OP_LOGD("MulTiling", "Enter no cast branch.");
         return DoTiling<MulOp<int32_t>::OpDag>(tiling->GetContext(), tiling->tilingKey_);
     }}};
} // namespace

ge::graphStatus MulTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }

bool MulTiling::IsCapable() { return true; }

ge::graphStatus MulTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDType = outputDesc->GetDataType();

    OP_LOGD("MulTiling", "Input0DType is: %s, input1DType is: %s, outputDtype is: %s.",
            ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(input1DType).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputDType).c_str());

    DtypeCombination key = {input0DType, input1DType, outputDType};
    auto it = DTYPE_MAP.find(key);
    if (it != DTYPE_MAP.end()) {
        return it->second(this);
    }

    std::string dtypeMsg = Ops::Base::ToString(input0DType) + ", " + Ops::Base::ToString(input1DType) + " and " +
                           Ops::Base::ToString(outputDType);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2 and y", dtypeMsg.c_str(),
                                           "The dtypes of x1, x2 and y must all be fp16, fp32, int32, uint8 or int8");
    return ge::GRAPH_FAILED;
}

ge::graphStatus MulTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MulTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus MulTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MulTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MulTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const BroadcastCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(context_->GetNodeName(), "Get ubSize from compileInfo is: %ld", ubSize_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
        OP_LOGD(context_->GetNodeName(), "Get ubSize from ascendcPlatform is: %ld", ubSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMul(gert::TilingContext* context)
{
    OP_LOGI("MulTiling", "Enter arch31 TilingForMul");
    if (context == nullptr) {
        OP_LOGE("MulTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context, "Enter arch31 ascendc MulTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMul([[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(Mul).Tiling(TilingForMul).TilingParse<BroadcastCompileInfo>(TilingPrepareForMul);

REGISTER_OPS_TILING_TEMPLATE(Mul, MulTiling, MUL_COMMON_TILING_PRIORITY);
} // namespace optiling
