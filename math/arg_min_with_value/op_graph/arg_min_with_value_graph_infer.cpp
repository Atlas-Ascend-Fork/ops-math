/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include <cstdlib>
#include <set>
#include <string>

using namespace ge;
namespace ops {
constexpr size_t ATTR_IDX = 2;
constexpr int32_t NPU_ARCH_DECIMAL_BASE = 10;

static const std::set<NpuArch> SUPPORTED_NPU_ARCH_LIST = {NpuArch::DAV_3510, NpuArch::DAV_9201};

static bool IsSupportedNpuArch()
{
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
        ge::GRAPH_SUCCESS) {
        OP_LOGW("ArgMinWithValue", "Get platform info failed.");
        return false;
    }
    std::string npuArchStr;
    if (!platformInfos.GetPlatformRes("version", "NpuArch", npuArchStr)) {
        OP_LOGW("ArgMinWithValue", "Get NpuArch from platform info failed.");
        return false;
    }
    auto npuArch = static_cast<NpuArch>(std::strtoul(npuArchStr.c_str(), nullptr, NPU_ARCH_DECIMAL_BASE));
    bool isSupported = SUPPORTED_NPU_ARCH_LIST.count(npuArch) != 0U;
    OP_LOGD("ArgMinWithValue", "Current NpuArch is %s, supported=%d", npuArchStr.c_str(),
            static_cast<int>(isSupported));
    return isSupported;
}

static ge::graphStatus InferDataType4ArgMinWithValue(gert::InferDataTypeContext* context)
{
    OP_LOGD("Begin InferDataType4ArgMinWithValue");
    auto input_x_dtype = context->GetInputDataType(0);
    ge::DataType indiceDtype = ge::DT_INT32;

    if (IsSupportedNpuArch()) {
        auto attrs = context->GetAttrs();
        if (attrs == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
            return ge::GRAPH_FAILED;
        }
        auto indice_dtype_ptr = attrs->GetAttrPointer<int64_t>(ATTR_IDX);
        if (indice_dtype_ptr == nullptr) { // 仅 DAV_3510/DAV_9201 要求属性存在
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "indice_dtype");
            return ge::GRAPH_FAILED;
        }
        indiceDtype = static_cast<ge::DataType>(*indice_dtype_ptr);
        OP_LOGI(context->GetNodeName(), "argminwithvalueOutputDtypeInfer 2.0, output dtype is %d",
                static_cast<int>(indiceDtype));
    }

    context->SetOutputDataType(0, indiceDtype);
    context->SetOutputDataType(1, input_x_dtype);
    OP_LOGD("End InferDataType4ArgMinWithValue");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ArgMinWithValue).InferDataType(InferDataType4ArgMinWithValue);
} // namespace ops
