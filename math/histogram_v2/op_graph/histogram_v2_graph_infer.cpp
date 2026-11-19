/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_graph_infer.cpp
 * \brief histogram_v2 operater graph infer resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

namespace ops {

namespace {
// regbase 短名集合(short_soc_version 域), 与 histogram_fusion_pass 的
// MeetRequirements 同域: 图编译路线下 PlatformInfo 的 arch_type 不填(-1),
// 只能靠短名判定
bool IsRegBaseShortSoc(const std::string& shortSoc)
{
    return shortSoc == "Ascend950" || shortSoc == "Ascend350" || shortSoc == "Ascend960DT" || shortSoc == "Ascend960PR";
}

// 取平台信息并判定 regbase; 取不到平台信息时返回 false, 由调用方走旧逻辑兜底
bool IsRegBasePlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    return IsRegBaseShortSoc(platformInfo.str_info.short_soc_version);
}

} // namespace

static ge::graphStatus HistogramV2InferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramV2InferDataTypeFunc");
    auto inputDtype = context->GetInputDataType(0);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());

    ge::DataType outDtype = ge::DT_INT32;
    // 平台分叉(先例: range_graph_infer / group_norm_silu_infershape):
    // regbase 输出 dtype 跟随 x 的就近表内源(浮点->DT_FLOAT, 整数->DT_INT32),
    // 上层无需设置 y_dtype attr; 旧平台(910b 等)或取不到平台信息时保持
    // y_dtype attr 二值旧逻辑, 保证存量行为不变
    const bool isRegBase = IsRegBasePlatform();
    if (isRegBase) {
        outDtype = (inputDtype == ge::DT_FLOAT || inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16) ?
                       ge::DT_FLOAT :
                       ge::DT_INT32;
        // 显式设置的 y_dtype 必须与推导结果一致, 冲突时报错退出,
        // 避免属性与推导矛盾时静默选错编译组合; 未显式设置时按推导执行
        auto attrs = context->GetAttrs();
        if (attrs != nullptr) {
            auto yDtypePtr = attrs->GetAttrPointer<int64_t>(1);
            if (yDtypePtr != nullptr && *yDtypePtr != static_cast<int64_t>(outDtype)) {
                OP_LOGE(context,
                        "The y_dtype attr [%ld] does not match the derived output dtype [%s] (input dtype "
                        "[%s]). The y_dtype attr should be left unset or set to the derived dtype.",
                        *yDtypePtr, Ops::Base::ToString(outDtype).c_str(), Ops::Base::ToString(inputDtype).c_str());
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        // y_dtype属性指定输出类型，默认为DT_INT32(3)，可设置为DT_FLOAT(0)
        auto attrs = context->GetAttrs();
        if (attrs != nullptr) {
            auto yDtypePtr = attrs->GetAttrPointer<int64_t>(1);
            OP_LOGD(context, "yDtypePtr: %p", yDtypePtr);
            if (yDtypePtr != nullptr) {
                OP_LOGD(context, "yDtypePtr value: %ld", *yDtypePtr);
                if (*yDtypePtr == ge::DT_FLOAT) {
                    outDtype = ge::DT_FLOAT;
                }
            }
        }
    }

    OP_LOGD(context, "before set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context, "after set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    OP_LOGD(context, "End to do HistogramV2InferDataTypeFunc end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(HistogramV2).InferDataType(HistogramV2InferDataTypeFunc);

}; // namespace ops
