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
 * \file histogram_graph_infer.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"

namespace ops {

namespace {
static bool IsRegBasePlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::GRAPH_SUCCESS) {
        return false;
    }
    const std::string& shortSoc = platformInfo.str_info.short_soc_version;
    return shortSoc == "Ascend950" || shortSoc == "Ascend350" || shortSoc == "Ascend960DT" || shortSoc == "Ascend960PR";
}
} // namespace

static ge::graphStatus HistogramInferDataTypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramInferDataTypeFunc");
    auto inputDtype = context->GetInputDataType(0);
    OP_LOGD(context, "x dtype: %s", Ops::Base::ToString(inputDtype).c_str());

    ge::DataType outDtype = inputDtype;
    if (!IsRegBasePlatform() && inputDtype == ge::DT_FLOAT16) {
        outDtype = ge::DT_FLOAT;
    }

    OP_LOGD(context, "before set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context, "after set y dtype: %s", Ops::Base::ToString(context->GetOutputDataType(0)).c_str());
    OP_LOGD(context, "End to do HistogramInferDataTypeFunc end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Histogram).InferDataType(HistogramInferDataTypeFunc);

} // namespace ops
