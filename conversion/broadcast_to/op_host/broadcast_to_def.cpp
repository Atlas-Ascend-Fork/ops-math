/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_def.cpp
 * \brief op config of broadcast_to
 */

#include "register/op_def_registry.h"
#include "common/inc/op_host/op_const_def_math.h"

namespace ops {
static const std::vector<ge::DataType> BASE_DATA_TYPE_ARCH92 = {ge::DT_INT8,
                                                                ge::DT_INT32,
                                                                ge::DT_INT64,
                                                                ge::DT_UINT8,
                                                                ge::DT_FLOAT16,
                                                                ge::DT_FLOAT,
                                                                ge::DT_BF16,
                                                                ge::DT_UINT32,
                                                                ge::DT_BOOL,
                                                                ge::DT_HIFLOAT8,
                                                                ge::DT_FLOAT8_E5M2,
                                                                ge::DT_FLOAT8_E4M3FN,
                                                                ge::DT_FLOAT4_E2M1,
                                                                ge::DT_FLOAT4_E1M2,
                                                                Ops::Math::DT_HIFLOAT4,
                                                                Ops::Math::DT_HIFLOAT4_SCALE,
                                                                ge::DT_INT16,
                                                                ge::DT_INT8,
                                                                ge::DT_INT32,
                                                                ge::DT_INT64,
                                                                ge::DT_UINT8,
                                                                ge::DT_FLOAT16,
                                                                ge::DT_FLOAT,
                                                                ge::DT_BF16,
                                                                ge::DT_UINT32,
                                                                ge::DT_BOOL,
                                                                ge::DT_HIFLOAT8,
                                                                ge::DT_FLOAT8_E5M2,
                                                                ge::DT_FLOAT8_E4M3FN,
                                                                ge::DT_FLOAT4_E2M1,
                                                                ge::DT_FLOAT4_E1M2,
                                                                Ops::Math::DT_HIFLOAT4,
                                                                Ops::Math::DT_HIFLOAT4_SCALE,
                                                                ge::DT_INT16};
static const std::vector<ge::Format> BASE_FORMAT_ARCH92 = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> BASE_CONST_DTYPE_ARCH92 = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::DataType> xDType = {
    ge::DT_INT8,          ge::DT_INT32,  ge::DT_INT64,  ge::DT_UINT8,    ge::DT_FLOAT16,     ge::DT_FLOAT,
    ge::DT_BF16,          ge::DT_UINT32, ge::DT_BOOL,   ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_INT16,         ge::DT_INT8,   ge::DT_INT32,  ge::DT_INT64,    ge::DT_UINT8,       ge::DT_FLOAT16,
    ge::DT_FLOAT,         ge::DT_BF16,   ge::DT_UINT32, ge::DT_BOOL,     ge::DT_HIFLOAT8,    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN, ge::DT_INT16};
static const std::vector<ge::Format> xFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> constDType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

class BroadcastTo : public OpDef {
public:
    explicit BroadcastTo(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType(constDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .ValueDepend(OPTIONAL);
        this->Output("y").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "broadcast_to_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
        this->AICore().AddConfig("ascend960dt", aicoreConfig);

        OpAICoreConfig CONFIG_ARCH92;
        CONFIG_ARCH92.Input("x")
            .ParamType(REQUIRED)
            .DataType(BASE_DATA_TYPE_ARCH92)
            .Format(BASE_FORMAT_ARCH92)
            .UnknownShapeFormat(BASE_FORMAT_ARCH92);
        CONFIG_ARCH92.Input("shape")
            .ParamType(REQUIRED)
            .DataType(BASE_CONST_DTYPE_ARCH92)
            .Format(BASE_FORMAT_ARCH92)
            .UnknownShapeFormat(BASE_FORMAT_ARCH92)
            .ValueDepend(OPTIONAL);
        CONFIG_ARCH92.Output("y")
            .ParamType(REQUIRED)
            .DataType(BASE_DATA_TYPE_ARCH92)
            .Format(BASE_FORMAT_ARCH92)
            .UnknownShapeFormat(BASE_FORMAT_ARCH92);
        CONFIG_ARCH92.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "broadcast_to_apt");
        this->AICore().AddConfig("ascend960dt", CONFIG_ARCH92);
        this->AICore().AddConfig("ascend960pr", CONFIG_ARCH92);
    }
};

OP_ADD(BroadcastTo);
} // namespace ops
