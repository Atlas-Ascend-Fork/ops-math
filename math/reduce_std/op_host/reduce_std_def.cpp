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
 * \file reduce_std_def.cpp
 * \brief aicore info for reduce std op
 */
#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT16, ge::DT_FLOAT};

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND};

class ReduceStd : public OpDef {
public:
    explicit ReduceStd(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Output("y1").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Output("y2").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Attr("dim").AttrType(OPTIONAL).ListInt();
        this->Attr("unbiased").AttrType(OPTIONAL).Bool(true);
        this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "reduce_std");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, aicoreConfig);
    }
};

OP_ADD(ReduceStd);
} // namespace ops
