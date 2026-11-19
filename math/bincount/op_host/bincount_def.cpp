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
 * \file bincount_def.cpp
 * \brief bincount def
 */

#include "op_common/op_host/util/op_arch_helper.h"
#include "register/op_def_registry.h"

namespace ops {
class Bincount : public OpDef {
public:
    explicit Bincount(const char* name) : OpDef(name)
    {
        this->Input("array").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("size").ParamType(REQUIRED).ValueDepend(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("bins").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "bincount");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, aicoreConfig);
    }
};
OP_ADD(Bincount);
} // namespace ops
