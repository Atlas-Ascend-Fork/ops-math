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
 * \file hypot_def.cpp
 * \brief hypot operator definition
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {

// dtype combinations: 3 same-dtype + 6 mixed (mixed always promotes to FP32 output; the
// kernel reads DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT per combination, so no TilingKey dtype encoding)
static const std::vector<ge::DataType> selfDataTypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT,
};

static const std::vector<ge::DataType> otherDataTypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,  ge::DT_BF16, ge::DT_FLOAT16,
    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
};

static const std::vector<ge::DataType> outDataTypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
};

static const std::vector<ge::Format> formats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

class Hypot : public OpDef {
public:
    explicit Hypot(const char* name) : OpDef(name)
    {
        this->Input("self").ParamType(REQUIRED).DataType(selfDataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("other").ParamType(REQUIRED).DataType(otherDataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("out").ParamType(REQUIRED).DataType(outDataTypes).Format(formats).UnknownShapeFormat(formats);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Hypot);
} // namespace ops
