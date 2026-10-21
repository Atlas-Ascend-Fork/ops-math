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
 * \file squeeze_def.cpp
 * \brief op config of Squeeze
 */

#include "register/op_def_registry.h"
#include <array>
#include <vector>

namespace ops {
// x data type: 14 types
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_BOOL,   ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT4,  ge::DT_INT8,   ge::DT_UINT8,  ge::DT_INT16,
    ge::DT_UINT16, ge::DT_INT32,   ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_DOUBLE, ge::DT_COMPLEX64,
};
// format: each data type supports ND and FRACTAL_NZ
static constexpr std::array FORMAT_ALL{ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ};
// total combos: 14 * 2 = 28
static constexpr size_t DATA_TYPE_COMBINE_COUNT = VALUE_DATA_TYPE_ALL.size() * FORMAT_ALL.size();

static constexpr std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT> CombineDataTypes()
{
    std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT> dataTypes{};
    std::size_t idx = 0;
    for (std::size_t ifmt = 0; ifmt < FORMAT_ALL.size(); ++ifmt) {
        for (std::size_t idt = 0; idt < VALUE_DATA_TYPE_ALL.size(); ++idt) {
            dataTypes[idx] = VALUE_DATA_TYPE_ALL[idt];
            ++idx;
        }
    }
    return dataTypes;
}

static constexpr std::array<ge::Format, DATA_TYPE_COMBINE_COUNT> CombineFormats()
{
    std::array<ge::Format, DATA_TYPE_COMBINE_COUNT> formats{};
    std::size_t idx = 0;
    for (std::size_t ifmt = 0; ifmt < FORMAT_ALL.size(); ++ifmt) {
        for (std::size_t idt = 0; idt < VALUE_DATA_TYPE_ALL.size(); ++idt) {
            formats[idx] = FORMAT_ALL[ifmt];
            ++idx;
        }
    }
    return formats;
}

static constexpr auto DATA_TYPE_LIST = CombineDataTypes();
static constexpr auto FORMAT_LIST = CombineFormats();
static const std::vector<ge::DataType> X_DATA_TYPE_LIST(DATA_TYPE_LIST.begin(), DATA_TYPE_LIST.end());
static const std::vector<ge::Format> X_FORMAT_LIST(FORMAT_LIST.begin(), FORMAT_LIST.end());

class Squeeze : public OpDef {
public:
    explicit Squeeze(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(X_DATA_TYPE_LIST)
            .Format(X_FORMAT_LIST)
            .UnknownShapeFormat(X_FORMAT_LIST);

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(X_DATA_TYPE_LIST)
            .Format(X_FORMAT_LIST)
            .UnknownShapeFormat(X_FORMAT_LIST);

        this->Attr("axis").AttrType(OPTIONAL).ListInt({});

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(Squeeze);
} // namespace ops
