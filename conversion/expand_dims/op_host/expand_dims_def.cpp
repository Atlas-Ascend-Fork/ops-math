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
 * \file expand_dims_def.cpp
 * \brief Operator info library of ExpandDims.
 */
#include "register/op_def_registry.h"
#include <array>

namespace ops {
// x/y data type (14 types), from the source ini entry of ai_core/ascend950
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_BOOL,   ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT4,  ge::DT_INT8,   ge::DT_UINT8,  ge::DT_INT16,
    ge::DT_UINT16, ge::DT_INT32,   ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_DOUBLE, ge::DT_COMPLEX64,
};
// axis data type (int32, int64)
static constexpr std::array INDEX_DATA_TYPE_ALL{ge::DT_INT32, ge::DT_INT64};
// supported formats
static constexpr std::array FORMAT_ALL{ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ};
// total combos: 14 * 2 * 2 = 56
static constexpr size_t DATA_TYPE_COMBINE_COUNT = VALUE_DATA_TYPE_ALL.size() * INDEX_DATA_TYPE_ALL.size() *
                                                  FORMAT_ALL.size();

static constexpr std::array<std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT>, 3> CombineDataTypes()
{
    std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT> xDTs{}, axisDTs{}, yDTs{};
    std::size_t idx = 0;
    for (std::size_t fm = 0; fm < FORMAT_ALL.size(); ++fm) {
        for (std::size_t ic = 0; ic < INDEX_DATA_TYPE_ALL.size(); ++ic) {
            for (std::size_t iv = 0; iv < VALUE_DATA_TYPE_ALL.size(); ++iv) {
                xDTs[idx] = VALUE_DATA_TYPE_ALL[iv];
                axisDTs[idx] = INDEX_DATA_TYPE_ALL[ic];
                yDTs[idx] = VALUE_DATA_TYPE_ALL[iv];
                ++idx;
            }
        }
    }
    return {xDTs, axisDTs, yDTs};
}
static constexpr auto DATA_TYPE_LIST = CombineDataTypes();
static constexpr auto& X_DATA_TYPE_LIST = std::get<0>(DATA_TYPE_LIST);
static constexpr auto& AXIS_DATA_TYPE_LIST = std::get<1>(DATA_TYPE_LIST);
static constexpr auto& Y_DATA_TYPE_LIST = std::get<2>(DATA_TYPE_LIST);

static constexpr std::array<ge::Format, DATA_TYPE_COMBINE_COUNT> CombineFormats()
{
    std::array<ge::Format, DATA_TYPE_COMBINE_COUNT> fmts{};
    std::size_t idx = 0;
    for (std::size_t fm = 0; fm < FORMAT_ALL.size(); ++fm) {
        for (std::size_t ic = 0; ic < INDEX_DATA_TYPE_ALL.size(); ++ic) {
            for (std::size_t iv = 0; iv < VALUE_DATA_TYPE_ALL.size(); ++iv) {
                fmts[idx] = FORMAT_ALL[fm];
                ++idx;
            }
        }
    }
    return fmts;
}
static constexpr auto FORMAT_LIST = CombineFormats();

class ExpandDims : public OpDef {
public:
    explicit ExpandDims(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(X_DATA_TYPE_LIST.begin(), X_DATA_TYPE_LIST.end()))
            .Format(std::vector<ge::Format>(FORMAT_LIST.begin(), FORMAT_LIST.end()))
            .UnknownShapeFormat(std::vector<ge::Format>(FORMAT_LIST.begin(), FORMAT_LIST.end()));

        this->Input("axis")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(AXIS_DATA_TYPE_LIST.begin(), AXIS_DATA_TYPE_LIST.end()))
            .Format(std::vector<ge::Format>(FORMAT_LIST.begin(), FORMAT_LIST.end()))
            .UnknownShapeFormat(std::vector<ge::Format>(FORMAT_LIST.begin(), FORMAT_LIST.end()))
            .ValueDepend(REQUIRED);

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(Y_DATA_TYPE_LIST.begin(), Y_DATA_TYPE_LIST.end()))
            .Format(std::vector<ge::Format>(FORMAT_LIST.begin(), FORMAT_LIST.end()))
            .UnknownShapeFormat(std::vector<ge::Format>(FORMAT_LIST.begin(), FORMAT_LIST.end()));

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "Null");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};

OP_ADD(ExpandDims);
} // namespace ops
