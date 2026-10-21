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
 * \file bitcast_def.cpp
 * \brief op config of Bitcast
 */

#include "register/op_def_registry.h"
#include <vector>

namespace ops {
// input types x output types x formats = 14 x 3 x 4 + 2 x 2 + 2 = 174 combos
static const std::vector<ge::DataType> INPUT_DTYPE = {
    ge::DT_BOOL,  ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT4,  ge::DT_INT8,   ge::DT_INT32,  ge::DT_UINT32,
    ge::DT_UINT8, ge::DT_INT64,   ge::DT_UINT64, ge::DT_INT16, ge::DT_UINT16, ge::DT_DOUBLE, ge::DT_COMPLEX64,
};
static const std::vector<ge::DataType> OUTPUT_DTYPE = {ge::DT_INT4, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
static const std::vector<ge::Format> FORMAT_LIST = {
    ge::FORMAT_FRACTAL_NZ_C0_32,
    ge::FORMAT_FRACTAL_NZ_C0_16,
    ge::FORMAT_ND,
    ge::FORMAT_FRACTAL_NZ,
};
// c0_08 group: uint8 + FRACTAL_NZ_C0_08 -> int4/float4_e2m1 + FRACTAL_NZ_C0_16
static const std::vector<ge::DataType> X_DTYPE_C0_08 = {ge::DT_UINT8, ge::DT_UINT8};
static const std::vector<ge::DataType> Y_DTYPE_C0_08 = {ge::DT_INT4, ge::DT_FLOAT4_E2M1};
static constexpr size_t MAIN_COUNT = 14 * 3 * 4;                            // 168
static constexpr size_t HF8_COUNT = 2 * 2;                                  // 4
static constexpr size_t C0_08_COUNT = 2;                                    // 2
static constexpr size_t TOTAL_COUNT = MAIN_COUNT + HF8_COUNT + C0_08_COUNT; // 174

static std::vector<ge::DataType> GenXDType()
{
    std::vector<ge::DataType> v;
    v.reserve(TOTAL_COUNT);
    for (size_t f = 0; f < FORMAT_LIST.size(); ++f) {
        for (size_t i = 0; i < INPUT_DTYPE.size(); ++i) {
            for (size_t o = 0; o < OUTPUT_DTYPE.size(); ++o) {
                v.push_back(INPUT_DTYPE[i]);
            }
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        v.push_back(i == 0 ? ge::DT_INT8 : ge::DT_UINT8);
        v.push_back(i == 0 ? ge::DT_INT8 : ge::DT_UINT8);
    }
    // c0_08 group
    for (size_t i = 0; i < C0_08_COUNT; ++i) {
        v.push_back(X_DTYPE_C0_08[i]);
    }
    return v;
}

static std::vector<ge::DataType> GenYDType()
{
    std::vector<ge::DataType> v;
    v.reserve(TOTAL_COUNT);
    for (size_t f = 0; f < FORMAT_LIST.size(); ++f) {
        for (size_t i = 0; i < INPUT_DTYPE.size(); ++i) {
            for (size_t o = 0; o < OUTPUT_DTYPE.size(); ++o) {
                v.push_back(OUTPUT_DTYPE[o]);
            }
        }
    }
    for (size_t i = 0; i < HF8_COUNT; ++i) {
        v.push_back(ge::DT_HIFLOAT8);
    }
    // c0_08 group
    for (size_t i = 0; i < C0_08_COUNT; ++i) {
        v.push_back(Y_DTYPE_C0_08[i]);
    }
    return v;
}

static std::vector<ge::Format> GenXFormat()
{
    std::vector<ge::Format> v;
    v.reserve(TOTAL_COUNT);
    for (size_t f = 0; f < FORMAT_LIST.size(); ++f) {
        for (size_t i = 0; i < INPUT_DTYPE.size() * OUTPUT_DTYPE.size(); ++i) {
            v.push_back(FORMAT_LIST[f]);
        }
    }
    for (size_t i = 0; i < 2; ++i) {
        v.push_back(ge::FORMAT_ND);
        v.push_back(ge::FORMAT_FRACTAL_NZ);
    }
    // c0_08 group
    for (size_t i = 0; i < C0_08_COUNT; ++i) {
        v.push_back(ge::FORMAT_FRACTAL_NZ_C0_8);
    }
    return v;
}

static const std::vector<ge::DataType> xDType = GenXDType();
static const std::vector<ge::DataType> yDType = GenYDType();
static const std::vector<ge::Format> xFormat = GenXFormat();

static std::vector<ge::Format> GenYFormat()
{
    std::vector<ge::Format> v;
    v.reserve(TOTAL_COUNT);
    for (size_t i = 0; i < xFormat.size(); ++i) {
        if (xFormat[i] == ge::FORMAT_FRACTAL_NZ_C0_16 && (xDType[i] == ge::DT_INT8 || xDType[i] == ge::DT_UINT8) &&
            yDType[i] == ge::DT_FLOAT4_E2M1) {
            v.push_back(ge::FORMAT_FRACTAL_NZ_C0_32);
        } else if (xFormat[i] == ge::FORMAT_FRACTAL_NZ_C0_8 && xDType[i] == ge::DT_UINT8) {
            v.push_back(ge::FORMAT_FRACTAL_NZ_C0_16);
        } else {
            v.push_back(xFormat[i]);
        }
    }
    return v;
}
static const std::vector<ge::Format> yFormat = GenYFormat();

class Bitcast : public OpDef {
public:
    explicit Bitcast(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);

        this->Output("y").ParamType(REQUIRED).DataType(yDType).Format(yFormat).UnknownShapeFormat(yFormat);

        this->Attr("type").AttrType(REQUIRED).Int();
        this->Attr("keep_dim").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "Null");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Bitcast);
} // namespace ops
