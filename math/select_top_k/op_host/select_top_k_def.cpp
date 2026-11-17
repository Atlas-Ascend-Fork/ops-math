/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"
#include "../op_kernel/select_top_k_config.h"

namespace ops {
class SelectTopK : public OpDef {
public:
    // 注册扩展算子的输入、输出、属性；A2/A3 共用 arch22 配置和实现。
    explicit SelectTopK(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> values = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::DataType> indices = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                   ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::vector<ge::DataType> bounds(values.size(), ge::DT_INT32);
        const std::vector<ge::Format> formats(values.size(), ge::FORMAT_ND);
        Input("x").ParamType(REQUIRED).DataType(values).Format(formats).UnknownShapeFormat(formats);
        Input("begin").ParamType(OPTIONAL).DataType(bounds).Format(formats).UnknownShapeFormat(formats);
        Input("end").ParamType(OPTIONAL).DataType(bounds).Format(formats).UnknownShapeFormat(formats);
        Output("values").ParamType(REQUIRED).DataType(values).Format(formats).UnknownShapeFormat(formats);
        Output("indices").ParamType(REQUIRED).DataType(indices).Format(formats).UnknownShapeFormat(formats);
        Attr("k").AttrType(REQUIRED).Int();
        Attr("largest").AttrType(OPTIONAL).Bool(true);
        Attr("order").AttrType(OPTIONAL).Int(::SelectTopK::ORDER_VALUE);
        Attr("indices_dtype").AttrType(OPTIONAL).Int(ge::DT_INT64);
        Attr("indices_offset").AttrType(OPTIONAL).Int(0);
        Attr("output_mode").AttrType(OPTIONAL).Int(::SelectTopK::OUTPUT_BOTH);
        Attr("value_fill").AttrType(OPTIONAL).Float(0.0f);
        Attr("index_fill").AttrType(OPTIONAL).Int(-1);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "select_top_k")
            .ExtendCfgInfo("opInterface.value", "select_top_k");
        AICore().AddConfig("ascend910b", config);
        AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(SelectTopK);
} // namespace ops
