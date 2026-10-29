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
namespace ops {
class ReduceStdWithMeanV2 : public OpDef {
public:
    explicit ReduceStdWithMeanV2(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("meanOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("dim").AttrType(OPTIONAL).ListInt();
        this->Attr("unbiased").AttrType(OPTIONAL).Bool();
        this->Attr("keepdim").AttrType(OPTIONAL).Bool();
        this->Attr("invert").AttrType(OPTIONAL).Bool();
        this->Attr("eps").AttrType(OPTIONAL).Float();
        this->Attr("correction").AttrType(OPTIONAL).Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReduceStdWithMeanV2);

class ReduceStdWithMeanV2Partial : public OpDef {
public:
    explicit ReduceStdWithMeanV2Partial(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("partial")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("dim").AttrType(OPTIONAL).ListInt();
        this->Attr("partialK").AttrType(OPTIONAL).Int();
        this->Attr("keepdim").AttrType(OPTIONAL).Bool();
        this->Attr("invert").AttrType(OPTIONAL).Bool();
        this->Attr("eps").AttrType(OPTIONAL).Float();
        this->Attr("correction").AttrType(OPTIONAL).Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReduceStdWithMeanV2Partial);
class ReduceStdWithMeanV2Merge : public OpDef {
public:
    explicit ReduceStdWithMeanV2Merge(const char* name) : OpDef(name)
    {
        this->Input("partial")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("meanOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("dim").AttrType(OPTIONAL).ListInt();
        this->Attr("partialK").AttrType(OPTIONAL).Int();
        this->Attr("keepdim").AttrType(OPTIONAL).Bool();
        this->Attr("invert").AttrType(OPTIONAL).Bool();
        this->Attr("eps").AttrType(OPTIONAL).Float();
        this->Attr("correction").AttrType(OPTIONAL).Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReduceStdWithMeanV2Merge);
} // namespace ops
