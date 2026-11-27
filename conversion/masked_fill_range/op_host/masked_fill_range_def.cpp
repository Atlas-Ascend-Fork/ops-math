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
 * \file masked_fill_range_def.cpp
 * \brief MaskedFillRange 算子的定义
 */

#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
/*!
 * \brief MaskedFillRange 算子类定义
 */
class MaskedFillRange : public OpDef {
public:
    explicit MaskedFillRange(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("axis").AttrType(REQUIRED).Int(-1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            // opFile.value 必须与 PyPTO kernel 文件名(op_kernel/masked_fill_range.py)
            // 以及 enable_pypto_kernel(masked_fill_range) 保持一致
            .ExtendCfgInfo("opFile.value", "masked_fill_range");
        // 芯片注册按 NpuArch（AddConfigByArch 在 opbase 内按 g_archToSocNames 展开
        // 逐 SoC 调用 AddConfig，注册集合与逐 SoC 写法一致）：
        // DAV_3510 → ascend950 + ascend350（同 arch 共享 a5 产物）；
        // DAV_9201 → ascend960dt；DAV_9202 → ascend960pr。A2/A3
        // （ascend910b/ascend910_93，arch22）无本算子的 PyPTO 实现，不注册，
        // 其构建/运行不受影响。
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, aicoreConfig);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, aicoreConfig);
    }
};
OP_ADD(MaskedFillRange); // 注册算子到算子信息库
} // namespace ops
