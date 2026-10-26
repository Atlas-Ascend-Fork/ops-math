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
 * \file if_tf_plugin.cpp
 * \brief TensorFlow plugin for mapping functional control-flow op variants to If.
 */
#include "register/register.h"

namespace domi {
static Status AutoMappingFnIf(const google::protobuf::Message* op_src, ge::Operator& op)
{
    std::map<std::string, std::pair<std::string, std::string>> value;
    value["in"] = std::pair<std::string, std::string>("input", "Tin");
    value["out"] = std::pair<std::string, std::string>("output", "Tout");
    if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseSubgraphPostFnIf(const std::string& subgraph_name, const ge::Graph& graph)
{
    return AutoMappingSubgraphIndex(
        graph, [](int32_t data_index) { return data_index + 1; }, [](int32_t retval_index) { return retval_index; });
}

REGISTER_CUSTOM_OP("If")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"If", "_If", "StatelessIf"})
    .ParseParamsFn(AutoMappingFnIf)
    .ParseSubgraphPostFn(ParseSubgraphPostFnIf)
    .ImplyType(ImplyType::GELOCAL);
} // namespace domi
