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
 * \file while_tf_plugin.cpp
 * \brief TensorFlow plugin for mapping functional control-flow op variants to While.
 */
#include "register/register.h"

namespace domi {
static Status AutoMappingFnWhile(const google::protobuf::Message* op_src, ge::Operator& op)
{
    std::map<std::string, std::pair<std::string, std::string>> value;
    value["in"] = std::pair<std::string, std::string>("input", "T");
    value["out"] = std::pair<std::string, std::string>("output", "T");
    if (AutoMappingFnDynamic(op_src, op, value) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseSubgraphPostFnWhileInput(int32_t data_index, int32_t& parent_index)
{
    parent_index = data_index;
    return SUCCESS;
}

static Status ParseSubgraphPostFnWhileOutput(const std::string& subgraph_name, int32_t retval_index,
                                             int32_t& parent_index)
{
    if (subgraph_name == "cond") {
        parent_index = -1;
    } else {
        parent_index = retval_index;
    }
    return SUCCESS;
}

static Status ParseSubgraphPostFnWhile(const std::string& subgraph_name, const ge::Graph& graph)
{
    return AutoMappingSubgraphIndex(
        graph, ParseSubgraphPostFnWhileInput, [&](int32_t retval_index, int32_t& parent_index) -> Status {
            return ParseSubgraphPostFnWhileOutput(subgraph_name, retval_index, parent_index);
        });
}

REGISTER_CUSTOM_OP("While")
    .FrameworkType(TENSORFLOW)
    .OriginOpType({"While", "_While", "StatelessWhile"})
    .ParseParamsFn(AutoMappingFnWhile)
    .ParseSubgraphPostFn(ParseSubgraphPostFnWhile)
    .ImplyType(ImplyType::GELOCAL);
} // namespace domi
