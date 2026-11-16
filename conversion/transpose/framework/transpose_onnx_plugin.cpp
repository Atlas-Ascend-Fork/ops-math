/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "conversion/transpose/op_graph/transpose_proto.h"

using namespace ge;
using ge::Operator;
namespace domi {

static Status ParseParamsTranspose(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    // update attribute perm if given
    std::vector<int32_t> perm = {};
    for (auto attr : node->attribute()) {
        if (attr.name() == "perm") {
            for (auto axis : attr.ints()) {
                perm.push_back(axis);
            }
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("perm", perm);

    // add dynamic input and out
    op_dest.DynamicInputRegister("args", node->input_size());
    op_dest.DynamicOutputRegister("output", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::Transpose");
    return SUCCESS;
}

static Status ParseOpToGraphTranspose(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    std::vector<int32_t> perm;
    if (op.GetAttr("perm", perm) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Failed to get perm from operator");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;

    // ONNX Transpose allows an absent perm attribute, whose default semantics is the reverse of the
    // input axes, i.e. perm = [rank-1, ..., 0]. The input rank is unknown at parse time, so the empty
    // perm is NOT rejected here: keep an explicitly empty perm Const (shape {0}) as a placeholder, and
    // let TransposeEmptyPermFusionPass fill the reverse perm after InferShape (when the rank is known).
    ge::Tensor trans_perm;
    if (perm.empty()) {
        OP_LOGD(GetOpName(op).c_str(), "Transpose perm is empty, keep empty perm Const for default reverse perm.");
        ge::TensorDesc emptyPermDesc(ge::Shape(std::vector<int64_t>{0}), ge::FORMAT_ND, ge::DT_INT32);
        trans_perm.SetTensorDesc(emptyPermDesc);
    } else {
        int32_t dim = static_cast<int32_t>(perm.size());
        trans_perm = Vec2Tensor<int32_t>(perm, {dim}, ge::DT_INT32);
    }
    auto const_perm = op::Const((ori_name + "_Const").c_str()).set_attr_value(trans_perm);
    auto transpose = op::Transpose((ori_name + "_Transpose").c_str()).set_input_x(data0).set_input_perm(const_perm);

    output_indexs.emplace_back(transpose, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}
// register Transpose op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Transpose"), ge::AscendString("ai.onnx::9::Transpose"),
                   ge::AscendString("ai.onnx::10::Transpose"), ge::AscendString("ai.onnx::11::Transpose"),
                   ge::AscendString("ai.onnx::12::Transpose"), ge::AscendString("ai.onnx::13::Transpose"),
                   ge::AscendString("ai.onnx::14::Transpose"), ge::AscendString("ai.onnx::15::Transpose"),
                   ge::AscendString("ai.onnx::16::Transpose"), ge::AscendString("ai.onnx::17::Transpose"),
                   ge::AscendString("ai.onnx::18::Transpose")})
    .ParseParamsFn(ParseParamsTranspose)
    .ParseOpToGraphFn(ParseOpToGraphTranspose)
    .ImplyType(ImplyType::TVM);
} // namespace domi
