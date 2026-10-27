/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <string>
#include <vector>
#include "graph/operator_factory.h"
#include "register/register.h"
#include "stub_ops.h"
#include "../op_graph/ragged_bin_count_proto.h"

namespace domi {
namespace {
constexpr uint32_t SIZE_INPUT_INDEX = 2U;
constexpr std::array<const char*, 4> INPUT_NAMES = {"splits", "values", "size", "weights"};

Status ParseParamsRaggedBinCount(const ge::Operator& source, ge::Operator& destination)
{
    ge::DataType weightsType;
    ge::DataType indexType;
    if (source.GetAttr("T", weightsType) != ge::GRAPH_SUCCESS ||
        source.GetAttr("Tidx", indexType) != ge::GRAPH_SUCCESS) {
        return FAILED;
    }
    if ((indexType != ge::DT_INT32 && indexType != ge::DT_INT64) ||
        (weightsType != ge::DT_FLOAT && weightsType != ge::DT_DOUBLE && weightsType != ge::DT_INT32 &&
         weightsType != ge::DT_INT64)) {
        return FAILED;
    }
    if (AutoMappingByOpFn(source, destination) != SUCCESS) {
        return FAILED;
    }
    // Preserve the TF type attributes for the graph expansion. Integer and FP64
    // weights must continue to use the original AI_CPU implementation.
    destination.SetAttr("tf_weights_type", weightsType);
    destination.SetAttr("tf_index_type", indexType);
    destination.SetAttr("original_type", "RaggedBincount");
    return SUCCESS;
}

Status ParseOpToGraphRaggedBinCount(const ge::Operator& op, ge::Graph& graph)
{
    ge::AscendString name;
    ge::DataType weightsType;
    ge::DataType indexType;
    bool binaryOutput = false;
    if (op.GetName(name) != ge::GRAPH_SUCCESS || op.GetAttr("tf_weights_type", weightsType) != ge::GRAPH_SUCCESS ||
        op.GetAttr("tf_index_type", indexType) != ge::GRAPH_SUCCESS ||
        op.GetAttr("binary_output", binaryOutput) != ge::GRAPH_SUCCESS) {
        return FAILED;
    }

    std::vector<ge::Operator> inputs;
    for (uint32_t i = 0U; i < INPUT_NAMES.size(); ++i) {
        auto data = ge::op::Data(INPUT_NAMES[i]).set_attr_index(i);
        const auto desc = op.GetInputDesc(i);
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        inputs.push_back(data);
    }

    const bool useNative = weightsType == ge::DT_FLOAT;
    auto result = ge::OperatorFactory::CreateOperator(name.GetString(),
                                                      useNative ? "RaggedBinCount" : "RaggedBincount");
    if (result.GetInputsSize() != INPUT_NAMES.size() || result.GetOutputsSize() != 1U) {
        return FAILED;
    }
    for (uint32_t i = 0U; i < INPUT_NAMES.size(); ++i) {
        if (!useNative || i != SIZE_INPUT_INDEX) {
            result.SetInput(INPUT_NAMES[i], inputs[i]);
        }
        result.UpdateInputDesc(INPUT_NAMES[i], op.GetInputDesc(i));
    }
    result.SetAttr("binary_output", binaryOutput);
    result.UpdateOutputDesc("output", op.GetOutputDesc(0U));

    if (useNative) {
        // TF size is scalar; the native GE interface expects [1]. A real
        // Reshape preserves the value dependency and the INT32/INT64 payload.
        const int64_t vectorShape = 1;
        const ge::TensorDesc shapeDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
        const ge::Tensor shapeValue(shapeDesc, reinterpret_cast<const uint8_t*>(&vectorShape), sizeof(vectorShape));
        auto shape = ge::op::Const("size_vector_shape").set_attr_value(shapeValue);
        shape.update_output_desc_y(shapeDesc);

        auto reshape = ge::OperatorFactory::CreateOperator("size_to_vector", "Reshape");
        if (reshape.GetInputsSize() != 2U || reshape.GetOutputsSize() != 1U) {
            return FAILED;
        }
        reshape.SetInput("x", inputs[SIZE_INPUT_INDEX]);
        reshape.SetInput("shape", shape);
        reshape.UpdateInputDesc("x", op.GetInputDesc(SIZE_INPUT_INDEX));
        reshape.UpdateInputDesc("shape", shapeDesc);
        auto sizeDesc = op.GetInputDesc(SIZE_INPUT_INDEX);
        sizeDesc.SetShape(ge::Shape({1}));
        sizeDesc.SetOriginShape(ge::Shape({1}));
        sizeDesc.SetDataType(indexType);
        reshape.UpdateOutputDesc("y", sizeDesc);
        result.SetInput("size", reshape);
        result.UpdateInputDesc("size", sizeDesc);
    }

    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs = {{result, {0U}}};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}
} // namespace

REGISTER_CUSTOM_OP("RaggedBinCount")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RaggedBincount")
    .ParseParamsByOperatorFn(ParseParamsRaggedBinCount)
    .ParseOpToGraphFn(ParseOpToGraphRaggedBinCount)
    .ImplyType(ImplyType::TVM);
} // namespace domi
