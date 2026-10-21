/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0(the "License").
 * Please refer to the License for details.You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ge_api.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "../../op_graph/bessel_i1e_proto.h"

namespace {

constexpr int32_t SUCCESS = 0;
constexpr int32_t FAILED = -1;
constexpr float ATOL = 1.0e-4f;
constexpr float RTOL = 2.0e-4f;

std::string ShapeString(const std::vector<int64_t>& shape)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << (i == 0 ? "" : ",") << shape[i];
    }
    os << "]";
    return os.str();
}

size_t ElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

// Values cross the |x| = 3.75 polynomial branch boundary and include the
// special inputs NaN, +/-inf and +/-0 whose outputs are NaN, 0 and 0.
std::vector<float> MakeInput(const std::vector<int64_t>& shape)
{
    static const std::array<float, 14> values = {-1.0f,
                                                 -0.0f,
                                                 0.0f,
                                                 0.5f,
                                                 1.0f,
                                                 2.0f,
                                                 3.75f,
                                                 4.0f,
                                                 8.0f,
                                                 32.0f,
                                                 100.0f,
                                                 std::numeric_limits<float>::infinity(),
                                                 -std::numeric_limits<float>::infinity(),
                                                 std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> input(ElementCount(shape));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = values[i % values.size()];
    }
    return input;
}

ge::Tensor MakeTensor(const std::vector<int64_t>& shape, std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}

ge::Graph MakeGraph(const std::string& graphName, const std::vector<int64_t>& declaredShape)
{
    ge::Graph graph(graphName.c_str());
    auto data = ge::op::Data((graphName + "_data").c_str()).set_attr_index(0);
    ge::TensorDesc inputDesc(ge::Shape(declaredShape), ge::FORMAT_ND, ge::DT_FLOAT);
    data.update_input_desc_x(inputDesc);
    data.update_output_desc_y(inputDesc);
    auto besselI1e = ge::op::BesselI1e((graphName + "_bessel_i1e").c_str());
    besselI1e.set_input_x(data);
    besselI1e.update_input_desc_x(inputDesc);
    besselI1e.update_output_desc_y(inputDesc);
    graph.AddOp(data);
    graph.SetInputs(std::vector<ge::Operator>{data}).SetOutputs(std::vector<ge::Operator>{besselI1e});
    return graph;
}

bool ShapeEquals(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected)
{
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

// Independent reference (double-precision power series for |x| < 20 and the
// Abramowitz & Stegun 9.7.1 asymptotic expansion for |x| >= 20), not the
// kernel algorithm.
float Reference(float x)
{
    if (std::isnan(x)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (std::isinf(x)) {
        return 0.0f;
    }
    const double ax = std::fabs(static_cast<double>(x));
    const double sign = (x < 0.0f) ? -1.0 : 1.0;
    double i1e;
    if (ax < 20.0) {
        // I1(ax) = (ax/2) * sum_{k>=0} (ax^2/4)^k / (k! * (k+1)!)
        // Converges in ~34 terms at ax = 20, well inside the 60-iteration cap.
        double term = 1.0;
        double sum = 1.0;
        for (int k = 1; k <= 60; ++k) {
            term *= (ax * ax / 4.0) / (static_cast<double>(k) * static_cast<double>(k + 1));
            sum += term;
            if (term < 1e-17 * std::fabs(sum)) {
                break;
            }
        }
        i1e = (ax / 2.0) * sum * std::exp(-ax);
    } else {
        // i1e(x) ~ (1/sqrt(2*pi*x)) * (1 - 3/(8x) + 15/(128x^2) - 315/(3072x^3)
        //                                + 14175/(98304x^4))
        // Abramowitz & Stegun 9.7.1 with mu=4ν²=4: -(mu-1)/(8x) - (mu-1)(mu-9)/(2!(8x)²)
        // - (mu-1)(mu-9)(mu-25)/(3!(8x)³) - (mu-1)(mu-9)(mu-25)(mu-49)/(4!(8x)⁴)
        // Substituting mu=4: -3/(8x) + 15/(128x²) - 315/(3072x³) + 14175/(98304x⁴)
        // (signs alternate because (mu-9), (mu-25), (mu-49) are negative)
        // Truncation error <= ~6.0e-4 for x >= 20, inside the fp32 tolerance.
        const double inv = 1.0 / ax;
        const double poly = 1.0 +
                            inv * (-0.375 + inv * (0.1171875 + inv * (-0.1025390625 + inv * (14175.0 / 98304.0))));
        i1e = 0.3989422804014327 * std::sqrt(inv) * poly;
    }
    return static_cast<float>(sign * i1e);
}

bool Verify(const ge::Tensor& output, const std::vector<int64_t>& expectedShape, const std::vector<float>& input)
{
    const ge::TensorDesc& desc = output.GetTensorDesc();
    const std::vector<int64_t> actualShape = desc.GetShape().GetDims();
    if (desc.GetDataType() != ge::DT_FLOAT || !ShapeEquals(actualShape, expectedShape) ||
        ElementCount(actualShape) != input.size()) {
        std::cerr << "Output descriptor mismatch for " << ShapeString(expectedShape) << std::endl;
        return false;
    }
    const auto* actual = reinterpret_cast<const float*>(output.GetData());
    if (actual == nullptr) {
        std::cerr << "Output data is null" << std::endl;
        return false;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        const float expected = Reference(input[i]);
        const bool sameNan = std::isnan(expected) && std::isnan(actual[i]);
        const bool bothZero = (expected == 0.0f) && (actual[i] == 0.0f);
        const float limit = ATOL + RTOL * std::fabs(expected);
        if (!sameNan && !bothZero && !(std::isfinite(expected) && std::fabs(actual[i] - expected) <= limit)) {
            std::cerr << "Value mismatch at " << i << ": input=" << input[i] << ", actual=" << actual[i]
                      << ", expected=" << expected << std::endl;
            return false;
        }
    }
    std::cout << "Shape, dtype and values PASSED for " << ShapeString(expectedShape) << std::endl;
    return true;
}

// RunOne centralizes ge::Session::RunGraph and prints "Shape, dtype and values PASSED" only after
// full verification.
bool RunOne(ge::Session& session, uint32_t graphId, const std::vector<int64_t>& shape)
{
    std::vector<float> values = MakeInput(shape);
    ge::Tensor tensor = MakeTensor(shape, values);
    std::vector<ge::Tensor> inputs{tensor};
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graphId, inputs, outputs) != ge::SUCCESS || outputs.size() != 1U) {
        return false;
    }
    return Verify(outputs[0], shape, values);
}

// RunScenario runs one declared shape against several concrete shapes on the same graph/session.
bool RunScenario(const std::string& scenario, uint32_t graphId, const std::vector<int64_t>& declaredShape,
                 const std::vector<std::vector<int64_t>>& concreteShapes)
{
    std::cout << "Scenario " << scenario << ", declared shape " << ShapeString(declaredShape) << std::endl;
    const std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    ge::Graph graph = MakeGraph("bessel_i1e_" + scenario, declaredShape);
    const std::map<ge::AscendString, ge::AscendString> graphOptions = {
        {"ge.exec.dynamicGraphExecuteMode", "dynamic_execute"}};
    if (session.AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        return false;
    }
    size_t passed = 0;
    for (const auto& shape : concreteShapes) {
        std::cout << "Run concrete shape " << ShapeString(shape) << std::endl;
        passed += RunOne(session, graphId, shape) ? 1U : 0U;
    }
    std::cout << "Scenario " << scenario << " summary: " << passed << "/" << concreteShapes.size() << " passed"
              << std::endl;
    return passed == concreteShapes.size();
}

} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"},
                                                                        {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        return FAILED;
    }
    // -1: unknown dims (rank known); -2: unknown rank (dims resolved at runtime).
    const bool unknownDim = RunScenario("unknown_dim_minus_1", 0, {-1, -1}, {{4, 2}, {1, 8}, {3, 5}});
    const bool unknownRank = RunScenario("unknown_rank_minus_2", 1, {-2}, {{}, {8}, {4, 2}, {2, 3, 4}});
    const bool finalized = (ge::GEFinalize() == ge::SUCCESS);
    if (!unknownDim || !unknownRank || !finalized) {
        return FAILED;
    }
    std::cout << "BesselI1e dynamic GEIR verification PASSED (-1 and -2)" << std::endl;
    return SUCCESS;
}
