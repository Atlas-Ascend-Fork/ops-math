/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// End-to-end GE-IR example for Power on Ascend 950 (arch35).
//
// Power is a graph-only operator (no aclnn API). This example builds a
// one-node graph  Data -> Power -> y  and runs it through a GE session,
// where attributes power / scale / shift are injected on the Power node.
//
// Each case computes y = (scale * x + shift) ^ power element-wise and checks
// the device output against a pre-computed expected value. Cases cover the
// host-side culType branches documented in README.md:
//   LINEAR            power == 1
//   SQUARE            power == 2
//   CUBE              power == 3 (including negative base)
//   GENERIC_POW_POS   power > 0, power not in {1, 2, 3}
//   GENERIC_POW_NEG   power < 0
//   BROADCAST_SCALAR  power == 0, or scale == 0 with power != 0
// plus the default-attribute identity mapping, and dtypes fp32 / fp16 / bf16.
//
// The example exits with status 0 only if every case passes.

#include <iostream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/power_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t t;
    time(&t);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return buf;
}

static uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    return 1;
}

// IEEE-754 fp32 -> half. Normal range uses round-to-nearest-even (ties to
// even, with sticky bits); subnormal inputs/results flush to signed zero.
static uint16_t FloatToHalfBits(float v)
{
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp32 = (x >> 23) & 0xff;
    uint32_t mant = x & 0x7fffff;
    uint16_t out;
    if (exp32 == 0xff) {
        out = (uint16_t)((sign << 15) | 0x7c00 | (mant ? 0x200 : 0));
    } else if (exp32 == 0) {
        out = (uint16_t)(sign << 15);
    } else {
        int32_t newExp = (int32_t)exp32 - 127 + 15;
        if (newExp >= 31) {
            out = (uint16_t)((sign << 15) | 0x7c00);
        } else if (newExp <= 0) {
            out = (uint16_t)(sign << 15);
        } else {
            uint32_t roundBit = (mant >> 12) & 0x1;
            uint32_t stickyBits = mant & 0xfff;
            uint16_t m = (uint16_t)(mant >> 13);
            out = (uint16_t)((sign << 15) | (newExp << 10) | m);
            if (roundBit && (stickyBits != 0 || (out & 0x1) != 0))
                out++;
        }
    }
    return out;
}

static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp16 = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp16 == 0) {
        out = sign << 31;
    } else if (exp16 == 31) {
        out = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        uint32_t newExp = exp16 - 15 + 127;
        out = (sign << 31) | (newExp << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

// fp32 -> bfloat16 (round-to-nearest-even on the truncated 16 low bits).
static uint16_t FloatToBfloat16Bits(float v)
{
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    uint32_t lsb = (x >> 16) & 0x1;
    uint32_t rounding = 0x7fff + lsb;
    return (uint16_t)((x + rounding) >> 16);
}

static float Bfloat16BitsToFloat(uint16_t h)
{
    uint32_t x = (uint32_t)h << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// Fill a tensor with a single value of the given dtype. The Tensor ctor
// deep-copies the buffer, so a local std::vector keeps the memory managed.
static int32_t GenConstTensor(const vector<int64_t>& shape, DataType dtype, double value, Tensor& tensor,
                              TensorDesc& desc)
{
    desc.SetRealDimCnt(shape.size());
    size_t numel = 1;
    for (auto d : shape)
        numel *= (size_t)d;
    size_t bytes = numel * GetDataTypeSize(dtype);

    if (dtype == ge::DT_FLOAT) {
        std::vector<float> buf(numel, (float)value);
        tensor = Tensor(desc, reinterpret_cast<const uint8_t*>(buf.data()), bytes);
    } else if (dtype == ge::DT_FLOAT16) {
        uint16_t v = FloatToHalfBits((float)value);
        std::vector<uint16_t> buf(numel, v);
        tensor = Tensor(desc, reinterpret_cast<const uint8_t*>(buf.data()), bytes);
    } else if (dtype == ge::DT_BF16) {
        uint16_t v = FloatToBfloat16Bits((float)value);
        std::vector<uint16_t> buf(numel, v);
        tensor = Tensor(desc, reinterpret_cast<const uint8_t*>(buf.data()), bytes);
    } else {
        return FAILED;
    }
    return SUCCESS;
}

struct CaseSpec {
    string tag;
    DataType dtype;
    vector<int64_t> shape;
    double inVal;
    bool setAttrs;
    float power;
    float scale;
    float shift;
    double expected;
    double absTol;
};

// Build a graph computing y = (scale * x + shift) ^ power with one input.
static int BuildPowerGraph(const CaseSpec& spec, Graph& graph, std::vector<ge::Tensor>& inputData,
                           std::vector<Operator>& inputs, std::vector<Operator>& outputs)
{
    auto op = ge::op::Power("power_" + spec.tag);
    if (spec.setAttrs) {
        op.set_attr_power(spec.power);
        op.set_attr_scale(spec.scale);
        op.set_attr_shift(spec.shift);
    }

    auto ph = ge::op::Data("ph_x_" + spec.tag).set_attr_index(0);
    TensorDesc d(ge::Shape(spec.shape), FORMAT_ND, spec.dtype);
    d.SetPlacement(ge::kPlacementHost);
    d.SetFormat(FORMAT_ND);
    Tensor t;
    if (GenConstTensor(spec.shape, spec.dtype, spec.inVal, t, d) != SUCCESS) {
        printf("Generate input data failed\n");
        return FAILED;
    }
    ph.update_input_desc_x(d);
    ph.update_output_desc_y(d);
    inputData.push_back(t);
    graph.AddOp(ph);
    op.set_input_x(ph);
    inputs.push_back(ph);

    TensorDesc out(ge::Shape(spec.shape), FORMAT_ND, spec.dtype);
    op.update_output_desc_y(out);
    outputs.push_back(op);
    return SUCCESS;
}

static double ReadOutElement(const uint8_t* raw, DataType dtype, int64_t idx)
{
    if (dtype == ge::DT_FLOAT) {
        return (double)reinterpret_cast<const float*>(raw)[idx];
    }
    if (dtype == ge::DT_FLOAT16) {
        return (double)HalfBitsToFloat(reinterpret_cast<const uint16_t*>(raw)[idx]);
    }
    return (double)Bfloat16BitsToFloat(reinterpret_cast<const uint16_t*>(raw)[idx]);
}

static int RunOneCase(const CaseSpec& spec)
{
    printf("\n%s - INFO - [XIR]: ===== case %s start =====\n", GetTime().c_str(), spec.tag.c_str());
    Graph graph(("g_" + spec.tag).c_str());
    std::vector<ge::Tensor> inputData;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;
    if (BuildPowerGraph(spec, graph, inputData, inputs, outputs) != SUCCESS) {
        printf("BuildPowerGraph failed\n");
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    // Dump the graph so the dispatched node/kernel can be verified afterwards.
    // Dump failure only costs evidence, it does not invalidate the value check.
    string dumpPath = "./dump_power_" + spec.tag;
    if (aclgrphDumpGraph(graph, dumpPath.c_str(), dumpPath.length()) != ge::GRAPH_SUCCESS) {
        printf("  WARNING: aclgrphDumpGraph failed for case %s (dump evidence unavailable)\n", spec.tag.c_str());
    }

    std::map<AscendString, AscendString> buildOpts;
    ge::Session* session = new (std::nothrow) Session(buildOpts);
    if (session == nullptr) {
        printf("Create session failed\n");
        return FAILED;
    }

    std::map<AscendString, AscendString> graphOpts;
    uint32_t graphId = 1;
    if (session->AddGraph(graphId, graph, graphOpts) != SUCCESS) {
        printf("AddGraph failed\n");
        delete session;
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    Status ret = session->RunGraph(graphId, inputData, output);
    if (ret != SUCCESS) {
        printf("RunGraph failed\n");
        delete session;
        return FAILED;
    }

    int64_t expNumel = 1;
    for (auto d : spec.shape)
        expNumel *= d;

    int failCount = 0;
    if (output.size() != outputs.size()) {
        printf("  OUTPUT COUNT MISMATCH: got %zu expected %zu, result unverified\n", output.size(), outputs.size());
        failCount++;
    }
    for (size_t i = 0; i < output.size(); ++i) {
        int64_t numel = output[i].GetTensorDesc().GetShape().GetShapeSize();
        if (numel != expNumel) {
            printf("  SHAPE MISMATCH: output numel=%ld expected=%ld\n", (long)numel, (long)expNumel);
            failCount++;
        }
        uint8_t* raw = output[i].GetData();
        for (int64_t j = 0; j < numel; ++j) {
            double got = ReadOutElement(raw, spec.dtype, j);
            double diff = std::abs(got - spec.expected);
            if (diff > spec.absTol) {
                printf("  MISMATCH @%ld: got=%.6f expected=%.6f diff=%.6f\n", (long)j, got, spec.expected, diff);
                failCount++;
            }
        }
    }
    delete session;

    if (failCount != 0) {
        printf("%s - FAIL - case %s: %d mismatches (numel=%ld)\n", GetTime().c_str(), spec.tag.c_str(), failCount,
               (long)expNumel);
        return FAILED;
    }
    printf("%s - PASS - case %s (expected=%.4f)\n", GetTime().c_str(), spec.tag.c_str(), spec.expected);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::map<AscendString, AscendString> globalOpts = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOpts) != SUCCESS) {
        printf("GEInitialize failed\n");
        return FAILED;
    }

    std::vector<CaseSpec> cases = {
        // tag                    dtype            shape   inVal  setAttrs power scale shift expected absTol
        // LINEAR: y = scale * x + shift
        {"fp32_linear", ge::DT_FLOAT, {4}, 2.0, true, 1.0f, 2.0f, 1.0f, 5.0, 1e-5},
        // default attrs (power=1, scale=1, shift=0): identity mapping
        {"fp32_default_attrs", ge::DT_FLOAT, {4}, 2.5, false, 0.0f, 0.0f, 0.0f, 2.5, 1e-5},
        // SQUARE: y = x^2, 2-D shape
        {"fp32_square", ge::DT_FLOAT, {2, 2}, 3.0, true, 2.0f, 1.0f, 0.0f, 9.0, 1e-5},
        // CUBE with negative base: y = (-2)^3 = -8
        {"fp32_cube_neg_base", ge::DT_FLOAT, {4}, -2.0, true, 3.0f, 1.0f, 0.0f, -8.0, 1e-5},
        // GENERIC_POW_POS: y = 4^0.5 = 2
        {"fp32_generic_pos", ge::DT_FLOAT, {4}, 4.0, true, 0.5f, 1.0f, 0.0f, 2.0, 1e-5},
        // GENERIC_POW_NEG: y = 2^-1 = 0.5
        {"fp32_generic_neg", ge::DT_FLOAT, {4}, 2.0, true, -1.0f, 1.0f, 0.0f, 0.5, 1e-5},
        // BROADCAST_SCALAR: power == 0 -> y = 1
        {"fp32_power_zero", ge::DT_FLOAT, {4}, 7.0, true, 0.0f, 1.0f, 0.0f, 1.0, 1e-6},
        // BROADCAST_SCALAR: scale == 0 -> y = shift^power = 3^2 = 9
        {"fp32_scale_zero", ge::DT_FLOAT, {2, 2}, 7.0, true, 2.0f, 0.0f, 3.0f, 9.0, 1e-5},
        // scalar input handled as shape [1]
        {"fp32_scalar_shape", ge::DT_FLOAT, {1}, 5.0, true, 2.0f, 1.0f, 0.0f, 25.0, 1e-5},
        // fp16 SQUARE: y = 2^2 = 4
        {"fp16_square", ge::DT_FLOAT16, {4}, 2.0, true, 2.0f, 1.0f, 0.0f, 4.0, 1e-2},
        // fp16 GENERIC_POW_POS: y = 4^0.5 = 2
        {"fp16_generic_pos", ge::DT_FLOAT16, {2}, 4.0, true, 0.5f, 1.0f, 0.0f, 2.0, 1e-2},
        // bf16 SQUARE: y = 3^2 = 9
        {"bf16_square", ge::DT_BF16, {4}, 3.0, true, 2.0f, 1.0f, 0.0f, 9.0, 5e-2},
        // bf16 LINEAR: y = 2 * 1.5 + 0.5 = 3.5
        {"bf16_linear", ge::DT_BF16, {4}, 1.5, true, 1.0f, 2.0f, 0.5f, 3.5, 2e-2},
    };

    int allFail = 0;
    for (const auto& c : cases) {
        if (RunOneCase(c) != SUCCESS) {
            allFail++;
        }
    }

    ge::AscendString errMsg = ge::GEGetErrorMsgV2();
    const char* errCStr = errMsg.GetString();
    if (errCStr != nullptr && errCStr[0] != '\0') {
        std::cout << "GE error: " << errCStr << std::endl;
    }

    ge::AscendString warnMsg = ge::GEGetWarningMsgV2();
    const char* warnCStr = warnMsg.GetString();
    if (warnCStr != nullptr && warnCStr[0] != '\0') {
        std::cout << "GE warning: " << warnCStr << std::endl;
    }

    ge::GEFinalize();

    if (allFail != 0) {
        printf("\n%s - FAIL - [XIR]: %d / %zu cases failed\n", GetTime().c_str(), allFail, cases.size());
        return FAILED;
    }
    printf("\n%s - PASS - [XIR]: all %zu power cases passed\n", GetTime().c_str(), cases.size());
    return SUCCESS;
}
