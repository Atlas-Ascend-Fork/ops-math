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
 * \file test_geir_histogram_fusion.cpp
 * \brief 老 Histogram(attr min/max) GEIR 端到端验证: 经 HistogramFusionPass 融合为
 *        HistogramV2 后的 CUDA 语义对齐检查。覆盖: auto-range(min==max 数据重算)、
 *        常数数据 kernel nextafter 展开兜底、min<max 透传、M3a 超值域标量提升、
 *        静态空输入。编译运行见 ops-math build.sh --run_example histogram_v2 graph。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "tensor.h"
#include "ge_api.h"
// 老 Histogram proto(attr min/max, 全 dtype 集)来自 vendor 包 op_proto/inc
#include "histogram_proto.h"

// 建图用的 op::Data: 套用 CANN 头同款宏守卫, 避免与 proto 头内注册重复定义
#ifndef OPS_PROTO_DEF_DATA
#define OPS_PROTO_DEF_DATA
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge
#endif

using namespace ge;

namespace {
int g_pass = 0;
int g_fail = 0;

// host 侧 bf16 位模式包装: RNE 截断 float->bf16, 读回零扩展移位
struct Bf16Host {
    uint16_t bits;
    static Bf16Host FromFloat(float f)
    {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        u += 0x7FFFu + ((u >> 16) & 1u); // round-to-nearest-even
        return Bf16Host{static_cast<uint16_t>(u >> 16)};
    }
    float ToFloat() const
    {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
    operator float() const { return ToFloat(); }
    operator double() const { return static_cast<double>(ToFloat()); }
};

// 浮点: nextafter 邻接值与 ±1 取更远者(与 kernel MinMaxExpander<float> 逐点一致)
void ExpandEqual(float& lo, float& hi)
{
    float m = lo;
    float down = std::nextafterf(m, -INFINITY);
    float up = std::nextafterf(m, INFINITY);
    lo = std::min(down, m - 1.0f);
    hi = std::max(up, m + 1.0f);
}

// 整数: int64 裸 ±1(与 kernel 整数路径一致)
void ExpandEqual(int64_t& lo, int64_t& hi)
{
    lo -= 1;
    hi += 1;
}

template <typename T>
void CountInto(std::vector<int64_t>& counts, const std::vector<T>& data, int64_t bins, double minV, double maxV)
{
    for (T v : data) {
        double dv = static_cast<double>(v);
        if (dv >= minV && dv <= maxV) {
            int64_t idx = static_cast<int64_t>((dv - minV) * bins / (maxV - minV));
            if (idx == bins) {
                idx = bins - 1;
            }
            counts[idx]++;
        }
    }
}

// CPU golden: 与 kernel 同口径(仅 [min,max] 内计数, idx=trunc((v-min)*bins/(max-min)),
// idx==bins 收敛末 bin); min==max 时先 aminmax 重算, 仍相等则按 CPU #163506 展开。
template <typename T, typename CT>
std::vector<int64_t> GoldenHist(const std::vector<T>& data, int64_t bins, double minV, double maxV)
{
    std::vector<int64_t> counts(bins, 0);
    if (minV == maxV) {
        if (data.empty()) {
            return counts; // 空输入: 全零(CUDA 语义)
        }
        CT lo = static_cast<CT>(data[0]);
        CT hi = lo;
        for (T v : data) {
            lo = std::min(lo, static_cast<CT>(v));
            hi = std::max(hi, static_cast<CT>(v));
        }
        if (lo == hi) {
            ExpandEqual(lo, hi);
        }
        minV = static_cast<double>(lo);
        maxV = static_cast<double>(hi);
    }
    CountInto(counts, data, bins, minV, maxV);
    return counts;
}

// 读回输出 tensor 并统一转成 int64 计数(输出 desc dtype 由 GE infer 决定)
bool ReadCounts(const Tensor& out, std::vector<int64_t>& counts)
{
    auto desc = out.GetTensorDesc();
    int64_t numel = desc.GetShape().GetShapeSize();
    DataType dt = desc.GetDataType();
    const uint8_t* raw = out.GetData();
    counts.assign(numel, 0);
    if (dt == DT_INT32) {
        const int32_t* p = reinterpret_cast<const int32_t*>(raw);
        for (int64_t i = 0; i < numel; i++)
            counts[i] = p[i];
    } else if (dt == DT_INT64) {
        const int64_t* p = reinterpret_cast<const int64_t*>(raw);
        for (int64_t i = 0; i < numel; i++)
            counts[i] = p[i];
    } else if (dt == DT_FLOAT) {
        const float* p = reinterpret_cast<const float*>(raw);
        for (int64_t i = 0; i < numel; i++)
            counts[i] = static_cast<int64_t>(p[i]);
    } else if (dt == DT_INT8) {
        const int8_t* p = reinterpret_cast<const int8_t*>(raw);
        for (int64_t i = 0; i < numel; i++)
            counts[i] = p[i];
    } else if (dt == DT_BF16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
        for (int64_t i = 0; i < numel; i++)
            counts[i] = static_cast<int64_t>(Bf16Host{p[i]}.ToFloat());
    } else {
        printf("  [ERROR] unexpected output dtype %d\n", static_cast<int>(dt));
        return false;
    }
    std::cout << "  output dtype: " << static_cast<int>(dt) << std::endl;
    return true;
}

template <typename T, typename CT>
void RunCase(Session* session, uint32_t graphId, const char* name, DataType dtype, int64_t bins, float minAttr,
             float maxAttr, const std::vector<int64_t>& shape, const std::vector<T>& data)
{
    std::cout << "[CASE] " << name << std::endl;
    Graph graph(name);

    TensorDesc xDesc(ge::Shape(shape), FORMAT_ND, dtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    auto x = op::Data("x").set_attr_index(0);
    x.update_input_desc_x(xDesc);
    x.update_output_desc_y(xDesc);

    auto hist = op::Histogram("hist");
    hist.set_input_x(x);
    hist.set_attr_bins(bins);
    hist.set_attr_min(minAttr);
    hist.set_attr_max(maxAttr);

    std::vector<Operator> graphInputs = {x};
    std::vector<Operator> graphOutputs = {hist};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    Status ret = session->AddGraph(graphId, graph, std::map<AscendString, AscendString>());
    if (ret != SUCCESS) {
        printf("  [FAIL] AddGraph failed (ret=%u)\n", ret);
        g_fail++;
        return;
    }

    Tensor xTensor(xDesc, reinterpret_cast<uint8_t*>(const_cast<T*>(data.data())), data.size() * sizeof(T));
    std::vector<Tensor> outputs;
    ret = session->RunGraph(graphId, {xTensor}, outputs);
    if (ret != SUCCESS) {
        printf("  [FAIL] RunGraph failed (ret=%u)\n", ret);
        g_fail++;
        return;
    }

    std::vector<int64_t> actual;
    if (outputs.empty() || !ReadCounts(outputs[0], actual)) {
        printf("  [FAIL] read output failed\n");
        g_fail++;
        return;
    }
    std::vector<int64_t> expect = GoldenHist<T, CT>(data, bins, minAttr, maxAttr);

    bool ok = (actual == expect);
    std::cout << "  expect: [";
    for (auto v : expect)
        std::cout << v << ",";
    std::cout << "]" << std::endl;
    std::cout << "  actual: [";
    for (auto v : actual)
        std::cout << v << ",";
    std::cout << "]" << std::endl;
    printf("  [%s]\n", ok ? "PASS" : "FAIL");
    ok ? g_pass++ : g_fail++;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("[ERROR] GEInitialize failed\n");
        return -1;
    }
    Session* session = new Session(std::map<AscendString, AscendString>());
    uint32_t graphId = 0;

    // 1. auto-range fp32 非常数数据: min==max=0 -> ReduceMin/Max 重算
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = static_cast<float>((i * 7) % 100);
        RunCase<float, float>(session, graphId++, "autorange_fp32", DT_FLOAT, 10, 0.0f, 0.0f, {4, 8}, data);
    }
    // 2. auto-range 常数数据: reduce 后仍 min==max -> kernel nextafter 展开兜底
    {
        std::vector<float> data(5, 7.0f);
        RunCase<float, float>(session, graphId++, "autorange_const_fp32", DT_FLOAT, 4, 0.0f, 0.0f, {5}, data);
    }
    // 3. auto-range int32
    {
        std::vector<int32_t> data(32);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = static_cast<int32_t>(i % 20);
        RunCase<int32_t, int64_t>(session, graphId++, "autorange_int32", DT_INT32, 6, 0.0f, 0.0f, {4, 8}, data);
    }
    // 4. auto-range int64
    {
        std::vector<int64_t> data(32);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = static_cast<int64_t>(i % 20);
        RunCase<int64_t, int64_t>(session, graphId++, "autorange_int64", DT_INT64, 6, 0.0f, 0.0f, {4, 8}, data);
    }
    // 5. min<max fp32 正常透传
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = static_cast<float>((i * 7) % 100);
        RunCase<float, float>(session, graphId++, "range_fp32", DT_FLOAT, 10, 0.0f, 100.0f, {4, 8}, data);
    }
    // 6. M3a: int8 + 超值域标量 ±1000 -> Cast 提升 int32
    {
        std::vector<int8_t> data(32);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = static_cast<int8_t>(i % 100);
        RunCase<int8_t, int64_t>(session, graphId++, "widen_int8", DT_INT8, 3, -1000.0f, 1000.0f, {4, 8}, data);
    }
    // 7. 静态空输入 auto-range: 输出全零(CUDA 语义)
    {
        std::vector<float> data;
        RunCase<float, float>(session, graphId++, "empty_fp32", DT_FLOAT, 4, 0.0f, 0.0f, {0, 8}, data);
    }
    // 8. bf16 auto-range 常数数据: reduce 后仍 min==max -> kernel float 域 nextafter 展开
    {
        std::vector<Bf16Host> data(5, Bf16Host::FromFloat(5.0f));
        RunCase<Bf16Host, float>(session, graphId++, "autorange_const_bf16", DT_BF16, 4, 0.0f, 0.0f, {5}, data);
    }
    // 9. bf16 常规 min<max 透传(输出经 fp32 中间 Cast 回 bf16)
    {
        std::vector<Bf16Host> data(32);
        for (size_t i = 0; i < data.size(); i++)
            data[i] = Bf16Host::FromFloat(static_cast<float>((i * 7) % 100));
        RunCase<Bf16Host, float>(session, graphId++, "range_bf16", DT_BF16, 10, 0.0f, 100.0f, {4, 8}, data);
    }

    printf("[SUMMARY] PASS=%d FAIL=%d\n", g_pass, g_fail);
    delete session;
    ge::GEFinalize();
    return g_fail == 0 ? 0 : -1;
}
