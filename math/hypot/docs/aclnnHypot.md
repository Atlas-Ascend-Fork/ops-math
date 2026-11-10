# aclnnHypot

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：逐元素计算两个输入张量所对应的直角三角形斜边长度（二维欧几里得范数）。
- 计算公式：

  $$
  out_i = \sqrt{self_i^{2} + other_i^{2}}
  $$

- 计算原理：算子结果为$\sqrt{self_i^{2} + other_i^{2}}$，且不得因中间结果$self_i^{2}$、$other_i^{2}$的上溢或下溢而失真。本算子采用以绝对值较大一方为基准的稳定变形计算：

  $$
  out_i =
  \begin{cases}
  |self_i| \cdot \sqrt{1 + \left( \dfrac{other_i}{self_i} \right)^{2}}, & |self_i| \ge |other_i| \\
  |other_i| \cdot \sqrt{1 + \left( \dfrac{self_i}{other_i} \right)^{2}}, & |self_i| \lt |other_i|
  \end{cases}
  $$

  其中$self_i$、$other_i$为输入在第$i$个元素位置的取值，$out_i$为输出在第$i$个元素位置的取值，当$self_i = other_i = 0$时按$out_i = 0$处理。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnHypotGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnHypot”接口执行计算。

```Cpp
aclnnStatus aclnnHypotGetWorkspaceSize(
  const aclTensor*     self,
  const aclTensor*     other,
  aclTensor*           out,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnHypot(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnHypotGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的self。</td>
      <td>
        <ul>
          <li>支持空Tensor。</li>
          <li>数据类型与other的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</li>
          <li>shape需要与other满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的other。</td>
      <td>
        <ul>
          <li>支持空Tensor。</li>
          <li>数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</li>
          <li>shape需要与self满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>
        <ul>
          <li>支持空Tensor。</li>
          <li>数据类型需要是self与other推导之后可转换的数据类型（参见<a href="../../../docs/zh/context/deduction_relationship.md" target="_blank">互推导关系</a>）。</li>
          <li>shape需要是self与other broadcast之后的shape。</li>
        </ul>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、other或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>self或other的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self和other无法做数据类型推导，或推导结果不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为指定输出out的类型。</td>
    </tr>
    <tr>
      <td>self或other的shape维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self和other的shape不满足broadcast规则。</td>
    </tr>
  </tbody>
  </table>

## aclnnHypot

- **参数说明：**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnHypotGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnHypot默认确定性实现。
- 数据类型约束：self、other、out仅支持FLOAT、FLOAT16、BFLOAT16。
- 数据类型推导约束：self与other的数据类型允许不一致，接口按[互推导关系](../../../docs/zh/context/deduction_relationship.md)推导出计算类型。
- shape与broadcast约束：self与other的shape必须满足[broadcast关系](../../../docs/zh/context/broadcast_relationship.md)，输出shape为逐维取两个输入该维较大值的结果。
- 维度约束：输入支持0-8维ND Tensor，0维标量参与广播。
- 空Tensor约束：支持空Tensor，输出为广播后shape的空Tensor，不执行计算且不报错。
- 非连续Tensor约束：self、other、out均支持[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)。
- 数值范围与溢出约束：采用以较大量级为基准的稳定算法，避免计算过程中因平方提前上溢或下溢；仅当数学结果本身超出输出dtype的可表示范围时输出inf。
- 特殊值约束：输出恒为非负值，各输入取值组合的输出语义如下。

  | self | other | 输出out | 说明 |
  |----|----|-------|------|
  | 0 / ±0 | 0 / ±0 | +0 | 两个输入均为零时输出正零，包括两个输入均为带符号零的情形。 |
  | ±0 | 非零值（含±inf） | abs(other) | 退化为另一输入的绝对值。 |
  | 非零值（含±inf） | ±0 | abs(self) | 退化为另一输入的绝对值。 |
  | ±inf | 任意值（含NaN） | +inf | inf优先于NaN，另一输入为NaN时仍输出inf。 |
  | NaN | 有限值 | NaN | 无inf输入时NaN向输出传播。 |
  | 有限值 | 有限值 | sqrt(self^2 + other^2) | 按稳定算法计算，结果恒≥0。 |

- 精度约束：FLOAT16、BFLOAT16输入在Kernel内以FLOAT32计算后回转原dtype。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

完整可编译运行示例见算子目录 `examples/test_aclnn_hypot.cpp`。核心代码如下：

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_aclnn_hypot.cpp
 * aclnnHypot two-stage call example (eager mode).
 *
 * Covered scenarios (all FLOAT32):
 *   1) basic same-shape elementwise:  out = sqrt(self^2 + other^2)
 *   2) broadcast:                     self (3,1) x other (1,4) -> out (3,4)
 *   3) special values:                inf / NaN / signed zero / 1e30 stability demonstration
 *
 * Usage: ./test_aclnn_hypot [deviceId]   (default deviceId = 0)
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_hypot.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    // Fixed ACL resource initialization sequence
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                           aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Allocate device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Copy host data to device
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Strides of a contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// Golden reference: double-precision std::hypot.
static double GoldenHypot(float a, float b) { return std::hypot(static_cast<double>(a), static_cast<double>(b)); }

// Map a linear index in the broadcast output back to the linear index of one input,
// following numpy broadcast rules (input dims of size 1 contribute offset 0).
static int64_t BroadcastLinearIndex(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                                    int64_t outLinear)
{
    size_t outRank = outShape.size();
    size_t inRank = inShape.size();
    std::vector<int64_t> outStrides(outRank, 1);
    for (int64_t d = static_cast<int64_t>(outRank) - 2; d >= 0; --d) {
        outStrides[d] = outStrides[d + 1] * outShape[d + 1];
    }
    std::vector<int64_t> inStrides(inRank, 1);
    for (int64_t d = static_cast<int64_t>(inRank) - 2; d >= 0; --d) {
        inStrides[d] = inStrides[d + 1] * inShape[d + 1];
    }
    size_t pad = outRank - inRank; // input is left-padded with 1s
    int64_t inLinear = 0;
    for (size_t od = 0; od < outRank; ++od) {
        int64_t coord = (outStrides[od] > 0) ? (outLinear / outStrides[od]) % outShape[od] : 0;
        if (od >= pad) {
            size_t id = od - pad;
            if (inShape[id] != 1) {
                inLinear += coord * inStrides[id];
            }
        }
    }
    return inLinear;
}

// Verify one element against golden; fp32 output tolerance: |a-g| <= atol + rtol*|g|.
static bool CheckValue(float actual, double golden, double rtol, double atol, std::string* reason)
{
    if (std::isnan(golden)) {
        if (std::isnan(actual)) {
            return true;
        }
        *reason = "expect NaN";
        return false;
    }
    if (std::isnan(actual)) {
        *reason = "got NaN, expect finite/inf";
        return false;
    }
    if (std::isinf(golden)) {
        if (std::isinf(actual) && (std::signbit(actual) == std::signbit(golden))) {
            return true;
        }
        *reason = "expect same-sign inf";
        return false;
    }
    if (std::isinf(actual)) {
        *reason = "got inf, expect finite";
        return false;
    }
    double err = std::fabs(static_cast<double>(actual) - golden);
    if (err <= atol + rtol * std::fabs(golden)) {
        return true;
    }
    *reason = "abs err " + std::to_string(err) + " exceeds tolerance";
    return false;
}

// Run one aclnnHypot scenario end to end (two-stage call) and verify against golden.
// Broadcast semantics: out[i] pairs self/other by numpy broadcast over selfShape/otherShape.
static bool RunScenario(const char* name, const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                        const std::vector<int64_t>& outShape, const std::vector<float>& selfData,
                        const std::vector<float>& otherData, aclrtStream stream)
{
    LOG_PRINT("------------------------------------------------------------\n");
    LOG_PRINT("[scenario] %s\n", name);

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> outData(GetShapeSize(outShape), 0.0F);

    int ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create self tensor failed\n"); return false);
    ret = CreateAclTensor(otherData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create other tensor failed\n"); return false);
    ret = CreateAclTensor(outData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create out tensor failed\n"); return false);

    // Stage 1: query workspace size and obtain the executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHypotGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnHypotGetWorkspaceSize failed. ERROR: %d\n", ret);
        return false;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return false);
    }
    // Stage 2: launch the computation (executor is consumed by this call)
    ret = aclnnHypot(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnHypot failed. ERROR: %d\n", ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return false);

    // Copy result back to host
    auto outSize = GetShapeSize(outShape);
    std::vector<float> resultData(outSize, 0.0F);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr, outSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return false);

    // Broadcast-aware golden comparison
    const double rtol = 1e-4;
    const double atol = 1e-5;
    bool pass = true;
    for (int64_t i = 0; i < outSize; i++) {
        int64_t selfIdx = BroadcastLinearIndex(selfShape, outShape, i);
        int64_t otherIdx = BroadcastLinearIndex(otherShape, outShape, i);
        double golden = GoldenHypot(selfData[selfIdx], otherData[otherIdx]);
        std::string reason;
        bool ok = CheckValue(resultData[i], golden, rtol, atol, &reason);
        if (!ok) {
            pass = false;
        }
        LOG_PRINT("  out[%ld] = %.8g (golden %.8g) %s%s\n", i, resultData[i], golden,
                  ok ? "OK" : "MISMATCH: ", ok ? "" : reason.c_str());
    }

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    LOG_PRINT("[scenario] %s : %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char* argv[])
{
    int32_t deviceId = 0;
    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }
    LOG_PRINT("=== aclnnHypot example (two-stage aclnn call, FLOAT32) ===\n");
    LOG_PRINT("deviceId: %d\n", deviceId);

    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    bool allPass = true;

    // Scenario 1: basic same-shape elementwise (Pythagorean triples for easy eyeballing)
    allPass &= RunScenario("basic same-shape (4,2)", {4, 2}, {4, 2}, {4, 2}, {3, 4, 5, 12, 8, 15, 7, 24},
                           {4, 3, 12, 5, 15, 8, 24, 7}, stream);

    // Scenario 2: broadcast (3,1) x (1,4) -> (3,4)
    allPass &= RunScenario("broadcast (3,1)x(1,4)->(3,4)", {3, 1}, {1, 4}, {3, 4}, {3, 5, 9}, {4, 12, 40, 7}, stream);

    // Scenario 3: special values: inf priority over NaN, NaN propagation, signed zero,
    // 1e30 numerical stability (no overflow of intermediate squares)
    const float kInf = std::numeric_limits<float>::infinity();
    const float kQnan = std::numeric_limits<float>::quiet_NaN();
    allPass &= RunScenario("special values inf/NaN/zero/1e30", {8}, {8}, {8},
                           {kInf, kQnan, 0.0F, -0.0F, 1.0e30F, -3.0F, kInf, -kInf},
                           {kQnan, 2.0F, -0.0F, 5.0F, 1.0e30F, -4.0F, -7.0F, kQnan}, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("------------------------------------------------------------\n");
    LOG_PRINT("aclnnHypot example overall: %s\n", allPass ? "PASS" : "FAIL");
    return allPass ? 0 : 1;
}
```
