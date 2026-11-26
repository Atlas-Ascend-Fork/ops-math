# aclnnHypot

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Description

- API function: Computes the length of the hypotenuse (two-dimensional Euclidean norm) of the right triangle corresponding to the two input tensors element by element.
- Formula:

  $$
  out_i = \sqrt{self_i^{2} + other_i^{2}}
  $$

- Calculation principle: The operator result is $\sqrt{self_i^{2} + other_i^{2}}$, and it must not be distorted by the overflow or underflow of the intermediate results $self_i^{2}$ or $other_i^{2}$. This operator uses a stable transformed computation based on the operand with the larger absolute value:

  $$
  out_i =
  \begin{cases}
  |self_i| \cdot \sqrt{1 + \left( \dfrac{other_i}{self_i} \right)^{2}}, & |self_i| \ge |other_i| \\
  |other_i| \cdot \sqrt{1 + \left( \dfrac{self_i}{other_i} \right)^{2}}, & |self_i| \lt |other_i|
  \end{cases}
  $$

  Here, $self_i$ and $other_i$ are the values of the inputs at the i-th element, $out_i$ is the value of the output at the i-th element, and when $self_i = other_i = 0$, $out_i$ is handled as 0.

## Prototype

Each operator is split into a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnHypotGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnHypot" API to perform the computation.

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

- **Parameters:**

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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>
        <ul>
          <li>Supports empty tensors.</li>
          <li>The data types of self and other must comply with the data type deduction rules (see the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>
        <ul>
          <li>Supports empty tensors.</li>
          <li>The data types of self and other must comply with the data type deduction rules (see the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>
        <ul>
          <li>Supports empty tensors.</li>
          <li>The data type must be convertible from the data type deduced from self and other (see the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li>
          <li>The shape must be the shape resulting from the broadcast of self and other.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value:**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API validates the input parameters and reports an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of self or other is not within the supported range.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed on self and other, or the deduction result is not within the supported range.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The number of shape dimensions of self or other is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and other do not satisfy the broadcast rules.</td>
    </tr>
  </tbody>
  </table>

## aclnnHypot

- **Parameters:**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device. This value is obtained by the first-phase API aclnnHypotGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value:**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism: aclnnHypot defaults to a deterministic implementation.
- Data type constraint: self, other, and out support only FLOAT, FLOAT16, and BFLOAT16.
- Data type deduction constraint: The data types of self and other are allowed to be different, and the API deduces the computation type based on the [Deduction Relationship](../../../docs/en/context/deduction_relationship.md).
- Shape and broadcast constraint: The shapes of self and other must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md), and the output shape is obtained by taking the larger value of the two inputs dimension by dimension.
- Dimension constraint: The input supports 0- to 8-dimensional ND tensors, and a 0-dimensional scalar participates in broadcasting.
- Empty tensor constraint: Empty tensors are supported. The output is an empty tensor of the broadcast shape, and no computation is performed and no error is reported.
- Non-contiguous tensor constraint: self, other, and out support [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md).
- Numeric range and overflow constraint: A stable algorithm based on the larger magnitude is used to avoid premature overflow or underflow caused by squaring during the computation. The output is inf only when the mathematical result itself exceeds the representable range of the output dtype.
- Special value constraint: The output is always non-negative. The output semantics for various input value combinations are as follows:

  | self | other | output out | Description |
  |----|----|-------|------|
  | 0 / ±0 | 0 / ±0 | +0 | When both inputs are zero, positive zero is output, including the case where both inputs are signed zeros. |
  | ±0 | non-zero value (including ±inf) | abs(other) | Degenerates to the absolute value of the other input. |
  | non-zero value (including ±inf) | ±0 | abs(self) | Degenerates to the absolute value of the other input. |
  | ±inf | any value (including NaN) | +inf | inf takes precedence over NaN; inf is still output when the other input is NaN. |
  | NaN | finite value | NaN | NaN propagates to the output when no inf input is present. |
  | finite value | finite value | sqrt(self^2 + other^2) | Computed using the stable algorithm, and the result is always ≥ 0. |

- Precision constraint: For FLOAT16 and BFLOAT16 inputs, the computation is performed in FLOAT32 in the Kernel and then converted back to the original dtype.

## Examples

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

For the complete compilable and runnable sample, see the `examples/test_aclnn_hypot.cpp` file in the operator directory. The core code is as follows:

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
