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
