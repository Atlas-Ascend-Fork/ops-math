/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "aclnn_fused_exponential.h"

namespace {
class Resources {
public:
    bool initialized = false;
    bool deviceSet = false;
    int32_t device = 0;
    aclrtStream stream = nullptr;
    void* inputData = nullptr;
    void* outputData = nullptr;
    void* workspace = nullptr;
    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    aclScalar* rate = nullptr;
    aclOpExecutor* executor = nullptr;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        if (rate != nullptr) {
            aclDestroyScalar(rate);
        }
        if (input != nullptr) {
            aclDestroyTensor(input);
        }
        if (output != nullptr) {
            aclDestroyTensor(output);
        }
        for (auto address : {workspace, outputData, inputData}) {
            if (address != nullptr) {
                aclrtFree(address);
            }
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(device);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

bool Check(int status, const char* operation)
{
    if (status == 0) {
        return true;
    }
    std::cerr << operation << " failed: " << status << '\n';
    return false;
}

int Run()
{
    Resources resources;
    const char* deviceText = std::getenv("DEVICE");
    if (deviceText != nullptr) {
        char* end = nullptr;
        errno = 0;
        const long device = std::strtol(deviceText, &end, 10);
        if (errno != 0 || end == deviceText || *end != '\0' || device < 0 ||
            device > std::numeric_limits<int32_t>::max()) {
            std::cerr << "DEVICE must be a nonnegative logical device index\n";
            return 1;
        }
        resources.device = static_cast<int32_t>(device);
    }
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!Check(aclrtSetDevice(resources.device), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }
    // Non-aligned length exercises the final vector tile; include the small-u boundary.
    const int64_t length = 65;
    const int64_t stride = 1;
    std::vector<float> input(length);
    std::vector<float> output(length);
    for (int64_t i = 0; i < length; ++i) {
        input[i] = static_cast<float>(i) / static_cast<float>(length);
    }
    const size_t bytes = input.size() * sizeof(float);
    if (!Check(aclrtMalloc(&resources.inputData, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "input allocation") ||
        !Check(aclrtMalloc(&resources.outputData, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "output allocation") ||
        !Check(aclrtMemcpy(resources.inputData, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "input copy")) {
        return 1;
    }
    resources.input = aclCreateTensor(&length, 1, ACL_FLOAT, &stride, 0, ACL_FORMAT_ND, &length, 1,
                                      resources.inputData);
    resources.output = aclCreateTensor(&length, 1, ACL_FLOAT, &stride, 0, ACL_FORMAT_ND, &length, 1,
                                       resources.outputData);
    double rate = 2.0;
    resources.rate = aclCreateScalar(&rate, ACL_DOUBLE);
    if (resources.input == nullptr || resources.output == nullptr || resources.rate == nullptr) {
        std::cerr << "Tensor/scalar descriptor creation failed\n";
        return 1;
    }
    uint64_t workspaceSize = 0;
    if (!Check(aclnnFusedExponentialGetWorkspaceSize(resources.input, resources.rate, resources.output, &workspaceSize,
                                                     &resources.executor),
               "aclnnFusedExponentialGetWorkspaceSize")) {
        return 1;
    }
    // Explicit ownership permits cleanup on allocation or submission failure.
    if (!Check(aclSetAclOpExecutorRepeatable(resources.executor), "executor ownership")) {
        return 1;
    }
    if (workspaceSize != 0 &&
        !Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation")) {
        return 1;
    }
    if (!Check(aclnnFusedExponential(resources.workspace, workspaceSize, resources.executor, resources.stream),
               "execution") ||
        !Check(aclrtSynchronizeStream(resources.stream), "stream synchronization") ||
        !Check(aclrtMemcpy(output.data(), bytes, resources.outputData, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
               "output copy")) {
        return 1;
    }
    const double epsilon = std::numeric_limits<float>::epsilon() / 2.0;
    const double tolerance = 1.0 / 8192.0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double value = input[i] < epsilon ? epsilon : input[i];
        const double expected = -std::log1p(-value) / rate;
        if (!std::isfinite(output[i]) || std::abs(output[i] - expected) > tolerance * (1.0 + std::abs(expected))) {
            std::cerr << "Mismatch at " << i << ": actual=" << output[i] << ", expected=" << expected << '\n';
            return 1;
        }
    }
    std::cout << "FusedExponential example passed: " << length << " values, workspace=" << workspaceSize << '\n';
    return 0;
}
} // namespace

int main() { return Run(); }
