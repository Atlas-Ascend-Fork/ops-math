/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_select_top_k.h"

namespace {
constexpr int64_t ROW_COUNT = 2;
constexpr int64_t AXIS_LENGTH = 5;
constexpr int64_t OUTPUT_COUNT = 3;
constexpr int64_t LAST_AXIS = -1;
constexpr int64_t INDEX_FILL = -1;
using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

// 创建本示例的连续二维张量，并把主机数据复制到设备。
template <typename T>
int CreateTensor(const std::vector<T>& data, int64_t columns, aclDataType type, DeviceMemPtr& memory, TensorPtr& tensor)
{
    const size_t bytes = data.size() * sizeof(T);
    void* address = nullptr;
    auto status = aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        return status;
    }
    memory.reset(address);
    status = aclrtMemcpy(address, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) {
        return status;
    }
    const std::vector<int64_t> shape = {ROW_COUNT, columns};
    const std::vector<int64_t> strides = {columns, 1};
    tensor.reset(aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                 shape.size(), address));
    return tensor ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

// 将输出读回主机，与无重复值输入的确定结果逐项比较。
template <typename T>
int CheckOutput(void* memory, const std::vector<T>& expected)
{
    std::vector<T> actual(expected.size());
    const size_t bytes = actual.size() * sizeof(T);
    const auto status = aclrtMemcpy(actual.data(), bytes, memory, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (status != ACL_SUCCESS) {
        return status;
    }
    if (actual != expected) {
        std::printf("SelectTopK output mismatch.\n");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

// 演示全轴、双输出、INT64索引、零偏移的两段式调用。
int RunExample(aclrtStream stream)
{
    DeviceMemPtr inputMemory(nullptr, &aclrtFree);
    DeviceMemPtr valuesMemory(nullptr, &aclrtFree);
    DeviceMemPtr indicesMemory(nullptr, &aclrtFree);
    DeviceMemPtr workspace(nullptr, &aclrtFree);
    TensorPtr input(nullptr, &aclDestroyTensor);
    TensorPtr values(nullptr, &aclDestroyTensor);
    TensorPtr indices(nullptr, &aclDestroyTensor);
    const std::vector<float> data = {9, 2, 7, 1, 5, -3, 8, 4, 6, 0};
    auto status = CreateTensor(data, AXIS_LENGTH, ACL_FLOAT, inputMemory, input);
    if (status != ACL_SUCCESS) {
        return status;
    }
    status = CreateTensor(std::vector<float>(ROW_COUNT * OUTPUT_COUNT), OUTPUT_COUNT, ACL_FLOAT, valuesMemory, values);
    if (status != ACL_SUCCESS) {
        return status;
    }
    status = CreateTensor(std::vector<int64_t>(ROW_COUNT * OUTPUT_COUNT), OUTPUT_COUNT, ACL_INT64, indicesMemory,
                          indices);
    if (status != ACL_SUCCESS) {
        return status;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    status = aclnnSelectTopKGetWorkspaceSize(input.get(), OUTPUT_COUNT, LAST_AXIS, true, ACL_INT64,
                                             ACLNN_SELECT_TOP_K_ORDER_VALUE, 0, ACLNN_SELECT_TOP_K_OUTPUT_BOTH, nullptr,
                                             nullptr, nullptr, INDEX_FILL, values.get(), indices.get(), &workspaceSize,
                                             &executor);
    if (status != ACL_SUCCESS) {
        return status;
    }
    if (workspaceSize != 0) {
        void* address = nullptr;
        status = aclrtMalloc(&address, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            return status;
        }
        workspace.reset(address);
    }
    status = aclnnSelectTopK(workspace.get(), workspaceSize, executor, stream);
    // 即使任务提交返回错误，也先等待已提交任务结束，再析构张量及设备内存。
    const auto syncStatus = aclrtSynchronizeStream(stream);
    if (status != ACL_SUCCESS) {
        return status;
    }
    if (syncStatus != ACL_SUCCESS) {
        return syncStatus;
    }
    status = CheckOutput<float>(valuesMemory.get(), {9, 7, 5, 8, 6, 4});
    if (status != ACL_SUCCESS) {
        return status;
    }
    return CheckOutput<int64_t>(indicesMemory.get(), {0, 2, 4, 1, 3, 2});
}
} // namespace

// 初始化设备和流，执行示例后按相反顺序释放资源。
int main()
{
    constexpr int32_t DEVICE_ID = 0;
    auto status = aclInit(nullptr);
    if (status != ACL_SUCCESS) {
        return status;
    }
    bool deviceSet = false;
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(DEVICE_ID);
        }
        aclFinalize();
    });
    status = aclrtSetDevice(DEVICE_ID);
    if (status != ACL_SUCCESS) {
        return status;
    }
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    status = aclrtCreateStream(&rawStream);
    if (status != ACL_SUCCESS) {
        return status;
    }
    StreamPtr stream(rawStream, &aclrtDestroyStream);
    status = RunExample(stream.get());
    std::printf("SelectTopK example: %s (status=%d).\n", status == ACL_SUCCESS ? "PASS" : "FAIL", status);
    return status;
}
