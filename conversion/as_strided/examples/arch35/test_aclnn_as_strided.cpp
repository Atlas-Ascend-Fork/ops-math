/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_as_strided.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize(); // SetDevice 失败时回收 aclInit 已申请的全局资源
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId); // 回收已设置的 device
        aclFinalize();
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr); // 回收本函数已申请的 device 内存，避免失败路径泄漏
        *deviceAddr = nullptr;
        return ret;
    }

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2.构造输入与输出，需要根据API的接口自定义构造
    // x 为源数据张量；按 size={2,3}、stride={4,2}、storageOffset=1 物化生成输出，
    // 即 y[i0,i1] = x[1 + i0*4 + i1*2]
    std::vector<int64_t> xShape = {16};
    std::vector<int64_t> yShape = {2, 3};
    std::vector<int64_t> sizeValue = {2, 3};
    std::vector<int64_t> strideValue = {4, 2};
    int64_t storageOffset = 1;
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclIntArray* size = nullptr;
    aclIntArray* stride = nullptr;
    aclOpExecutor* executor = nullptr;
    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> yHostData(GetShapeSize(yShape), 0);
    int retCode = 0;

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    if (ret != ACL_SUCCESS) {
        retCode = ret;
        goto cleanup;
    }
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    if (ret != ACL_SUCCESS) {
        retCode = ret;
        goto cleanup;
    }
    // 创建size/stride aclIntArray
    size = aclCreateIntArray(sizeValue.data(), sizeValue.size());
    CHECK_RET(size != nullptr, retCode = ACL_ERROR_INVALID_PARAM; goto cleanup);
    stride = aclCreateIntArray(strideValue.data(), strideValue.size());
    CHECK_RET(stride != nullptr, retCode = ACL_ERROR_INVALID_PARAM; goto cleanup);

    // 3.调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnAsStrided第一段接口
    ret = aclnnAsStridedGetWorkspaceSize(x, size, stride, storageOffset, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnAsStridedGetWorkspaceSize failed. ERROR: %d\n", ret);
        retCode = ret;
        goto cleanup;
    }
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            retCode = ret;
            goto cleanup;
        }
    }
    // 调用aclnnAsStrided第二段接口
    ret = aclnnAsStrided(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnAsStrided failed. ERROR: %d\n", ret);
        retCode = ret;
        goto cleanup;
    }

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        retCode = ret;
        goto cleanup;
    }

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    {
        auto sizeOut = GetShapeSize(yShape);
        std::vector<float> resultData(sizeOut, 0);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                          sizeOut * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
            retCode = ret;
            goto cleanup;
        }
        for (int64_t i = 0; i < sizeOut; i++) {
            LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
        }
    }

cleanup:
    // 6.释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (size != nullptr) {
        aclDestroyIntArray(size);
    }
    if (stride != nullptr) {
        aclDestroyIntArray(stride);
    }

    // 7.释放Device资源，需要根据具体API的接口定义修改（正常与失败路径统一回收）
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return retCode;
}
