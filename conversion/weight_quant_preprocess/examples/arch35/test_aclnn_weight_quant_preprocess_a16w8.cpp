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
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

// 设备侧 Tensor 持有者：析构时按创建逆序自动释放 tensor 与 device 内存
class DeviceTensor {
public:
    DeviceTensor(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
                 const std::vector<int64_t>& strides, aclDataType dtype, aclFormat format, double bytesPerElem)
    {
        int64_t storageSize = GetShapeSize(storageShape);
        bytes_ = static_cast<int64_t>(storageSize * bytesPerElem);

        std::vector<int8_t> hostData(bytes_, 0);
        auto ret = aclrtMalloc(&deviceAddr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc device memory failed" << std::endl; return);
        ret = aclrtMemcpy(deviceAddr_, bytes_, hostData.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy H2D failed" << std::endl; return);
        const int64_t* stridesData = strides.empty() ? nullptr : strides.data();
        tensor_ = aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stridesData, 0, format,
                                  storageShape.data(), storageShape.size(), deviceAddr_);
        CHECK_RET(tensor_ != nullptr, std::cout << "Create tensor failed" << std::endl; return);
    }

    ~DeviceTensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
            tensor_ = nullptr;
        }
        if (deviceAddr_ != nullptr) {
            aclrtFree(deviceAddr_);
            deviceAddr_ = nullptr;
        }
    }

    aclTensor* Get() const { return tensor_; }
    bool IsValid() const { return tensor_ != nullptr && deviceAddr_ != nullptr; }

private:
    aclTensor* tensor_ = nullptr;
    void* deviceAddr_ = nullptr;
    int64_t bytes_ = 0;
};

int RunPreprocess(aclTensor* weight, aclTensor* weightScale, aclTensor* weightOffset, aclDataType xDtype,
                  int64_t kGroupSize, aclTensor* outWeight, aclTensor* outWeightScale, aclTensor* outWeightOffset,
                  aclTensor* bias, aclTensor* outBias, aclrtStream stream)
{
    // A16W8 数据流无 xScale 语义，xScaleDtype 固定传 ACL_DT_UNDEFINED
    aclDataType xScaleDtype = ACL_DT_UNDEFINED;

    // 1. 获取 workspace 与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, weightOffset, bias, xDtype, xScaleDtype,
                                                          kGroupSize, outWeight, outWeightScale, outWeightOffset,
                                                          outBias, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed, ret=" << ret << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. 执行计算
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed, ret=" << ret << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);
    return ACL_SUCCESS;
}

// ===== 场景一：MM_A16INT8_PERTENSOR（INT8 pertensor，非转置 weight，ND 直拷，offset/bias 透传） =====
int TestA16INT8PerTensorNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: INT8，非转置连续 [K, N]；1 字节元素无打包约束
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: pertensor 单元素 {1}，FP16；pertensor 场景 kGroupSize 必须为 0
    DeviceTensor weightScale({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // weightOffsetOptional: A16INT8 数据流支持透传，要求与 weightScale 同形同 dtype
    DeviceTensor weightOffset({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightOffset.IsValid(), return ACL_ERROR_FAILURE);

    // bias: {N}，FP16，contiguous
    DeviceTensor bias({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(bias.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷，format 为 ND，viewShape/storageShape 与 weight 相同
    DeviceTensor outWeight({k, n}, {k, n}, {}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outWeightOffset({1}, {1}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightOffset.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outBias({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outBias.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), weightOffset.Get(), ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), outWeightOffset.Get(), bias.Get(), outBias.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16INT8PerTensorNd failed" << std::endl; return ret);
    std::cout << "TestA16INT8PerTensorNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景二：MM_A16INT8_PERTENSOR（INT8 pertensor，转置 weight，ND 直拷） =====
int TestA16INT8PerTensorTransNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;

    // weight: INT8，转置 [K, N]（viewShape {K,N}，storage {N,K}，strides [1,K]）
    DeviceTensor weight({k, n}, {n, k}, {1, k}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: pertensor 单元素 {1,1}，FP16
    DeviceTensor weightScale({1, 1}, {1, 1}, {1, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷，与 weight 保持相同 viewShape/strides（转置视图）
    DeviceTensor outWeight({k, n}, {n, k}, {1, k}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({1, 1}, {1, 1}, {1, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16INT8PerTensorTransNd failed" << std::endl; return ret);
    std::cout << "TestA16INT8PerTensorTransNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景三：MM_A16INT8_PERCHANNEL（INT8 perchannel，非转置 weight，ND 直拷） =====
int TestA16INT8PerChannelNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: INT8，非转置连续 [K, N]
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {N}，FP16；perchannel 场景 kGroupSize 必须为 0
    DeviceTensor weightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // weightOffsetOptional: 与 weightScale 同形同 dtype 透传
    DeviceTensor weightOffset({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightOffset.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷，与 weight 保持相同 viewShape/strides
    DeviceTensor outWeight({k, n}, {k, n}, {n, 1}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);
    DeviceTensor outWeightOffset({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightOffset.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), weightOffset.Get(), ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), outWeightOffset.Get(), nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16INT8PerChannelNd failed" << std::endl; return ret);
    std::cout << "TestA16INT8PerChannelNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景四：MM_A16INT8_PERCHANNEL（INT8 perchannel，转置 weight，ND 直拷） =====
int TestA16INT8PerChannelTransNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;

    // weight: INT8，转置 [K, N]
    DeviceTensor weight({k, n}, {n, k}, {1, k}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {1, N}，FP16
    DeviceTensor weightScale({1, n}, {1, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷（转置视图透传）
    DeviceTensor outWeight({k, n}, {n, k}, {1, k}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({1, n}, {1, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16INT8PerChannelTransNd failed" << std::endl; return ret);
    std::cout << "TestA16INT8PerChannelTransNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景五：MM_A16INT8_PERGROUP（INT8 pergroup，非转置 weight，ND 直拷） =====
int TestA16INT8PerGroupNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;
    int64_t kGroupSize = 64;
    int64_t g = CEIL_DIV(k, kGroupSize);

    // weight: INT8，非转置连续 [K, N]
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: pergroup {ceildiv(K, kGroupSize), N}，FP16
    DeviceTensor weightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷
    DeviceTensor outWeight({k, n}, {k, n}, {}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_FLOAT16, kGroupSize, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16INT8PerGroupNd failed" << std::endl; return ret);
    std::cout << "TestA16INT8PerGroupNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景六：MM_A16INT8_PERGROUP（INT8 pergroup，转置 weight，ND 直拷） =====
int TestA16INT8PerGroupTransNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;
    int64_t kGroupSize = 64;
    int64_t g = CEIL_DIV(k, kGroupSize);

    // weight: INT8，转置 [K, N]
    DeviceTensor weight({k, n}, {n, k}, {1, k}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: pergroup {G, N}，FP16
    DeviceTensor weightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷（转置视图透传）
    DeviceTensor outWeight({k, n}, {n, k}, {1, k}, ACL_INT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_FLOAT16, kGroupSize, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16INT8PerGroupTransNd failed" << std::endl; return ret);
    std::cout << "TestA16INT8PerGroupTransNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景七：MM_A16FP8HIF8_PERCHANNEL（FP8_E4M3FN perchannel，非转置 weight，ND 直拷） =====
int TestA16FP8HIF8PerChannelE4M3Nd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 256;

    // weight: FLOAT8_E4M3FN，非转置连续 [K, N]
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {N}，FP16；FP8 场景 kGroupSize 必须为 0，offset 必须为 nullptr
    DeviceTensor weightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // bias: {N}，FP16
    DeviceTensor bias({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(bias.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷
    DeviceTensor outWeight({k, n}, {k, n}, {}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outBias({n}, {n}, {1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outBias.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), nullptr, bias.Get(), outBias.Get(), stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16FP8HIF8PerChannelE4M3Nd failed" << std::endl; return ret);
    std::cout << "TestA16FP8HIF8PerChannelE4M3Nd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景八：MM_A16FP8HIF8_PERCHANNEL（FP8_E4M3FN perchannel，转置 weight，ND 直拷） =====
int TestA16FP8HIF8PerChannelE4M3TransNd(aclrtStream stream)
{
    int64_t k = 256;
    int64_t n = 128;

    // weight: FLOAT8_E4M3FN，转置 [K, N]
    DeviceTensor weight({k, n}, {n, k}, {1, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {1, N}，FP16
    DeviceTensor weightScale({1, n}, {1, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷（转置视图透传）
    DeviceTensor outWeight({k, n}, {n, k}, {1, k}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({1, n}, {1, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_FLOAT16, 0, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16FP8HIF8PerChannelE4M3TransNd failed" << std::endl; return ret);
    std::cout << "TestA16FP8HIF8PerChannelE4M3TransNd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景九：MM_A16FP8HIF8_PERCHANNEL（HIFLOAT8 perchannel，非转置 weight，ND 直拷） =====
int TestA16FP8HIF8PerChannelHif8Nd(aclrtStream stream)
{
    int64_t k = 128;
    int64_t n = 128;

    // weight: HIFLOAT8，非转置连续 [K, N]
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_HIFLOAT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {N}，BF16（x 为 BF16）
    DeviceTensor weightScale({n}, {n}, {1}, ACL_BF16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷
    DeviceTensor outWeight({k, n}, {k, n}, {}, ACL_HIFLOAT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({n}, {n}, {1}, ACL_BF16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_BF16, 0, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16FP8HIF8PerChannelHif8Nd failed" << std::endl; return ret);
    std::cout << "TestA16FP8HIF8PerChannelHif8Nd success" << std::endl;
    return ACL_SUCCESS;
}

// ===== 场景十：MM_A16FP8HIF8_PERCHANNEL（HIFLOAT8 perchannel，转置 weight，ND 直拷） =====
int TestA16FP8HIF8PerChannelHif8TransNd(aclrtStream stream)
{
    int64_t k = 128;
    int64_t n = 128;

    // weight: HIFLOAT8，转置 [K, N]
    DeviceTensor weight({k, n}, {n, k}, {1, k}, ACL_HIFLOAT8, ACL_FORMAT_ND, 1);
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: perchannel {N}，BF16（x 为 BF16）
    DeviceTensor weightScale({n}, {n}, {1}, ACL_BF16, ACL_FORMAT_ND, 2);
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷（转置视图透传）
    DeviceTensor outWeight({k, n}, {n, k}, {1, k}, ACL_HIFLOAT8, ACL_FORMAT_ND, 1);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({n}, {n}, {1}, ACL_BF16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), nullptr, ACL_BF16, 0, outWeight.Get(),
                             outWeightScale.Get(), nullptr, nullptr, nullptr, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA16FP8HIF8PerChannelHif8TransNd failed" << std::endl; return ret);
    std::cout << "TestA16FP8HIF8PerChannelHif8TransNd success" << std::endl;
    return ACL_SUCCESS;
}

// WeightQuantPreprocess A16W8 currently supports Ascend 950 only.
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    // A16INT8：pertensor / perchannel / pergroup × 转置/非转置（6 场景）
    ret = TestA16INT8PerTensorNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16INT8PerTensorTransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16INT8PerChannelNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16INT8PerChannelTransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16INT8PerGroupNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16INT8PerGroupTransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // A16FP8HIF8：E4M3FN / HIFLOAT8 perchannel × 转置/非转置（4 场景）
    ret = TestA16FP8HIF8PerChannelE4M3Nd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16FP8HIF8PerChannelE4M3TransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16FP8HIF8PerChannelHif8Nd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA16FP8HIF8PerChannelHif8TransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::cout << "All A16W8 examples run success" << std::endl;
    return 0;
}
