/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
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

uint16_t FloatToFp16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
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
    // bytesPerElem=0.5 表示 4-bit 紧凑打包（FP4，每字节 2 个值），内存按 numel/2 字节申请；
    // hostData 非空时以其实际内容初始化 device 内存，否则零填充
    DeviceTensor(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
                 const std::vector<int64_t>& strides, aclDataType dtype, aclFormat format, double bytesPerElem,
                 const void* hostData = nullptr)
    {
        int64_t storageSize = GetShapeSize(storageShape);
        bytes_ = static_cast<int64_t>(storageSize * bytesPerElem);

        std::vector<uint8_t> zeroData;
        const void* src = hostData;
        if (src == nullptr) {
            zeroData.assign(bytes_, 0);
            src = zeroData.data();
        }
        auto ret = aclrtMalloc(&deviceAddr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc device memory failed" << std::endl; return);
        ret = aclrtMemcpy(deviceAddr_, bytes_, src, bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
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

    int CopyToHost(void* dst, int64_t bytes) const
    {
        auto ret = aclrtMemcpy(dst, bytes, deviceAddr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy D2H failed" << std::endl; return ret);
        return ACL_SUCCESS;
    }

private:
    aclTensor* tensor_ = nullptr;
    void* deviceAddr_ = nullptr;
    int64_t bytes_ = 0;
};

int RunPreprocess(aclTensor* weight, aclTensor* weightScale, int64_t kGroupSize, aclTensor* outWeight,
                  aclTensor* outWeightScale, aclrtStream stream)
{
    // A8F4 T-CG：x 固定 FP8_E4M3FN；preprocess 无 xScale 语义，xScaleDtype 固定传 ACL_DT_UNDEFINED
    aclDataType xDtype = ACL_FLOAT8_E4M3FN;
    aclDataType xScaleDtype = ACL_DT_UNDEFINED;

    // 1. 获取 workspace 与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, nullptr, // weightOffsetOptional
                                                          nullptr,                      // biasOptional
                                                          xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale,
                                                          nullptr, // outWeightOffsetOptional
                                                          nullptr, // outBiasOptional
                                                          &workspaceSize, &executor);
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

// 与期望 buffer 逐字节比对（ND 直拷 weight / scale 为物理透传，要求 bit-wise 一致）
bool VerifyBitWise(const char* tag, const DeviceTensor& out, const void* expected, int64_t bytes)
{
    std::vector<uint8_t> actual(bytes);
    auto ret = out.CopyToHost(actual.data(), bytes);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    if (std::memcmp(actual.data(), expected, bytes) != 0) {
        std::cout << tag << " bit-wise compare FAIL" << std::endl;
        return false;
    }
    std::cout << tag << " bit-wise compare PASS" << std::endl;
    return true;
}

// NZ_C0_32 紧凑 4-bit 物理布局校验（非转置 N 块在前）：storage 为 {ceildiv(N,32), ceildiv(K,16), 16, 32}，
// nibble 线性下标 p = ((n1 * kBlocks + k1) * 16 + k0) * 32 + n0，低 nibble 为偶数 p
bool VerifyNzC032(const char* tag, const DeviceTensor& out, const std::vector<uint8_t>& weightNibbles, int64_t k,
                  int64_t n)
{
    int64_t kBlocks = CEIL_DIV(k, 16);
    int64_t nBlocks = CEIL_DIV(n, 32);
    std::vector<uint8_t> expected(kBlocks * 16 * nBlocks * 32 / 2, 0);
    for (int64_t n1 = 0; n1 < nBlocks; n1++) {
        for (int64_t k1 = 0; k1 < kBlocks; k1++) {
            for (int64_t k0 = 0; k0 < 16; k0++) {
                for (int64_t n0 = 0; n0 < 32; n0++) {
                    int64_t kk = k1 * 16 + k0;
                    int64_t nn = n1 * 32 + n0;
                    uint8_t nib = (kk < k && nn < n) ? weightNibbles[kk * n + nn] : 0;
                    int64_t p = ((n1 * kBlocks + k1) * 16 + k0) * 32 + n0;
                    expected[p / 2] |= static_cast<uint8_t>(nib << ((p % 2) * 4));
                }
            }
        }
    }
    return VerifyBitWise(tag, out, expected.data(), kBlocks * 16 * nBlocks * 32 / 2);
}

// 公共输入：逻辑 weight {K,N} nibbles、非转置物理打包（沿 N，低 nibble = 偶数 n）
void MakeWeight(int64_t k, int64_t n, std::vector<uint8_t>& weightNibbles, std::vector<uint8_t>& weightPacked)
{
    weightNibbles.resize(k * n);
    for (int64_t i = 0; i < k * n; i++) {
        weightNibbles[i] = static_cast<uint8_t>((i * 7 + 3) % 15);
    }
    weightPacked.assign(k * n / 2, 0);
    for (int64_t kk = 0; kk < k; kk++) {
        for (int64_t nn = 0; nn + 1 < n; nn += 2) {
            weightPacked[(kk * n + nn) / 2] = (weightNibbles[kk * n + nn + 1] << 4) | weightNibbles[kk * n + nn];
        }
    }
}

// per-group FP16 scale：{ceildiv(K, kGroupSize), N}，连续
void MakeScaleFp16(int64_t g, int64_t n, std::vector<uint16_t>& scale)
{
    scale.resize(g * n);
    for (int64_t i = 0; i < g * n; i++) {
        scale[i] = FloatToFp16(0.5f + 0.5f * static_cast<float>(i % 4));
    }
}

// 场景一：MM_A8F4_TCG 转置（末两维严格转置 stride [1, K] -> ND 直拷透传，out 与入同 view shape/strides）
int TestA8F4TcgTransNd(aclrtStream stream)
{
    int64_t k = 64;
    int64_t n = 128;
    int64_t kGroupSize = 32; // T-CG 场景 kGroupSize 固定为 32
    int64_t g = CEIL_DIV(k, kGroupSize);

    std::vector<uint8_t> weightNibbles, weightPackedNonTrans;
    MakeWeight(k, n, weightNibbles, weightPackedNonTrans);

    // 转置 weight：view {K, N} strides {1, K}，物理 {N, K/2} 沿 K 打包（低 nibble = 偶数 k），K 须为偶数
    std::vector<uint8_t> weightPacked(k * n / 2, 0);
    for (int64_t nn = 0; nn < n; nn++) {
        for (int64_t kk = 0; kk + 1 < k; kk += 2) {
            weightPacked[(nn * k + kk) / 2] = (weightNibbles[(kk + 1) * n + nn] << 4) | weightNibbles[kk * n + nn];
        }
    }
    DeviceTensor weight({k, n}, {n, k}, {1, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5, weightPacked.data());
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    // weightScale: per-group 2-D {ceildiv(K, 32), N}（分组数须大于 1），FP16
    std::vector<uint16_t> scale;
    MakeScaleFp16(g, n, scale);
    DeviceTensor weightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2, scale.data());
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: ND 直拷，view shape/strides/storageShape 与输入 weight 完全一致（纯透传）
    DeviceTensor outWeight({k, n}, {n, k}, {1, k}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), kGroupSize, outWeight.Get(), outWeightScale.Get(),
                             stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA8F4TcgTransNd failed" << std::endl; return ret);

    // ND 直拷为物理透传：outWeight/outWeightScale 与输入 buffer 逐字节一致
    bool ok = VerifyBitWise("  outWeight(ND direct copy)", outWeight, weightPacked.data(), k * n / 2);
    ok = VerifyBitWise("  outWeightScale", outWeightScale, scale.data(), g * n * 2) && ok;
    CHECK_RET(ok, std::cout << "TestA8F4TcgTransNd verify failed" << std::endl; return ACL_ERROR_FAILURE);
    std::cout << "TestA8F4TcgTransNd success" << std::endl;
    return ACL_SUCCESS;
}

// 场景二：MM_A8F4_TCG 非转置（连续 [K, N] -> FRACTAL_NZ_C0_32 输出，N 块在前，scale 直拷透传）
int TestA8F4TcgNonTransNz(aclrtStream stream)
{
    int64_t k = 64;
    int64_t n = 128;
    int64_t kGroupSize = 32;
    int64_t g = CEIL_DIV(k, kGroupSize);

    // weight: FLOAT4_E2M1，非转置连续 [K, N]；紧凑 4-bit 打包维（N）须为偶数
    std::vector<uint8_t> weightNibbles, weightPacked;
    MakeWeight(k, n, weightNibbles, weightPacked);
    DeviceTensor weight({k, n}, {k, n}, {n, 1}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, 0.5, weightPacked.data());
    CHECK_RET(weight.IsValid(), return ACL_ERROR_FAILURE);

    std::vector<uint16_t> scale;
    MakeScaleFp16(g, n, scale);
    DeviceTensor weightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2, scale.data());
    CHECK_RET(weightScale.IsValid(), return ACL_ERROR_FAILURE);

    // outWeight: viewShape 与 weight 相同；NZ_C0_32 非转置为 N 块在前
    // {ceildiv(N, 32), ceildiv(K, 16), 16, 32}（区别于 MX A8W4 转置的 K 块在前）
    std::vector<int64_t> outStorageShape = {CEIL_DIV(n, 32), CEIL_DIV(k, 16), 16, 32};
    DeviceTensor outWeight({k, n}, outStorageShape, {}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, 0.5);
    CHECK_RET(outWeight.IsValid(), return ACL_ERROR_FAILURE);

    DeviceTensor outWeightScale({g, n}, {g, n}, {n, 1}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    CHECK_RET(outWeightScale.IsValid(), return ACL_ERROR_FAILURE);

    auto ret = RunPreprocess(weight.Get(), weightScale.Get(), kGroupSize, outWeight.Get(), outWeightScale.Get(),
                             stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "TestA8F4TcgNonTransNz failed" << std::endl; return ret);

    bool ok = VerifyNzC032("  outWeight(NZ_C0_32)", outWeight, weightNibbles, k, n);
    ok = VerifyBitWise("  outWeightScale", outWeightScale, scale.data(), g * n * 2) && ok;
    CHECK_RET(ok, std::cout << "TestA8F4TcgNonTransNz verify failed" << std::endl; return ACL_ERROR_FAILURE);
    std::cout << "TestA8F4TcgNonTransNz success" << std::endl;
    return ACL_SUCCESS;
}

// WeightQuantPreprocess A8F4 T-CG currently supports Ascend 950 only.
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    ret = TestA8F4TcgTransNd(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = TestA8F4TcgNonTransNz(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::cout << "All A8F4 T-CG examples run success" << std::endl;
    return 0;
}
