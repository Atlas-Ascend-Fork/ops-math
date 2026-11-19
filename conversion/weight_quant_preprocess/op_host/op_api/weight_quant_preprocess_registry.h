/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WEIGHT_QUANT_PREPROCESS_REGISTRY_H_
#define WEIGHT_QUANT_PREPROCESS_REGISTRY_H_

#include "opdev/common_types.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#define QUANT_DATA_FLOW_LIST        \
    X(UNKNOWN, 0)                   \
    X(MM_MX_A8W4, 1)                \
    X(GMM_MX_A8W4, 2)               \
    X(MM_A16S4_PERCHANNEL, 3)       \
    X(MM_A16S4_PERGROUP, 4)         \
    X(MM_A16S4_PERTENSOR, 5)        \
    X(MM_A16F4_PERGROUP, 6)         \
    X(MM_A16MXF4, 7)                \
    X(MM_MX_A16HIF4, 8)             \
    X(MM_C064_A16W4, 9)             \
    X(MM_C064_PerG_A16W4, 10)       \
    X(MM_A16INT8_PERTENSOR, 11)     \
    X(MM_A16INT8_PERCHANNEL, 12)    \
    X(MM_A16INT8_PERGROUP, 13)      \
    X(MM_A16FP8HIF8_PERCHANNEL, 14) \
    X(MM_A8F4_TCG, 15)              \
    X(MM_MX_A8HIF4, 16)

enum class QuantDataFlow : uint32_t {
#define X(name, value) name = value,
    QUANT_DATA_FLOW_LIST
#undef X
};

inline const char* QuantDataFlowToString(QuantDataFlow flow)
{
    static const char* QUANT_DATA_FLOW_NAMES[] = {
#define X(name, value) #name,
        QUANT_DATA_FLOW_LIST
#undef X
    };
    uint32_t idx = static_cast<uint32_t>(flow);
    if (idx < sizeof(QUANT_DATA_FLOW_NAMES) / sizeof(QUANT_DATA_FLOW_NAMES[0])) {
        return QUANT_DATA_FLOW_NAMES[idx];
    }
    return "INVALID";
}

struct QuantContext {
    const aclTensor* weight = nullptr;
    const aclTensor* weightScale = nullptr;
    const aclTensor* weightOffsetOptional = nullptr;
    const aclTensor* biasOptional = nullptr;
    const op::DataType xDtype = op::DataType::DT_UNDEFINED;
    const op::DataType xScaleDtype = op::DataType::DT_UNDEFINED;
    const int64_t kGroupSize = -1;
    NpuArch npuArch = NpuArch::DAV_RESV;
    QuantDataFlow dataFlow = QuantDataFlow::UNKNOWN;
    aclTensor* outWeight = nullptr;
    aclTensor* outWeightScale = nullptr;
    aclTensor* outWeightOffsetOptional = nullptr;
    aclTensor* outBiasOptional = nullptr;
    aclOpExecutor* executor = nullptr;
};

using JudgeFunc = std::function<bool(QuantContext&)>;
using CheckFunc = std::function<aclnnStatus(const QuantContext&)>;
using ProcessFunc = std::function<aclnnStatus(QuantContext&)>;

struct DataFlowEntry {
    JudgeFunc judge;
    std::vector<CheckFunc> checks;
    std::vector<ProcessFunc> processes;
};

namespace WeightQuantPreprocess {

inline constexpr size_t DIMS_1 = 1;
inline constexpr size_t DIMS_2 = 2;
inline constexpr size_t DIMS_3 = 3;
inline constexpr size_t IDX_0 = 0;
inline constexpr size_t IDX_1 = 1;
inline constexpr size_t IDX_2 = 2;
inline constexpr size_t IDX_3 = 3;
inline constexpr size_t IDX_4 = 4;
inline constexpr size_t SINGLE = 1;
inline constexpr size_t DOUBLE = 2;
inline constexpr int64_t KGROUP_SIZE_MX = 32;
inline constexpr int64_t KGROUP_SIZE_64 = 64;
inline constexpr int64_t KGROUP_SIZE_128 = 128;
inline constexpr int64_t NZ_16 = 16;
inline constexpr int64_t NZ_C0_2 = 2;
inline constexpr int64_t NZ_C0_16 = 16;
inline constexpr int64_t NZ_C0_32 = 32;
inline constexpr int64_t NZ_C0_64 = 64;

inline constexpr int64_t B4_NUMS_PER_BYTE = 2; // 1 字节打包 2 个 4-bit 值（INT4/FP4 通用）

int64_t CeilDiv(int64_t a, int64_t b);

bool IsTensor2DTransposed(const aclTensor* tensor2D);

bool IsTensor2DNonTransposed(const aclTensor* tensor2D);

bool IsTensor3DTransposed(const aclTensor* tensor3D);

bool IsTensor3DNonTransposed(const aclTensor* tensor3D);

bool IsB4TypeNormal(const aclTensor* tensorLocal);

} // namespace WeightQuantPreprocess

extern const std::unordered_map<NpuArch, std::vector<DataFlowEntry>> NPU_DATA_FLOW_REGISTRY_MAP;

#endif // WEIGHT_QUANT_PREPROCESS_REGISTRY_H_
