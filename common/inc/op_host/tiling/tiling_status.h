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
 * \file tiling_status.h
 * \brief 通用 Tiling 策略三态返回枚举（算子可复用，header-only）
 *
 * 三态语义与调度器动作（所有继承通用策略框架的算子共享，语义与 transpose 原实现完全一致）：
 *   - SUCCESS  ：策略命中且完成 tiling，blockDim/tilingKey/workspace 已设置 → 停止调度，返回成功
 *   - FALLBACK ：策略准入，但执行中发现不适合继续（UB 预算回溯失败、硬件约束不满足等可退避路径）
 *                → 调度器尝试下一个策略
 *   - FAILURE  ：策略准入，但执行中发生确定性错误（非法 shape 组合、内部计算错误等硬错误）
 *                → 整体返回失败，不再尝试后续策略
 *
 * 命名空间：Ops::Math::OpTiling（与 ops-math 公共头文件统一的 Ops::Math 命名空间保持一致，
 * Tiling 相关公共能力置于 OpTiling 子命名空间；算子侧可通过
 * `using TilingStatus = Ops::Math::OpTiling::TilingStatus;` 取别名）。
 */

#ifndef OPS_MATH_COMMON_OP_HOST_TILING_TILING_STATUS_H
#define OPS_MATH_COMMON_OP_HOST_TILING_TILING_STATUS_H

#include <cstdint>

namespace Ops::Math::OpTiling {

/**
 * @brief 策略执行三态返回枚举
 */
enum class TilingStatus : int32_t {
    SUCCESS = 0,  ///< 命中且完成 tiling
    FALLBACK = 1, ///< 准入但执行中发现不适合 → 调度器尝试下一策略
    FAILURE = 2,  ///< 准入但执行中确定性失败 → 调度器整体返回失败
};

} // namespace Ops::Math::OpTiling

#endif // OPS_MATH_COMMON_OP_HOST_TILING_TILING_STATUS_H
