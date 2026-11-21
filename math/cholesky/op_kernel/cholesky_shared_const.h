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
 * \file cholesky_shared_const.h
 * \brief Cholesky host tiling 与 device kernel 共享的调度契约常量（仅 A5/arch35 使用）
 *
 * 当前调度常量与 cholesky_arch35_tiling_key.h 中的 TilingKey 共同约束 host 与 kernel，取值不得单侧变更：
 *   CHOLESKY_PANEL_WIDTH
 *     kernel 面板算法的串行段宽度，同时决定 host 侧 needCoreNum 与 UB 缓冲尺寸的前提。
 *   CHOLESKY_SINGLE_MATRIX_SINGLE_CORE_MAX
 *     host 单矩阵降为单核 launch 的上界；kernel 单核时 SyncAll 到达数为 1，与之自洽。
 *
 * CHOLESKY_BATCH_COOP_MIN_M 是历史阈值，目前 host 与 kernel 均不消费；
 * arch35 的 cooperative_ 恒为 true，M=96/97 不构成当前协同路径切换边界。
 *
 * 本头文件必须能被 host 与 kernel 两侧同时安全包含，禁止 include 任何 host 专属或
 * kernel 专属头文件，故仅用 #define 常量，不依赖 uint32_t 等类型定义
 * （与 cholesky_arch35_tiling_key.h 同一约束）。
 */

#ifndef CHOLESKY_SHARED_CONST_H
#define CHOLESKY_SHARED_CONST_H

#define CHOLESKY_PANEL_WIDTH 64U
#define CHOLESKY_BATCH_COOP_MIN_M 96U
#define CHOLESKY_SINGLE_MATRIX_SINGLE_CORE_MAX 4U

#endif // CHOLESKY_SHARED_CONST_H
