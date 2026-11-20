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
 * \file tiling_strategy_base.h
 * \brief 通用 Tiling 策略基类模板（算子可复用，header-only）
 *
 * 本文件定义通用策略基类模板 TilingStrategyBase<TPlatformInfo, TOpParam>：
 *   - TPlatformInfo：运行时平台信息结构（算子侧定义，如核数/UB 大小/缓存行/门控标志）；
 *   - TOpParam       ：运行时算子参数结构（算子侧定义，如简化后 shape/阈值/缓冲数等
 *                       算子特有上下文，由算子主类公共前置产出后注入）。
 *
 * 基类只提供真正通用的内容：
 *   - 纯虚接口 IsSupport() const（准入检查，纯判定、无副作用）；
 *   - Execute() 虚函数默认实现（模板方法）：调用纯虚 protected DoExecute() 执行策略实际逻辑，
 *     成功后先 SetContext() 写回上下文，再 LogTilingData() 记录日志；SetContext() 失败返回 FAILURE；
 *   - 上下文注入：构造注入 context + platform + opParam（只读，成员为值拷贝，算子子类可通过自己的 setter
 * 修改自身持有的副本）；
 *   - SetContext()：依次调用 SetTilingData()（填写注册 TilingData，虚，默认空）
 *     → SetBlockDim(GetUsedCoreNum()) + SetTilingKey(GetTilingKey())
 *     → SetWorkspace()（设置 workspace，虚，默认 1 个 workspace 且置 0）；任一步失败返回 false；
 *   - GetUsedCoreNum()/GetTilingKey()：纯虚接口，由各策略提供实际 blockDim/tilingKey
 *     （返回类型直接对齐写回接口：uint32_t / uint64_t，SetContext 中无需再做类型转换）；
 *   - LogTilingData()：日志虚接口（默认空，在 SetContext 之后调用；仅排查信息可追溯性，不产出 Tiling 输出）；
 *   - 通用成员 context_/platform_/opParam_（protected，只读约定由使用方遵守）。
 *
 * 注意：算子特有的分阶段实现（如 transpose 各策略内的 UB 切分/分核/维度扩展等）
 * 属于算子侧策略类自身，不下沉到本通用基类。
 */

#ifndef OPS_MATH_COMMON_OP_HOST_TILING_TILING_STRATEGY_BASE_H
#define OPS_MATH_COMMON_OP_HOST_TILING_TILING_STRATEGY_BASE_H

#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"
#include "tiling_status.h"

namespace Ops::Math::OpTiling {

/**
 * @brief 通用 Tiling 策略基类模板
 *
 * @tparam TPlatformInfo 平台信息结构（算子侧定义）
 * @tparam TOpParam      算子参数结构（算子侧定义）
 */
template <typename TPlatformInfo, typename TOpParam>
class TilingStrategyBase {
public:
    TilingStrategyBase(gert::TilingContext* context, const TPlatformInfo& platform, const TOpParam& opParam)
        : context_(context), platform_(platform), opParam_(opParam)
    {}
    virtual ~TilingStrategyBase() = default;

    // 准入检查（const）。纯判定、无副作用、不写任何 tiling 结果；返回 false 时调度器尝试下一策略
    virtual bool IsSupport() const = 0;

    /**
     * @brief 执行 tiling，返回三态（模板方法）
     *
     * 入口先校验 context_ 非空，为空时打印错误日志并返回 FAILURE。
     * 模板方法统一收尾：DoExecute() 返回非 SUCCESS（FALLBACK/FAILURE）时直接透传，不做任何收尾；
     * 成功时先 SetContext()（SetTilingData → SetBlockDim/SetTilingKey → SetWorkspace）写回上下文，
     * 写回成功后再 LogTilingData() 记录日志（保证日志反映已落盘的最终结果）；
     * SetContext() 返回 false（写回失败）时统一返回 FAILURE，不再尝试下一策略。
     * 子类只实现纯虚 protected DoExecute()，无需重复实现公共收尾路径。
     */
    virtual TilingStatus Execute()
    {
        // 空指针校验：后续 DoExecute()/SetContext() 均依赖 context_
        if (context_ == nullptr) {
            OP_LOGE("TilingStrategyBase", "Tiling context is nullptr");
            return TilingStatus::FAILURE;
        }
        TilingStatus status = DoExecute(); // 纯虚 protected，策略实际逻辑
        if (status != TilingStatus::SUCCESS) {
            return status; // FALLBACK/FAILURE 直接返回，不做收尾
        }
        if (!SetContext()) {
            return TilingStatus::FAILURE; // 写回失败统一返回 FAILURE（收尾阶段不允许回退）
        }
        LogTilingData(); // SetContext 之后才记录日志
        return TilingStatus::SUCCESS;
    }

protected:
    // 策略实际执行逻辑（纯虚，子类实现；返回三态，SUCCESS 后由 Execute() 统一收尾）
    virtual TilingStatus DoExecute() = 0;

    // 日志（仅排查信息可追溯性，不产出 Tiling 输出、不影响行为等价）
    virtual void LogTilingData() {}

    /**
     * @brief 填写注册 TilingData（公共行为：SetContext 中调用）
     *
     * 子类可覆盖。Tiling 码流容量不足时 GetTilingData<T>() 返回 nullptr，
     * 子类必须返回 false（不得静默跳过），由 SetContext 透传给 Execute。
     * 默认实现返回 true（无需填写注册 TilingData 的策略，如包装独立模板类的策略）。
     */
    virtual bool SetTilingData() { return true; }

    // 设置 workspace（虚；默认 1 个 workspace 且置 0；需要不同设置的算子覆盖；失败返回 false）
    virtual bool SetWorkspace()
    {
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            OP_LOGE(context_, "workspaces is nullptr!");
            return false;
        }
        workspaces[0] = 0;
        return true;
    }

    /**
     * @brief 写回上下文：SetTilingData → SetBlockDim/SetTilingKey → SetWorkspace
     *
     * 任一步失败（返回 false 或非 ge::GRAPH_SUCCESS）时打印错误日志并返回 false。
     */
    bool SetContext()
    {
        // ① 公共行为：填写注册 TilingData（虚）；码流不足时子类返回 false
        if (!SetTilingData()) {
            return false;
        }
        // ② 写回实际使用核数；接口返回值即 SetBlockDim 所需类型，无需中间转换
        const uint32_t usedCoreNum = GetUsedCoreNum();
        if (context_->SetBlockDim(usedCoreNum) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "SetBlockDim failed, usedCoreNum: %u.", usedCoreNum);
            return false;
        }
        // ③ 写回 tilingKey；接口返回值即 SetTilingKey 所需类型，无需中间转换
        const uint64_t tilingKey = GetTilingKey();
        if (context_->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "SetTilingKey failed, tilingKey: %lu.", tilingKey);
            return false;
        }
        return SetWorkspace(); // ④ 设置 workspace（虚）
    }

    // 子类提供实际结果（纯虚：每个策略必须给出实际 blockDim / tilingKey，类型与写回接口一致）
    virtual uint32_t GetUsedCoreNum() const = 0;
    virtual uint64_t GetTilingKey() const = 0;

    // 通用上下文（protected 以便算子子类访问；写回 tilingKey/blockDim/workspace 需要非 const 指针）
    gert::TilingContext* context_{nullptr}; ///< 上下文（策略只读，SetContext 阶段写回）
    TPlatformInfo platform_;                ///< 平台信息（值拷贝注入，只读约定）
    TOpParam opParam_;                      ///< 算子参数（值拷贝注入，只读约定）
};

} // namespace Ops::Math::OpTiling

#endif // OPS_MATH_COMMON_OP_HOST_TILING_TILING_STRATEGY_BASE_H
