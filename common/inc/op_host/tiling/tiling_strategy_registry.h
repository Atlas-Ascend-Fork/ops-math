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
 * \file tiling_strategy_registry.h
 * \brief 通用 Tiling 策略编译期注册类模板（算子可复用，header-only）
 *
 * 本文件定义：
 *   1. StrategyTypeList<Ts...>：编译期策略类型序列（策略集合与注册顺序在编译期固定）；
 *   2. StrategyRegistry<TStrategyBase, TPlatformInfo, TOpParam, TStrategyList>：
 *      静态 Dispatch 接口，运行期把 context + platform + opParam 传入，遍历编译期类型序列，
 *      对每个策略类型在栈上按值构造临时对象并三态调度。
 *
 * 设计要点：
 *   - 通用层只提供“调度机制”，不定义任何具体策略集合/顺序——策略顺序由算子侧
 *     （算子 tiling 主类头文件 / 算子侧注册文件）以 StrategyTypeList 声明后作为
 *     TStrategyList 传入；
 *   - 每个策略在调度时栈上按值构造（TStrategy strategy(context, platform, opParam)），
 *     构造签名取策略基类约定，用完即销毁，不长期持有；
 *   - 跨步骤共享的中间态全部经 TPlatformInfo / TOpParam 参数传入。
 *
 * 调度语义（三态）：
 *   - IsSupport() 不满足 → 下一个策略
 *   - Execute() 返回 SUCCESS → 停止调度，返回成功
 *   - Execute() 返回 FALLBACK → 可退避，尝试下一个策略
 *   - Execute() 返回 FAILURE → 硬失败，整体返回失败
 *   - 兜底策略必须可命中；仍无成功 → 整体失败
 *
 */

#ifndef OPS_MATH_COMMON_OP_HOST_TILING_TILING_STRATEGY_REGISTRY_H
#define OPS_MATH_COMMON_OP_HOST_TILING_TILING_STRATEGY_REGISTRY_H

#include <cstdint>
#include <type_traits>
#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"
#include "log/log.h"
#include "tiling_status.h"

namespace Ops::Math::OpTiling {

/**
 * @brief 编译期策略类型序列（策略集合与注册顺序在编译期固定）
 *
 * 类型参数包按注册顺序排列；对每个类型在调度时栈上按值构造策略对象。
 */
template <typename... Ts>
struct StrategyTypeList {};

namespace detail {

/**
 * @brief 单策略调度步骤结果（编译期遍历内部状态）
 *
 *   - CONTINUE     ：不满足准入或执行返回 FALLBACK → 调度器尝试下一个策略
 *   - STOP_SUCCESS ：执行返回 SUCCESS → 停止调度，整体成功
 *   - STOP_FAILURE ：执行返回 FAILURE → 停止调度，整体失败
 */
enum class DispatchStepStatus { CONTINUE, STOP_SUCCESS, STOP_FAILURE };

/**
 * @brief 单策略调度：栈上按值构造 + IsSupport/Execute 三态
 *
 * 构造签名取策略基类约定：TStrategy(context, platform, opParam)；
 * 策略对象用完即销毁（无指针、无 vector、不长期持有）。
 */
template <typename TStrategy, typename TPlatformInfo, typename TOpParam>
DispatchStepStatus DispatchOne(gert::TilingContext* context, const TPlatformInfo& platform, const TOpParam& opParam)
{
    // 栈上按值构造策略临时对象（无指针、无 vector、用完即销毁）
    TStrategy strategy(context, platform, opParam);
    if (!strategy.IsSupport()) {
        return DispatchStepStatus::CONTINUE; // 不满足准入 → 下一个
    }
    TilingStatus status = strategy.Execute();
    if (status == TilingStatus::SUCCESS) {
        return DispatchStepStatus::STOP_SUCCESS;
    }
    if (status == TilingStatus::FALLBACK) {
        OP_LOGD(context->GetNodeName(), "Strategy fallback, try next strategy.");
        return DispatchStepStatus::CONTINUE;
    }
    OP_LOGE(context->GetNodeName(), "Strategy failure, return failed.");
    return DispatchStepStatus::STOP_FAILURE;
}

/**
 * @brief 编译期策略类型序列遍历（模板递归，注册顺序 = 类型参数包顺序）
 *
 * 对 StrategyTypeList<Head, Tail...> 先调度 Head，CONTINUE 时递归调度 Tail...；
 * 任一 STOP_* 短路返回。空列表（兜底之后仍无成功）返回 CONTINUE。
 */
template <typename TStrategyBase, typename TPlatformInfo, typename TOpParam, typename List>
struct StrategyDispatcher;

template <typename TStrategyBase, typename TPlatformInfo, typename TOpParam, typename Head, typename... Tail>
struct StrategyDispatcher<TStrategyBase, TPlatformInfo, TOpParam, StrategyTypeList<Head, Tail...>> {
    // 编译期约束：注册进类型序列的策略必须继承自 TStrategyBase（DispatchOne 的模板参数不含基类，断言需加在此处）
    static_assert(std::is_base_of<TStrategyBase, Head>::value, "Head must derive from TStrategyBase");

    static DispatchStepStatus Dispatch(gert::TilingContext* context, const TPlatformInfo& platform,
                                       const TOpParam& opParam)
    {
        DispatchStepStatus status = DispatchOne<Head, TPlatformInfo, TOpParam>(context, platform, opParam);
        if (status != DispatchStepStatus::CONTINUE) {
            return status;
        }
        return StrategyDispatcher<TStrategyBase, TPlatformInfo, TOpParam, StrategyTypeList<Tail...>>::Dispatch(
            context, platform, opParam);
    }
};

template <typename TStrategyBase, typename TPlatformInfo, typename TOpParam>
struct StrategyDispatcher<TStrategyBase, TPlatformInfo, TOpParam, StrategyTypeList<>> {
    // 空序列终止条件：参数仅为与通用 Dispatch 签名保持一致，此处不使用，故省略形参名
    static DispatchStepStatus Dispatch(gert::TilingContext* /* context */, const TPlatformInfo& /* platform */,
                                       const TOpParam& /* opParam */)
    {
        return DispatchStepStatus::CONTINUE;
    }
};

} // namespace detail

/**
 * @brief 通用策略注册类模板（编译期常量注册，运行时参数注入）
 *
 * 静态 Dispatch 接口：运行期把 context + platform + opParam 传入；
 * 遍历 TStrategyList（编译期类型序列），对每个策略类型在栈上按值构造临时对象，
 * 调用 IsSupport()/Execute()，按三态语义调度。
 *
 * @tparam TStrategyBase 算子策略基类（策略构造签名约定为 (context, platform, opParam)）
 * @tparam TPlatformInfo 平台信息结构
 * @tparam TOpParam      算子参数结构
 * @tparam TStrategyList 算子侧声明的策略类型序列（顺序即优先级）
 */
template <typename TStrategyBase, typename TPlatformInfo, typename TOpParam, typename TStrategyList>
class StrategyRegistry {
public:
    /**
     * @brief 运行期策略调度入口（编译期策略序列 + 运行时参数注入）
     *
     * @param context Tiling 上下文（写回 tilingKey/blockDim/workspace；为空时直接返回 ge::GRAPH_FAILED）
     * @param platform 平台信息
     * @param opParam 算子参数（主类公共前置产出）
     * @return ge::GRAPH_SUCCESS 命中并完成 tiling；ge::GRAPH_FAILED 硬失败或兜底未命中
     */
    static ge::graphStatus Dispatch(gert::TilingContext* context, const TPlatformInfo& platform,
                                    const TOpParam& opParam)
    {
        // 统一入口判空：策略的 IsSupport() 先于 Execute() 执行，且后续日志均会解引用 context；
        // 为空直接返回 ge::GRAPH_FAILED（Execute 内判空保留作最后防线）
        OP_CHECK_NULL_WITH_CONTEXT(context, context);
        // 遍历编译期策略类型序列（注册顺序即优先级）
        detail::DispatchStepStatus
            status = detail::StrategyDispatcher<TStrategyBase, TPlatformInfo, TOpParam, TStrategyList>::Dispatch(
                context, platform, opParam);
        if (status == detail::DispatchStepStatus::STOP_SUCCESS) {
            OP_LOGD(context->GetNodeName(), "Strategy dispatch success.");
            return ge::GRAPH_SUCCESS;
        }
        if (status == detail::DispatchStepStatus::STOP_FAILURE) {
            OP_LOGE(context->GetNodeName(), "Strategy failure, return failed.");
            return ge::GRAPH_FAILED;
        }
        // 兜底策略必须可命中；仍无成功 → FAILURE
        OP_LOGE(context->GetNodeName(), "No strategy hit in strategy registry.");
        return ge::GRAPH_FAILED;
    }
};

} // namespace Ops::Math::OpTiling

#endif // OPS_MATH_COMMON_OP_HOST_TILING_TILING_STRATEGY_REGISTRY_H
