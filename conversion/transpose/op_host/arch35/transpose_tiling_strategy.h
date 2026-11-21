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
 * \file transpose_tiling_strategy.h
 * \brief Transpose Tiling 策略基类（继承通用策略基类模板）与算子上下文参数
 *
 * 分层约定：
 *   - 本文件定义 transpose 算子自己的策略基类 TransposeTilingStrategy：只承载
 *     所有模板共有的上下文注入与只读访问，不包含任何模板专属状态/算法。
 *     模板专属内容（切分中间态、分核、UB 内 shape、区间、TilingData 回填等）
 *     全部下沉到对应策略子类，由只有该模板才会执行的代码自行持有。
 *   - TransposeNddmaGroupStrategy：NDDMA 模板族（NDDMA/NDDMA_BIG_DIM）共有的缓冲数，
 *     仅被这两个子类继承，同样不进入通用策略基类。
 *
 * TPlatformInfo = TransposePlatformInfo
 * TOpParam      = TransposeOpParam
 *
 * 上下文注入方式：统一使用 (context, platform, opParam) 三元组构造。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_H

#include <cstdint>
#include <initializer_list>
#include "transpose_tiling_base.h"
#include "conversion/transpose/op_kernel/arch35/transpose_tiling_key.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "common/inc/op_host/tiling/tiling_strategy_base.h"

namespace optiling {

/**
 * @brief 统一引用通用三态枚举
 */
using TilingStatus = Ops::Math::OpTiling::TilingStatus;

/**
 * @brief 运行时平台信息（策略构造注入的只读上下文）
 *
 * 只放平台固有的信息。
 * 模板专属参数由对应策略在自身内部推导。
 */
struct TransposePlatformInfo {
    int64_t coreNum = 0;                     ///< AI Vector 核数
    int64_t ubSize = 0;                      ///< UB 大小（字节）
    int64_t cacheLineSize = 0;               ///< = GetCacheLineSize()（256B）
    int64_t ubBlockSize = 0;                 ///< = GetUbBlockSize()（32B）
    ::NpuArch npuArch = Ops::Base::DAV_RESV; ///< 当前平台架构
    bool isSupportSimt = true;               ///< 平台是否支持 SIMT（决定小 shape 兜底路径）
};

/**
 * @brief 运行时算子参数（主类公共前置产出的只读上下文）
 *
 * 只放所有策略都会消费的通用运行时参数
 */
struct TransposeOpParam {
    StrategyShapeInfo shapeInfo;     ///< 策略侧精简 shape/perm（主类前置转换，只读）
    bool isRelatedTranspose = false; ///< 融合内联调用门控（运行期调用形态，非平台信息）
};

/**
 * @brief 取注册 TilingData（10000-10005 模板共用 TransposeOpTilingData）
 *
 * 各 10000-10005 策略子类共用本工具函数取码流指针：容量不足时记录错误并返回 nullptr。
 * 只做取址/校验，不包含任何模板专属字段映射。
 */
TransposeOpTilingData* GetTransposeOpTilingDataBuffer(gert::TilingContext* context);

/**
 * @brief Transpose 策略基类（继承通用策略基类模板）
 *
 * 只承载所有 transpose 模板共有的内容：
 *   - 上下文注入（context/精简 shape/平台信息/融合内联门控）与只读访问成员；
 *   - 纯虚准入/执行接口 IsSupport() const/DoExecute()。
 * 不持有任何模板专属状态，也不提供任何模板专属阶段实现——这些由对应策略子类自负。
 * 执行收尾（Execute 模板方法：DoExecute → SetContext → LogTilingData）与
 * SetTilingData/SetWorkspace/GetUsedCoreNum/GetTilingKey 由通用基类提供，
 * 子类按需覆盖 SetTilingData/LogTilingData/GetUsedCoreNum/GetTilingKey。
 */
class TransposeTilingStrategy
    : public Ops::Math::OpTiling::TilingStrategyBase<TransposePlatformInfo, TransposeOpParam> {
public:
    using Base = Ops::Math::OpTiling::TilingStrategyBase<TransposePlatformInfo, TransposeOpParam>;

    // 3 参构造（context/platform/opParam，构造签名取通用基类约定）
    TransposeTilingStrategy(gert::TilingContext* context, const TransposePlatformInfo& platform,
                            const TransposeOpParam& opParam);
    virtual ~TransposeTilingStrategy() = default;

    // 准入检查。纯判定、无副作用、不写任何 tiling 结果；返回 false 时调度器尝试下一策略
    virtual bool IsSupport() const = 0;

protected:
    /**
     * @brief 判断 perm 取值（同时判维数）
     *
     * 策略只传自己要求的 perm 取值，例如 2D 转置 `{1, 0}`、3D 021 转置 `{0, 2, 1}`；
     * 维数（perm 元素个数）随之在内部一并判定，不再各自散落 dim/perm 判断。
     *
     * @param perm 期望的 perm 取值（顺序即各轴映射）
     * @return true 维数等于 perm 长度且各轴取值逐一相等
     */
    bool IsPermValue(std::initializer_list<int64_t> perm) const;

    // 策略实际执行逻辑（纯虚，子类实现）
    TilingStatus DoExecute() override = 0;

    // 保护成员访问：只读引用，绑定到 opParam_/platform_ 对应字段
    const StrategyShapeInfo& shapeInfo_; ///< = opParam_.shapeInfo
    const int64_t& coreNum_;             ///< = platform_.coreNum
    const int64_t& ubSize_;              ///< = platform_.ubSize
    const int64_t& cacheLineSize_;       ///< = platform_.cacheLineSize
    const int64_t& ubBlockSize_;         ///< = platform_.ubBlockSize
    const ::NpuArch& npuArch_;           ///< = platform_.npuArch
    const bool& isSupportSimt_;          ///< = platform_.isSupportSimt
    const bool& isRelatedTranspose_;     ///< = opParam_.isRelatedTranspose
};

/**
 * @brief NDDMA 模板族策略基类（NDDMA/NDDMA_BIG_DIM 共用）
 *
 * 承载 NDDMA 模板族专属的缓冲数（ubElement 预算的分母），只有 NDDMA 与 NDDMA_BIG_DIM 消费。
 * 缓冲数完全由平台架构推导（DAV_9201/DAV_9202 双缓冲=2，其他=1），不提供显式注入接口：
 * 单测/策略直调需要不同缓冲数时，只需改 platformInfo 的 npuArch。
 */
class TransposeNddmaGroupStrategy : public TransposeTilingStrategy {
public:
    TransposeNddmaGroupStrategy(gert::TilingContext* context, const TransposePlatformInfo& platform,
                                const TransposeOpParam& opParam);
    virtual ~TransposeNddmaGroupStrategy() = default;

protected:
    // NDDMA 缓冲数：按平台架构推导（DAV_9201/DAV_9202=2，其他=1）
    int64_t GetNddmaBufferNum() const;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_H
