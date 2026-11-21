/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_tiling_arch35.cpp
 * \brief Transpose Tiling 调度器实现
 *
 * 本文件实现 TransposeNddmaTiling 调度器：
 *   1. 公共前置：获取输入 shape/perm/dtype、校验 shape/perm 有效性
 *   2. 轴简化：消除 size=1 的轴（RemoveAxisV2）、合并 perm 中连续的轴（MergeAxisV2）
 *   3. 公共前置产出运行时算子参数 TransposeOpParam（精简 shape + 融合内联门控）
 *   4. 经编译期注册类顺序调度（TransposeStrategyRegistry::Dispatch）
 *
 * 策略集合与注册顺序由本文件的编译期类型序列（TransposeStrategyList）唯一表达，运行期把 context + TransposePlatformInfo
 * + TransposeOpParam 传入注册类 Dispatch， 每个策略在调度时栈上按值构造、用完销毁（无 vector、无指针持有策略）。
 *
 * 各策略的 UB 切分、分核、5 维扩展、UB shape、区间边界与 TilingData 填充均在独立
 * 策略类（transpose_tiling_strategy_*.cpp）中实现，主类只保留参数解析/校验与调度。
 */

#include <array>
#include <sstream>
#include <set>
#include <limits>
#include "util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "transpose_tiling_base.h"
#include "transpose_tiling_arch35.h"
#include "transpose_tiling_strategy.h"
#include "transpose_tiling_strategy_transto5hd_10_perf.h"
#include "transpose_tiling_strategy_transto5hd_10.h"
#include "transpose_tiling_strategy_transto5hd_021.h"
#include "transpose_tiling_strategy_b4.h"
#include "transpose_tiling_strategy_gather.h"
#include "transpose_tiling_strategy_tensor_move.h"
#include "transpose_tiling_strategy_n_last.h"
#include "transpose_tiling_strategy_nddma.h"
#include "transpose_tiling_strategy_nddma_big_dim.h"
#include "transpose_tiling_strategy_small_shape.h"
#include "common/inc/op_host/tiling/tiling_strategy_registry.h"
#include "common/inc/op_host/math_log.h"
#include "op_host/tiling_util.h"

namespace optiling {
namespace {
static constexpr uint64_t INPUT_IDX_X = 0;    ///< 输入 x 的索引
static constexpr uint64_t OUTPUT_IDX_Y = 0;   ///< 输出 y 的索引
static constexpr uint64_t INPUT_IDX_PERM = 1; ///< 输入 perm 的索引

/**
 * @brief Transpose 策略列表
 */
using TransposeStrategyList = Ops::Math::OpTiling::StrategyTypeList<
    TransposeTransTo5HD10PerfStrategy, // 2D perm=[1,0] 全 tile 转置（DAV_3510/DAV_5102）
    TransposeTransTo5HD10Strategy,     // DAV_5102、2D perm=[1,0] 且 16bit
    TransposeTransTo5HD021Strategy,    // DAV_5102、3D perm=[0,2,1]
    TransposeB4Strategy,               // 4bit 类型且尾轴转置
    TransposeGatherStrategy,           // 非融合内联的尾轴转置（非 4bit）
    TransposeTensorMoveStrategy,       // 1 维纯数据搬运
    TransposeNLastStrategy,            // 尾轴不转置且尾轴元素数不少于 32
    TransposeSmallShapeStrategy,       // SIMT 小 shape
    TransposeNddmaStrategy,            // dim<=5，NDDMA 单轴/双轴切分
    TransposeNddmaBigDimStrategy       // dim>5，压缩为 5 维 NDDMA
    >;

/**
 * @brief Transpose 策略注册类（通用注册类模板实例化）
 *
 * 通用注册类（Ops::Math::OpTiling::StrategyRegistry）只提供调度机制；
 * 策略集合与注册顺序由上面的 TransposeStrategyList 传入。
 */
using TransposeStrategyRegistry = Ops::Math::OpTiling::StrategyRegistry<TransposeTilingStrategy, TransposePlatformInfo,
                                                                        TransposeOpParam, TransposeStrategyList>;
} // namespace

static constexpr int64_t B4_TYPE_ELES_PER_BYTE = 2;
static constexpr int64_t B4_BIT_SIZE = 4;
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,   ge::DT_UINT8, ge::DT_INT16,    ge::DT_UINT16,      ge::DT_INT32,
    ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64,   ge::DT_FLOAT,       ge::DT_FLOAT16,
    ge::DT_BF16,   ge::DT_BOOL,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
};

static std::string FormatShape(const std::vector<int64_t>& shape, int64_t dimNum)
{
    gert::Shape tensorShape;
    for (int64_t i = 0; i < dimNum; ++i) {
        tensorShape.AppendDim(shape[i]);
    }
    return Ops::Base::ToString(tensorShape);
}

ge::graphStatus TransposeNddmaTiling::Init(const int64_t& coreNum, const int64_t& ubSize)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init TransposeNddmaTiling.");
    coreNum_ = coreNum;
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    ubSize_ = ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    cacheLineSize_ = Ops::Base::GetCacheLineSize(tilingContext_);
    OP_CHECK_IF((cacheLineSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get cache line size."),
                return ge::GRAPH_FAILED);

    ubBlockSize_ = Ops::Base::GetUbBlockSize(tilingContext_);
    OP_CHECK_IF((ubBlockSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub block size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 主 Tiling 流程
 *
 * 执行完整的 Tiling 计算流程，包括 shape 预处理、策略选型、
 * UB 切分、Block 切分、5维扩展、TilingData 填充。
 *
 * @return ge::GRAPH_SUCCESS 成功；ge::GRAPH_FAILED 失败
 */
ge::graphStatus TransposeNddmaTiling::RunTranposelTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start running Tiling4Transpose.");
    if (!isRelatedTranspose_) {
        CHECK_RET_SUCC(GetShapeInfo());
    }

    auto ret = CheckShapeInfo();
    CHECK_RET_SUCC(ret);
    // 消除 size=1 的轴（减少不必要的维度）
    RemoveAxisV2(shapeInfo_);
    // 合并 perm 中连续的轴（进一步降低维度数）
    MergeAxisV2(shapeInfo_);
    // check reduced shape
    ret = CheckReducedShapeInfo();
    CHECK_RET_SUCC(ret);

    // 公共前置：打包通用算子参数（shapeInfo/related 门控）供注册类注入各策略。
    // 模板专属参数（SIMT 小 shape 阈值、NDDMA 缓冲数）由对应策略按平台架构/形状自行推导，
    // 不经 TransposeOpParam 传递；尾轴转置判定由消费它的策略在自身 IsSupport/执行中完成。
    SetIsLastAxisTranspose();
    // 4bit 非尾轴转置按 int8 视图处理（尾轴转置的 4bit 由 B4 策略处理，此处自动跳过）
    AdjustShapeForB4();
    TransposeOpParam opParam = PrepareStrategyContext();

    // 编译期常量注册 + 运行时参数注入（注册顺序即优先级）
    return DispatchStrategies(opParam);
}

/**
 * @brief 公共前置：把完整 ShapeInfo 裁剪为策略侧精简结构并打包 TransposeOpParam
 *
 * 只打包所有策略都会消费的运行时参数：shapeInfo 与 isRelatedTranspose 门控。
 * 模板专属参数不在本方法传递：
 *   - SIMT 小 shape 阈值由 TransposeSmallShapeStrategy 按平台架构/形状推导；
 *   - NDDMA 缓冲数由 TransposeNddmaGroupStrategy 按平台架构推导（DAV_9201/9202=2）。
 *
 * @return TransposeOpParam 主类公共前置产出的只读算子参数
 */
TransposeOpParam TransposeNddmaTiling::PrepareStrategyContext()
{
    TransposeOpParam opParam;
    // 完整 ShapeInfo → 策略侧精简 StrategyShapeInfo（策略只消费精简字段）
    opParam.shapeInfo = StrategyShapeInfo(shapeInfo_);
    opParam.isRelatedTranspose = isRelatedTranspose_;
    return opParam;
}

/**
 * @brief 策略调度：编译期常量注册 + 运行时参数注入
 *
 * 平台信息打包（平台固有信息 + 平台能力：架构/SIMT 支持）后交给
 * TransposeStrategyRegistry::Dispatch，由注册类对每个策略类型在栈上按值构造临时对象
 * 并三态调度：
 *   - IsSupport() 不满足 → 下一个策略
 *   - Execute() 返回 SUCCESS → 停止调度，返回成功
 *   - Execute() 返回 FALLBACK → 可退避，尝试下一个策略
 *   - Execute() 返回 FAILURE → 硬失败，整体返回失败
 * 兜底策略（SMALL_SHAPE/NDDMA/NDDMA_BIG_DIM）必须可命中；仍无成功 → 整体失败。
 *
 * @param opParam 主类公共前置产出的只读算子参数
 * @return ge::GRAPH_SUCCESS 成功；ge::GRAPH_FAILED 失败
 */
ge::graphStatus TransposeNddmaTiling::DispatchStrategies(const TransposeOpParam& opParam)
{
    // 平台信息打包（平台固有信息 + 平台能力：架构/SIMT 支持）
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    auto arch = ascendcPlatform.GetCurNpuArch();

    TransposePlatformInfo platformInfo;
    platformInfo.coreNum = coreNum_;
    platformInfo.ubSize = ubSize_;
    platformInfo.cacheLineSize = cacheLineSize_;
    platformInfo.ubBlockSize = ubBlockSize_;
    platformInfo.npuArch = arch;
    platformInfo.isSupportSimt = Ops::Math::OpTiling::IsSimtSupported(tilingContext_);

    return TransposeStrategyRegistry::Dispatch(tilingContext_, platformInfo, opParam);
}

ge::graphStatus TransposeNddmaTiling::AdjustShapeForB4()
{
    if (!is4bit_) {
        return ge::GRAPH_SUCCESS;
    }
    if (shapeInfo_.isLastAxisTranspose) {
        return ge::GRAPH_SUCCESS;
    }

    // 4bit 类型，非尾轴转置，则视为 int8
    shapeInfo_.eleLenInBytes = 1; // INT8 view
    shapeInfo_.reducedInShape[shapeInfo_.dim - 1] /= B4_TYPE_ELES_PER_BYTE;
    shapeInfo_.reducedOutShape[shapeInfo_.dim - 1] /= B4_TYPE_ELES_PER_BYTE;
    shapeInfo_.totalVolumeActual /= B4_TYPE_ELES_PER_BYTE;
    OP_LOGD(tilingContext_->GetNodeName(), "[B4] GetShapeInfo: non-last-axis, lastAxis halved to %ld, eleLenInBytes=1",
            shapeInfo_.reducedInShape[shapeInfo_.dim - 1]);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
bool TransposeNddmaTiling::GetPerm(const gert::Tensor* permTensor)
{
    int64_t dims = permTensor->GetShapeSize();
    if (dims < 1 || dims > TP_MAX_AXIS_NUM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            tilingContext_->GetNodeName(), "perm", std::to_string(dims).c_str(),
            ("The total number of elements of perm must be within the range [1, " + std::to_string(TP_MAX_AXIS_NUM) +
             "]")
                .c_str());
        return false;
    }
    const T* permValue = permTensor->GetData<T>();
    if (!permValue) {
        OP_LOGE(tilingContext_->GetNodeName(), "Perm GetData is nullptr");
        return false;
    }
    for (int64_t i = 0; i < dims; i++) {
        // 非法负轴保留原值，以便报错时展示调用方的实际输入。
        shapeInfo_.perm[i] = permValue[i] < 0 && permValue[i] >= -dims ? permValue[i] + dims : permValue[i];
    }
    return true;
}

void TransposeNddmaTiling::SetIsLastAxisTranspose()
{
    int64_t dim = shapeInfo_.dim;
    shapeInfo_.isLastAxisTranspose = shapeInfo_.reducedPerm[dim - 1] != dim - 1 ? true : false;
}

ge::graphStatus TransposeNddmaTiling::GetShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering GetShapeInfo.");

    const gert::Tensor* permTensor = tilingContext_->GetInputTensor(INPUT_IDX_PERM);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, permTensor);
    shapeInfo_.permSize = permTensor->GetShapeSize();

    auto permDtype = tilingContext_->GetInputDesc(INPUT_IDX_PERM)->GetDataType();
    if (permDtype == ge::DT_INT32) {
        if (!GetPerm<int32_t>(permTensor)) {
            return ge::GRAPH_FAILED;
        }
    } else if (permDtype == ge::DT_INT64) {
        if (!GetPerm<int64_t>(permTensor)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "perm",
                                  ge::TypeUtils::DataTypeToSerialString(permDtype).c_str(), "INT32 or INT64");
        return ge::GRAPH_FAILED;
    }

    auto outputY = tilingContext_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputY);
    auto yShape = outputY->GetStorageShape();
    auto yDims = yShape.GetDimNum();
    auto inputX = tilingContext_->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputX);
    auto xShape = inputX->GetStorageShape();
    auto xDims = xShape.GetDimNum();

    auto xDtype = tilingContext_->GetInputDesc(INPUT_IDX_X)->GetDataType();
    shapeInfo_.eleLenInBytes = ge::GetSizeByDataType(xDtype);
    shapeInfo_.inShapeSize = xDims;
    shapeInfo_.outShapeSize = yDims;
    shapeInfo_.dim = xDims;
    shapeInfo_.origDim = xDims;
    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        shapeInfo_.inShape[i] = xShape[i];
        shapeInfo_.outShape[i] = yShape[i];
    }

    is4bit_ = shapeInfo_.eleLenInBytes == ge::kDataTypeSizeBitOffset + B4_BIT_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckShapeDims()
{
    const int64_t inDims = shapeInfo_.inShapeSize;
    const int64_t outDims = shapeInfo_.outShapeSize;
    const int64_t permDims = shapeInfo_.permSize;
    OP_CHECK_IF(
        inDims < 1 || inDims > TP_MAX_AXIS_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            tilingContext_->GetNodeName(), "x", std::to_string(inDims).c_str(),
            ("The shape dim of x must be within the range [1, " + std::to_string(TP_MAX_AXIS_NUM) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inDims != outDims,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "x and y",
                                                          Ops::Math::Join(inDims, outDims).c_str(),
                                                          "The shape dims of x and y must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inDims != permDims,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    tilingContext_->GetNodeName(), "perm", std::to_string(permDims).c_str(),
                    "The total number of elements of perm must be equal to the shape dim of x"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验 perm 每个元素的合法性（取值范围与重复值）
 *
 * 使用 std::set 去重，避免双重循环，复杂度 O(nlogn)：
 * 1. perm[i] 必须在 [0, shape dim of x) 范围内
 * 2. perm 中各值不允许重复
 *
 * @return ge::GRAPH_SUCCESS 成功；ge::GRAPH_FAILED 失败
 */
ge::graphStatus TransposeNddmaTiling::CheckPermValue()
{
    std::set<int64_t> permSet;
    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.perm[i] < 0 || shapeInfo_.perm[i] >= shapeInfo_.inShapeSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                tilingContext_->GetNodeName(), "perm", std::to_string(shapeInfo_.perm[i]).c_str(),
                ("Each value of perm must be within the range [-" + std::to_string(shapeInfo_.inShapeSize) + ", " +
                 std::to_string(shapeInfo_.inShapeSize) + ")")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (!permSet.insert(shapeInfo_.perm[i]).second) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "perm",
                                                  std::to_string(shapeInfo_.perm[i]).c_str(),
                                                  "The axes selected by perm must not be duplicated after normalizing "
                                                  "negative indices");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckB4ShapeInfo()
{
    if (!is4bit_) {
        return ge::GRAPH_SUCCESS;
    }

    // 4 bit 类型，输入输出尾轴应该为偶数
    int64_t lastAxisIdx = shapeInfo_.inShapeSize - 1;
    if (lastAxisIdx >= 0 && shapeInfo_.inShape[lastAxisIdx] % B4_TYPE_ELES_PER_BYTE != 0) {
        std::string reasonMsg = "Last axis of x must be exactly divided by " + std::to_string(B4_TYPE_ELES_PER_BYTE) +
                                ", when the type of x is 4-bit dtype";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                              FormatShape(shapeInfo_.inShape, shapeInfo_.inShapeSize).c_str(),
                                              reasonMsg);
        return ge::GRAPH_FAILED;
    }
    lastAxisIdx = shapeInfo_.outShapeSize - 1;
    if (lastAxisIdx >= 0 && shapeInfo_.outShape[lastAxisIdx] % B4_TYPE_ELES_PER_BYTE != 0) {
        std::string reasonMsg = "Last axis of y must be exactly divided by " + std::to_string(B4_TYPE_ELES_PER_BYTE) +
                                ", when the type of y is 4-bit dtype";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "y",
                                              FormatShape(shapeInfo_.outShape, shapeInfo_.outShapeSize).c_str(),
                                              reasonMsg);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CheckShapeInfo.");
    CHECK_RET_SUCC(CheckShapeDims());
    CHECK_RET_SUCC(CheckPermValue());
    if (shapeInfo_.eleLenInBytes <= 0) {
        const auto* inputDesc = tilingContext_->GetInputDesc(INPUT_IDX_X);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "x",
                                  ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str(),
                                  Ops::Math::JoinArray<Ops::Math::ItemConj::OR>(VALUE_DATA_TYPE_ALL).c_str());
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.inShape[shapeInfo_.perm[i]] != shapeInfo_.outShape[i]) {
            std::ostringstream oss;
            oss << "The shape of y must be the same as the shape consisting of the axes of x. "
                << "The " << i << "-th axis is determined by value " << shapeInfo_.perm[i] << " of perm. "
                << "When the value of perm is a negative number, the " << i << "-th axis is equal to the "
                << std::abs(shapeInfo_.perm[i]) << "-th axis counted from the end of shape of x";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                tilingContext_->GetNodeName(), "x and y",
                Ops::Math::Join(shapeInfo_.inShape[shapeInfo_.perm[i]], shapeInfo_.outShape[i]).c_str(),
                oss.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }

    gert::Shape inputShape;
    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.inShape[i] <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                                  std::to_string(shapeInfo_.inShape[i]).c_str(),
                                                  "All axes of x must be positive numbers");
            return ge::GRAPH_FAILED;
        }
        // 一致性校验已保证 outShape[i] == inShape[perm[i]] 且 inShape 全为正，y 轴非正不可达；
        // 保留防御性校验并以 LCOV_EXCL 标记该不可达分支。
        // LCOV_EXCL_START
        if (shapeInfo_.outShape[i] <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "y",
                                                  std::to_string(shapeInfo_.outShape[i]).c_str(),
                                                  "All axes of y must be positive numbers");
            return ge::GRAPH_FAILED;
        }
        // LCOV_EXCL_STOP
        inputShape.AppendDim(shapeInfo_.inShape[i]);
    }

    // 正维度检查与溢出检查分开，元素数量溢出由 SDK 的 Shape 接口检测。
    const int64_t totalVolume = inputShape.GetShapeSize();
    OP_CHECK_IF(
        totalVolume == gert::Shape::kInvalidDimValue,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext_->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
            ("The total number of elements of x must not exceed " + std::to_string(std::numeric_limits<int64_t>::max()))
                .c_str()),
        return ge::GRAPH_FAILED);
    // 4bit 类型的元素字节数是 SDK 编码值，不能作为实际字节数参与校验。
    OP_CHECK_IF(
        !is4bit_ && totalVolume > std::numeric_limits<int64_t>::max() / shapeInfo_.eleLenInBytes,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext_->GetNodeName(), "x", Ops::Base::ToString(inputShape).c_str(),
            ("The total size of x in bytes must not exceed " + std::to_string(std::numeric_limits<int64_t>::max()))
                .c_str()),
        return ge::GRAPH_FAILED);
    // 删除单例轴和合并连续轴均保持元素总数，后续直接复用已通过溢出校验的结果。
    shapeInfo_.totalVolumeActual = totalVolume;

    return CheckB4ShapeInfo();
}

ge::graphStatus TransposeNddmaTiling::CheckReducedShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CheckReducedShapeInfo.");
    auto dim = shapeInfo_.dim;
    if (dim < 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "The dim of reducedShape is invalid, dim = %ld", dim);
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo_.reducedInShape[i] <= 0 || shapeInfo_.reducedOutShape[i] <= 0) {
            OP_LOGE(tilingContext_->GetNodeName(), "Invalid shape, index is %ld, inShape is %ld, outShape is %ld", i,
                    shapeInfo_.reducedInShape[i], shapeInfo_.reducedOutShape[i]);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 关联转置场景的 Tiling 入口
 *
 * 解析输入 shape/perm 与编译信息，完成公共前置后按注册顺序调度策略；
 * 命中的策略负责写回 TilingData、tilingKey 与 blockDim。
 */
ge::graphStatus TransposeNddmaTiling::TilingForRelatedTranspose(gert::TilingContext* context,
                                                                const TransposeCompilerInfo& compilerInfo,
                                                                ShapeInfo& opInput)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    // compilerInfo 为引用，无需判空
    OP_LOGD(context->GetNodeName(), "Start TilingForRelatedTranspose.");
    TransposeNddmaTiling tilingObject(context);
    tilingObject.tilingContext_ = context;
    tilingObject.shapeInfo_ = opInput;

    tilingObject.isRelatedTranspose_ = true;
    if (tilingObject.Init(compilerInfo.coreNum, compilerInfo.ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "tilingObject.isRelatedTranspose_: %d", tilingObject.isRelatedTranspose_);
    return tilingObject.RunTranposelTiling();
}

static ge::graphStatus TransposeTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForTranspose");
    auto compilerInfo = context->GetCompileInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compilerInfo);
    TransposeNddmaTiling tilingObject(context);
    if (tilingObject.Init(compilerInfo->coreNum, compilerInfo->ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunTranposelTiling();
}

ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareTransposeForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Transpose Op get coreNum:%ld, ubSize:%ld.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Transpose)
    .Tiling(TransposeTilingForAscendC)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareTransposeForAscendC);

} // namespace optiling
