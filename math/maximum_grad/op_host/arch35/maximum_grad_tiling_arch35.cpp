/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file maximum_grad_tiling.cpp
 * \brief MaximumGrad 算子 Host Tiling 实现
 *
 * 支持三种输出模式：
 *   outputCount=1: 只输出 y1 (grad_x=True, grad_y=False)
 *   outputCount=2: 只输出 y2 (grad_x=False, grad_y=True)
 *   outputCount=3: 输出 y1 和 y2 (grad_x=True, grad_y=True)
 *
 */

#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/infershape_broadcast_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../op_kernel/arch35/maximum_grad_dag.h"
#include "../../op_kernel/arch35/maximum_grad_struct.h"
#include "maximum_grad_tiling_arch35.h"

namespace optiling {

using namespace ge;
using namespace NsMaximumGrad;
using namespace Ops::Base;

// 校验单个 shape 的合法性（使用 StorageShape）
static bool ValidateShape(const gert::StorageShape* storageShape, const char* name, const gert::TilingContext* context)
{
    if (storageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), name);
        return false;
    }

    // 使用 origin shape 进行校验
    const gert::Shape& shape = storageShape->GetOriginShape();

    // 校验 dims 数量（atvoss broadcast 最多支持 8 维）
    size_t dimNum = shape.GetDimNum();
    if (dimNum > BROADCAST_MAX_DIMS_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), name, std::to_string(dimNum).c_str(),
                                     std::to_string(BROADCAST_MAX_DIMS_NUM).c_str());
        return false;
    }

    // 校验每个 dim 值合法：-1 表示动态 shape 允许
    for (size_t i = 0; i < dimNum; i++) {
        int64_t dim = shape.GetDim(i);
        if (dim < -1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), name,
                                                     ("dim[" + std::to_string(i) + "]=" + std::to_string(dim)).c_str(),
                                                     "it must be > 0 or -1");
            return false;
        }
    }

    return true;
}

// 校验输入 shape 合法性
static bool ValidateInputShapes(const gert::TilingContext* context)
{
    // 获取三个输入的 StorageShape
    const gert::StorageShape* gradsStorageShape = context->GetInputShape(0);
    const gert::StorageShape* x1StorageShape = context->GetInputShape(1);
    const gert::StorageShape* x2StorageShape = context->GetInputShape(2);

    // 校验每个输入 shape
    if (!ValidateShape(gradsStorageShape, "grads", context)) {
        return false;
    }
    if (!ValidateShape(x1StorageShape, "x1", context)) {
        return false;
    }
    if (!ValidateShape(x2StorageShape, "x2", context)) {
        return false;
    }

    return true;
}

// 校验广播合法性并返回 broadcast shape
static bool ValidateBroadcastShape(const gert::TilingContext* context, gert::Shape* broadcastShape)
{
    const gert::StorageShape* gradsStorageShape = context->GetInputShape(0);
    const gert::StorageShape* x1StorageShape = context->GetInputShape(1);
    const gert::StorageShape* x2StorageShape = context->GetInputShape(2);

    // 获取 origin shapes 用于广播计算
    const gert::Shape& gradsShape = gradsStorageShape->GetOriginShape();
    const gert::Shape& x1Shape = x1StorageShape->GetOriginShape();
    const gert::Shape& x2Shape = x2StorageShape->GetOriginShape();

    std::vector<const gert::Shape*> inShapes = {&gradsShape, &x1Shape, &x2Shape};
    if (!Ops::Base::BroadcastShape(inShapes, broadcastShape)) {
        const std::string shapeMsg = "grads: " + Ops::Base::ToString(gradsShape) +
                                     ", x1: " + Ops::Base::ToString(x1Shape) + ", x2: " + Ops::Base::ToString(x2Shape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grads, x1, x2", shapeMsg.c_str(),
                                               "the shapes cannot be broadcast together");
        return false;
    }

    return true;
}

// 校验输出 shape 与期望一致（无论 grad_x/grad_y 属性值，两个输出都需要校验）
static bool ValidateOutputShapes(const gert::TilingContext* context, const gert::Shape& expectedShape)
{
    // 校验 y1 shape 与 broadcast shape 一致（允许动态 shape -1）
    const gert::StorageShape* y1StorageShape = context->GetOutputShape(0);
    if (y1StorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y1");
        return false;
    }
    const gert::Shape& y1Shape = y1StorageShape->GetOriginShape();

    size_t dimNum = expectedShape.GetDimNum();
    if (y1Shape.GetDimNum() != dimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "y1", std::to_string(y1Shape.GetDimNum()).c_str(),
                                     std::to_string(dimNum).c_str());
        return false;
    }
    for (size_t i = 0; i < dimNum; i++) {
        int64_t expectedDim = expectedShape.GetDim(i);
        int64_t actualDim = y1Shape.GetDim(i);
        // 允许动态 shape (-1)，其他情况必须一致
        if (expectedDim != -1 && actualDim != -1 && expectedDim != actualDim) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "y1",
                                         ("dim[" + std::to_string(i) + "]=" + std::to_string(actualDim)).c_str(),
                                         ("dim[" + std::to_string(i) + "]=" + std::to_string(expectedDim)).c_str());
            return false;
        }
    }

    // 校验 y2 shape 与 broadcast shape 一致
    const gert::StorageShape* y2StorageShape = context->GetOutputShape(1);
    if (y2StorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y2");
        return false;
    }
    const gert::Shape& y2Shape = y2StorageShape->GetOriginShape();

    if (y2Shape.GetDimNum() != dimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "y2", std::to_string(y2Shape.GetDimNum()).c_str(),
                                     std::to_string(dimNum).c_str());
        return false;
    }
    for (size_t i = 0; i < dimNum; i++) {
        int64_t expectedDim = expectedShape.GetDim(i);
        int64_t actualDim = y2Shape.GetDim(i);
        if (expectedDim != -1 && actualDim != -1 && expectedDim != actualDim) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "y2",
                                         ("dim[" + std::to_string(i) + "]=" + std::to_string(actualDim)).c_str(),
                                         ("dim[" + std::to_string(i) + "]=" + std::to_string(expectedDim)).c_str());
            return false;
        }
    }

    return true;
}

// 空 Tensor 早返回。ATVOSS 的 BroadcastBaseTiling 在合轴之后会显式拒绝 0 元素
// （broadcast_tiling.h: "tensor check is empty, check failed"），不能落到 DoTiling，
// 因此这里自己出一份 tiling：blockDim = 1，schMode 选自定义 999 分支，kernel 侧
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, uint64_t outputCount)
{
    OP_LOGD(context, "MaximumGrad: empty tensor, skip kernel computation.");
    auto* tilingData = context->GetTilingData<MaximumGradEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->numel = 0;

    // workspace：内核 wrapper 在进入计算分支前无条件执行 SetSysWorkspaceForce(workspace)
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());

    context->SetBlockDim(1);
    context->SetTilingKey(GET_TPL_TILING_KEY(TPL_SCH_MODE_EMPTY, outputCount));
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus DoTilingWithOutputCount(gert::TilingContext* context, uint64_t outputCount)
{
    uint64_t schMode = 0;
    ge::graphStatus ret;
    if (outputCount == TPL_OUTPUT_Y1_ONLY) {
        BroadcastBaseTiling<NsMaximumGrad::DagTypes::OutputY1<T>> brcTiling(context);
        ret = brcTiling.DoTiling();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "BroadcastBaseTiling(Y1) DoTiling failed"), return ret);
        schMode = brcTiling.GetSchMode();
    } else if (outputCount == TPL_OUTPUT_Y2_ONLY) {
        BroadcastBaseTiling<NsMaximumGrad::DagTypes::OutputY2<T>> brcTiling(context);
        ret = brcTiling.DoTiling();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "BroadcastBaseTiling(Y2) DoTiling failed"), return ret);
        schMode = brcTiling.GetSchMode();
    } else {
        BroadcastBaseTiling<NsMaximumGrad::DagTypes::TwoOutput<T>> brcTiling(context);
        ret = brcTiling.DoTiling();
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "BroadcastBaseTiling(Both) DoTiling failed"),
                    return ret);
        schMode = brcTiling.GetSchMode();
    }

    // 编码 TilingKey: schMode + outputCount
    if (context->SetTilingKey(GET_TPL_TILING_KEY(schMode, outputCount)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "failed to set tiling key");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "schMode=%lu, outputCount=%lu, tilingKey=%lu", schMode, outputCount,
            GET_TPL_TILING_KEY(schMode, outputCount));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaximumGradTilingFunc(gert::TilingContext* context)
{
    // 获取属性：grad_x / grad_y
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const bool* gradXPtr = attrs->GetBool(0);
    const bool* gradYPtr = attrs->GetBool(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYPtr);

    bool gradXAttr = *gradXPtr;
    bool gradYAttr = *gradYPtr;

    // 校验：grad_x 和 grad_y 不能同时为 False
    if (!gradXAttr && !gradYAttr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "grad_x, grad_y", "false, false",
                                               "grad_x and grad_y cannot both be false");
        return ge::GRAPH_FAILED;
    }

    // 确定输出模式（空 Tensor 分支同样需要）
    uint64_t outputCount;
    if (gradXAttr && !gradYAttr) {
        outputCount = TPL_OUTPUT_Y1_ONLY; // 只输出 y1
    } else if (!gradXAttr && gradYAttr) {
        outputCount = TPL_OUTPUT_Y2_ONLY; // 只输出 y2
    } else {
        outputCount = TPL_OUTPUT_BOTH; // 双输出
    }

    if (!ValidateInputShapes(context)) {
        return ge::GRAPH_FAILED;
    }

    // 校验广播合法性并计算 broadcast shape
    gert::Shape broadcastShape;
    if (!ValidateBroadcastShape(context, &broadcastShape)) {
        return ge::GRAPH_FAILED;
    }

    // 空 Tensor 早返回：广播结果元素数为 0 时无数据可算，走自定义空分支
    if (broadcastShape.GetShapeSize() == 0) {
        return HandleEmptyTensor(context, outputCount);
    }

    // 校验输出 shape 与期望一致（无论 grad_x/grad_y 属性值）
    if (!ValidateOutputShapes(context, broadcastShape)) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "grad_x=%d, grad_y=%d, outputCount=%lu, broadcast dims=%zu", gradXAttr, gradYAttr, outputCount,
            broadcastShape.GetDimNum());

    // 获取输入 dtype
    auto inputDesc = context->GetInputDesc(0); // grads 输入
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    // 根据 dtype 执行 Tiling（outputCount 通过 TilingKey 传递）
    switch (dtype) {
        case ge::DT_FLOAT16:
            return DoTilingWithOutputCount<half>(context, outputCount);
        case ge::DT_FLOAT:
            return DoTilingWithOutputCount<float>(context, outputCount);
        case ge::DT_BF16:
            return DoTilingWithOutputCount<bfloat16_t>(context, outputCount);
        case ge::DT_INT32:
            return DoTilingWithOutputCount<int32_t>(context, outputCount);
        default:
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(context->GetNodeName(), "grads",
                                             std::to_string(static_cast<int>(dtype)).c_str(),
                                             "float16/float32/bfloat16/int32");
            return ge::GRAPH_FAILED;
    }
}

ge::graphStatus TilingParseForMaximumGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    compileInfoPtr->isAscendC = true;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaximumGrad)
    .Tiling(MaximumGradTilingFunc)
    .TilingParse<Ops::Base::BroadcastCompileInfo>(TilingParseForMaximumGrad);

} // namespace optiling
