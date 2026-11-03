/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "op_graph/graph_infer_comm_funcs.h"
#include "graph/infer_format_context.h"

using namespace ge;
namespace ops {

static void SetBothFormatToNd(gert::StorageFormat* fmt1, gert::StorageFormat* fmt2)
{
    fmt1->SetOriginFormat(FORMAT_ND);
    fmt1->SetStorageFormat(FORMAT_ND);
    fmt2->SetOriginFormat(FORMAT_ND);
    fmt2->SetStorageFormat(FORMAT_ND);
}

static std::vector<int64_t> ReadPermValues(const gert::Tensor* permTensor)
{
    const size_t permSize = static_cast<size_t>(permTensor->GetShapeSize());
    std::vector<int64_t> permList;
    if (permTensor->GetDataType() == DT_INT32) {
        const int32_t* data = permTensor->GetData<int32_t>();
        for (size_t i = 0; i < permSize; ++i) {
            permList.push_back(data[i]);
        }
    } else {
        const int64_t* data = permTensor->GetData<int64_t>();
        for (size_t i = 0; i < permSize; ++i) {
            permList.push_back(data[i]);
        }
    }
    return permList;
}

static void ResolveNchwNhwc(const std::vector<int64_t>& permList, const ge::Format inOrigin, const ge::Format outOrigin,
                            ge::Format& newIn, ge::Format& newOut)
{
    const std::vector<int64_t> nchwToNhwc = {0, 2, 3, 1};
    const std::vector<int64_t> nhwcToNchw = {0, 3, 1, 2};
    const bool isNhwcToNchw = (permList == nhwcToNchw);
    const bool isNchwToNhwc = (permList == nchwToNhwc);

    if (inOrigin == FORMAT_ND) {
        if (outOrigin == FORMAT_NCHW && isNhwcToNchw) {
            newIn = FORMAT_NHWC;
        } else if (outOrigin == FORMAT_NHWC && isNchwToNhwc) {
            newIn = FORMAT_NCHW;
        } else {
            newOut = FORMAT_ND;
        }
    } else {
        if (inOrigin == FORMAT_NCHW && isNchwToNhwc) {
            newOut = FORMAT_NHWC;
        } else if (inOrigin == FORMAT_NHWC && isNhwcToNchw) {
            newOut = FORMAT_NCHW;
        } else {
            newIn = FORMAT_ND;
            newOut = FORMAT_ND;
        }
    }
}

static ge::graphStatus InferFormat4Transpose(gert::InferFormatContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    constexpr size_t kIdxX = 0;
    constexpr size_t kIdxY = 0;
    gert::StorageFormat* inFmt = context->GetInputFormat(kIdxX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inFmt);
    gert::StorageFormat* outFmt = context->GetOutputFormat(kIdxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, outFmt);

    const gert::Tensor* permTensor = context->GetInputTensor(1);
    if (permTensor == nullptr || permTensor->GetAddr() == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const ge::DataType permDtype = permTensor->GetDataType();
    if (permDtype != DT_INT32 && permDtype != DT_INT64) {
        SetBothFormatToNd(inFmt, outFmt);
        return ge::GRAPH_SUCCESS;
    }

    const ge::Format inOrigin = inFmt->GetOriginFormat();
    const ge::Format outOrigin = outFmt->GetOriginFormat();
    if (inOrigin == FORMAT_ND && outOrigin == FORMAT_ND) {
        return ge::GRAPH_SUCCESS;
    }

    ge::Format newIn = inOrigin;
    ge::Format newOut = outOrigin;
    ResolveNchwNhwc(ReadPermValues(permTensor), inOrigin, outOrigin, newIn, newOut);

    inFmt->SetOriginFormat(newIn);
    inFmt->SetStorageFormat(newIn);
    outFmt->SetOriginFormat(newOut);
    outFmt->SetStorageFormat(newOut);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Transpose).InferDataType(InferDataTypeOutputSameAsInput).InferFormat(InferFormat4Transpose);
} // namespace ops
