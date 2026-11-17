/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "select_top_k_tiling_common.h"
namespace optiling {
// 统一规划普通路径与大 K 的 GM 路径，通过 tiling key 选择设备实现。
static ge::graphStatus TileSelectTopK(gert::TilingContext* context) { return TileSelectTopKPlan(context); }
struct SelectTopKCompileInfo {};
// 扩展模块无需额外编译期数据，返回解析成功。
static ge::graphStatus ParseSelectTopK(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(SelectTopK).Tiling(TileSelectTopK).TilingParse<SelectTopKCompileInfo>(ParseSelectTopK);
} // namespace optiling
