/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACCUMULATE_NV2_GRAPH_INFER_H_
#define ACCUMULATE_NV2_GRAPH_INFER_H_

#include "graph/ge_error_codes.h"

namespace ge {

// InferShape is served by the V2 gert implementation in
// op_host/accumulate_nv2_infershape.cpp (core logic in
// accumulate_nv2_infershape.h). The legacy V1 registration is removed:
// it was verified through the GEIR V1 route against the same V2 core and then
// dropped. This module only provides the InferDataType registration.

} // namespace ge

#endif // ACCUMULATE_NV2_GRAPH_INFER_H_
