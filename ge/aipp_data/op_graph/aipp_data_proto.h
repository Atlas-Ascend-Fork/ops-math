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
 * \file aipp_data_proto.h
 * \brief Definition of the AippData operator.
 */
#ifndef AIPP_DATA_PROTO_H_
#define AIPP_DATA_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Performs this op is for dynamic aipp.If you set aipp-mode to dynamic
 in aipp config file, framework will auto add one input node to graph at last. \n

 *@par Inputs:
 *data: An NCHW or NHWC tensor of type uint8, specifying the input to the data layer. \n

 *@par Attributes:
 *index: specify aipp serial num \n

 *@par Outputs:
 *out: The AIPP-processed output tensor of all types. \n

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator AippData.
 *@par Restrictions:
 *Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it
 directly.
 */

#ifndef OPS_PROTO_DEF_AIPPDATA
#define OPS_PROTO_DEF_AIPPDATA
REG_OP(AippData)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(AippData)
#endif
} // namespace ge
#endif // AIPP_DATA_PROTO_H_
