# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # tiling sources needed by the kernel UT (tiling tmp so for ExecuteTiling)
    set(hypot_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch22/hypot_tiling_arch22.cpp
        )
    # param1: operator name (kernel-style snake name)
    # param2: soc versions, ';'-separated (arch22 covers Atlas A2/A3)
    # param3: extra compile options (DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT stand in for the per-combination
    #         aliases the opc framework stamps on device; the launcher TU exercises other dtype
    #         combinations through explicit template arguments)
    # param4: tiling source files
    AddOpTestCase(hypot "ascend910b" "-DDTYPE_SELF=float -DDTYPE_OTHER=float -DDTYPE_OUT=float" "${hypot_tiling_files}")
endif()
