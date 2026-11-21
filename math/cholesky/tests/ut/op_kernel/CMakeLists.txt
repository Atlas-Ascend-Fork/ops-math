# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    if (ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
        set(cholesky_tiling_files
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/cholesky_tiling_arch35.cpp)

        # AddOpsTestCase's recursive glob only resolves a basename, while the product route
        # value "arch35/cholesky_arch35" never matches on the UT side. Keep this local
        # override paired with a configure-time assertion so A5 cannot compile the arch22 entry;
        # the arch suffix keeps the basename unique across the whole operator directory.
        list(INSERT KERNEL_SRC_LIST 0 "Cholesky ascend950 cholesky_arch35")
        file(GLOB cholesky_ut_kernel_entries
            "${PROJECT_SOURCE_DIR}/*/cholesky/op_kernel/*/cholesky_arch35.cpp")
        list(LENGTH cholesky_ut_kernel_entries cholesky_ut_kernel_entry_num)
        if(NOT cholesky_ut_kernel_entry_num EQUAL 1)
            message(FATAL_ERROR "cholesky ascend950 UT must use exactly op_kernel/arch35/cholesky_arch35.cpp, "
                                "got [${cholesky_ut_kernel_entries}]")
        endif()
        AddOpTestCase(cholesky "ascend950" "-D__CCE_UT_TEST__" "${cholesky_tiling_files}")
    elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93")
        set(cholesky_tiling_files
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch22/cholesky_tiling_arch22.cpp)
        list(INSERT KERNEL_SRC_LIST 0 "Cholesky ascend910_93 cholesky_arch22")
        AddOpTestCase(cholesky "ascend910_93" "-D__CCE_UT_TEST__" "${cholesky_tiling_files}")
    elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
        set(cholesky_tiling_files
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch22/cholesky_tiling_arch22.cpp)
        list(INSERT KERNEL_SRC_LIST 0 "Cholesky ascend910b cholesky_arch22")
        AddOpTestCase(cholesky "ascend910b" "-D__CCE_UT_TEST__" "${cholesky_tiling_files}")
    endif()
endif()
