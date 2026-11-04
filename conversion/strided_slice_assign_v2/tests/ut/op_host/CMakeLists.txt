# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    # tiling 目录单一来源：按 ASCEND_COMPUTE_UNIT 查算子 CMakeLists 的
    # SUPPORT_COMPUTE_UNIT / SUPPORT_TILING_DIR 映射表（父目录作用域可见），
    # 避免 arch35 平台清单在算子构建与 UT 两处漂移
    list(FIND SUPPORT_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}" _SSA_CU_IDX)
    if(_SSA_CU_IDX GREATER_EQUAL 0)
        list(GET SUPPORT_TILING_DIR ${_SSA_CU_IDX} SSA_UT_TILING_DIR)
    else()
        # 未在映射表中的 compute unit（UT 默认基线平台）回退 arch22
        set(SSA_UT_TILING_DIR arch22)
    endif()
    add_modules_ut_sources(
        UT_NAME ${OP_TILING_MODULE_NAME}
        MODE PRIVATE
        DIR ${CMAKE_CURRENT_SOURCE_DIR}
        TILING_DIR ${SSA_UT_TILING_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
