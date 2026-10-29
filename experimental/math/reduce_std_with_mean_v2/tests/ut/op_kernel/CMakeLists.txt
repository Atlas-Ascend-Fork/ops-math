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
    if(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
        list(FIND ASCEND_OP_NAME reduce_std_with_mean_v2 INDEX)
        if("${ASCEND_OP_NAME}" STREQUAL "" OR NOT INDEX EQUAL -1)
            # Workaround: 不用公共 AddOpTestCase——其 gen_tiling_head_file.py 依赖
            # CANN tbe python 包（CI 的 CANN 9.2.0 缺 tvm._ffi._cy3.core 会失败）。
            # 用预生成的通用 tiling 宏头（reduce_std_with_mean_v2_tiling_data_gen.h，
            # 与 AddOpTestCase 的 gen 产物同源）替代 gen 步骤；具体结构
            # ReduceStdWithMeanV2TilingData 由 kernel 实现 include 的
            # op_kernel/reduce_std_with_mean_v2_tiling_data.h 提供。
            set(socVersion Ascend910B1)
            set(tilingFile ${CMAKE_CURRENT_SOURCE_DIR}/reduce_std_with_mean_v2_tiling_data_gen.h)
            file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_std_with_mean_v2*.cpp)
            # kernel 实现由用例源 #include（test 文件 include op_kernel/reduce_std_with_mean_v2.cpp），
            # 与 AddOpTestCase 相同；无需单独编译 kernel 源。
            add_library(reduce_std_with_mean_v2_${socVersion}_cases_obj OBJECT ${OPKERNEL_CASES_SRC})
            target_compile_options(reduce_std_with_mean_v2_${socVersion}_cases_obj PRIVATE
                -g -include ${tilingFile} -DDTYPE_SELF=half -DUT_SOC_VERSION="${socVersion}")
            target_include_directories(reduce_std_with_mean_v2_${socVersion}_cases_obj PRIVATE
                ${GTEST_INCLUDE}
                ${ASCEND_DIR}/include/base/context_builder
                ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
                ${PROJECT_SOURCE_DIR}/tests/ut/common
                ${OP_TILING_INCLUDE}
                ${dlog_MATH_INCLUDE_DIR}
                ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/
                ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/
                ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/
                ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/)
            target_link_libraries(reduce_std_with_mean_v2_${socVersion}_cases_obj PRIVATE
                $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                tikicpulib::${socVersion} GTest::gtest)
            # 聚合到公共 op_kernel UT 用例对象库（与 AddOpTestCase 相同的组装机制）
            if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
                add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT
                    $<TARGET_OBJECTS:reduce_std_with_mean_v2_${socVersion}_cases_obj>)
            else()
                target_sources(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE
                    $<TARGET_OBJECTS:reduce_std_with_mean_v2_${socVersion}_cases_obj>)
            endif()
            target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE
                $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                $<TARGET_OBJECTS:reduce_std_with_mean_v2_${socVersion}_cases_obj>)
            set(fastOpTestSocVersions ${fastOpTestSocVersions} ${socVersion}
                CACHE STRING "fastOp Test SocVersions" FORCE)
        endif()
    endif()
endif()
