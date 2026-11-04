# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 需要将Tiling依赖的文件添加到CMakeLists.txt中
    # set(elewise_common_tiling_files
    #         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
    #         )
    # 算子自己的tiling文件路径
    if (ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
        set(sinkhorn_tiling_files
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/sinkhorn_tiling_arch35.cpp
            )
    else()
        set(sinkhorn_tiling_files
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch22/sinkhorn_tiling_arch22.cpp
            )
    endif()
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本，多个以分号分隔，例如："ascend950;ascend910b"
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合，不需要则传入空字符串，例如："-DDTYPE_X=float"，多个使用空格分隔，例如："-DDTYPE_X=float -DDTYPE_Y=float"
    # param4：该算子依赖的所有tiling源码文件
    if (ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
        # cmake/ut.cmake:524-527 用 "op_kernel/**/<KERNEL_SRC>.cpp" 定位入口，而 CMake file(GLOB)
        # 的 "**" 只匹配恰好一层目录：产品路由值 "arch35/sinkhorn" 在 UT 侧恒不命中，根入口
        # "op_kernel/sinkhorn.cpp" 也恒不命中。故在 UT 目录作用域内插入 basename 条目，使 glob
        # 唯一命中 op_kernel/arch35/sinkhorn.cpp。list(INSERT) 只创建遮蔽 cache 的局部变量，
        # 不回写 CACHE INTERNAL，产品构建（cmake/opbuild.cmake:200）仍取 kernelSrc=arch35/sinkhorn。
        # 硬约束：op_kernel/ 的一级子目录中只允许存在 arch35/sinkhorn.cpp 这一个入口，
        #         arch22 实现文件必须留在 op_kernel/ 根且不得命名为 sinkhorn.cpp。
        list(INSERT KERNEL_SRC_LIST 0 "Sinkhorn ascend950 sinkhorn")
        file(GLOB sinkhorn_ut_kernel_entries
             "${PROJECT_SOURCE_DIR}/*/sinkhorn/op_kernel/*/sinkhorn.cpp")
        list(LENGTH sinkhorn_ut_kernel_entries sinkhorn_ut_kernel_entry_num)
        if(NOT sinkhorn_ut_kernel_entry_num EQUAL 1 OR
           NOT sinkhorn_ut_kernel_entries MATCHES "/op_kernel/arch35/sinkhorn\\.cpp$")
            message(FATAL_ERROR
                "sinkhorn ascend950 UT kernel entry must resolve to exactly "
                "op_kernel/arch35/sinkhorn.cpp, got [${sinkhorn_ut_kernel_entries}]")
        endif()
        AddOpTestCase(sinkhorn "ascend950" "-DDTYPE_X=float" "${sinkhorn_tiling_files}")
    elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910_93")
        AddOpTestCase(sinkhorn "ascend910_93" "-DDTYPE_X=float" "${sinkhorn_tiling_files}")
    elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
        AddOpTestCase(sinkhorn "ascend910b" "-DDTYPE_X=float" "${sinkhorn_tiling_files}")
    endif()

endif()
