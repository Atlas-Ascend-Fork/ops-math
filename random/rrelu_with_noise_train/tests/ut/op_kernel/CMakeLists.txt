# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/rrelu_with_noise_train/arch35)
    # rrelu_with_noise_train.h 以 "../../random_common/..." 相对路径引用公共头,
    # 通过 staging 符号链接 + -I 使其在 UT 编译时可解析(同 stateless_uniform)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/random/random_common/op_kernel
        ${KERNEL_STAGING_DIR}/random_common)

    # tiling 实现尚未落地(op_host/arch35/rrelu_with_noise_train_tiling_arch35.cpp 由 tiling 侧
    # 按约定提供), 落地前仅以 random_common 基类源编译 tiling 临时 so;
    # 落地后经 EXISTS 守卫自动纳入, 与 stateless_uniform 等同仓算子保持一致
    set(rrelu_with_noise_train_tiling_files
        ${PROJECT_SOURCE_DIR}/random/random_common/op_host/arch35/random_tiling_arch35.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/rrelu_with_noise_train_tiling_arch35.cpp)
        list(INSERT rrelu_with_noise_train_tiling_files 0
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/rrelu_with_noise_train_tiling_arch35.cpp)
    endif()
    AddOpTestCase(
        rrelu_with_noise_train
        "ascend950"
        "-DDTYPE_X=float -DTestUtDefaultTilingStruct=RandomUnifiedSimtTilingDataStruct -I${KERNEL_STAGING_DIR}/rrelu_with_noise_train/arch35"
        "${rrelu_with_noise_train_tiling_files}")
endif()
