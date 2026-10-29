# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# PyPTO 内核（op_kernel/pdist.py 与 op_host/arch35 的 tiling、根目录 def/infershape）仅覆盖
# Ascend 950（reg-base 指令集，运行期由 arch35/pdist_tiling.cpp 的 PdistIsRegbaseArch 校验）；
# 其余平台沿用 CANN 内置 Pdist 实现，op_api 组合层保持 master 既有全平台编译行为。
#
# PyPTO算子接入两步（参照 examples/add_example_pypto，仅 950 生效）：
#   1) require_pypto_pro：环境中没有pypto_pro时告警并跳过本算子，避免configure失败
#   2) enable_pypto_kernel：configure阶段对op_kernel/pdist.py做codegen，生成tiling/tilingkey
#      头文件并force-include到本算子的tiling源文件；同时把本算子登记为PyPTO算子，
#      使其wrapper改为调用pypto_compile_op
# 注意 enable_pypto_kernel 必须在 add_all_modules_sources 之前调用
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    require_pypto_pro(pdist)
    enable_pypto_kernel(pdist)
    # tiling 位于 op_host/arch35（对齐 math 仓 arch35 布局）；kernel（pdist.py）保留在
    # op_kernel 根目录——enable_pypto_kernel 的 codegen 与 opc 交付路径均按根目录定位
    add_all_modules_sources(OPTYPE pdist ACLNNTYPE aclnn_exclude
                            COMPUTE_UNIT ascend950 TILING_DIR arch35)
else()
    # 非 950：不编译 PyPTO 的 op_host（tiling/def/infershape 交由内置算子实现承接），
    # 仅编译 op_api 组合层（与 master 行为一致），并自行处理 --ops 过滤。
    get_filename_component(OP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    if(NEED_COMPILE_OPS AND (NOT OP_NAME IN_LIST NEED_COMPILE_OPS))
        return()
    endif()
    if(OP_NAME IN_LIST COMPILED_OPS)
        message(STATUS "already compiled ${OP_NAME}, skip")
        return()
    endif()
    set(COMPILED_OPS
        ${COMPILED_OPS} ${OP_NAME}
        CACHE STRING "Compiled Ops" FORCE
      )
    set(COMPILED_OP_DIRS
        ${COMPILED_OP_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
        CACHE STRING "Compiled Ops Dirs" FORCE
      )
    add_opbase_modules()
    file(GLOB OPAPI_L0_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_api/*.cpp)
    list(FILTER OPAPI_L0_SRCS EXCLUDE REGEX ".*/aclnn_[^/]*$")
    if(OPAPI_L0_SRCS)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L0_SRCS})
    endif()
    file(GLOB OPAPI_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_*.h)
    if(OPAPI_HEADERS)
        target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
    endif()
    file(GLOB OPAPI_L2_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_*.cpp)
    if(OPAPI_L2_SRCS)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L2_SRCS})
    endif()
endif()
