# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(OPTYPE reduce_std_with_mean_v2 ACLNNTYPE aclnn DEPENDENCIES reduce_std_v2)

# Workaround: asc_opc imports "dynamic.ReduceStdWithMeanV2" (PascalCase) but
# TBE generates "reduce_std_with_mean_v2.py". Create a symlink.
# The target file doesn't need to exist yet for a symlink.
set(_DST_DIR ${CMAKE_BINARY_DIR}/tbe/dynamic)
file(MAKE_DIRECTORY ${_DST_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  reduce_std_with_mean_v2.py
  ${_DST_DIR}/ReduceStdWithMeanV2.py
  WORKING_DIRECTORY ${_DST_DIR}
  ERROR_QUIET
)

# two-stage kernels (ReduceStdWithMeanV2Partial/Merge) live in this
# module's op_kernel/ dir, but the impl-gen (ascendc_impl_build.py) resolves
# each op type's kernel source at ${tbe/ascendc}/{op_file}/{kernel_src}.cpp.
# Mirror the entry + headers into the per-op-type dirs at configure time so
# the dynamic compile drivers (dynamic/reduce_std_with_mean_v2_partial.py
# / _merge.py) and the kernel binaries are generated.
set(_TWO_STAGE_IMPL_DIR ${CMAKE_BINARY_DIR}/tbe/ascendc)
set(_TWO_STAGE_OP_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
foreach(_op_type reduce_std_with_mean_v2_partial reduce_std_with_mean_v2_merge)
  set(_op_dir ${_TWO_STAGE_IMPL_DIR}/${_op_type})
  file(MAKE_DIRECTORY ${_op_dir})
  file(COPY ${_TWO_STAGE_OP_KERNEL}/${_op_type}.cpp
            ${_TWO_STAGE_OP_KERNEL}/${_op_type}_tiling_key.h
            ${_TWO_STAGE_OP_KERNEL}/reduce_std_with_mean_v2_two_stage_kernels.h
            ${_TWO_STAGE_OP_KERNEL}/reduce_std_with_mean_v2_tiling_data.h
       DESTINATION ${_op_dir})
endforeach()

# Compile the two-stage kernel binaries for the package build. The standard flow only
# compiles the module's eponymous op type, so Partial/Merge are compiled with
# asc_opc directly (self binary configs in op_host/config/ascend910b/), after
# the impl-gen has produced the dynamic drivers (.impl_timestamp).
if(ENABLE_PACKAGE)
  add_custom_target(reduce_std_with_mean_v2_two_stage_kernels ALL
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/op_host/config/ascend910b/build_two_stage_kernels.sh
            ${CMAKE_SOURCE_DIR}
    DEPENDS ${CMAKE_BINARY_DIR}/tbe/.impl_timestamp
    COMMENT "Compile two-stage kernel binaries (Partial/Merge)"
    VERBATIM
  )
  # Install the two-stage kernel binaries, their runtime config jsons and the
  # regenerated binary_info_config.json (the standard flow only installs the
  # module's eponymous op type).
  set(_TWO_STAGE_BIN_DIR ${CMAKE_BINARY_DIR}/binary/ascend910b/bin/ascend910b)
  set(_TWO_STAGE_CFG_DIR ${CMAKE_BINARY_DIR}/binary/ascend910b/bin/config/ascend910b)
  install(DIRECTORY ${_TWO_STAGE_BIN_DIR}/reduce_std_with_mean_v2_partial
          DESTINATION ${BIN_KERNEL_INSTALL_DIR}/ascend910b)
  install(DIRECTORY ${_TWO_STAGE_BIN_DIR}/reduce_std_with_mean_v2_merge
          DESTINATION ${BIN_KERNEL_INSTALL_DIR}/ascend910b)
  install(FILES ${_TWO_STAGE_CFG_DIR}/reduce_std_with_mean_v2_partial.json
                ${_TWO_STAGE_CFG_DIR}/reduce_std_with_mean_v2_merge.json
          DESTINATION ${BIN_KERNEL_CONFIG_INSTALL_DIR}/ascend910b)
  install(FILES ${_TWO_STAGE_CFG_DIR}/binary_info_config.json
          DESTINATION ${BIN_KERNEL_CONFIG_INSTALL_DIR}/ascend910b
          OPTIONAL)
endif()

# ---------------------------------------------------------------------------
# Workaround: UT 模式（-u）不构建/安装 custom opp 包（opp/op_proto/custom），
# L2 API 内部 l0op 调用的自定义算子（ReduceStdWithMeanV2 等）inferShape 注册
# 缺失，op_api UT 会报 "The inferShape function does not exist"（CI 无 custom
# 包环境必现）。此处把 infer 源文件编入 opapi 对象库（${OPHOST_NAME}_opapi_obj，
# 创建于本模块的祖先目录，UT 模式被 libopapi_math_ut.so 引用），使 UT 不依赖
# 已安装的 custom 包即可完成进程内注册；非 UT 构建（--pkg）不触发。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Workaround: UT 模式（-u）不构建/安装 custom opp 包，L2 API 内部 l0op 调用的
# 自定义算子（ReduceStdWithMeanV2 等）inferShape 注册缺失，op_api UT 报
# "The inferShape function does not exist"（CI 无 custom 包环境必现）。
# build.sh -u 会把 ASCEND_CUSTOM_OPP_PATH 指向 build 目录，UT 运行时
# LoadAllOppPackage 从该路径加载 opp 包库（含 inferShape 注册）。此处构建
# 一个仅含本模块 infer 注册的 opp 包库并拷入 build 目录的 op_proto/custom
# 路径，使 UT 不依赖已安装的 custom 包；非 UT 构建（--pkg）不触发。
# ---------------------------------------------------------------------------
if(ENABLE_TEST)
  add_library(rswmv2_infer_reg SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/reduce_std_with_mean_v2_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/reduce_std_with_mean_v2_two_stage_infershape.cpp)
  target_include_directories(rswmv2_infer_reg PRIVATE ${OP_PROTO_INCLUDE})
  target_compile_definitions(rswmv2_infer_reg PRIVATE
    ASCEND_OPSPROTO_UT OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO")
  target_compile_options(rswmv2_infer_reg PRIVATE -Dgoogle=ascend_private)
  target_link_directories(rswmv2_infer_reg PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
  target_link_libraries(rswmv2_infer_reg PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> $<BUILD_INTERFACE:dlog_headers>
    opp_registry exe_graph
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>)
  set(_RSWMV2_OPP_DST ${CMAKE_BINARY_DIR}/op_proto/custom/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  add_custom_command(TARGET rswmv2_infer_reg POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_RSWMV2_OPP_DST}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rswmv2_infer_reg> ${_RSWMV2_OPP_DST}/
    COMMENT "Copy rswmv2_infer_reg.so to UT opp package path"
    VERBATIM)
  # CI 只构建 op_api UT 目标，需保证本库先于其构建（经用例对象库传递依赖）。
  if(TARGET ${OP_API_MODULE_NAME}_cases_obj)
    add_dependencies(${OP_API_MODULE_NAME}_cases_obj rswmv2_infer_reg)
    # 测试用例在启动时 dlopen 该注册库（LoadAllOppPackage 只加载
    # opp/vendors 路径的包，UT 构建产物不可达，需进程内加载完成注册）。
    target_compile_definitions(${OP_API_MODULE_NAME}_cases_obj PRIVATE
      RSWMV2_INFER_REG_PATH="${_RSWMV2_OPP_DST}/librswmv2_infer_reg.so")
  endif()
endif()
