# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(view_copy_header LANGUAGES CXX)
get_filename_component(OPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(OPHOST_NAME test)
include(${OPS_ROOT}/cmake/func.cmake)
include(${OPS_ROOT}/cmake/opbuild.cmake)
add_library(opbase_util_objs INTERFACE)
add_library(test_opapi_obj OBJECT EXCLUDE_FROM_ALL)
foreach(kind aclnn inner exclude)
  if(kind STREQUAL "inner")
    add_library(test_opdef_aclnn_inner_obj INTERFACE)
  elseif(kind STREQUAL "exclude")
    add_library(test_opdef_aclnn_exclude_obj INTERFACE)
  else()
    add_library(test_opdef_aclnn_obj INTERFACE)
  endif()
endforeach()
add_library(test_aclnn_exclude_headers INTERFACE)
set(NEED_COMPILE_OPS contiguous)
set(COMPILED_OPS "" CACHE STRING "Compiled Ops" FORCE)
add_subdirectory(${OPS_ROOT}/conversion/contiguous/op_host contiguous)
set(ASCEND_AUTOGEN_PATH ${CMAKE_CURRENT_BINARY_DIR}/autogen)
set(ENABLE_PACKAGE ON)
set(ACLNN_INC_INSTALL_DIR include/aclnnop)
set(ACLNN_INC_LEVEL2_INSTALL_DIR include/aclnnop/level2)
set(ASCEND_PYTHON_EXECUTABLE python3)
# Use the real aggregate-header template without copying files into the source tree.
set(CMAKE_CURRENT_SOURCE_DIR ${OPS_ROOT})
set(PROJECT_SOURCE_DIR ${OPS_ROOT})
if(ENABLE_CUSTOM)
  set(VENDOR_NAME header_test)
  set(ACLNN_INC_INSTALL_DIR vendors/header_test_math/op_api/include)
endif()
gen_aclnn_with_opdef()
# Installation must run before building the consumer; it only uses staged headers.
if(ENABLE_CUSTOM)
  set(master_header aclnn_ops_math_header_test.h)
else()
  set(master_header aclnn_ops_math.h)
endif()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp
  "#include <${master_header}>\n#include <type_traits>\n"
  "using Phase1 = aclnnStatus (*)(const aclTensorList*, const aclTensorList*, uint64_t*, aclOpExecutor**);\n"
  "using Phase2 = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);\n"
  "static_assert(std::is_same_v<decltype(&aclnnViewCopyGetWorkspaceSize), Phase1>);\n"
  "static_assert(std::is_same_v<decltype(&aclnnViewCopy), Phase2>);\n"
  "#include <aclnn_view_copy.h>\n"
  "Phase1 phase1 = &aclnnViewCopyGetWorkspaceSize;\nPhase2 phase2 = &aclnnViewCopy;\n"
  "int main() { return phase1 && phase2 ? 0 : 1; }\n")
add_executable(consumer ${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp)
target_compile_features(consumer PRIVATE cxx_std_17)
target_include_directories(consumer PRIVATE
  ${CMAKE_INSTALL_PREFIX}/${ACLNN_INC_INSTALL_DIR}
  ${CANN_ROOT}/include ${CANN_ROOT}/include/aclnnop)
find_library(OPAPI_MATH opapi_math PATHS ${CANN_ROOT}/lib64 NO_DEFAULT_PATH REQUIRED)
target_link_libraries(consumer PRIVATE ${OPAPI_MATH})
