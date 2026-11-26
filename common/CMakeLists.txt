# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(NOT ENABLE_EXPERIMENTAL)
  # --ops 指定算子时为选择性构建；all/ALL/*（忽略大小写与首尾空白）表示全量构建
  set(_is_selective_ops FALSE)
  if(DEFINED ASCEND_OP_NAME)
      string(STRIP "${ASCEND_OP_NAME}" _ops_norm)
      string(TOLOWER "${_ops_norm}" _ops_lc)
      if(NOT "${_ops_norm}" STREQUAL ""
         AND NOT "${_ops_lc}" STREQUAL "all"
         AND NOT "${_ops_norm}" STREQUAL "*")
          set(_is_selective_ops TRUE)
      endif()
  endif()
  file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
  foreach(SUB_DIR ${CURRENT_DIRS})
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/framework/CMakeLists.txt")
          add_subdirectory(${SUB_DIR}/framework)
      endif()
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_graph/CMakeLists.txt")
          # 选择性构建时跳过 common/src/op_graph 的 legacy 融合桩，避免桩指向空实现
          if(NOT ("${SUB_DIR}" STREQUAL "src" AND _is_selective_ops))
              add_subdirectory(${SUB_DIR}/op_graph)
          else()
              message(STATUS "skip common/src/op_graph (selective --ops=${ASCEND_OP_NAME})")
          endif()
      endif()
  endforeach()
endif()
