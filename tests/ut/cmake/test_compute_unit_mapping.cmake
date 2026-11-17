# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

# Run without CANN: cmake -P tests/ut/cmake/test_compute_unit_mapping.cmake
cmake_minimum_required(VERSION 3.16)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/soc_arch_variables.cmake")
include("${REPO_ROOT}/cmake/func.cmake")

function(check_arch soc expected)
  get_npu_arch_by_soc_version("${soc}" actual)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR "${soc}: expected arch '${expected}', got '${actual}'")
  endif()
endfunction()

check_arch(mc62 dav_5102)
check_arch(mc62cm12a dav_5102)
check_arch(MC62CM12A dav_5102)
check_arch(mc62cm12aa dav_5102)
check_arch(MC62CM12AA dav_5102)
check_arch(ascend950 dav_3510)
check_arch(unmapped_soc "")
get_soc_version_by_npu_arch(dav_5102 canonical_socs)
if(NOT "${canonical_socs}" STREQUAL "mc62;ascend630")
  message(FATAL_ERROR "Aliases must not change canonical SoC enumeration: ${canonical_socs}")
endif()

# Isolate target creation while exercising the real Host macro and its UT forwarding.
macro(add_opbase_modules)
endmacro()
macro(add_ext_def_sources)
endmacro()
macro(add_tiling_sources dirs disable_in_opp)
  set(captured_host_dirs "${dirs}")
endmacro()
macro(add_all_ut_sources)
  cmake_parse_arguments(TEST "" "OP_NAME" "UT_TILING_DIR" ${ARGN})
  set(captured_ut_dirs "${TEST_UT_TILING_DIR}")
endmacro()

function(check_tiling soc units dirs expected)
  # This fixture has no sources, so all source globs are empty.
  set(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/compute_unit_fixture")
  set(ASCEND_COMPUTE_UNIT "${soc}")
  set(COMPILED_OPS "" CACHE STRING "Compiled Ops" FORCE)
  set(NEED_COMPILE_OPS "")
  add_all_modules_sources(OPTYPE test ACLNNTYPE aclnn_exclude
      COMPUTE_UNIT ${units} TILING_DIR ${dirs})
  if(NOT "${captured_host_dirs}" STREQUAL "${expected}")
    message(FATAL_ERROR "${soc}: expected Host dirs '${expected}', got '${captured_host_dirs}'")
  endif()
  if(NOT "${captured_ut_dirs}" STREQUAL "${expected}")
    message(FATAL_ERROR "${soc}: expected UT dirs '${expected}', got '${captured_ut_dirs}'")
  endif()
endfunction()

# complex_abs's declaration must work for both full platform names and the short name.
set(complex_abs_units ascend950 mc62cm12a ascend960dt ascend350)
set(complex_abs_dirs arch35 arch35 arch35 arch35)
check_tiling(mc62cm12a "${complex_abs_units}" "${complex_abs_dirs}" arch35)
check_tiling(MC62CM12A "${complex_abs_units}" "${complex_abs_dirs}" arch35)
check_tiling(mc62cm12aa "${complex_abs_units}" "${complex_abs_dirs}" arch35)
check_tiling(mc62 "${complex_abs_units}" "${complex_abs_dirs}" arch35)
check_tiling(ascend950 "${complex_abs_units}" "${complex_abs_dirs}" arch35)
check_tiling(mc62cm12a dav_5102 arch35 arch35)
check_tiling(mc62 "mc62;mc62cm12a;mc62cm12aa" "arch35;arch35;arch35" arch35)
# Unmapped names retain exact literal matching, with consistent case normalization.
check_tiling(unmapped_soc unmapped_soc arch_custom arch_custom)
check_tiling(UNMAPPED_SOC unmapped_soc arch_custom arch_custom)
check_tiling(unmapped_soc UNMAPPED_SOC arch_custom arch_custom)
check_tiling(unmapped_soc "unmapped_soc;unmapped_soc" "arch_custom;common" "arch_custom;common")
check_tiling(unmapped_soc different_soc arch_custom "")
message(STATUS "Compute unit mapping regression checks passed")
