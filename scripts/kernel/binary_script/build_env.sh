#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

declare -A SOC_MAP
SOC_MAP=([Ascend310P]="Ascend310P3"
          [Ascend310B]="Ascend310B1"
          [Ascend910]="Ascend910A"
          [Ascend910B]="Ascend910B1"
          [Ascend910_93]="Ascend910_9391"
          [Ascend950]="Ascend950PR_9599"
          [Ascend960DT]="Ascend960DT_968B7"
          [Ascend960PR]="Ascend960PR_969A1"
          [Ascend350]="Ascend350_355e"
          [KirinX90]="KirinX90"
          [Kirin9030]="Kirin9030"
          [Ascend5162]="Ascend5162A"
)

OPC_TASK_NAME="opc_cmd.sh"
OUT_TASK_NAME="out_cmd.sh"
OP_CATEGORY_LIST="math conversion examples"

function get_binary_config_file() {
  local workdir="$1"
  local soc_version_lower="$2"
  local op_file_name="${3%_apt}"
  local topdir
  topdir=$(readlink -f -- "${workdir}/../../..") || return 1

  local primary_config="${topdir}/build/tbe/config/${soc_version_lower}/${op_file_name}/${op_file_name}_binary.json"
  if [[ -f "${primary_config}" ]]; then
    printf '%s\n' "${primary_config}"
    return 0
  fi

  local category category_dir candidates
  for category in ${OP_CATEGORY_LIST}; do
    category_dir="${topdir}/${category}"
    [[ -d "${category_dir}" ]] || continue
    if ! candidates=$(find "${category_dir}" -xtype f \
      -path "*/op_host/config/${soc_version_lower}/${op_file_name}_binary.json" | LC_ALL=C sort); then
      printf '[WARNING] Could not search all of %s for binary configs\n' "${category_dir}" >&2
    fi
    if [[ -n "${candidates}" ]]; then
      local selected_config="${candidates%%$'\n'*}"
      if [[ "${candidates}" != "${selected_config}" ]]; then
        printf '[WARNING] Multiple binary configs for %s on %s in %s, using %s\n' \
          "${op_file_name}" "${soc_version_lower}" "${category}" "${selected_config}" >&2
      fi
      printf '%s\n' "${selected_config}"
      return 0
    fi
  done
  return 0
}

function generate_source_binary_config() {
  local python_arg="$1"
  local config="$2"
  local derived_config="$3"
  local impl_mode="$4"
  local task_count="$5"
  ${python_arg} gen_opc_json_with_impl_mode.py "${config}" "${derived_config}" "${impl_mode}" || return 1
  [[ -s "${derived_config}" ]] || return 1
  ${python_arg} gen_opc_json_with_threadnum.py "${derived_config}" "${task_count}" || return 1

  local index
  for ((index = 0; index < task_count; index++)); do
    [[ -s "${derived_config}_${index}" ]] || return 1
  done
}

function reset_binary_compile_json() {
  local op_type="$1"
  local manifest="$2"
  [[ -f "${manifest}" ]] || return 0

  printf '[WARNING] op:%s will clean %s\n' "${op_type}" "${manifest}"
  if ! rm -f -- "${manifest}"; then
    # A shared output directory can allow file writes while denying removal.
    printf '[WARNING] Resetting %s in place\n' "${manifest}" >&2
    printf '%s\n' '{"binList":[]}' > "${manifest}" || return 1
  fi
}

function trans_soc() {
  local _soc_input=$1
  local _soc_input_lower=${_soc_input,,}
  for key in ${!SOC_MAP[*]}; do
    key_lower=${key,,}
    if [ "${_soc_input_lower}" == "${key_lower}" ]; then
      echo ${SOC_MAP[$key]}
      return
    fi
  done

  echo ${_soc_input}
}
