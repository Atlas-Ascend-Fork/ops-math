#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 算子自定义 ST 执行脚本模板（置于 <算子>/tests/st/<arch>/ 下，替代 ops_st_test.sh
# 里写死的 python3 -m ttk 命令）。用法见: bash st_test.sh --help
# 退出码 0=通过 非0=存在失败。刻意不用 set -e：单用例失败要继续跑完剩余用例。
set -uo pipefail

usage() {
    cat <<'EOF'
用法: bash st_test.sh [选项]

算子自定义 ST 执行脚本模板。未指定的参数使用默认值:
在用例目录下按文件名关键字扫描测试 csv 并全部执行。

选项:
  --test_path <dir>      用例扫描目录，默认脚本所在目录。
                         在该目录下按文件名关键字归类: 含kernel→kernel用例, 含aclnn→aclnn用例, 含e2e→e2e用例
                         (自动排除 *_result.csv 结果文件)
  --ttk_path <dir>       ops-test-kit 目录，默认 当前目录/ops_test_kit
  --output_path <dir>    结果输出目录，默认脚本所在目录，结果命名为 <用例csv文件名>_result.csv
  --exec_type <type>     执行类型 kernel|aclnn|e2e|all, 默认 all
  -h, --help             显示本帮助

示例:
  bash st_test.sh                                        # 跑脚本同级目录全部用例(all)
  bash st_test.sh --exec_type kernel                     # 只跑 kernel
  bash st_test.sh --test_path ./arch35 --output_path /tmp/st_log
EOF
}

# stage 1 脚本参数初始化
# 获取当前脚本所在目录
current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 设置参数初始值
ttk_path="${current_dir}/ops_test_kit"
test_path="${current_dir}"
plugin="${current_dir}/../.."        # 当前算子所在的 test 目录
output_path="${current_dir}"
exec_type="all"

# stage 2 脚本输入参数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ttk_path)    ttk_path="$2";    shift 2;;
        --test_path)   test_path="$2";   shift 2;;
        --output_path) output_path="$2"; shift 2;;
        --exec_type)   exec_type="$2";   shift 2;;
        -h|--help)     usage; exit 0;;
        *)             echo "[st_test] WARNING: 未知参数 $1, 已忽略" >&2; shift;;
    esac
done

# stage 2.5 参数校验：空输出路径直接拒绝，避免落到 mkdir 底层报错
[[ -n "${output_path}" ]] || { echo "[st_test] ERROR: --output_path 不能为空" >&2; exit 1; }

# 相对路径转绝对（stage 4 会 cd 到 ttk_path，必须在此之前完成）
to_abs() { [[ "$1" == /* ]] && echo "$1" || echo "$(pwd)/$1"; }
ttk_path="$(to_abs "${ttk_path}")"
test_path="$(to_abs "${test_path}")"
output_path="$(to_abs "${output_path}")"
plugin="$(to_abs "${plugin}")"

# stage 3 检查参数的路径是否真实存在
if [[ -d "${test_path}" ]]; then
    echo "[st_test] 用例路径是${test_path}" >&2
else
    echo "[st_test] ERROR: 用例路径不存在: ${test_path}" >&2
    exit 1
fi
if [[ -d "${ttk_path}" ]]; then
    echo "[st_test] ttk路径是${ttk_path}" >&2
else
    echo "[st_test] ERROR: ttk路径不存在: ${ttk_path}" >&2
    exit 1
fi
[[ -n "${output_path}" ]] && echo "[st_test] output_path是${output_path}" >&2
[[ -n "${exec_type}" ]] && echo "[st_test] exec_type是${exec_type}" >&2

# 在 test_path 下按关键字扫描用例 csv（排除 *_result.csv），存在就打印回显
kernel_csv=()
for f in "${test_path}"/*kernel*.csv; do
    if [[ -f "$f" && "$f" != *_result.csv ]]; then
        kernel_csv+=("$f")
        echo "[st_test] kernel测试csv路径是$f" >&2
    fi
done
aclnn_csv=()
for f in "${test_path}"/*aclnn*.csv; do
    if [[ -f "$f" && "$f" != *_result.csv ]]; then
        aclnn_csv+=("$f")
        echo "[st_test] aclnn测试csv路径是$f" >&2
    fi
done
e2e_csv=()
for f in "${test_path}"/*e2e*.csv; do
    if [[ -f "$f" && "$f" != *_result.csv ]]; then
        e2e_csv+=("$f")
        echo "[st_test] e2e测试csv路径是$f" >&2
    fi
done

# stage 4 执行用例
cd "${ttk_path}" || { echo "[st_test] ERROR: 无法进入ttk路径: ${ttk_path}" >&2; exit 1; }

rc=0
run_case_list() {    # $1=执行类型 其余=该类型的用例csv全路径
    local mode="$1" csv out
    shift
    if [[ $# -eq 0 ]]; then
        echo "[st_test] 没有${mode}测试文件, 没有执行ttk测试命令" >&2
        return 0
    fi
    local flags=""
    case "${mode}" in
        kernel) flags="-d=false -b=release" ;;
        e2e)    flags="--run 1" ;;
        aclnn)  flags="" ;;
    esac
    for csv in "$@"; do
        out="${output_path}/$(basename "${csv}" .csv)_result.csv"
        echo "[st_test] 执行${mode}测试: ${csv}" >&2
        # shellcheck disable=SC2086
        python3 -m ttk "${mode}" -i "${csv}" -o "${out}" --plugin "${plugin}" ${flags} \
            --pc=8 --warmup=false --compare close --task-prof false --no-memory-check || rc=1
    done
}

case "${exec_type}" in
    kernel) run_case_list kernel ${kernel_csv[@]+"${kernel_csv[@]}"} ;;
    aclnn)  run_case_list aclnn ${aclnn_csv[@]+"${aclnn_csv[@]}"} ;;
    e2e)    run_case_list e2e ${e2e_csv[@]+"${e2e_csv[@]}"} ;;
    all)    run_case_list kernel ${kernel_csv[@]+"${kernel_csv[@]}"}
            run_case_list aclnn ${aclnn_csv[@]+"${aclnn_csv[@]}"}
            run_case_list e2e ${e2e_csv[@]+"${e2e_csv[@]}"} ;;
    *)      echo "[st_test] 执行类型错误，没有执行ttk测试命令" >&2 ; rc=1 ;;
esac

exit ${rc}
