#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""算子变更解析公共库：filelist解析、算子目录遍历、依赖展开，供各CI检查脚本共用。

路径约定：
- 主类目: <category>/<op>/...，category须在算子类目集合中，至少3段路径
- experimental: experimental/<category>/<op>/...，category须在算子类目集合中，至少4段路径
- 类目公共目录约定记为'<category>.common'
- 依赖读取本仓add_all_modules_sources的字面DEPENDENCIES，无需CANN或cmake configure
- pr_filelist仅接受git diff --name-only产出的纯路径列表（每行一个路径），不支持带状态列的格式
"""

import glob
import os
import re
import sys
from typing import Iterator, List, Optional, Set, Tuple

# 公共库目录名（不算子）
COMMON_LIB_NAMES = {"common", "sort_lib", "index_check", "foreach_utils", "pool_utils"}


def get_op_categories(repo_root: str) -> List[str]:
    """读取多行OPS_CATEGORY_LIST（兼容OP_CATEGORY_LIST）。"""
    variables_cmake = os.path.join(repo_root, "cmake", "variables.cmake")
    with open(variables_cmake, encoding="utf-8") as file:
        content = re.sub(r"#[^\n]*", "", file.read())
    match = re.search(r"set\s*\(\s*OPS?_CATEGORY_LIST\s+([^)]*)\)", content)
    if match:
        return match.group(1).replace('"', "").replace(";", " ").split()
    return []


def resolve_changed_op(
    parts: List[str], categories: Set[str], include_experimental: bool = True
) -> Optional[Tuple[str, str, bool]]:
    """按路径段解析变更路径所属算子。

    返回(prefix, second, is_common)：prefix为'<category>'或'experimental/<category>'，
    second为算子目录名（类目公共目录时second为'common'且is_common=True）；
    不属于任何类目返回None。仅做路径段判定，不做目录存在性校验。
    include_experimental=False时，experimental路径返回None。
    """
    if len(parts) < 2:
        return None
    category, second = parts[0], parts[1]
    if category in categories and len(parts) >= 3:
        return (category, second, second == "common")
    if (
        include_experimental
        and category == "experimental"
        and len(parts) >= 4
        and parts[1] in categories
    ):
        # prefix须用parts[1]拼接——此处category仍为'experimental'，不能直接引用
        return (f"experimental/{parts[1]}", parts[2], parts[2] == "common")
    return None


def is_common_lib(name: str) -> bool:
    """判断目录名是否为公共库目录（不算子）。"""
    return (
        name in COMMON_LIB_NAMES
        or name.endswith("_common")
        or name.endswith("_utils")
        or name.endswith("_util")
    )


def has_def_file(op_path: str) -> bool:
    """判断算子op_host下是否存在def文件，无def文件为接口类算子。"""
    op_host_dir = os.path.join(op_path, "op_host")
    if not os.path.isdir(op_host_dir):
        return False
    return bool(
        glob.glob(os.path.join(op_host_dir, f"{os.path.basename(op_path)}_def*.cpp"))
        or glob.glob(os.path.join(op_host_dir, "*_def*.cpp"))
    )


def iter_category_dirs(repo_root: str) -> Iterator[str]:
    """遍历所有算子类目目录（含experimental下的二级类目）。"""
    for prefix in ("", "experimental"):
        for category in get_op_categories(repo_root):
            path = os.path.join(repo_root, prefix, category)
            if os.path.isdir(path):
                yield path


def iter_op_dirs(repo_root: str) -> Iterator[str]:
    """遍历仓库内所有算子目录（不含公共库目录与无def文件的接口类算子）。"""
    for category in iter_category_dirs(repo_root):
        for entry in sorted(os.listdir(category)):
            op_path = os.path.join(category, entry)
            if not os.path.isdir(op_path) or is_common_lib(entry):
                continue
            if not has_def_file(op_path):
                continue
            if os.path.isdir(os.path.join(op_path, "op_host")) or os.path.isdir(
                os.path.join(op_path, "op_kernel")
            ):
                yield op_path


def parse_filelist_ops(
    repo_root: str, pr_filelist: str, categories: Optional[Set[str]] = None
) -> Tuple[dict, Set[str], Set[str]]:
    """解析pr_filelist（纯路径列表），返回(name_prefixes, involved_dirs, involved_names)。

    name_prefixes: 算子名 -> 命名空间前缀集合（'<category>'或'experimental/<category>'）；
    involved_dirs: 实际存在的算子或公共库目录绝对路径集合；
    involved_names: 算子名集合（类目公共目录记为'<category>.common'）。
    """
    if categories is None:
        categories = set(get_op_categories(repo_root))
    name_prefixes: dict = {}
    involved_dirs: Set[str] = set()
    involved_names: Set[str] = set()
    with open(pr_filelist, encoding="utf-8") as file:
        for line in file:
            file_path = line.strip()
            if not file_path or file_path.startswith("#"):
                continue
            if os.path.isabs(file_path):
                file_path = os.path.relpath(file_path, repo_root)
            resolved = resolve_changed_op(file_path.split("/"), categories)
            if resolved is None:
                continue
            prefix, second, is_common = resolved
            op_dir = os.path.join(repo_root, prefix, second)
            if not os.path.isdir(op_dir):
                continue
            involved_dirs.add(op_dir)
            if is_common:
                involved_names.add(f"{prefix.split('/')[-1]}.common")
                continue
            involved_names.add(second)
            name_prefixes.setdefault(second, set()).add(prefix)
    return name_prefixes, involved_dirs, involved_names


# add_all_modules_sources 的参数边界，与cmake/func.cmake一致。
MODULE_KEYS = {
    "OPTYPE",
    "ACLNNTYPE",
    "DEPENDENCIES",
    "COMPUTE_UNIT",
    "NPU_ARCH",
    "TILING_DIR",
    "DISABLE_IN_OPP",
    "HOSTCPU",
}


def declared_dependencies(op_dir: str) -> Set[str]:
    """读取本仓当前使用的字面DEPENDENCIES列表，不执行CMake代码。

    支持多行、引号和分号分隔；动态变量/生成器表达式不作猜测，发出警告。
    """
    cmake_file = os.path.join(op_dir, "CMakeLists.txt")
    try:
        with open(cmake_file, encoding="utf-8") as file:
            content = re.sub(r"#[^\n]*", "", file.read())
    except FileNotFoundError:
        return set()
    dependencies = set()
    for match in re.finditer(r"add_all_modules_sources\s*\(([^)]*)\)", content):
        in_dependencies = False
        for token in match.group(1).replace('"', "").replace(";", " ").split():
            if token in MODULE_KEYS:
                in_dependencies = token == "DEPENDENCIES"
            elif in_dependencies:
                if re.fullmatch(r"[A-Za-z0-9_]+", token):
                    dependencies.add(token)
                else:
                    print(
                        f"警告: {cmake_file} 的动态依赖 {token!r} 无法静态解析，已跳过",
                        file=sys.stderr,
                    )
    return dependencies


def dependency_graph(repo_root: str) -> Tuple[dict, dict]:
    """按目录构建正反依赖，experimental查找规则与add_dependent_ops一致。

    常规算子只查常规目录；experimental算子优先experimental，找不到则回退常规。
    无def的接口/公共目录也保留为图节点，以便传播其反向影响。
    """
    nodes = []
    index = {}
    for category_dir in iter_category_dirs(repo_root):
        for name in sorted(os.listdir(category_dir)):
            path = os.path.join(category_dir, name)
            if not os.path.isdir(path):
                continue
            nodes.append(path)
            namespace = os.path.relpath(path, repo_root).startswith("experimental/")
            index.setdefault((namespace, name), []).append(path)
    forward = {path: set() for path in nodes}
    reverse = {path: set() for path in nodes}
    for path in nodes:
        experimental = os.path.relpath(path, repo_root).startswith("experimental/")
        for name in declared_dependencies(path):
            candidates = index.get((experimental, name), [])
            if not candidates and experimental:
                candidates = index.get((False, name), [])
            if len(candidates) != 1:
                print(
                    f"警告: {os.path.relpath(path, repo_root)} 的依赖 {name!r} "
                    "不存在或目录不唯一，已跳过",
                    file=sys.stderr,
                )
                continue
            forward[path].add(candidates[0])
            reverse[candidates[0]].add(path)
        common = os.path.join(os.path.dirname(path), "common")
        if common in forward and path != common:
            forward[path].add(common)
            reverse[common].add(path)
    return forward, reverse


def dependency_closure(seeds: Set[str], graph: dict) -> Set[str]:
    """遍历依赖图，访问集使环形依赖安全终止。"""
    visited = set(seeds)
    pending = list(seeds)
    while pending:
        node = pending.pop()
        for dependency in graph.get(node, ()):
            if dependency not in visited:
                visited.add(dependency)
                pending.append(dependency)
    return visited


def expand_ops_with_dependencies(
    repo_root: str,
    involved_dirs: Set[str],
    involved_names: Set[str],
    name_prefixes: dict,
) -> Tuple[List[str], List[str]]:
    """正向+反向展开，返回(全部算子目录, 直接涉及算子目录)。

    只把存在def的算子纳入报告。目录级图避免主类目与experimental同名算子互相误伤。
    依赖展开仅提示，默认门禁仍限于直接涉及算子。
    """
    direct = {
        path
        for path in involved_dirs
        if has_def_file(path) and not is_common_lib(os.path.basename(path))
    }
    seeds = set(involved_dirs)
    if not seeds:
        return [], sorted(direct)
    forward, reverse = dependency_graph(repo_root)
    involved = dependency_closure(seeds, forward) | dependency_closure(seeds, reverse)
    reportable = set(iter_op_dirs(repo_root))
    return sorted(involved & reportable), sorted(direct)
