# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 本算子仅在 ascend910b/ascend910_93/ascend950 注册 tiling，未注册 soc 不编译 tiling 源。
# add_kernel_sources 仅在 add_all_modules_sources 未提前 return（即本算子被选中）时执行，
# 故无需额外的 NEED_COMPILE_OPS 守卫；AUTO_SYNC 缺省即 true，与 ascendc_config.json 一致，无需显式声明。
set(SUPPORT_COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950")
set(SUPPORT_TILING_DIR "arch22" "arch22" "arch35")
add_all_modules_sources(
    OPTYPE sinkhorn
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
)

add_kernel_sources(
    KERNEL_SRC arch35/sinkhorn_arch35.cpp
    COMPUTE_UNITS ascend950
)
