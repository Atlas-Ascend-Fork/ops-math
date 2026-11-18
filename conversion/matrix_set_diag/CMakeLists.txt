# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    OPTYPE matrix_set_diag
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT dav_3510 dav_9201 dav_9202
    TILING_DIR arch35 arch35 arch35
    DEPENDENCIES matrix_set_diag_v2
)

add_kernel_sources(
    KERNEL_SRC matrix_set_diag_apt.cpp
    COMPUTE_UNITS dav_3510 dav_9201
    AUTO_SYNC false
    OPTIONS "-mllvm -cce-aicore-dcci-before-kernel-end=false"
)
