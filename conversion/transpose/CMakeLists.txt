# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    OPTYPE transpose
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT dav_2002 dav_2201 dav_3510 mc62 dav_9201 dav_9202 kirinx90 kirin9030 ascend5162
    TILING_DIR arch22 arch22 arch35 arch35 arch35 arch35 arch31 arch31 arch35
    DISABLE_IN_OPP TRUE
)

add_kernel_sources(
    KERNEL_SRC arch31/transpose_apt.cpp
    COMPUTE_UNITS kirinx90 kirin9030
    AUTO_SYNC false
)
