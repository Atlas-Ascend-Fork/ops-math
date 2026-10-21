# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_modules_sources(OPTYPE squeeze ACLNNTYPE aclnn_exclude)

# collect the def source introduced by the migration, the legacy add_modules_sources has no
# def collection (mirrors the opdef target of add_all_modules_sources)
file(GLOB OPDEF_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/squeeze_def*.cpp)
if(OPDEF_SRCS)
    target_sources(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE ${OPDEF_SRCS})
endif()
