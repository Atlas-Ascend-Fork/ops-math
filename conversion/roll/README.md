# Roll

## 功能说明

- 算子功能：沿给定维度按给定的移位量循环移动Tensor中的元素。

接口参数、数据类型及产品支持情况详见[aclnnRoll](./docs/aclnnRoll.md)。

## 交付件

| 交付件 | 位置 |
| :--- | :--- |
| aclnn接口 | [aclnn_roll.h](./op_api/aclnn_roll.h)、[aclnn_roll.cpp](./op_api/aclnn_roll.cpp)，共享实现[roll.h](./op_api/roll.h)、[roll.cpp](./op_api/roll.cpp) |
| AscendC实现 | [roll_apt.cpp](./op_kernel/roll_apt.cpp)（arch35，依赖[roll_struct.h](./op_kernel/arch35/roll_struct.h)、[roll_simd.h](./op_kernel/arch35/roll_simd.h)、[roll_simt.h](./op_kernel/arch35/roll_simt.h)、[roll_gather_simd.h](./op_kernel/arch35/roll_gather_simd.h)、[roll_hsplit.h](./op_kernel/arch35/roll_hsplit.h)、[roll_unaligned_simd.h](./op_kernel/arch35/roll_unaligned_simd.h)） |
| 算子原型 | [roll_proto.h](./op_graph/roll_proto.h)、[roll_graph_infer.cpp](./op_graph/roll_graph_infer.cpp) |
| 接口文档 | [aclnnRoll.md](./docs/aclnnRoll.md) |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_roll](./examples/test_aclnn_roll.cpp) | 通过[aclnnRoll](./docs/aclnnRoll.md)接口方式调用Roll算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/roll_proto.h)构图方式调用Roll算子。 |
