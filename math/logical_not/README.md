# LogicalNot

## 功能说明

- 算子功能：计算输入Tensor的逐元素逻辑非，零视为False，非零视为True。

- 计算公式：

  $$
  out[i]=
  \begin{cases}
  1, & \text{if } self[i]=0 \\
  0, & \text{if } self[i] \ne 0
  \end{cases}
  $$

接口参数、数据类型及产品支持情况详见[aclnnLogicalNot&aclnnInplaceLogicalNot](./docs/aclnnLogicalNot&aclnnInplaceLogicalNot.md)。

## 交付件

| 交付件 | 位置 |
| :--- | :--- |
| aclnn接口 | [aclnn_logical_not.h](./op_api/aclnn_logical_not.h)、[aclnn_logical_not.cpp](./op_api/aclnn_logical_not.cpp) |
| AscendC实现 | [logical_not_apt.cpp](./op_kernel/logical_not_apt.cpp)（arch35，依赖[logical_not_dag.h](./op_kernel/arch35/logical_not_dag.h)） |
| 算子原型 | [logical_not_proto.h](./op_graph/logical_not_proto.h) |
| 接口文档 | [aclnnLogicalNot&aclnnInplaceLogicalNot.md](./docs/aclnnLogicalNot&aclnnInplaceLogicalNot.md) |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_logical_not](./examples/test_aclnn_logical_not.cpp) | 通过aclnnLogicalNot或aclnnInplaceLogicalNot接口调用LogicalNot算子。 |
