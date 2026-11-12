# LogicalNot

English | [简体中文](./README.md)

## Function Description

- Operator function: computes the element-wise logical NOT of the input tensor. Zero is regarded as False and non-zero as True.

- Formula:

  $$
  out[i]=
  \begin{cases}
  1, & \text{if } self[i]=0 \\
  0, & \text{if } self[i] \ne 0
  \end{cases}
  $$

For API parameters, data types, and product support, see [aclnnLogicalNot&aclnnInplaceLogicalNot_en](./docs/aclnnLogicalNot&aclnnInplaceLogicalNot_en.md).

## Deliverables

| Deliverable | Location |
| :--- | :--- |
| aclnn API | [aclnn_logical_not.h](./op_api/aclnn_logical_not.h), [aclnn_logical_not.cpp](./op_api/aclnn_logical_not.cpp) |
| Ascend C implementation | [logical_not_apt.cpp](./op_kernel/logical_not_apt.cpp) (arch35, depends on [logical_not_dag.h](./op_kernel/arch35/logical_not_dag.h)) |
| Operator prototype | [logical_not_proto.h](./op_graph/logical_not_proto.h) |
| API documentation | [aclnnLogicalNot&aclnnInplaceLogicalNot_en.md](./docs/aclnnLogicalNot&aclnnInplaceLogicalNot_en.md) |

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_logical_not](./examples/test_aclnn_logical_not.cpp) | Invokes the LogicalNot operator through the aclnnLogicalNot or aclnnInplaceLogicalNot API. |
| Graph-mode invocation | - | Invokes the LogicalNot operator by building a graph through the [operator IR](./op_graph/logical_not_proto.h). |
