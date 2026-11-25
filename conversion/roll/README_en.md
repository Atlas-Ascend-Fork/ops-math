# Roll

English | [简体中文](./README.md)

## Function Description

- Operator function: rolls the elements of the input tensor along the given dimensions by the given shifts.

For API parameters, data types, and product support, see [aclnnRoll_en](./docs/aclnnRoll_en.md).

## Deliverables

| Deliverable | Location |
| :--- | :--- |
| aclnn API | [aclnn_roll.h](./op_api/aclnn_roll.h), [aclnn_roll.cpp](./op_api/aclnn_roll.cpp), with shared implementation [roll.h](./op_api/roll.h), [roll.cpp](./op_api/roll.cpp) |
| Ascend C implementation | [roll_apt.cpp](./op_kernel/roll_apt.cpp) (arch35, depends on [roll_struct.h](./op_kernel/arch35/roll_struct.h), [roll_simd.h](./op_kernel/arch35/roll_simd.h), [roll_simt.h](./op_kernel/arch35/roll_simt.h), [roll_gather_simd.h](./op_kernel/arch35/roll_gather_simd.h), [roll_hsplit.h](./op_kernel/arch35/roll_hsplit.h), [roll_unaligned_simd.h](./op_kernel/arch35/roll_unaligned_simd.h)) |
| Operator prototype | [roll_proto.h](./op_graph/roll_proto.h), [roll_graph_infer.cpp](./op_graph/roll_graph_infer.cpp) |
| API documentation | [aclnnRoll_en.md](./docs/aclnnRoll_en.md) |

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_roll](./examples/test_aclnn_roll.cpp) | Invokes the Roll operator through the [aclnnRoll](./docs/aclnnRoll_en.md) API. |
| Graph-mode invocation | - | Invokes the Roll operator by building a graph through the [operator IR](./op_graph/roll_proto.h). |
