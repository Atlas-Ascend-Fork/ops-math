# ReduceMax

English | [简体中文](./README.md)

## Function Description

- Operator function: computes the maximum value of each slice along the given dimensions, the maximum of all elements, or the maximum over values in the same channel.

For API parameters, data types, and product support, see [aclnnAmax_en](./docs/aclnnAmax_en.md), [aclnnGlobalMaxPool_en](./docs/aclnnGlobalMaxPool_en.md), [aclnnMax_en](./docs/aclnnMax_en.md), and [aclnnMaxV2_en](./docs/aclnnMaxV2_en.md).

## Deliverables

| Deliverable | Location |
| :--- | :--- |
| aclnn API | [aclnn_amax.h](./op_api/aclnn_amax.h), [aclnn_amax.cpp](./op_api/aclnn_amax.cpp), [aclnn_global_max_pool.h](./op_api/aclnn_global_max_pool.h), [aclnn_global_max_pool.cpp](./op_api/aclnn_global_max_pool.cpp), [aclnn_max.h](./op_api/aclnn_max.h), [aclnn_max.cpp](./op_api/aclnn_max.cpp), [aclnn_max_v2.h](./op_api/aclnn_max_v2.h), [aclnn_max_v2.cpp](./op_api/aclnn_max_v2.cpp), with shared implementation [reduce_max.h](./op_api/reduce_max.h), [reduce_max.cpp](./op_api/reduce_max.cpp) |
| Ascend C implementation | [reduce_max_apt.cpp](./op_kernel/reduce_max_apt.cpp) (arch35, depends on [reduce_max_dag.h](./op_kernel/arch35/reduce_max_dag.h), [reduce_max_tiling_key.h](./op_kernel/arch35/reduce_max_tiling_key.h)) |
| Operator prototype | [reduce_max_proto.h](./op_graph/reduce_max_proto.h), [reduce_max_graph_infer.cpp](./op_graph/reduce_max_graph_infer.cpp) |
| API documentation | [aclnnAmax_en.md](./docs/aclnnAmax_en.md), [aclnnGlobalMaxPool_en.md](./docs/aclnnGlobalMaxPool_en.md), [aclnnMax_en.md](./docs/aclnnMax_en.md), [aclnnMaxV2_en.md](./docs/aclnnMaxV2_en.md) |

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_amax](./examples/test_aclnn_amax.cpp) | Invokes the ReduceMax operator through the [aclnnAmax](./docs/aclnnAmax_en.md) API. |
| aclnn invocation | [test_aclnn_global_max_pool](./examples/test_aclnn_global_max_pool.cpp) | Invokes the ReduceMax operator through the [aclnnGlobalMaxPool](./docs/aclnnGlobalMaxPool_en.md) API. |
| aclnn invocation | [test_aclnn_max](./examples/test_aclnn_max.cpp) | Invokes the ReduceMax operator through the [aclnnMax](./docs/aclnnMax_en.md) API. |
| aclnn invocation | [test_aclnn_max_v2](./examples/test_aclnn_max_v2.cpp) | Invokes the ReduceMax operator through the [aclnnMaxV2](./docs/aclnnMaxV2_en.md) API. |
| Graph-mode invocation | - | Invokes the ReduceMax operator by building a graph through the [operator IR](./op_graph/reduce_max_proto.h). |
