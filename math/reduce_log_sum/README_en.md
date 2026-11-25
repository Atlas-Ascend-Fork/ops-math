# ReduceLogSum

English | [简体中文](./README.md)

## Function Description

- Operator function: returns the logarithm of the sum of the input tensor elements along the given dimensions.

For API parameters, data types, and product support, see [aclnnReduceLogSum_en](./docs/aclnnReduceLogSum_en.md).

## Deliverables

| Deliverable | Location |
| :--- | :--- |
| aclnn API | [aclnn_reduce_log_sum.h](./op_api/aclnn_reduce_log_sum.h), [aclnn_reduce_log_sum.cpp](./op_api/aclnn_reduce_log_sum.cpp), with shared implementation [reduce_log_sum.h](./op_api/reduce_log_sum.h), [reduce_log_sum.cpp](./op_api/reduce_log_sum.cpp) |
| Ascend C implementation | [reduce_log_sum_apt.cpp](./op_kernel/reduce_log_sum_apt.cpp) (arch35, depends on [reduce_log_sum_dag.h](./op_kernel/arch35/reduce_log_sum_dag.h), [reduce_log_sum_tiling_key.h](./op_kernel/arch35/reduce_log_sum_tiling_key.h)) |
| API documentation | [aclnnReduceLogSum_en.md](./docs/aclnnReduceLogSum_en.md) |

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_reduce_log_sum](./examples/test_aclnn_reduce_log_sum.cpp) | Invokes the ReduceLogSum operator through the [aclnnReduceLogSum](./docs/aclnnReduceLogSum_en.md) API. |
