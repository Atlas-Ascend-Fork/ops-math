# ReduceLogSum

## 功能说明

- 算子功能：返回给定维度中输入张量每行的和，再对和取对数。

接口参数、数据类型及产品支持情况详见[aclnnReduceLogSum](./docs/aclnnReduceLogSum.md)。

## 交付件

| 交付件 | 位置 |
| :--- | :--- |
| aclnn接口 | [aclnn_reduce_log_sum.h](./op_api/aclnn_reduce_log_sum.h)、[aclnn_reduce_log_sum.cpp](./op_api/aclnn_reduce_log_sum.cpp)，共享实现[reduce_log_sum.h](./op_api/reduce_log_sum.h)、[reduce_log_sum.cpp](./op_api/reduce_log_sum.cpp) |
| AscendC实现 | [reduce_log_sum_apt.cpp](./op_kernel/reduce_log_sum_apt.cpp)（arch35，依赖[reduce_log_sum_dag.h](./op_kernel/arch35/reduce_log_sum_dag.h)、[reduce_log_sum_tiling_key.h](./op_kernel/arch35/reduce_log_sum_tiling_key.h)） |
| 接口文档 | [aclnnReduceLogSum.md](./docs/aclnnReduceLogSum.md) |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_reduce_log_sum](./examples/test_aclnn_reduce_log_sum.cpp) | 通过[aclnnReduceLogSum](./docs/aclnnReduceLogSum.md)接口方式调用ReduceLogSum算子。 |
