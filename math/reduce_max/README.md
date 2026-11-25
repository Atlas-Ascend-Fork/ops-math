# ReduceMax

## 功能说明

- 算子功能：求取输入张量在指定维度上每个切片的最大值、所有元素中的最大值，或同一通道中的最大值。

接口参数、数据类型及产品支持情况详见[aclnnAmax](./docs/aclnnAmax.md)、[aclnnGlobalMaxPool](./docs/aclnnGlobalMaxPool.md)、[aclnnMax](./docs/aclnnMax.md)、[aclnnMaxV2](./docs/aclnnMaxV2.md)。

## 交付件

| 交付件 | 位置 |
| :--- | :--- |
| aclnn接口 | [aclnn_amax.h](./op_api/aclnn_amax.h)、[aclnn_amax.cpp](./op_api/aclnn_amax.cpp)、[aclnn_global_max_pool.h](./op_api/aclnn_global_max_pool.h)、[aclnn_global_max_pool.cpp](./op_api/aclnn_global_max_pool.cpp)、[aclnn_max.h](./op_api/aclnn_max.h)、[aclnn_max.cpp](./op_api/aclnn_max.cpp)、[aclnn_max_v2.h](./op_api/aclnn_max_v2.h)、[aclnn_max_v2.cpp](./op_api/aclnn_max_v2.cpp)，共享实现[reduce_max.h](./op_api/reduce_max.h)、[reduce_max.cpp](./op_api/reduce_max.cpp) |
| AscendC实现 | [reduce_max_apt.cpp](./op_kernel/reduce_max_apt.cpp)（arch35，依赖[reduce_max_dag.h](./op_kernel/arch35/reduce_max_dag.h)、[reduce_max_tiling_key.h](./op_kernel/arch35/reduce_max_tiling_key.h)） |
| 算子原型 | [reduce_max_proto.h](./op_graph/reduce_max_proto.h)、[reduce_max_graph_infer.cpp](./op_graph/reduce_max_graph_infer.cpp) |
| 接口文档 | [aclnnAmax.md](./docs/aclnnAmax.md)、[aclnnGlobalMaxPool.md](./docs/aclnnGlobalMaxPool.md)、[aclnnMax.md](./docs/aclnnMax.md)、[aclnnMaxV2.md](./docs/aclnnMaxV2.md) |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_amax](./examples/test_aclnn_amax.cpp) | 通过[aclnnAmax](./docs/aclnnAmax.md)接口方式调用ReduceMax算子。 |
| aclnn调用 | [test_aclnn_global_max_pool](./examples/test_aclnn_global_max_pool.cpp) | 通过[aclnnGlobalMaxPool](./docs/aclnnGlobalMaxPool.md)接口方式调用ReduceMax算子。 |
| aclnn调用 | [test_aclnn_max](./examples/test_aclnn_max.cpp) | 通过[aclnnMax](./docs/aclnnMax.md)接口方式调用ReduceMax算子。 |
| aclnn调用 | [test_aclnn_max_v2](./examples/test_aclnn_max_v2.cpp) | 通过[aclnnMaxV2](./docs/aclnnMaxV2.md)接口方式调用ReduceMax算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/reduce_max_proto.h)构图方式调用ReduceMax算子。 |
