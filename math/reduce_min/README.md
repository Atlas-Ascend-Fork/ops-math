# ReduceMin

## 功能说明

- 算子功能：求取输入张量在指定维度上每个切片的最小值，或在指定维度、所有维度上同时返回最小值与最大值，或返回所有元素中的最小值。

接口参数、数据类型及产品支持情况详见[aclnnAmin](./docs/aclnnAmin.md)、[aclnnAminmax](./docs/aclnnAminmax.md)、[aclnnAminmaxAll](./docs/aclnnAminmaxAll.md)、[aclnnAminmaxDim](./docs/aclnnAminmaxDim.md)、[aclnnMin](./docs/aclnnMin.md)。

## 交付件

| 交付件 | 位置 |
| :--- | :--- |
| aclnn接口 | [aclnn_amin.h](./op_api/aclnn_amin.h)、[aclnn_amin.cpp](./op_api/aclnn_amin.cpp)、[aclnn_aminmax.h](./op_api/aclnn_aminmax.h)、[aclnn_aminmax.cpp](./op_api/aclnn_aminmax.cpp)、[aclnn_aminmax_all.h](./op_api/aclnn_aminmax_all.h)、[aclnn_aminmax_all.cpp](./op_api/aclnn_aminmax_all.cpp)、[aclnn_aminmax_dim.h](./op_api/aclnn_aminmax_dim.h)、[aclnn_aminmax_dim.cpp](./op_api/aclnn_aminmax_dim.cpp)、[aclnn_min.h](./op_api/aclnn_min.h)、[aclnn_min.cpp](./op_api/aclnn_min.cpp)，共享实现[reduce_min.h](./op_api/reduce_min.h)、[reduce_min.cpp](./op_api/reduce_min.cpp) |
| AscendC实现 | [reduce_min_apt.cpp](./op_kernel/reduce_min_apt.cpp)（arch35，依赖[reduce_min_dag.h](./op_kernel/arch35/reduce_min_dag.h)、[reduce_min_tiling_key.h](./op_kernel/arch35/reduce_min_tiling_key.h)） |
| 算子原型 | [reduce_min_proto.h](./op_graph/reduce_min_proto.h)、[reduce_min_graph_infer.cpp](./op_graph/reduce_min_graph_infer.cpp) |
| 接口文档 | [aclnnAmin.md](./docs/aclnnAmin.md)、[aclnnAminmax.md](./docs/aclnnAminmax.md)、[aclnnAminmaxAll.md](./docs/aclnnAminmaxAll.md)、[aclnnAminmaxDim.md](./docs/aclnnAminmaxDim.md)、[aclnnMin.md](./docs/aclnnMin.md) |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_amin](./examples/test_aclnn_amin.cpp) | 通过[aclnnAmin](./docs/aclnnAmin.md)接口方式调用ReduceMin算子。 |
| aclnn调用 | [test_aclnn_aminmax](./examples/test_aclnn_aminmax.cpp) | 通过[aclnnAminmax](./docs/aclnnAminmax.md)接口方式调用ReduceMin算子。 |
| aclnn调用 | [test_aclnn_aminmax_all](./examples/test_aclnn_aminmax_all.cpp) | 通过[aclnnAminmaxAll](./docs/aclnnAminmaxAll.md)接口方式调用ReduceMin算子。 |
| aclnn调用 | [test_aclnn_aminmax_dim](./examples/test_aclnn_aminmax_dim.cpp) | 通过[aclnnAminmaxDim](./docs/aclnnAminmaxDim.md)接口方式调用ReduceMin算子。 |
| aclnn调用 | [test_aclnn_min](./examples/test_aclnn_min.cpp) | 通过[aclnnMin](./docs/aclnnMin.md)接口方式调用ReduceMin算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/reduce_min_proto.h)构图方式调用ReduceMin算子。 |
