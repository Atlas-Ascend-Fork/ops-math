# ReduceMin

English | [简体中文](./README.md)

## Function Description

- Operator function: computes the minimum value of each slice along the given dimensions, the minimum and maximum along the given or all dimensions, or the minimum of all elements.

For API parameters, data types, and product support, see [aclnnAmin_en](./docs/aclnnAmin_en.md), [aclnnAminmax_en](./docs/aclnnAminmax_en.md), [aclnnAminmaxAll_en](./docs/aclnnAminmaxAll_en.md), [aclnnAminmaxDim_en](./docs/aclnnAminmaxDim_en.md), and [aclnnMin_en](./docs/aclnnMin_en.md).

## Deliverables

| Deliverable | Location |
| :--- | :--- |
| aclnn API | [aclnn_amin.h](./op_api/aclnn_amin.h), [aclnn_amin.cpp](./op_api/aclnn_amin.cpp), [aclnn_aminmax.h](./op_api/aclnn_aminmax.h), [aclnn_aminmax.cpp](./op_api/aclnn_aminmax.cpp), [aclnn_aminmax_all.h](./op_api/aclnn_aminmax_all.h), [aclnn_aminmax_all.cpp](./op_api/aclnn_aminmax_all.cpp), [aclnn_aminmax_dim.h](./op_api/aclnn_aminmax_dim.h), [aclnn_aminmax_dim.cpp](./op_api/aclnn_aminmax_dim.cpp), [aclnn_min.h](./op_api/aclnn_min.h), [aclnn_min.cpp](./op_api/aclnn_min.cpp), with shared implementation [reduce_min.h](./op_api/reduce_min.h), [reduce_min.cpp](./op_api/reduce_min.cpp) |
| Ascend C implementation | [reduce_min_apt.cpp](./op_kernel/reduce_min_apt.cpp) (arch35, depends on [reduce_min_dag.h](./op_kernel/arch35/reduce_min_dag.h), [reduce_min_tiling_key.h](./op_kernel/arch35/reduce_min_tiling_key.h)) |
| Operator prototype | [reduce_min_proto.h](./op_graph/reduce_min_proto.h), [reduce_min_graph_infer.cpp](./op_graph/reduce_min_graph_infer.cpp) |
| API documentation | [aclnnAmin_en.md](./docs/aclnnAmin_en.md), [aclnnAminmax_en.md](./docs/aclnnAminmax_en.md), [aclnnAminmaxAll_en.md](./docs/aclnnAminmaxAll_en.md), [aclnnAminmaxDim_en.md](./docs/aclnnAminmaxDim_en.md), [aclnnMin_en.md](./docs/aclnnMin_en.md) |

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_amin](./examples/test_aclnn_amin.cpp) | Invokes the ReduceMin operator through the [aclnnAmin](./docs/aclnnAmin_en.md) API. |
| aclnn invocation | [test_aclnn_aminmax](./examples/test_aclnn_aminmax.cpp) | Invokes the ReduceMin operator through the [aclnnAminmax](./docs/aclnnAminmax_en.md) API. |
| aclnn invocation | [test_aclnn_aminmax_all](./examples/test_aclnn_aminmax_all.cpp) | Invokes the ReduceMin operator through the [aclnnAminmaxAll](./docs/aclnnAminmaxAll_en.md) API. |
| aclnn invocation | [test_aclnn_aminmax_dim](./examples/test_aclnn_aminmax_dim.cpp) | Invokes the ReduceMin operator through the [aclnnAminmaxDim](./docs/aclnnAminmaxDim_en.md) API. |
| aclnn invocation | [test_aclnn_min](./examples/test_aclnn_min.cpp) | Invokes the ReduceMin operator through the [aclnnMin](./docs/aclnnMin_en.md) API. |
| Graph-mode invocation | - | Invokes the ReduceMin operator by building a graph through the [operator IR](./op_graph/reduce_min_proto.h). |
