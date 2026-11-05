# GetDynamicDims

English | [简体中文](./README.md)

## Function Description

The GetDynamicDims operator extracts actual dimension values from dynamic input tensors according to the unknown dimensions marked in `shape_info` and outputs the values.

## Parameter Description

| Parameter Name | Input/Output | Parameter Type | Data Type | Description |
| - | - | - | - | - |
| input | Input | Dynamic tensor input | int32, int64 | Input tensor. Each input represents a shape vector. |
| dims | Output | Tensor | int32, int64 | Outputs the actual values of the unknown dimensions. The data type is the same as that of `input`. |
| shape_info | Attribute | ListInt | int64 | Shape information of each input, encoded as `rank, dim0, dim1, ...`, where `-1` indicates an unknown dimension. |
| N | Attribute | Int | int64 | Number of dynamic inputs. |

## Constraint Description

- The number of `input` inputs must be consistent with the attribute `N`.
- The data types of all `input` inputs must be the same.
- The number of inputs and the rank of each input described in `shape_info` must be consistent with the actual inputs.

## Invocation Description

| Invocation Method | Sample Link | Description |
| - | - | - |
| Graph-mode invocation | [test_geir_get_dynamic_dims](./examples/test_geir_get_dynamic_dims.cpp) | Invokes the GetDynamicDims operator by building a graph through the [operator IR](./op_graph/get_dynamic_dims_proto.h). |
