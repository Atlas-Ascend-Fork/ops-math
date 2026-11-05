# ExpandDims

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>                                 |    √     |
| <term>Atlas A2 products</term>                                 |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                        |    √     |
| <term>Atlas inference products</term>                                |    √     |
| <term>Atlas training products</term>                                |    √     |

## Function Description

- Operator function: inserts a dimension of size 1 at the specified position in the shape of the input tensor. It only changes the shape of the tensor and does not change the underlying data. The operator is executed by the graph engine on the host side (Host CPU) and is processed as a memory reuse (reuse-input) operator. The output directly views the input memory.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | BOOL, FLOAT16, FLOAT, INT4, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, COMPLEX64 | ND, FRACTAL_NZ |
| axis | Input | Position index for inserting the dimension. It is a scalar or a single-element tensor and supports negative indices (`[-rank-1, rank]`). The value depends on the input. | INT32, INT64 | ND, FRACTAL_NZ |
| y | Output | Output tensor after the dimension is inserted. Its data type is the same as `x`. | Same as `x` | ND, FRACTAL_NZ |

## Constraint Description

- The value range of `axis` is `[-(rank(x)-1), rank(x)]`. If the value is out of the range, inference fails.
- `axis` must be a scalar or a single-element tensor.
- The data type of `axis` must be INT32 or INT64.
- If the value of `axis` is unknown at compile time, the output shape is the rank of the input plus 1 unknown dimensions (-1).
- For supported data types, refer to the IR definitions in [`expand_dims_proto.h`](op_graph/expand_dims_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_expand_dims](examples/test_geir_expand_dims.cpp) | Invokes the ExpandDims operator by building a graph through the [operator IR](op_graph/expand_dims_proto.h). |
