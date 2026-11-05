# BroadcastTo

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: broadcasts the input tensor to the specified shape. If the input tensor has the shape (1, 4) and the specified shape is (2, 4), the output is a tensor with the shape (2, 4).

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                                         | Data Type | Data Format |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | Input tensor       | Tensor to be broadcast.  | INT8, INT16, INT32, INT64, UINT8, UINT32, FLOAT16, BF16, FLOAT, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN   | ND       |
| shape   | Input tensor       | Shape of x after broadcasting.| INT64, INT32    | -        |
| y    | Output           | The number of dimensions does not exceed 8, the shape is determined by the shape input, and the dtype must be the same as x. | Same as x   | ND       |

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Graph-mode invocation | [test_geir_broadcast_to](./examples/test_geir_broadcast_to.cpp)   | Invokes the broadcast_to operator by building a graph through the [operator IR](./op_graph/broadcast_to_proto.h). |
